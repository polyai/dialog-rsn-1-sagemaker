/**
 * A local WebSocket that fronts a SageMaker bidirectional-streaming endpoint.
 *
 * SageMaker hands the caller no `wss://` URL: frames travel over
 * `InvokeEndpointWithBidirectionalStream`, HTTP/2 with SigV4. This listens on localhost, opens one
 * bidi stream per connection, and pumps frames across unchanged, so a stock OpenAI Realtime client
 * connects to `ws://127.0.0.1:PORT` and neither it nor the server knows the difference.
 *
 * Node rather than Python, and not by preference: `runtime.sagemaker.<region>.amazonaws.com`
 * serves HTTP/2 without advertising `h2` in ALPN. Node speaks h2 by prior knowledge, while awscrt
 * -- which the Python SDK requires for duplex -- decides the version from ALPN alone and refuses
 * with AWS_ERROR_HTTP_UNSUPPORTED_PROTOCOL. Verified against `awscrt.aio.http`: the same call
 * connects to a host that does advertise h2. There is no prior-knowledge option to reach for.
 */

import {
  InvokeEndpointWithBidirectionalStreamCommand,
  SageMakerRuntimeHTTP2Client,
} from "@aws-sdk/client-sagemaker-runtime-http2";
import { realpathSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { WebSocketServer } from "ws";

/** An async iterable a producer pushes into: the request Body the SDK reads frames from. */
function frameQueue() {
  const pending = [];
  let wake = null;
  let closed = false;

  return {
    push(frame) {
      pending.push(frame);
      wake?.();
      wake = null;
    },
    close() {
      closed = true;
      wake?.();
      wake = null;
    },
    async *[Symbol.asyncIterator]() {
      for (;;) {
        if (pending.length) {
          yield pending.shift();
        } else if (closed) {
          return;
        } else {
          await new Promise((resolve) => {
            wake = resolve;
          });
        }
      }
    },
  };
}

async function pump(ws, client, endpointName) {
  const body = frameQueue();
  // Closing the request Body is not enough on its own: the response loop below keeps reading until
  // the endpoint ends the stream, holding an HTTP/2 stream and a session slot until the idle close.
  // Aborting cancels the SageMaker stream as soon as the client goes.
  const abort = new AbortController();
  const hangUp = () => {
    body.close();
    abort.abort();
  };

  ws.on("message", (data, isBinary) => {
    body.push({
      PayloadPart: {
        Bytes: new Uint8Array(data),
        DataType: isBinary ? "BINARY" : "UTF8",
        // Each WebSocket frame is a whole payload; the protocol has no partial-frame concept.
        CompletionState: "COMPLETE",
      },
    });
  });
  ws.on("close", hangUp);
  ws.on("error", hangUp);

  try {
    const response = await client.send(
      new InvokeEndpointWithBidirectionalStreamCommand({
        EndpointName: endpointName,
        Body: body,
      }),
      { abortSignal: abort.signal },
    );

    for await (const event of response.Body) {
      const part = event.PayloadPart;
      if (part?.Bytes) {
        ws.send(Buffer.from(part.Bytes), { binary: part.DataType === "BINARY" });
        continue;
      }
      // A stream error is the endpoint failing mid-conversation. Surfacing it as a close with a
      // reason is the whole point of the bridge -- the alternative is a client that hangs.
      const failure = event.ModelStreamError ?? event.InternalStreamFailure;
      if (failure) {
        ws.close(1011, String(failure.Message ?? "stream error").slice(0, 120));
        return;
      }
    }
    ws.close(1000);
  } catch (error) {
    if (abort.signal.aborted) return; // the client left first; there is no one to tell
    ws.close(1011, String(error.message ?? error).slice(0, 120));
  }
}

/** Returns the listening WebSocketServer. `client` is injectable so the pump is testable. */
export function startBridge({ endpointName, port = 8079, region, client }) {
  const runtime = client ?? new SageMakerRuntimeHTTP2Client({ region });
  const wss = new WebSocketServer({ host: "127.0.0.1", port });
  wss.on("connection", (ws) => pump(ws, runtime, endpointName));
  return wss;
}

// Run directly, not imported by the tests. realpath because Node resolves a symlinked entry point
// in import.meta.url but not in argv; fileURLToPath because a file:// URL is not a Windows path.
if (process.argv[1] && realpathSync(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const endpointName = process.env.SAGEMAKER_ENDPOINT;
  if (!endpointName) {
    console.error("SAGEMAKER_ENDPOINT is required");
    process.exit(2);
  }
  const port = Number(process.env.PORT ?? 8079);
  startBridge({
    endpointName,
    port,
    region: process.env.AWS_REGION ?? "us-east-2",
  });
  console.error(`bridge: ws://127.0.0.1:${port} -> ${endpointName}`);
}
