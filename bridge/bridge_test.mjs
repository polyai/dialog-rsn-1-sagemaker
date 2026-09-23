/**
 * node bridge_test.mjs
 *
 * Drives the pump with a stub client, so everything except the AWS transport is covered without
 * an endpoint: frames reach the request Body, responses reach the socket, and a stream error
 * closes the socket instead of hanging it.
 */

import assert from "node:assert/strict";
import { WebSocket } from "ws";
import { startBridge } from "./bridge.mjs";

const PORT = 8477;

/**
 * Emits `lead` (a stream error, say), then echoes every request frame back.
 *
 * `lead` goes first because the echo loop only ends when the client closes the socket, so
 * anything emitted after it would never be reached by a test that waits to be closed instead.
 */
function stubClient(lead = []) {
  const seen = [];
  return {
    seen,
    async send(command) {
      const body = command.input.Body;
      return {
        Body: (async function* () {
          yield* lead;
          for await (const frame of body) {
            seen.push(frame);
            yield { PayloadPart: { Bytes: frame.PayloadPart.Bytes, DataType: "UTF8" } };
          }
        })(),
      };
    },
  };
}

function connect() {
  const ws = new WebSocket(`ws://127.0.0.1:${PORT}`);
  return new Promise((resolve) => ws.on("open", () => resolve(ws)));
}

async function withBridge(client, run) {
  const wss = startBridge({ endpointName: "stub", port: PORT, client });
  try {
    return await run();
  } finally {
    wss.close();
  }
}

async function testFramesRoundTrip() {
  const client = stubClient();
  await withBridge(client, async () => {
    const ws = await connect();
    const received = [];
    ws.on("message", (data) => received.push(data.toString()));

    ws.send('{"type":"session.update"}');
    ws.send('{"type":"response.create"}');
    await new Promise((r) => setTimeout(r, 100));
    ws.close();
    await new Promise((r) => setTimeout(r, 50));

    assert.deepEqual(received, ['{"type":"session.update"}', '{"type":"response.create"}']);
    assert.equal(client.seen.length, 2, "both frames should reach the request Body");
    assert.equal(client.seen[0].PayloadPart.DataType, "UTF8");
    assert.equal(client.seen[0].PayloadPart.CompletionState, "COMPLETE");
  });
}

async function testBinaryFramesKeepTheirType() {
  const client = stubClient();
  await withBridge(client, async () => {
    const ws = await connect();
    ws.send(Buffer.from([0, 1, 2, 3]));
    await new Promise((r) => setTimeout(r, 100));
    ws.close();
    await new Promise((r) => setTimeout(r, 50));

    assert.equal(client.seen[0].PayloadPart.DataType, "BINARY");
  });
}

async function testStreamErrorClosesTheSocket() {
  // The failure the bridge exists to make visible: without this the client just stops hearing back.
  const client = stubClient([{ ModelStreamError: { Message: "model exploded" } }]);
  await withBridge(client, async () => {
    const ws = await connect();
    const closed = new Promise((resolve) => ws.on("close", (code, reason) => resolve({ code, reason: reason.toString() })));

    ws.send('{"type":"response.create"}');
    const { code, reason } = await closed;

    assert.equal(code, 1011);
    assert.equal(reason, "model exploded");
  });
}

for (const test of [testFramesRoundTrip, testBinaryFramesKeepTheirType, testStreamErrorClosesTheSocket]) {
  await test();
  console.log(`ok - ${test.name}`);
}
