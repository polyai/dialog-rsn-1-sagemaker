# Dialog-RSN-1 on Amazon SageMaker

Sample code for the [Dialog-RSN-1](https://poly.ai) model package on AWS Marketplace.

Dialog-RSN-1 is PolyAI's audio-native language model for customer-service voice agents. It listens
to caller audio directly, decides when the caller has finished speaking, and answers with text or a
tool call in one pass. It does not synthesise speech. You pair it with the text-to-speech engine of
your choice.

The endpoint speaks the OpenAI Realtime wire protocol. Any OpenAI Realtime client works against it
by changing one thing, the base URL. This repository gives you the two pieces that make that true
on SageMaker, plus a notebook that walks through the whole thing.

## What is here

| Path | What it is |
| --- | --- |
| `dialog_rsn_1_sagemaker.ipynb` | The sample notebook. Subscribe, deploy, hold three conversations, delete everything. |
| `bridge/` | A small Node process that exposes a SageMaker endpoint as a local WebSocket. See below. |
| `audio/` | Short 16 kHz PCM16 WAV files the notebook speaks to the model. Synthetic voices. |
| `requirements.txt` | The Python side: the `openai` SDK, pinned, and `boto3`. |

## Why there is a bridge

SageMaker gives you no `wss://` URL. A bidirectional-streaming endpoint is reached by calling
`InvokeEndpointWithBidirectionalStream` over HTTP/2 with SigV4 signing. The AWS SDKs do that. An
OpenAI Realtime client opens a WebSocket.

The bridge closes that gap. It listens on `ws://127.0.0.1:8079`, opens one SageMaker stream per
connection, and copies frames across unchanged in both directions. Your client and the model each
see the protocol they expect. Credentials come from the standard AWS credential chain, so
there is nothing to configure beyond the endpoint name.

The bridge has no authentication of its own. It listens on 127.0.0.1 only, but any process on the
same machine can connect and spend your AWS credentials through it. Run it on a machine you
control, and stop it when you are done.

It is Node rather than Python because the SageMaker runtime serves HTTP/2 without advertising it
via ALPN. Node connects by prior knowledge. The Python AWS SDK cannot.

You do not have to use it. The bridge is a pure pass-through, so a client that calls
`InvokeEndpointWithBidirectionalStream` directly with `@aws-sdk/client-sagemaker-runtime-http2`
and carries the same JSON events in `PayloadPart` works too.

## Prerequisites

- An AWS account subscribed to Dialog-RSN-1 on AWS Marketplace, and the model package ARN for your
  region from the listing's configuration page.
- An IAM role with `AmazonSageMakerFullAccess`. On SageMaker Studio or a notebook instance that is
  the execution role. On a laptop, set `SAGEMAKER_ROLE_ARN`.
- Service quota of at least 1 for the instance type you deploy on, under "endpoint usage". A fresh
  account has 0 and the endpoint fails several minutes in rather than immediately.
- `ml.p4d.24xlarge` or `ml.p5.48xlarge`, with the inference AMI the notebook sets. SageMaker has
  at times refused that AMI on p4d with `Invalid combination: instance p4d and InferenceAmiVersion`
  and accepted the same request days later. If you see it, retry, or use p5.
- Node.js 20 or newer for the bridge, and Python 3.10 or newer for the notebook.

## Quick start

```bash
git clone https://github.com/polyai/dialog-rsn-1-sagemaker
cd dialog-rsn-1-sagemaker
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt jupyter
cp .env.example .env            # then fill in the values
jupyter notebook dialog_rsn_1_sagemaker.ipynb
```

Run the cells in order. The notebook creates the endpoint, starts the bridge for you, talks to the
model, and tears everything down at the end.

An endpoint bills per hour from the moment it is `InService` until you delete it, whether or not
anything is connected. The last section of the notebook deletes it. If you stop partway, delete it
yourself.

## Running the bridge on its own

```bash
cd bridge && npm install
SAGEMAKER_ENDPOINT=dialog-rsn-1 AWS_REGION=us-east-2 node bridge.mjs
# bridge: ws://127.0.0.1:8079 -> dialog-rsn-1
```

Then point any OpenAI Realtime client at it. In Python:

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key="not-required", websocket_base_url="ws://127.0.0.1:8079")
async with client.realtime.connect(model="dialog-rsn-1") as connection:
    ...
```

The SDK appends `/realtime` to the base URL itself. The bridge ignores the path and the API key.
Authentication happened already, on the SigV4 call the bridge made to SageMaker.

`npm test` runs the bridge's own tests against a stub, with no endpoint involved.

## What the model accepts and returns

| | |
| --- | --- |
| Audio in | PCM16, mono, 16 kHz or 24 kHz, base64 in `input_audio_buffer.append`. No G.711, Opus or stereo. |
| Text in | `conversation.item.create` with an `input_text` part also works. |
| Context | System instructions, function tools with JSON-schema arguments, knowledge topics via the `poly_knowledge` extension, and results of tool calls. |
| Out | Streamed reply text, or one function call with complete JSON arguments, then `response.done`. A transcript of the caller's speech arrives separately. |
| Not out | Audio. `output_modalities` is `["text"]` and a request for audio output is refused. |
| Context window | 32,000 tokens. Old audio is compacted to its transcript automatically; a conversation that is long in text gets `context_length_exceeded`. |
| Session | 30 minutes per connection, a hard cap SageMaker enforces. Reconnect before it and replay what matters. An idle connection is closed after 5 minutes. |
| Language | English by default. Set `poly_response_language` on the session to an ISO 639 tag such as `pt-BR`, or plain words such as "Southern US English", and the model answers in that language. |

The 30-minute cap is SageMaker's, not the model's. AWS does not document it; PolyAI measured it on
live endpoints. The 5-minute idle close is the model server's own.

PolyAI's additions to the protocol are fields on events the protocol already has, never new event
types, so a stock SDK reads them from `model_extra` and ignores what it does not know:

- `session.poly_knowledge`, a list of `{name, content}` topics the model may answer from.
  The same list can travel on a `function_call_output` item, which is what a knowledge lookup
  tool returns.
- `response.poly_cited_topics` on `response.done`, the topic names the answer came from.
- `response.poly_out_of_domain` on `response.done`, whether the model judged the request outside
  this deployment's scope.
- `session.poly_response_language`, the language the model answers in. Unset, the instructions
  decide.

## Building a voice agent on top

The notebook shows the conversation loop. A real agent adds three things:

1. **Text-to-speech.** Pass each `response.output_text.done` (or the deltas) to your synthesiser.
2. **Barge-in.** When `input_audio_buffer.speech_started` arrives while a reply is playing, stop
   playback. The model has already decided the caller is interrupting.
3. **Tool execution.** Run the function the model asked for in your own code, validate the
   arguments, and send the result back with `conversation.item.create`.

Validate exact values such as dates, reference numbers and amounts in your application before
acting on them.

## Licence

Apache 2.0. See `LICENSE`.
