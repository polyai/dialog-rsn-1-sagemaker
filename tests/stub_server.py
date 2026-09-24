"""A fake Realtime server: just enough of the protocol to drive the notebook's conversation cells.

Acknowledges `session.update`, counts the audio it is sent, and on `response.create` emits one
function call the first time tools are declared, canned text otherwise. Adds the `poly_*` fields
the notebook reads. It proves the client's protocol handling, not the model.

    python tests/stub_server.py 8079
"""

import asyncio
import base64
import json
import sys

import websockets


def log(*args) -> None:
    print("stub:", *args, file=sys.stderr, flush=True)


async def handle(ws) -> None:
    session = {
        "type": "realtime",
        "model": "dialog-rsn-1",
        "output_modalities": ["text"],
        "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000}}},
    }
    await ws.send(json.dumps({"type": "session.created", "session": session}))

    audio_bytes, n_responses, tool_calls_made, item_n = 0, 0, 0, 0
    pending_text = None

    async for raw in ws:
        event = json.loads(raw)
        kind = event["type"]

        if kind == "session.update":
            session.update(event["session"])
            await ws.send(json.dumps({"type": "session.updated", "session": session}))

        elif kind == "input_audio_buffer.append":
            if audio_bytes == 0:
                await ws.send(
                    json.dumps({"type": "input_audio_buffer.speech_started", "item_id": f"item_{item_n}", "audio_start_ms": 32})
                )
            audio_bytes += len(base64.b64decode(event["audio"]))

        elif kind == "conversation.item.create":
            item = event["item"]
            if item["type"] == "message":
                pending_text = item["content"][0]["text"]
            elif item["type"] == "function_call_output":
                log("tool result", item["output"])

        elif kind == "response.create":
            n_responses += 1
            response_id = f"resp_{n_responses}"
            audio_ms = audio_bytes // 32  # 16 kHz PCM16: 32 bytes per millisecond

            if audio_bytes:
                await ws.send(
                    json.dumps({"type": "input_audio_buffer.speech_stopped", "item_id": f"item_{item_n}", "audio_end_ms": audio_ms})
                )
                await ws.send(json.dumps({"type": "input_audio_buffer.committed", "item_id": f"item_{item_n}"}))
            await ws.send(
                json.dumps(
                    {"type": "response.created", "response": {"id": response_id, "status": "in_progress", "output_modalities": ["text"]}}
                )
            )

            tools = session.get("tools") or []
            if tools and tool_calls_made == 0:
                tool_calls_made += 1
                name = tools[0]["name"]
                args = json.dumps({"party_size": 4, "time": "19:30", "date": "Friday"})
                await ws.send(
                    json.dumps(
                        {
                            "type": "response.function_call_arguments.done",
                            "response_id": response_id,
                            "call_id": "call_1",
                            "name": name,
                            "arguments": args,
                        }
                    )
                )
                output = [{"type": "function_call", "name": name, "call_id": "call_1", "arguments": args, "status": "completed"}]
                extra = {"poly_out_of_domain": False}
            else:
                text = f"Stub reply {n_responses}" + (f" to {pending_text!r}" if pending_text else "")
                await ws.send(json.dumps({"type": "response.output_text.delta", "response_id": response_id, "delta": text[:4]}))
                await ws.send(json.dumps({"type": "response.output_text.done", "response_id": response_id, "text": text}))
                output = [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}]
                extra = {"poly_out_of_domain": False}
                if session.get("poly_knowledge"):
                    extra["poly_cited_topics"] = [session["poly_knowledge"][0]["name"]]

            if audio_bytes:
                await ws.send(
                    json.dumps(
                        {
                            "type": "conversation.item.input_audio_transcription.completed",
                            "item_id": f"item_{item_n}",
                            "transcript": f"<{audio_ms} ms of audio>",
                            "usage": {"type": "duration", "seconds": audio_ms / 1000},
                        }
                    )
                )
            await ws.send(
                json.dumps({"type": "response.done", "response": {"id": response_id, "status": "completed", "output": output, **extra}})
            )
            audio_bytes, pending_text, item_n = 0, None, item_n + 1

        else:
            log("ignored", kind)


def serve(port: int = 0):
    """An unstarted server; `async with serve() as server` listens. Port 0 picks a free one."""
    return websockets.serve(handle, "127.0.0.1", port)


async def main(port: int) -> None:
    async with serve(port):
        log(f"listening on {port}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 8079))
