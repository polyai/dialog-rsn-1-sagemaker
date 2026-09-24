"""Run the notebook's conversation cells against a Realtime server, without AWS.

The notebook's only endpoint-specific line is the base URL it hands the openai SDK. This swaps
that line and executes the client cells in order (imports, session helpers, the four
conversations), skipping the cells that create the endpoint, start the bridge and delete. Then it
checks that the conversations got what the notebook shows: a tool call, a cited topic, and no
errors.

    python tests/notebook_smoke.py                      # against the stub, started here
    python tests/notebook_smoke.py ws://127.0.0.1:8079  # against a running bridge or server

Run from the repo root so the audio/ paths resolve.
"""

import ast
import asyncio
import inspect
import os
import sys
from pathlib import Path

import nbformat

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stub_server  # noqa: E402

NOTEBOOK = Path(__file__).resolve().parent.parent / "dialog_rsn_1_sagemaker.ipynb"
BRIDGE_URL_LINE = 'BRIDGE_URL = f"ws://127.0.0.1:{BRIDGE_PORT}"'


def is_client_cell(src: str) -> bool:
    return (
        "import boto3" in src
        or "from openai import AsyncOpenAI" in src
        or "class Conversation" in src
        or "async with conversation(" in src
    )


async def run_cells(base_url: str) -> list[dict]:
    nb = nbformat.read(NOTEBOOK, as_version=4)
    ns: dict = {"BRIDGE_PORT": 0, "__name__": "__main__"}
    seen: list[dict] = []
    ran = 0

    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code" or not is_client_cell(cell.source):
            continue
        src = cell.source
        if "from openai import AsyncOpenAI" in src:
            # Fail loudly if the notebook changes the line this script depends on.
            assert BRIDGE_URL_LINE in src, f"cell {i}: expected {BRIDGE_URL_LINE!r}"
            src = src.replace(BRIDGE_URL_LINE, f'BRIDGE_URL = "{base_url}"')

        print(f"\n===== cell {i} =====")
        result = eval(compile(src, f"cell{i}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT), ns)
        if inspect.isawaitable(result):
            await result
        ran += 1

        if "def render" in src:
            # Every event a conversation reads goes through render, so wrapping it records them all.
            original = ns["render"]

            def recording_render(event: dict, original=original) -> None:
                seen.append(event)
                original(event)

            ns["render"] = recording_render

    assert ran >= 6, f"expected the setup, helper and four conversation cells, ran {ran}"
    return seen


def check(events: list[dict]) -> None:
    kinds = [e.get("type") for e in events]
    done = [e["response"] for e in events if e.get("type") == "response.done"]

    errors = [e for e in events if e.get("type") == "error"]
    assert not errors, f"server errors: {errors}"
    assert "response.function_call_arguments.done" in kinds, "no tool call reached the client"
    assert any(r.get("poly_cited_topics") for r in done), "no response.done carried poly_cited_topics"
    assert all(r.get("status") == "completed" for r in done), [r.get("status") for r in done]
    assert all("poly_out_of_domain" in r for r in done), "a response.done lacked poly_out_of_domain"
    print(f"\nok: {len(done)} responses, {kinds.count('response.function_call_arguments.done')} tool call(s)")


async def main(base_url: str | None) -> None:
    # The setup cell reads these, though nothing here touches AWS.
    os.environ.setdefault("SAGEMAKER_ROLE_ARN", "arn:aws:iam::000000000000:role/unused")
    os.environ.setdefault("AWS_REGION", "us-east-2")

    if base_url:
        check(await run_cells(base_url))
        return
    async with stub_server.serve() as server:
        port = server.sockets[0].getsockname()[1]
        check(await run_cells(f"ws://127.0.0.1:{port}"))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else None))
