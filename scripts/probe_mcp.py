"""Connect to each configured MCP server, list tools, and make one read-only call."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import clear_mcp_cache, google_maps_key, list_server_tools, mcp_targets

MUTATING = ("book", "purchase", "cancel", "pay", "delete", "charge")


def _sample_args(tool) -> dict:
    schema = getattr(tool, "args_schema", None) or {}
    if hasattr(schema, "model_json_schema"):
        schema = schema.model_json_schema()
    if not isinstance(schema, dict):
        return {}
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    args = {}
    for name, spec in properties.items():
        if name not in required and name not in {"query", "q", "origin", "destination", "city"}:
            continue
        if not isinstance(spec, dict):
            args[name] = "Lisbon"
            continue
        kind = spec.get("type")
        if kind in {"integer", "number"}:
            args[name] = 1
        elif kind == "boolean":
            args[name] = False
        elif kind == "array":
            args[name] = []
        elif kind == "object":
            args[name] = {}
        else:
            lowered = name.lower()
            if "date" in lowered:
                args[name] = "2026-03-15"
            elif "origin" in lowered or name.lower() in {"from", "fly_from"}:
                args[name] = "JFK"
            elif "dest" in lowered or name.lower() in {"to", "fly_to"}:
                args[name] = "LIS"
            else:
                args[name] = "Lisbon"
    return args


async def _try_call(tool) -> str:
    lowered = tool.name.lower()
    if any(word in lowered for word in MUTATING):
        return f"skipped mutating tool {tool.name}"
    args = _sample_args(tool)
    try:
        result = await asyncio.wait_for(tool.ainvoke(args), timeout=40)
    except Exception as exc:
        return f"CALL FAIL {type(exc).__name__}: {exc} args={args}"
    text = result if isinstance(result, str) else json.dumps(result, default=str)
    return f"CALL OK args={args} result={text[:500]}"


async def main() -> None:
    clear_mcp_cache()
    targets = mcp_targets()
    print("configured", ", ".join(targets))
    if "places" not in targets:
        print("places SKIPPED: set GOOGLE_MAPS_KEY or GOOGLE_MAPS_API_KEY (no key in this environment)")
        print("google_maps_key_set", bool(google_maps_key()))
    for name in targets:
        print(f"\n== {name} ==")
        try:
            tools = await list_server_tools(name)
        except Exception as exc:
            print(f"LIST FAIL {type(exc).__name__}: {exc}")
            continue
        print(f"LIST OK {len(tools)} tools: {', '.join(tool.name for tool in tools)}")
        if tools:
            print(await _try_call(tools[0]))


if __name__ == "__main__":
    asyncio.run(main())
