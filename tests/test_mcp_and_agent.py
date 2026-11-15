import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import ToolException
from mcp.shared.exceptions import MCPError

import agent
from agent import (
    BAR_PROMPT,
    CARS_ALT_URL,
    CARS_URL,
    HOTELS_ALT_URL,
    HOTELS_URL,
    KIWI_URL,
    PLACES_URL,
    RENTAL_PROMPT,
    call_with_retry,
    clear_mcp_cache,
    google_maps_key,
    invoke_with_fallback,
    load_with_fallback,
    mcp_targets,
    resolve_model_name,
    retry_mcp_tool,
    web_search,
)
from http_app import app


@pytest.fixture(autouse=True)
def _reset_caches(monkeypatch):
    monkeypatch.delenv("GOOGLE_MAPS_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("TRIP_PLANNER_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    clear_mcp_cache()
    yield
    clear_mcp_cache()


def test_module_has_no_top_level_await():
    tree = ast.parse(Path("agent.py").read_text())
    assert not any(isinstance(node, ast.Await) for node in tree.body)


def test_removed_broken_names():
    text = Path("agent.py").read_text()
    for name in ("travel_agent", "WeddingState", "search_venues", "suggest_playlist"):
        assert name not in text


def test_rental_prompt_is_not_the_hotel_prompt():
    assert "rental car specialist" in RENTAL_PROMPT
    assert "hotel specialist" not in RENTAL_PROMPT
    assert "bar and nightlife specialist" in BAR_PROMPT


def test_default_model_is_hosted_openai():
    assert resolve_model_name() == "openai:gpt-5-nano"
    assert "ollama" not in resolve_model_name()


def test_model_env_override(monkeypatch):
    monkeypatch.setenv("TRIP_PLANNER_MODEL", "openai:gpt-4.1-mini")
    assert resolve_model_name() == "openai:gpt-4.1-mini"


def test_places_omitted_without_key():
    targets = mcp_targets()
    assert "places" not in targets
    assert targets["kiwi"] == KIWI_URL
    assert targets["hotels"] == HOTELS_URL
    assert targets["hotels_alt"] == HOTELS_ALT_URL
    assert targets["cars"] == CARS_URL
    assert targets["cars_alt"] == CARS_ALT_URL
    assert google_maps_key() == ""


def test_places_uses_google_header(monkeypatch):
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "maps-test-key")
    target = mcp_targets()["places"]
    assert target.url == PLACES_URL or str(target.url) == PLACES_URL
    assert target.headers["X-Goog-Api-Key"] == "maps-test-key"


def test_google_maps_key_prefers_sketch_name(monkeypatch):
    monkeypatch.setenv("GOOGLE_MAPS_KEY", "from-sketch")
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "from-alias")
    assert google_maps_key() == "from-sketch"


@pytest.mark.asyncio
async def test_fallback_uses_alt_when_primary_fails(monkeypatch):
    calls = []

    async def fake_list(name):
        calls.append(name)
        if name == "hotels":
            raise TimeoutError("primary down")
        tool = SimpleNamespace(name="search_hotels")
        return [tool]

    monkeypatch.setattr(agent, "list_server_tools", fake_list)
    tools, source, errors = await load_with_fallback("hotels", "hotels_alt")
    assert calls == ["hotels", "hotels_alt"]
    assert source == "hotels_alt"
    assert tools[0].name == "search_hotels"
    assert errors and "hotels:" in errors[0]


@pytest.mark.asyncio
async def test_primary_success_skips_alt(monkeypatch):
    calls = []

    async def fake_list(name):
        calls.append(name)
        return [SimpleNamespace(name="search")]

    monkeypatch.setattr(agent, "list_server_tools", fake_list)
    tools, source, errors = await load_with_fallback("cars", "cars_alt")
    assert calls == ["cars"]
    assert source == "cars"
    assert tools and errors == []


@pytest.mark.asyncio
async def test_both_servers_down(monkeypatch):
    async def fake_list(name):
        raise ConnectionError(f"{name} unreachable")

    monkeypatch.setattr(agent, "list_server_tools", fake_list)
    tools, source, errors = await load_with_fallback("hotels", "hotels_alt")
    assert tools == []
    assert source is None
    assert len(errors) == 2
    assert "Source unavailable" not in errors[0]


@pytest.mark.asyncio
async def test_call_time_fallback_uses_same_tool_name():
    async def primary(_args):
        raise TimeoutError("timed out")

    class Alt:
        name = "search_hotels"

        async def ainvoke(self, args):
            return {"ok": args["city"]}

    result = await invoke_with_fallback(
        "search_hotels",
        primary,
        [Alt()],
        {"city": "Lisbon"},
    )
    assert result == {"ok": "Lisbon"}


@pytest.mark.asyncio
async def test_call_time_fallback_when_alt_has_no_matching_tool():
    async def primary(_args):
        raise RuntimeError("boom")

    result = await invoke_with_fallback("search_hotels", primary, [], {"city": "Lisbon"})
    assert "Source unavailable" in result
    assert "search_hotels" in result
    assert "boom" in result


@pytest.mark.asyncio
async def test_retry_then_succeed(monkeypatch):
    monkeypatch.setattr(agent, "_max_retries", lambda: 3)
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(agent.asyncio, "sleep", fake_sleep)
    state = {"n": 0}

    async def flaky():
        state["n"] += 1
        if state["n"] < 3:
            raise MCPError(-32603, "internal")
        return "ok"

    assert await call_with_retry(flaky) == "ok"
    assert sleeps == [1, 2]


@pytest.mark.asyncio
async def test_non_retryable_mcp_error_is_not_retried(monkeypatch):
    calls = {"n": 0}

    async def bad():
        calls["n"] += 1
        raise MCPError(-32602, "invalid params")

    with pytest.raises(MCPError):
        await call_with_retry(bad)
    assert calls["n"] == 1


def _request(name="search", mcp=True):
    return SimpleNamespace(
        tool_call={"name": name, "id": "call-1", "args": {}},
        tool=SimpleNamespace(metadata={"mcp": {"server": {"name": "kiwi"}}} if mcp else {}),
    )


@pytest.mark.asyncio
async def test_interceptor_retries_internal_error(monkeypatch):
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(agent.asyncio, "sleep", fake_sleep)
    state = {"n": 0}

    async def handler(_request):
        state["n"] += 1
        if state["n"] < 2:
            raise MCPError(-32603, "internal")
        return "done"

    result = await retry_mcp_tool(_request(), handler)
    assert result == "done"
    assert sleeps == [1]


@pytest.mark.asyncio
async def test_interceptor_returns_non_retryable_error(monkeypatch):
    async def handler(_request):
        raise MCPError(-32602, "bad arguments")

    result = await retry_mcp_tool(_request(), handler)
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "non-retryable" in result.content
    assert result.tool_call_id == "call-1"


@pytest.mark.asyncio
async def test_interceptor_skips_non_mcp_tools():
    async def handler(_request):
        return "plain"

    assert await retry_mcp_tool(_request(mcp=False), handler) == "plain"


@pytest.mark.asyncio
async def test_tool_exception_is_not_retried():
    calls = {"n": 0}

    async def handler(_request):
        calls["n"] += 1
        raise ToolException("nope")

    result = await retry_mcp_tool(_request(), handler)
    assert calls["n"] == 1
    assert result.status == "error"


@pytest.mark.asyncio
async def test_web_search_limit_does_not_call_tavily(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test")

    class Boom:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Tavily should not be constructed")

    monkeypatch.setattr(agent, "TavilySearch", Boom)
    result = await web_search.ainvoke(
        {"query": "lisbon", "search_number": 4, "max_search_number": 3}
    )
    assert "Search limit" in result["message"]


@pytest.mark.asyncio
async def test_web_search_without_key():
    result = await web_search.ainvoke(
        {"query": "lisbon events", "search_number": 1, "max_search_number": 3}
    )
    assert "TAVILY_API_KEY" in result["error"]


@pytest.mark.asyncio
async def test_places_skipped_and_primaries_wired(monkeypatch):
    seen = []

    async def fake_list(name):
        seen.append(name)
        from langchain_core.tools import StructuredTool

        async def _run(**kwargs):
            return "ok"

        return [
            StructuredTool.from_function(
                coroutine=_run,
                name=f"{name}_search",
                description=f"{name} search",
            )
        ]

    monkeypatch.setattr(agent, "list_server_tools", fake_list)
    specialists = await agent.build_specialists(model=agent.FakeTripModel())
    assert set(specialists) == {"flight", "hotel", "rental_car", "restaurant", "bar"}
    assert seen == ["kiwi", "hotels", "cars"]
    assert "hotels_alt" not in seen
    assert "places" not in seen


@pytest.mark.asyncio
async def test_hotel_fallback_server_is_used_when_primary_lists_fail(monkeypatch):
    seen = []

    async def fake_list(name):
        seen.append(name)
        from langchain_core.tools import StructuredTool

        if name == "hotels":
            raise TimeoutError("lastminute down")

        async def _run(**kwargs):
            return "ok"

        return [StructuredTool.from_function(coroutine=_run, name=f"{name}_search", description="d")]

    monkeypatch.setattr(agent, "list_server_tools", fake_list)
    await agent.build_specialists(model=agent.FakeTripModel())
    assert "hotels" in seen and "hotels_alt" in seen


@pytest.mark.asyncio
async def test_fake_planner_streams_a_lisbon_plan(monkeypatch):
    monkeypatch.setenv("TRIP_PLANNER_FAKE_LLM", "1")

    async def fake_list(name):
        raise ConnectionError(f"{name} offline in test")

    monkeypatch.setattr(agent, "list_server_tools", fake_list)
    planner = agent.build_graph()
    final = None
    async for state in planner.astream(
        {"messages": [{"role": "user", "content": "plan 3 days in Lisbon in March"}]},
        config={"recursion_limit": 40},
        stream_mode="values",
    ):
        final = state
    assert final is not None
    assert final.get("destination") == "Lisbon"
    assert final.get("origin") == "unspecified"
    text = agent.final_text(final)
    assert "Lisbon" in text
    assert "search_flights" in text
    assert "search_hotels" in text
    assert "suggest_rental_cars" in text
    assert "suggest_restaurants" in text
    assert "suggest_bars" in text
    called = {
        call["name"]
        for message in final["messages"]
        if isinstance(message, AIMessage)
        for call in (message.tool_calls or [])
    }
    assert {
        "update_state",
        "search_flights",
        "search_hotels",
        "suggest_rental_cars",
        "suggest_restaurants",
        "suggest_bars",
    } <= called


def test_health_endpoint():
    from starlette.testclient import TestClient

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_coordinator_tools():
    names = [tool.name for tool in agent.COORDINATOR_TOOLS]
    assert names == [
        "update_state",
        "search_flights",
        "search_hotels",
        "suggest_rental_cars",
        "suggest_restaurants",
        "suggest_bars",
        "web_search",
    ]


def test_graph_imports_without_credentials():
    assert agent.graph is not None
    assert agent.graph.name == "trip_planner"
