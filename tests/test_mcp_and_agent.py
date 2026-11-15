import ast
import asyncio
import json
import logging
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import ToolException
from mcp.shared.exceptions import MCPError

import agent
from agent import (
    BAR_PROMPT,
    CARS_ALT_URL,
    CARS_URL,
    HOTELS_URL,
    KIWI_URL,
    LASTMINUTE_URL,
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
    assert resolve_model_name() == "openai:gpt-4.1-mini"
    assert agent.DEFAULT_MODEL == "openai:gpt-4.1-mini"
    assert "ollama" not in resolve_model_name()


def test_model_env_override(monkeypatch):
    monkeypatch.setenv("TRIP_PLANNER_MODEL", "openai:gpt-4.1-mini")
    assert resolve_model_name() == "openai:gpt-4.1-mini"


def test_places_omitted_without_key():
    targets = mcp_targets()
    assert "places" not in targets
    assert targets["kiwi"] == KIWI_URL
    assert targets["hotels"] == HOTELS_URL
    assert "skiplagged.com" in HOTELS_URL
    assert targets["lastminute"] == LASTMINUTE_URL
    assert "hotels_alt" not in targets
    assert str(targets["cars"].url) == CARS_URL
    assert "trip-planner" in targets["cars"].headers["User-Agent"]
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
async def test_hotel_source_does_not_retry_lastminute(monkeypatch):
    seen = []

    async def fake_list(name):
        seen.append(name)
        from langchain_core.tools import StructuredTool

        if name == "hotels":
            raise TimeoutError("skiplagged down")

        async def _run(**kwargs):
            return "ok"

        return [StructuredTool.from_function(coroutine=_run, name=f"{name}_search", description="d")]

    monkeypatch.setattr(agent, "list_server_tools", fake_list)
    await agent.build_specialists(model=agent.FakeTripModel())
    assert seen == ["kiwi", "hotels", "cars"]
    assert "lastminute" not in seen
    assert "hotels_alt" not in seen


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


def test_prompts_match_live_corrections():
    assert "only look for one ticket, one way" not in agent.FLIGHT_PROMPT
    assert "return flight on the trip end date" in agent.FLIGHT_PROMPT
    assert "one specialist at a time" not in agent.COORDINATOR_PROMPT
    assert "in parallel" in agent.COORDINATOR_PROMPT
    assert "arrival airport" in agent.RENTAL_PROMPT
    assert agent.driver_age_note(None) == "assumed driver age 30 — tell me yours if different"
    assert "assumed driver age 30 — tell me yours if different" in agent.COORDINATOR_PROMPT
    assert "assumed driver age 30 — tell me yours if different" in agent.RENTAL_PROMPT
    assert "42" in agent.driver_age_note(42)
    assert "Assumed" not in agent.driver_age_note(42)
    note = agent._source_note("places", None, ["down"])
    assert "venues" in note and "ratings" in note and "hours" in note
    assert "Do not put trip dates" in agent.RESTAURANT_PROMPT
    assert "check hours before going" in " ".join(agent.BAR_PROMPT.split())
    assert "shrimp stew" in agent.RESTAURANT_PROMPT


def test_strip_trip_dates_leaves_the_place():
    assert agent.strip_trip_dates("lisbon restaurants on 2027-03-15") == "lisbon restaurants"
    assert "March" not in agent.strip_trip_dates("bars in Lisbon March 15, 2027")


@pytest.mark.asyncio
async def test_web_search_strips_dates_before_tavily(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test")
    seen = []

    class Capture:
        def __init__(self, *args, **kwargs):
            pass

        async def ainvoke(self, payload):
            seen.append(payload["query"])
            return "ok"

    monkeypatch.setattr(agent, "TavilySearch", Capture)
    await web_search.ainvoke(
        {"query": "lisbon restaurants 2027-03-15", "search_number": 1, "max_search_number": 3}
    )
    assert seen == ["lisbon restaurants"]


def test_grounding_lite_names_survive():
    payload = {
        "places": [
            {
                "attribution": {},
                "googleMapsLinks": {"placeUrl": "https://maps.google.com/?cid=1"},
                "id": "places/1",
                "location": {"latitude": 38.71, "longitude": -9.14},
                "place": {"attribution": {"title": "Cafe Portugal - Google Maps"}},
            },
            {
                "attribution": {},
                "googleMapsLinks": {"placeUrl": "https://maps.google.com/?cid=2"},
                "id": "places/2",
                "location": {"latitude": 38.71, "longitude": -9.15},
                "place": {"attribution": {"title": "Taberna do Bairro alto - Google Maps"}},
            },
            {
                "attribution": {},
                "googleMapsLinks": {"placeUrl": "https://maps.google.com/?cid=3"},
                "id": "places/3",
                "location": {"latitude": 38.71, "longitude": -9.14},
                "place": {
                    "attribution": {
                        "title": "Pink Bar - The smallest bar in town - Google Maps"
                    }
                },
            },
        ],
        "summary": (
            "Photo: Rafael Tonon\n"
            "**Cafe Portugal**\n4.6. EUR_20\n\n"
            "**Taberna do Bairro alto**\nHours 12:00-23:00.\n\n"
            "**Pink Bar - The smallest bar in town**\n"
        ),
        "contentToken": "token",
    }
    names = agent.extract_grounded_venues(json.dumps(payload))
    assert "Cafe Portugal" in names
    assert "Taberna do Bairro alto" in names
    assert "Pink Bar - The smallest bar in town" in names
    assert "Rafael Tonon" not in names
    assert all(not name.startswith("EUR") for name in names)
    prose = "Dinner at **Cafe Portugal** and **Pink Bar - The smallest bar in town**."
    grounded = agent.render_grounded_specialist(prose, names, json.dumps(payload), tools_called=True)
    assert "Source unavailable" not in grounded
    assert "Cafe Portugal" in grounded
    assert "Pink Bar - The smallest bar in town" in grounded


def test_credits_are_not_venues_and_headings_are():
    blob = (
        "Photo: Rafael Tonon\n"
        "## The 10 Best Restaurants in Lisbon\n"
        "## Cervejaria Ramiro\n"
        '{"displayName": {"text": "Santa Joana"}, "formattedAddress": "Lisbon", '
        '"rating": 4.2, "editorialSummary": "shrimp stew"}\n'
        "## Antiga Wine Bar\n"
        "Best for lunch."
    )
    names = agent.extract_grounded_venues(blob)
    assert "Cervejaria Ramiro" in names
    assert "Santa Joana" in names
    assert "Antiga Wine Bar" in names
    assert "Rafael Tonon" not in names
    assert all("Best" not in name for name in names)


def test_no_venue_data_names_nothing():
    text = agent.render_grounded_specialist(
        "Dinner at **Rafael Tonon**.",
        set(),
        "Photo: Rafael Tonon",
        tools_called=True,
    )
    assert "Rafael Tonon" not in text
    assert "unavailable" in text.lower()
    assert "VENUES: (none)" in text


def test_hours_cuisine_and_meal_are_not_overclaimed():
    evidence = (
        "## Santa Joana\nshrimp stew. No hours listed.\n"
        "## Antiga Wine Bar\nBest for lunch."
    )
    answer = (
        "**Santa Joana** is vegetarian and hours are confirmed for the trip.\n"
        "Dinner at **Antiga Wine Bar**."
    )
    text = agent.scrub_unconfirmed_hours(answer, evidence)
    text = agent.scrub_overclaims(text, evidence, {"Santa Joana", "Antiga Wine Bar"})
    assert "confirmed" not in text.lower()
    assert "check hours before going" in text.lower()
    assert "vegetarian" not in text.lower()
    assert "Dinner at" not in text
    assert "lunch" in text.lower()


def test_coordinator_drops_credit_and_keeps_hotels_and_grounded_venues():
    hotels = agent.format_structured(
        "HOTELS",
        {
            "city": "Lisbon",
            "checkin": "2027-03-15",
            "checkout": "2027-03-18",
            "hotels": [
                {
                    "name": "Ibis Parque das Nações",
                    "nightly": "90 EUR",
                    "total": "270 EUR",
                    "link": "https://hotels.example/ibis",
                }
            ],
        },
    )
    messages = [
        ToolMessage(
            hotels + "\n**Cancellation policy**\n**Booking link**",
            name="search_hotels",
            tool_call_id="h",
        ),
        ToolMessage(
            "**Cervejaria Ramiro**\nVENUES: Cervejaria Ramiro",
            name="suggest_restaurants",
            tool_call_id="r",
        ),
        ToolMessage("VENUES: (none)", name="suggest_bars", tool_call_id="b"),
    ]
    answer = (
        "Hotels\n**Cancellation policy**\n**Booking link**\n"
        "Dinner at **Rafael Tonon**.\nAlso **Cervejaria Ramiro**."
    )
    text = agent.apply_coordinator_checks(answer, messages)
    assert "Rafael Tonon" not in text
    assert "Cervejaria Ramiro" in text
    assert "Ibis Parque das Nações" in text
    assert "Hotels returned by the hotel specialist" not in text
    assert "Cancellation policy" not in text
    assert "Booking link" not in text
    assert "unavailable" in text.lower()


def test_kiwi_itinerary_fields_stay_together_and_warn():
    payload = {
        "itineraries": [
            {
                "price": 1805,
                "currency": "EUR",
                "bookingUrl": "https://kiwi.com/a",
                "route": [
                    {
                        "flyFrom": "JFK",
                        "flyTo": "LIS",
                        "local_departure": "2027-03-15T18:00",
                        "local_arrival": "2027-03-16T06:00",
                    }
                ],
            },
            {
                "price": 1878,
                "currency": "EUR",
                "bookingUrl": "https://kiwi.com/b",
                "route": [
                    {
                        "flyFrom": "LIS",
                        "flyTo": "JFK",
                        "local_departure": "2027-03-18T12:00",
                        "local_arrival": "2027-03-18T18:00",
                    }
                ],
            },
        ]
    }
    rendered = agent.render_kiwi_itineraries(
        payload, {"departureDate": "2027-03-15", "returnDate": "2027-03-18"}
    )
    first, second = rendered.split("ITINERARY 2", 1)
    assert "1805" in first and "https://kiwi.com/a" in first
    assert "https://kiwi.com/b" not in first
    assert "1878" in second and "https://kiwi.com/b" in second
    assert "https://kiwi.com/a" not in second
    assert "day or more after the trip start" in rendered
    mixed = "The 1805 EUR trip books at https://kiwi.com/b and the return is separate."
    repaired = agent.repair_flight_pairing(mixed, rendered)
    assert "intact" in repaired
    assert "https://kiwi.com/a" in repaired and "https://kiwi.com/b" in repaired


def _kiwi_itinerary(price, link, outbound_day, return_day, arrival_day=None):
    arrival_day = arrival_day or outbound_day
    return {
        "price": price,
        "currency": "EUR",
        "bookingUrl": link,
        "outbound": {
            "from": "SFO",
            "to": "LIS",
            "departureTime": f"{outbound_day}T08:25:00",
            "arrivalTime": f"{arrival_day}T10:10:00",
            "route": ["SFO", "YUL", "LIS"],
            "segments": [
                {
                    "from": "SFO",
                    "to": "YUL",
                    "departureTime": f"{outbound_day}T08:25:00",
                    "arrivalTime": f"{outbound_day}T16:42:00",
                    "carrierName": "Air Canada",
                    "flightNumber": "AC756",
                },
                {
                    "from": "YUL",
                    "to": "LIS",
                    "departureTime": f"{outbound_day}T23:30:00",
                    "arrivalTime": f"{arrival_day}T10:10:00",
                    "carrierName": "Air Canada",
                    "flightNumber": "AC812",
                },
            ],
        },
        "inbound": {
            "from": "LIS",
            "to": "SFO",
            "departureTime": f"{return_day}T12:15:00",
            "arrivalTime": f"{return_day}T22:00:00",
            "route": ["LIS", "SFO"],
            "segments": [
                {
                    "from": "LIS",
                    "to": "SFO",
                    "departureTime": f"{return_day}T12:15:00",
                    "arrivalTime": f"{return_day}T22:00:00",
                }
            ],
        },
    }


def test_kiwi_segments_keep_times_and_warn_on_late_arrival():
    payload = {"itineraries": [_kiwi_itinerary(765, "https://kiwi.com/u/njuby4v", "2027-03-15", "2027-03-18", "2027-03-16")]}
    rendered = agent.render_kiwi_itineraries(payload, {"departureDate": "15/03/2027", "returnDate": "18/03/2027"})
    assert "SFO -> YUL depart 2027-03-15T08:25:00 arrive 2027-03-15T16:42:00" in rendered
    assert "YUL -> LIS depart 2027-03-15T23:30:00 arrive 2027-03-16T10:10:00" in rendered
    assert "depart  arrive" not in rendered
    assert "day or more after the trip start" in rendered
    assert "765" in rendered and "https://kiwi.com/u/njuby4v" in rendered


def test_only_requested_flight_dates_are_the_plan():
    match = {"itineraries": [_kiwi_itinerary(765, "https://kiwi.com/match", "2027-03-15", "2027-03-18", "2027-03-16")]}
    early = {"itineraries": [_kiwi_itinerary(500, "https://kiwi.com/early", "2027-03-11", "2027-03-16")]}
    later = {"itineraries": [_kiwi_itinerary(510, "https://kiwi.com/later", "2027-03-13", "2027-03-18")]}
    result = {
        "messages": [
            HumanMessage(content="Book SFO to Lisbon outbound 15 March 2027 and return 18 March 2027."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search-flight",
                        "args": {"departureDate": "15/03/2027", "returnDate": "18/03/2027"},
                        "id": "match",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(json.dumps(match), name="search-flight", tool_call_id="match"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search-flight",
                        "args": {"departureDate": "11/03/2027", "returnDate": "16/03/2027"},
                        "id": "early",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(json.dumps(early), name="search-flight", tool_call_id="early"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search-flight",
                        "args": {"departureDate": "13/03/2027", "returnDate": "18/03/2027"},
                        "id": "later",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(json.dumps(later), name="search-flight", tool_call_id="later"),
        ]
    }
    text = agent.flight_specialist_text(result)
    assert text is not None
    primary, alternate = text.split("ALTERNATE DATES", 1)
    assert "https://kiwi.com/match" in primary
    assert "https://kiwi.com/early" not in primary
    assert "https://kiwi.com/later" not in primary
    assert "https://kiwi.com/early" in alternate
    assert "https://kiwi.com/later" in alternate
    assert "SFO -> YUL depart 2027-03-15T08:25:00" in primary
    plan = "Use https://kiwi.com/match and also https://kiwi.com/early."
    checked = agent.apply_coordinator_checks(
        plan, [ToolMessage(text, name="search_flights", tool_call_id="flights")]
    )
    assert "https://kiwi.com/early" not in checked
    assert "https://kiwi.com/match" in checked
    assert "2027-03-15T08:25:00" in checked


def test_car_names_prices_and_assumed_age_stay_in_the_plan():
    offer = {
        "results": [
            {
                "name": "Peugeot 2008",
                "vendor": "CRC Car Rental Company",
                "price": 49.4,
                "currency": "EUR",
                "booking_url": "https://cars.example/peugeot",
                "category": "Compact",
            }
        ]
    }
    specialist = agent.seal_car_tool_result(
        offer,
        {
            "location": "Lisbon Airport",
            "pickup_date": "2027-03-15",
            "dropoff_date": "2027-03-18",
        },
    )
    answer = "Rental Cars ():\n- https://cars.example/peugeot"
    text = agent.apply_coordinator_checks(
        answer,
        [
            HumanMessage(content="Plan Lisbon. No age was given."),
            ToolMessage(specialist, name="suggest_rental_cars", tool_call_id="cars"),
        ],
    )
    assert "Rental Cars ():" not in text
    assert "Rental Cars in Lisbon Airport (2027-03-15 to 2027-03-18):" in text
    assert "Peugeot 2008" in text
    assert "49.40" in text
    assert "CRC Car Rental Company" in text
    assert "https://cars.example/peugeot" in text
    assert "assumed driver age 30 — tell me yours if different" in text


def test_octotrip_offers_survive_a_heading_only_plan():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "octotrip_lisbon.json").read_text()
    )
    listed = {
        "results": [
            {
                "name": "Peugeot 208",
                "vendor": "CRC Car Rental Company",
                "price": 36.89,
                "currency": "EUR",
                "category": "Economy",
                "booking_url": "https://mcp.octotrip.app/rental-cars/r/208",
            },
            {
                "name": "Peugeot 308",
                "vendor": "CRC Car Rental Company",
                "price": 40.41,
                "currency": "EUR",
                "category": "Compact",
                "booking_url": "https://mcp.octotrip.app/rental-cars/r/308",
            },
            {
                "name": "BMW 1 Series",
                "vendor": "Autounion",
                "price": 64.94,
                "currency": "EUR",
                "category": "Compact Elite",
                "booking_url": "https://mcp.octotrip.app/rental-cars/r/bmw",
            },
            {
                "name": "VW Polo",
                "vendor": "Record",
                "price": 36.90,
                "currency": "EUR",
                "category": "Economy",
                "booking_url": "https://mcp.octotrip.app/rental-cars/r/polo",
            },
            {
                "name": "Audi A3",
                "vendor": "Masterkings",
                "price": 88.0,
                "currency": "EUR",
                "category": "Compact",
                "booking_url": "https://mcp.octotrip.app/rental-cars/r/a3",
            },
            {
                "name": "Ferrari Roma",
                "vendor": "Masterkings",
                "price": 900,
                "currency": "EUR",
                "category": "Luxury",
                "booking_url": "https://mcp.octotrip.app/rental-cars/r/ferrari",
            },
        ]
    }
    padded = "search note\n" + json.dumps(fixture) + (" " * 25000)
    parsed = agent.car_record_from_payload(padded, {"location": "Lisbon Airport"})
    assert parsed is not None
    assert [offer["name"] for offer in parsed["offers"]] == [
        "Peugeot 2008",
        "Peugeot 308",
        "Mercedes-Benz A-Class",
        "BMW 1 Series",
    ]
    assert parsed["offers"][0]["vendor"] == "CRC Car Rental Company"
    assert parsed["offers"][0]["booking_url"].startswith("https://mcp.octotrip.app/rental-cars/r/")
    specialist = (
        "Rental Cars (): the writeup names nothing useful.\n"
        + agent.seal_car_tool_result(
            listed,
            {
                "location": "Lisbon Airport",
                "pickup_date": "2027-03-15",
                "dropoff_date": "2027-03-18",
            },
        )
    )
    answer = (
        "Rental Cars ():\n"
        "- https://mcp.octotrip.app/rental-cars/r/208\n"
        "- https://mcp.octotrip.app/rental-cars/r/polo\n"
        "Driver age 30+."
    )
    text = agent.apply_coordinator_checks(
        answer,
        [
            HumanMessage(content="Plan a long weekend in Lisbon. I did not mention an age."),
            ToolMessage(specialist, name="suggest_rental_cars", tool_call_id="cars"),
        ],
    )
    for name in ("Peugeot 208", "Peugeot 308", "BMW 1 Series", "VW Polo", "Audi A3"):
        assert name in text
    assert "36.89" in text and "40.41" in text and "64.94" in text and "36.90" in text
    assert "Autounion" in text and "https://mcp.octotrip.app/rental-cars/r/polo" in text
    assert "Ferrari Roma" not in text
    assert "Rental Cars ():" not in text
    assert "Rental Cars in Lisbon Airport (2027-03-15 to 2027-03-18):" in text
    assert "assumed driver age 30 — tell me yours if different" in text
    assert "30+" not in text


def test_stated_driver_age_is_not_replaced_with_the_assumption():
    answer = "Rental car: Peugeot 208. assumed driver age 30 — tell me yours if different"
    text = agent.enforce_assumed_driver_age(
        answer,
        [HumanMessage(content="I am 42 years old.")],
        "CARS:\n- Peugeot 208 | 36.89 EUR",
    )
    assert "Driver age 42 was taken from the request." in text
    assert "assumed driver age 30" not in text


def test_hotel_field_labels_are_not_hotel_names():
    table = (
        Path(__file__).parent / "fixtures" / "skiplagged_lisbon_hotels.md"
    ).read_text()
    prose = (
        "**Neighborhood**\nAlfama\n**Nightly rate**\n€67\n"
        "**Cancellation**\nfree\n**Tradeoff**\ncentral but small\n"
    )
    names = agent.extract_hotel_names(prose)
    assert names == []
    record = agent.hotel_record_from_payload(
        table, {"city": "Lisbon", "checkin": "2027-03-15", "checkout": "2027-03-18"}
    )
    assert record is not None
    specialist = prose + "\n" + agent.format_structured("HOTELS", record)
    answer = "Hotels in Lisbon.\n**Cancellation policy**\n**Booking link**"
    text = agent.apply_coordinator_checks(
        answer,
        [ToolMessage(specialist, name="search_hotels", tool_call_id="hotels")],
    )
    assert "Inn Rossio Hotel" in text
    assert "Hotel ibis Lisboa Jose Malhoa" in text
    assert "Hotels returned by the hotel specialist" not in text
    for label in (
        "Neighborhood",
        "Nightly rate",
        "Cancellation",
        "Tradeoff",
        "Cancellation policy",
        "Booking link",
    ):
        assert label not in text


def test_flight_legs_follow_the_presented_itinerary_in_eur():
    itineraries = []
    for index in range(15):
        item = _kiwi_itinerary(
            765 + index,
            f"https://kiwi.com/u/{index}",
            "2027-03-15",
            "2027-03-18",
            "2027-03-16",
        )
        item["inbound"]["arrivalTime"] = "2027-03-19T12:45:00"
        item["priceFormatted"] = f"{765 + index} EUR"
        itineraries.append(item)
    rendered = agent.render_kiwi_itineraries(
        {"currency": "EUR", "itineraries": itineraries},
        {"departureDate": "15/03/2027", "returnDate": "18/03/2027"},
    )
    assert "price: 765 EUR" in rendered
    assert "outbound: next-day arrival (16 Mar)" in rendered
    assert "return: next-day arrival (19 Mar)" in rendered
    answer = "One option for the weekend."
    text = agent.ensure_flight_legs(answer, rendered)
    assert "https://kiwi.com/u/0" in text
    assert "https://kiwi.com/u/2" in text
    assert "https://kiwi.com/u/3" not in text
    assert "https://kiwi.com/u/14" not in text
    assert "2027-03-15T08:25:00" in text
    priced = agent.lock_flight_currency(
        agent.ensure_flight_legs("One option: $765 on the trip.", rendered),
        rendered,
    )
    assert "https://kiwi.com/u/0" in priced
    assert "https://kiwi.com/u/1" not in priced
    assert "$765" not in priced
    assert "765 EUR" in priced
    presented = "Book https://kiwi.com/u/4 for this trip at $769."
    kept = agent.lock_flight_currency(agent.ensure_flight_legs(presented, rendered), rendered)
    assert "https://kiwi.com/u/4" in kept
    assert "https://kiwi.com/u/0" not in kept
    assert "https://kiwi.com/u/14" not in kept
    assert "$769" not in kept
    assert "769 EUR" in kept
    assert "next-day arrival (19 Mar)" in kept


def test_flight_specialist_returns_locked_blocks():
    result = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search-flight",
                        "args": {"departureDate": "2027-03-15", "returnDate": "2027-03-18"},
                        "id": "c1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                json.dumps(
                    {
                        "price": 317,
                        "currency": "EUR",
                        "bookingUrl": "https://kiwi.com/u/one",
                        "route": [
                            {
                                "flyFrom": "JFK",
                                "flyTo": "LIS",
                                "local_departure": "2027-03-15T10:00",
                                "local_arrival": "2027-03-15T18:00",
                            }
                        ],
                    }
                ),
                name="search-flight",
                tool_call_id="c1",
            ),
        ]
    }
    text = agent.flight_specialist_text(result)
    assert text is not None
    assert "317" in text and "https://kiwi.com/u/one" in text
    assert "ITINERARY 1" in text


@pytest.mark.asyncio
async def test_empty_result_uses_same_name_fallback():
    async def primary(_args):
        return {"results": [], "status": "no_results"}

    class Alt:
        name = "search_hotels"

        async def ainvoke(self, args):
            return {"ok": args["city"]}

    result = await invoke_with_fallback("search_hotels", primary, [Alt()], {"city": "Lisbon"})
    assert result == {"ok": "Lisbon"}


@pytest.mark.asyncio
async def test_octotrip_empty_retries_airport_then_upfront(monkeypatch):
    from langchain_core.tools import StructuredTool

    calls = []

    async def search(location: str, pickup_date: str = "", dropoff_date: str = ""):
        calls.append(("search", location))
        return {"results": [], "status": "no_results"}

    async def get_locations(location: str = "", query: str = ""):
        calls.append(("get_locations", location or query))
        return [{"name": "LIS"}]

    async def get_offers(location: str = "", pickup_location: str = "", pickup_date: str = "", dropoff_date: str = ""):
        calls.append(("get_offers", location or pickup_location))
        return [{"vehicle": "Micra", "price": 100}]

    tool = StructuredTool.from_function(coroutine=search, name="search", description="search cars")
    wrapped = agent.attach_call_fallback(tool, "cars_alt")

    async def fake_list(name):
        assert name == "cars_alt"
        return [
            StructuredTool.from_function(coroutine=get_locations, name="get_locations", description="locs"),
            StructuredTool.from_function(coroutine=get_offers, name="get_offers", description="offers"),
        ]

    monkeypatch.setattr(agent, "list_server_tools", fake_list)
    result = await wrapped.ainvoke(
        {"location": "Lisbon", "pickup_date": "2027-03-15", "dropoff_date": "2027-03-18"}
    )
    assert calls[0] == ("search", "Lisbon")
    assert calls[1] == ("search", "Lisbon Airport")
    assert ("get_locations", "Lisbon Airport") in calls
    assert ("get_offers", "Lisbon Airport") in calls
    assert "Upfront" in str(result)
    assert "assumed driver age 30 — tell me yours if different" in str(result)


@pytest.mark.asyncio
async def test_list_tools_retries_before_fallback(monkeypatch):
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(agent.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(agent, "_max_retries", lambda: 3)
    state = {"n": 0}

    class Adapter:
        def __init__(self, target):
            self.target = target

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def list_tools(self, cache_mode="bypass"):
            state["n"] += 1
            if state["n"] < 3:
                raise MCPError(-32603, "internal")
            return ["kiwi-tool"]

    monkeypatch.setattr("langchain.mcp.MCPAdapter", Adapter)
    tools = await agent.list_server_tools("kiwi")
    assert tools == ["kiwi-tool"]
    assert state["n"] == 3
    assert sleeps == [1, 2]

    state["n"] = 0

    class BadAdapter(Adapter):
        async def list_tools(self, cache_mode="bypass"):
            state["n"] += 1
            raise MCPError(-32602, "bad")

    monkeypatch.setattr("langchain.mcp.MCPAdapter", BadAdapter)
    agent.clear_mcp_cache()
    with pytest.raises(MCPError):
        await agent.list_server_tools("kiwi")
    assert state["n"] == 1


class UnparseablePlanModel(agent.FakeTripModel):
    """Coordinator and specialists write car and hotel prose that cannot be parsed."""

    def _once(self, messages, name, args, prose):
        if name not in agent._tool_names(self.bound_tools):
            return AIMessage(content=prose)
        if name not in agent._called_tools(messages):
            return AIMessage(
                content="",
                tool_calls=[{"name": name, "args": args, "id": f"call_{name}", "type": "tool_call"}],
            )
        return AIMessage(content=prose)

    def _respond(self, messages):
        system = agent._system_text(messages)
        names = agent._tool_names(self.bound_tools)
        user = agent._human_text(messages)
        coordinator = "trip planner coordinator" in system or (
            "update_state" in names and "search_flights" in names
        )
        if coordinator:
            return self._bad_plan(messages, names, user)
        if "rental car specialist" in system:
            return self._once(
                messages,
                "search",
                {
                    "location": "Lisbon Airport",
                    "pickup_date": "2027-03-12",
                    "dropoff_date": "2027-03-15",
                },
                "Rental Cars in Lisbon (\n\n):\nnothing named here",
            )
        if "hotel specialist" in system:
            return self._once(
                messages,
                "sk_hotels_search",
                {"city": "Lisbon", "checkin": "2027-03-15", "checkout": "2027-03-18"},
                "**Cancellation policy**\n**Booking link**\n**Neighborhood**\n**Nightly rate**",
            )
        if "you are a travel agent" in system:
            return self._once(
                messages,
                "search-flight",
                {"departureDate": "15/03/2027", "returnDate": "18/03/2027"},
                "One flight. The price is left out on purpose.",
            )
        if "restaurant specialist" in system or "bar and nightlife" in system:
            return AIMessage(content="No places were returned.\nVENUES: (none)")
        return AIMessage(content="Offline specialist returned no structured data.")

    def _bad_plan(self, messages, names, user):
        called = agent._called_tools(messages)
        destination = agent._infer_destination(user)
        order = [
            "update_state",
            "search_flights",
            "search_hotels",
            "suggest_rental_cars",
            "suggest_restaurants",
            "suggest_bars",
        ]
        for name in order:
            if name not in names or name in called:
                continue
            if name == "update_state":
                args = {
                    "origin": agent._infer_origin(user),
                    "destination": destination,
                    "guest_count": agent._infer_guests(user),
                    "genre": "city break",
                }
            else:
                args = {"request": user or f"Plan the trip to {destination}."}
            return AIMessage(
                content="",
                tool_calls=[{"name": name, "args": args, "id": f"call_{name}", "type": "tool_call"}],
            )
        return AIMessage(
            content=(
                "Trip plan for Lisbon\n\n"
                "Flights\n"
                "- SFO to LIS https://kiwi.com/u/price1249\n\n"
                "Hotels\n"
                "**Cancellation policy**\n"
                "**Booking link**\n\n"
                "Rental Cars in Lisbon (\n\n):\n"
            )
        )


@pytest.mark.asyncio
async def test_graph_renders_cars_and_hotels_from_tool_payloads(monkeypatch):
    monkeypatch.setenv("TRIP_PLANNER_FAKE_LLM", "1")
    agent.clear_mcp_cache()
    fixture = json.loads((Path(__file__).parent / "fixtures" / "octotrip_lisbon.json").read_text())
    fixture["results"] = [
        {
            "name": "Fiat 500",
            "vendor": "Autounion",
            "price": 34.98,
            "currency": "EUR",
            "category": "Economy",
            "booking_url": "https://mcp.octotrip.app/rental-cars/r/fiat-500",
            "sipp": "EDMR",
        },
        *fixture["results"],
        {
            "name": "VW Polo",
            "vendor": "Record",
            "price": 36.90,
            "currency": "EUR",
            "category": "Economy",
            "booking_url": "https://mcp.octotrip.app/rental-cars/r/polo",
            "sipp": "EDMR",
        },
        {
            "name": "Ferrari Roma",
            "vendor": "Masterkings",
            "price": 900,
            "currency": "EUR",
            "category": "Luxury",
            "booking_url": "https://mcp.octotrip.app/rental-cars/r/ferrari",
            "sipp": "LDAR",
        },
    ]
    hotels_md = (Path(__file__).parent / "fixtures" / "skiplagged_lisbon_hotels.md").read_text()
    flight = {
        "currency": "EUR",
        "itineraries": [
            _kiwi_itinerary(1249, "https://kiwi.com/u/price1249", "2027-03-15", "2027-03-18", "2027-03-16")
        ],
    }

    from langchain_core.tools import StructuredTool

    async def search(location: str = "", pickup_date: str = "", dropoff_date: str = "", age: int = 30):
        return fixture

    async def sk_hotels_search(city: str, checkin: str, checkout: str, limit: int = 5):
        return hotels_md

    async def search_flight(departureDate: str = "", returnDate: str = ""):
        return flight

    async def fake_list(name):
        if name == "cars":
            return [StructuredTool.from_function(coroutine=search, name="search", description="OctoTrip search")]
        if name == "hotels":
            return [
                StructuredTool.from_function(
                    coroutine=sk_hotels_search, name="sk_hotels_search", description="Skiplagged hotels"
                )
            ]
        if name == "kiwi":
            return [
                StructuredTool.from_function(
                    coroutine=search_flight, name="search-flight", description="Kiwi flights"
                )
            ]
        raise ConnectionError(name)

    monkeypatch.setattr(agent, "list_server_tools", fake_list)
    monkeypatch.setattr(agent, "FakeTripModel", UnparseablePlanModel)
    # The live failure: the tool wrapper captured the record, then the specialist
    # ToolMessage never contained <<<STRUCTURED_CARS>>>. Only the out-of-band stash
    # (or a scan of a payload that is not on that message) can fill the plan.
    real_cars = agent.seal_car_tool_result
    real_hotels = agent.seal_hotel_tool_result

    def seal_cars_without_marker(result, args=None):
        real_cars(result, args)
        return "OctoTrip rows were not copied onto this tool message."

    def seal_hotels_without_marker(result, args=None):
        real_hotels(result, args)
        return "Skiplagged rows were not copied onto this tool message."

    monkeypatch.setattr(agent, "seal_car_tool_result", seal_cars_without_marker)
    monkeypatch.setattr(agent, "seal_hotel_tool_result", seal_hotels_without_marker)
    try:
        planner = agent.build_graph()
        final = await planner.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Plan 3 days in Lisbon from SFO for 2 people, "
                            "outbound 15 March 2027 and return 18 March 2027."
                        ),
                    }
                ]
            },
            config={"recursion_limit": 40},
        )
    finally:
        agent.clear_mcp_cache()
    text = agent.final_text(final)
    expected = {
        "Fiat 500": ("34.98", "Autounion", "https://mcp.octotrip.app/rental-cars/r/fiat-500"),
        "VW Polo": ("36.90", "Record", "https://mcp.octotrip.app/rental-cars/r/polo"),
        "Peugeot 2008": ("49.40", "CRC Car Rental Company", "https://mcp.octotrip.app/rental-cars/r/peugeot-2008"),
        "Peugeot 308": ("49.51", "CRC Car Rental Company", "https://mcp.octotrip.app/rental-cars/r/peugeot-308"),
        "Mercedes-Benz A-Class": ("95.88", "Masterkings", "https://mcp.octotrip.app/rental-cars/r/a-class"),
    }
    for name, (price, vendor, link) in expected.items():
        assert name in text
        assert price in text
        assert vendor in text
        assert link in text
    assert "Ferrari Roma" not in text
    assert "BMW 1 Series" not in text
    assert "Rental Cars ():" not in text
    assert "Rental Cars in Lisbon (\n" not in text
    assert "\n):" not in text
    assert "Rental Cars in Lisbon Airport (2027-03-12 to 2027-03-15):" in text
    assert "Inn Rossio Hotel" in text
    assert "Hotel ibis Lisboa Jose Malhoa" in text
    assert "Hotels in Lisbon (2027-03-15 to 2027-03-18):" in text
    assert "€67 per night" in text and "€231 total" in text
    assert "€47 per night" in text and "€174 total" in text
    assert "Hotels returned by the hotel specialist" not in text
    for label in ("Cancellation policy", "Booking link", "Neighborhood", "Nightly rate"):
        assert label not in text
    assert "1249 EUR" in text
    assert "https://kiwi.com/u/price1249" in text
    assert "assumed driver age 30 — tell me yours if different" in text


def test_blank_car_dates_do_not_render_empty_parens():
    text = agent.render_car_section(
        {
            "pickup": "Lisbon",
            "pickup_date": "\n\n",
            "dropoff_date": " \n ",
            "offers": [
                {
                    "name": "Fiat 500",
                    "price": "34.98",
                    "currency": "EUR",
                    "vendor": "Autounion",
                    "booking_url": "https://mcp.octotrip.app/rental-cars/r/fiat-500",
                }
            ],
        }
    )
    assert "()" not in text
    assert "(\n" not in text
    assert text.startswith("Rental Cars in Lisbon:")
    replaced = agent.replace_car_section(
        "Rental Cars in Lisbon (\n\n):\n",
        {
            "pickup": "Lisbon Airport",
            "pickup_date": "2027-03-12",
            "dropoff_date": "2027-03-15",
            "offers": [
                {
                    "name": "Fiat 500",
                    "price": "34.98",
                    "currency": "EUR",
                    "vendor": "Autounion",
                    "booking_url": "https://mcp.octotrip.app/rental-cars/r/fiat-500",
                }
            ],
        },
    )
    assert "Rental Cars in Lisbon Airport (2027-03-12 to 2027-03-15):" in replaced
    assert "Rental Cars in Lisbon (\n" not in replaced
    assert "\n):" not in replaced


def test_pub_crawl_is_dropped_from_bar_picks():
    evidence = (
        "**The Lisbon Pub Crawl**\n"
        "**Park Bar**\n"
        "VENUES: The Lisbon Pub Crawl | Park Bar"
    )
    text = agent.apply_coordinator_checks(
        "Night one\n**The Lisbon Pub Crawl**\n**Park Bar**",
        [ToolMessage(evidence, name="suggest_bars", tool_call_id="bars")],
    )
    assert "Pub Crawl" not in text
    assert "Park Bar" in text
    assert not agent.is_bar_venue("The Lisbon Pub Crawl")
    assert agent.is_bar_venue("Park Bar")


def test_hotel_stash_renders_nightly_and_total_when_the_tool_message_does_not():
    table = """
| Hotel | Price/night | Total | Booking |
| --- | --- | --- | --- |
| **The Lisbon Inn** | $52 | $157 | [View deal](https://skiplagged.com/hotel/1) |
| **Alfama Stay** | $88 | $263 | [View deal](https://skiplagged.com/hotel/2) |
| **River House** | $75 | $224 | [View deal](https://skiplagged.com/hotel/3) |
| **Bairro Alto Rooms** | $96 | $290 | [View deal](https://skiplagged.com/hotel/4) |
| **Chiado Hotel** | $106 | $319 | [View deal](https://skiplagged.com/hotel/5) |
"""
    agent.clear_capture()
    try:
        agent.seal_hotel_tool_result(
            table, {"city": "Lisbon", "checkin": "2027-03-12", "checkout": "2027-03-15"}
        )
        text = agent.apply_coordinator_checks(
            "Hotels\nThe Lisbon Inn, Alfama Stay, River House, Bairro Alto Rooms, and Chiado Hotel.",
            [
                ToolMessage(
                    "Skiplagged rows were not copied onto this tool message.",
                    name="search_hotels",
                    tool_call_id="hotels",
                )
            ],
        )
    finally:
        agent.clear_capture()
    assert "Hotels in Lisbon (2027-03-12 to 2027-03-15):" in text
    for total in ("$157", "$263", "$224", "$290", "$319"):
        assert total in text
    assert "$52 per night" in text
    assert "$106 per night" in text
    assert "https://skiplagged.com/hotel/1" in text


def _one_way_itinerary(origin, dest, day, price, link, flight="TP123"):
    return {
        "price": price,
        "currency": "EUR",
        "bookingUrl": link,
        "outbound": {
            "from": origin,
            "to": dest,
            "departureTime": f"{day}T08:00:00",
            "arrivalTime": f"{day}T16:00:00",
            "segments": [
                {
                    "from": origin,
                    "to": dest,
                    "departureTime": f"{day}T08:00:00",
                    "arrivalTime": f"{day}T16:00:00",
                    "carrierName": "TAP",
                    "flightNumber": flight,
                }
            ],
        },
    }


def _scripted_flight_model(script):
    class ScriptedFlightModel(agent.FakeTripModel):
        def _respond(self, messages):
            system = agent._system_text(messages)
            names = agent._tool_names(self.bound_tools)
            user = agent._human_text(messages)
            coordinator = "trip planner coordinator" in system or (
                "update_state" in names and "search_flights" in names
            )
            if coordinator:
                called = agent._called_tools(messages)
                if "update_state" in names and "update_state" not in called:
                    return AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "update_state",
                                "args": {
                                    "origin": "SFO",
                                    "destination": "Lisbon",
                                    "guest_count": "2",
                                    "genre": "city break",
                                },
                                "id": "call_update_state",
                                "type": "tool_call",
                            }
                        ],
                    )
                if "search_flights" in names and "search_flights" not in called:
                    return AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "search_flights",
                                "args": {"request": user},
                                "id": "call_search_flights",
                                "type": "tool_call",
                            }
                        ],
                    )
                return AIMessage(
                    content="Flights\nthere is no return flight on your specified date\n"
                )
            if "travel agent" in system:
                done = sum(
                    1
                    for message in messages
                    if isinstance(message, ToolMessage) and getattr(message, "name", "") == "search-flight"
                )
                if done < len(script):
                    return AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "search-flight",
                                "args": script[done],
                                "id": f"call_flight_{done}",
                                "type": "tool_call",
                            }
                        ],
                    )
                return AIMessage(content="No fare listed.")
            return AIMessage(content="Offline specialist.")

    return ScriptedFlightModel


async def _run_scripted_flights(monkeypatch, script, search):
    from langchain_core.tools import StructuredTool

    monkeypatch.setenv("TRIP_PLANNER_FAKE_LLM", "1")
    agent.clear_mcp_cache()

    async def fake_list(name):
        if name == "kiwi":
            return [
                StructuredTool.from_function(
                    coroutine=search, name="search-flight", description="Kiwi flights"
                )
            ]
        raise ConnectionError(name)

    monkeypatch.setattr(agent, "list_server_tools", fake_list)
    monkeypatch.setattr(agent, "FakeTripModel", _scripted_flight_model(script))
    try:
        planner = agent.build_graph()
        final = await planner.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Plan 3 days in Lisbon from SFO for 2 people, "
                            "outbound 12 March 2027 and return 15 March 2027."
                        ),
                    }
                ]
            },
            config={"recursion_limit": 40},
        )
    finally:
        agent.clear_mcp_cache()
    return agent.final_text(final)


@pytest.mark.asyncio
async def test_two_one_way_searches_become_the_12_to_15_itinerary(monkeypatch):
    async def search_flight(flyFrom: str = "", flyTo: str = "", departureDate: str = "", returnDate: str = ""):
        if flyFrom.upper() == "LIS":
            return {
                "currency": "EUR",
                "itineraries": [
                    _one_way_itinerary("LIS", "SFO", "2027-03-15", 1001.0, "https://kiwi.com/u/9bs2n3s")
                ],
            }
        return {
            "currency": "EUR",
            "itineraries": [
                _one_way_itinerary(
                    "SFO", "LIS", "2027-03-12", price, f"https://kiwi.com/u/out{price}", flight=f"TP{price}"
                )
                for price in (400, 500, 600, 900)
            ],
        }

    text = await _run_scripted_flights(
        monkeypatch,
        [
            {"flyFrom": "SFO", "flyTo": "LIS", "departureDate": "2027-03-12"},
            {"flyFrom": "LIS", "flyTo": "SFO", "departureDate": "2027-03-15"},
        ],
        search_flight,
    )
    assert "Flights (2027-03-12 to 2027-03-15):" in text
    assert "1001.0 EUR" in text
    assert "https://kiwi.com/u/9bs2n3s" in text
    assert "SFO -> LIS depart 2027-03-12T08:00:00" in text
    assert "LIS -> SFO depart 2027-03-15T08:00:00" in text
    assert "https://kiwi.com/u/out400" in text
    assert "https://kiwi.com/u/out500" in text
    assert "https://kiwi.com/u/out600" in text
    assert "https://kiwi.com/u/out900" not in text
    assert "no return flight" not in text.lower()
    assert "ALTERNATE DATES" not in text


@pytest.mark.asyncio
async def test_missing_return_date_is_filled_and_rendered(monkeypatch):
    seen = []

    async def search_flight(flyFrom: str = "", flyTo: str = "", departureDate: str = "", returnDate: str = ""):
        seen.append({"flyFrom": flyFrom, "flyTo": flyTo, "departureDate": departureDate, "returnDate": returnDate})
        if agent._parse_date(returnDate) and agent._parse_date(returnDate).isoformat() == "2027-03-15":
            item = _kiwi_itinerary(1249, "https://kiwi.com/u/tap1249", "2027-03-12", "2027-03-15")
            for segment in item["outbound"]["segments"]:
                segment["carrierName"] = "TAP"
            return {"currency": "EUR", "itineraries": [item]}
        return {"itineraries": []}

    text = await _run_scripted_flights(
        monkeypatch,
        [{"flyFrom": "SFO", "flyTo": "LIS", "departureDate": "2027-03-12"}],
        search_flight,
    )
    assert seen and agent._parse_date(seen[0]["returnDate"]).isoformat() == "2027-03-15"
    assert "Flights (2027-03-12 to 2027-03-15):" in text
    assert "1249 EUR" in text
    assert "https://kiwi.com/u/tap1249" in text
    assert "TAP" in text
    assert "SFO -> YUL depart 2027-03-12T08:25:00" in text
    assert "LIS -> SFO depart 2027-03-15T12:15:00" in text
    assert "no return flight" not in text.lower()


def _tap_round_trip(price: float, link: str) -> dict:
    return {
        "price": price,
        "currency": "EUR",
        "bookingUrl": link,
        "outbound": {
            "from": "LIS",
            "to": "OPO",
            "departureTime": "2027-03-15T08:00:00",
            "arrivalTime": "2027-03-15T09:05:00",
            "segments": [
                {
                    "from": "LIS",
                    "to": "OPO",
                    "departureTime": "2027-03-15T08:00:00",
                    "arrivalTime": "2027-03-15T09:05:00",
                    "carrierName": "TAP",
                    "flightNumber": "TP238",
                }
            ],
        },
        "inbound": {
            "from": "OPO",
            "to": "LIS",
            "departureTime": "2027-03-18T19:10:00",
            "arrivalTime": "2027-03-18T20:15:00",
            "segments": [
                {
                    "from": "OPO",
                    "to": "LIS",
                    "departureTime": "2027-03-18T19:10:00",
                    "arrivalTime": "2027-03-18T20:15:00",
                    "carrierName": "TAP",
                    "flightNumber": "TP237",
                }
            ],
        },
    }


def test_same_flight_legs_keep_the_cheapest_link():
    pricey = _tap_round_trip(510, "https://kiwi.com/duration-sort")
    cheap = _tap_round_trip(320, "https://kiwi.com/default-sort")
    other = _kiwi_itinerary(400, "https://kiwi.com/other", "2027-03-15", "2027-03-18")
    record = agent.assemble_flight_record(
        [
            {
                "args": {"departureDate": "2027-03-15", "returnDate": "2027-03-18"},
                "itineraries": [pricey, other],
                "currency": "EUR",
            },
            {
                "args": {"sort": "duration", "departureDate": "2027-03-15", "returnDate": "2027-03-18"},
                "itineraries": [cheap],
                "currency": "EUR",
            },
        ],
        date(2027, 3, 15),
        date(2027, 3, 18),
    )
    tap = [item for item in record["itineraries"] if "TP238" in " ".join(item["legs"])]
    assert len(tap) == 1
    assert tap[0]["booking_url"] == "https://kiwi.com/default-sort"
    assert "https://kiwi.com/duration-sort" not in str(record)
    assert any(item["booking_url"] == "https://kiwi.com/other" for item in record["itineraries"])
    assert "TP237" in " ".join(tap[0]["legs"])


def test_coordinator_does_not_restate_a_rendered_itinerary():
    record = agent.assemble_flight_record(
        [
            {
                "args": {"departureDate": "2027-03-15", "returnDate": "2027-03-18"},
                "itineraries": [_tap_round_trip(320, "https://kiwi.com/default-sort")],
                "currency": "EUR",
            }
        ],
        date(2027, 3, 15),
        date(2027, 3, 18),
    )
    rendered = agent.render_flight_section(record)
    answer = (
        rendered
        + "\n\n1. TAP TP238 / TP237 — 320 EUR https://kiwi.com/default-sort\n"
        + "2. Same flights via the duration search: https://kiwi.com/duration-sort\n"
    )
    text = agent.apply_coordinator_checks(
        answer,
        [
            ToolMessage(
                rendered + "\n" + agent.format_structured("FLIGHTS", record),
                name="search_flights",
                tool_call_id="flights",
            )
        ],
    )
    assert text.count("https://kiwi.com/default-sort") == 1
    assert "https://kiwi.com/duration-sort" not in text
    assert "\n1. TAP" not in text
    assert "TP238" in text and "TP237" in text
    assert rendered.splitlines()[0] in text


def test_coordinator_keeps_only_the_rendered_hotel_block():
    record = {
        "city": "Lisbon",
        "checkin": "2027-03-15",
        "checkout": "2027-03-18",
        "hotels": [
            {
                "name": "Inn Rossio Hotel",
                "nightly": "€67",
                "total": "€231",
                "link": "https://skiplagged.com/hotel/1",
            },
            {
                "name": "Hotel ibis Lisboa Jose Malhoa",
                "nightly": "€47",
                "total": "€174",
                "link": "https://skiplagged.com/hotel/2",
            },
        ],
    }
    rendered = agent.render_hotel_section(record)
    answer = (
        rendered
        + "\n\nHotels\n"
        + "1. Inn Rossio Hotel — €67 per night, €231 total\n"
        + "2. Hotel ibis Lisboa Jose Malhoa — €47 per night, €174 total\n"
        + "Dinner nearby is separate."
    )
    text = agent.apply_coordinator_checks(
        answer,
        [
            ToolMessage(
                rendered + "\n" + agent.format_structured("HOTELS", record),
                name="search_hotels",
                tool_call_id="hotels",
            )
        ],
    )
    assert text.count("Inn Rossio Hotel") == 1
    assert text.count("Hotel ibis Lisboa Jose Malhoa") == 1
    assert "\n1. " not in text
    assert "\n2. " not in text
    assert rendered.splitlines()[0] in text
    assert "https://skiplagged.com/hotel/1" in text
    assert "Dinner nearby is separate." in text


@pytest.mark.asyncio
async def test_lastminute_fails_fast_and_warns_once(monkeypatch, caplog):
    agent._lastminute_warned = False
    sleeps: list[float] = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(agent.asyncio, "sleep", fake_sleep)
    constructed = {"n": 0}

    class Adapter:
        def __init__(self, target):
            constructed["n"] += 1

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def list_tools(self, cache_mode="bypass"):
            raise AssertionError("lastminute must not open a retried MCP session")

    monkeypatch.setattr("langchain.mcp.MCPAdapter", Adapter)
    caplog.set_level(logging.WARNING, logger="trip_planner")
    with pytest.raises(MCPError) as first:
        await agent.list_server_tools("lastminute")
    with pytest.raises(MCPError):
        await agent.list_server_tools("lastminute")
    assert first.value.code == -32603
    assert constructed["n"] == 0
    assert sleeps == []
    warnings = [rec for rec in caplog.records if "lastminute" in rec.getMessage().casefold()]
    assert len(warnings) == 1


def test_start_config_binds_port_and_documents_env():
    script = Path("scripts/start-backend.sh").read_text()
    assert "--host 0.0.0.0" in script
    assert 'PORT="${PORT:-2024}"' in script
    assert '--port "$PORT"' in script
    assert "CORS_ALLOW_ORIGINS" in script
    assert 'CORS_ALLOW_ORIGINS="*"' not in script
    assert "http://localhost:3000" in script
    assert "start-backend.sh" in Path("Procfile").read_text()
    railway = Path("railway.toml").read_text()
    assert "start-backend.sh" in railway
    assert "/health" in railway
    ui_railway = Path("ui/railway.toml").read_text()
    ui_docker = Path("ui/Dockerfile").read_text()
    ui_package = Path("ui/package.json").read_text()
    assert "DOCKERFILE" in ui_railway
    assert "/health" in ui_railway
    assert "node server.js" in ui_docker or 'CMD ["node", "server.js"]' in ui_docker
    assert "HOSTNAME=0.0.0.0" in ui_docker
    assert "PORT=3000" in ui_docker
    assert "NEXT_PUBLIC_API_URL" in ui_docker
    assert "next start -H 0.0.0.0" in ui_package
    assert "${PORT:-3000}" in ui_package
    banner = Path("ui/src/components/thread/index.tsx").read_text()
    assert (
        "Illustrated coastal scenes: a resort pool, a seaside fort, and an observation tower"
        in banner
    )
    readme = Path("README.md").read_text()
    example = Path(".env.example").read_text()
    ui_example = Path("ui/.env.example").read_text()
    for key in ("OPENAI_API_KEY", "TAVILY_API_KEY", "GOOGLE_MAPS_API_KEY", "PORT", "CORS_ALLOW_ORIGINS"):
        assert key in readme
        assert key in example
    assert "NEXT_PUBLIC_API_URL" in readme
    assert "NEXT_PUBLIC_API_URL" in ui_example
    assert "langgraph dev" in readme
    assert "langgraph up" in readme
