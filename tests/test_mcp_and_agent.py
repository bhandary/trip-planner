import ast
import asyncio
import json
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
    assert targets["hotels_alt"] == HOTELS_ALT_URL
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


def test_prompts_match_live_corrections():
    assert "only look for one ticket, one way" not in agent.FLIGHT_PROMPT
    assert "return flight on the trip end date" in agent.FLIGHT_PROMPT
    assert "one specialist at a time" not in agent.COORDINATOR_PROMPT
    assert "in parallel" in agent.COORDINATOR_PROMPT
    assert "arrival airport" in agent.RENTAL_PROMPT
    assert "Assumed age" in agent.driver_age_note(None)
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
    messages = [
        ToolMessage(
            "**Ibis Parque das Nações** — 90 EUR",
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
    answer = "Skip the hotels.\nDinner at **Rafael Tonon**.\nAlso **Cervejaria Ramiro**."
    text = agent.apply_coordinator_checks(answer, messages)
    assert "Rafael Tonon" not in text
    assert "Cervejaria Ramiro" in text
    assert "Ibis Parque das Nações" in text
    assert "unavailable" in text.lower()


def test_hotel_omission_with_a_reason_is_kept():
    answer = "Excluded Ibis Parque das Nações because it is sold out."
    text = agent.ensure_hotels_kept(answer, ["Ibis Parque das Nações"])
    assert text == answer
    assert "Hotels returned" not in text


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
            }
        ]
    }
    specialist = json.dumps(offer) + "\n" + agent.driver_age_note(None)
    answer = "Reserve https://cars.example/peugeot"
    text = agent.ensure_cars_kept(answer, specialist)
    assert "Peugeot 2008" in text
    assert "49.4" in text
    assert "30 is an assumed age" in text


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
    assert "Assumed age 30" in str(result)


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
