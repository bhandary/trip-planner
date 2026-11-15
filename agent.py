"""Trip planner: a coordinator and five specialist agents.

``langchain-mcp-adapters`` (``MultiServerMCPClient``) is archived. MCP support
lives in ``langchain.mcp.MCPAdapter``, which is what this module uses. Each
specialist still loads only its own server, the same split as
``client.get_tools(server_name=...)``, then ``create_agent`` with that
server's tools plus Tavily ``web_search``.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import re
import uuid
from datetime import date, timedelta
from typing import Any, Awaitable, Callable, NamedTuple

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import after_model, before_model, wrap_tool_call
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool, ToolException
from langchain_tavily import TavilySearch
from langgraph.types import Command
from mcp.shared.exceptions import MCPError
from pydantic import Field
from typing_extensions import NotRequired

load_dotenv()

# Tool tasks often do not share the caller's context, and a large OctoTrip
# payload can be truncated before a trailer marker is stored on the ToolMessage.
# The parsed record is kept here for the specialist return and the final post-check.
_CAPTURE: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "trip_planner_capture", default=None
)
_RUN_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("trip_planner_run", default=None)
_RUNS: dict[str, dict[str, Any]] = {}
_LATEST: dict[str, Any] = {}
_WINDOW: dict[str, Any] = {"start": None, "end": None}

RETRYABLE_MCP_CODES = {-32603}
DEFAULT_MODEL = "openai:gpt-4.1-mini"

KIWI_URL = "https://mcp.kiwi.com"
# Skiplagged is the hotel source. lastminute.com's streamable-HTTP endpoint
# (https://mcp.lastminute.com/mcp) answers list_tools with HTTP 403 from a
# Cloudflare bot challenge. The MCP client turns that non-JSON body into
# JSON-RPC -32603, and retries only burn time. It is not a configured target.
HOTELS_URL = "https://mcp.skiplagged.com/mcp"
CARS_URL = "https://mcp.octotrip.app/rental-cars/mcp"
CARS_ALT_URL = "https://www.upfront-rentals.com/MCP/Streamable"
PLACES_URL = "https://mapstools.googleapis.com/mcp"

logger = logging.getLogger("trip_planner")
_lastminute_warned = False

_tool_cache: dict[str, list[BaseTool]] = {}
_specialists: dict[str, Any] | None = None
_specialist_lock: asyncio.Lock | None = None


def resolve_model_name() -> str:
    """Hosted model selected by env. Defaults to OpenAI, never a local runtime."""
    selected = os.getenv("TRIP_PLANNER_MODEL", "").strip()
    if selected:
        return selected
    openai_model = os.getenv("OPENAI_MODEL", "").strip()
    if openai_model:
        return openai_model if ":" in openai_model else f"openai:{openai_model}"
    return DEFAULT_MODEL


def _fake_llm_enabled() -> bool:
    return os.getenv("TRIP_PLANNER_FAKE_LLM", "").strip() == "1"


def _max_retries() -> int:
    raw = os.getenv("MCP_MAX_RETRIES", "3").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 3


def _timeout_seconds() -> float:
    raw = os.getenv("MCP_TIMEOUT_SECONDS", "30").strip()
    try:
        return float(raw)
    except ValueError:
        return 30.0


def google_maps_key() -> str:
    return os.getenv("GOOGLE_MAPS_KEY", "").strip() or os.getenv("GOOGLE_MAPS_API_KEY", "").strip()


def mcp_targets() -> dict[str, Any]:
    """Streamable HTTP MCP targets. ``places`` is omitted until a Maps key is set."""
    # OctoTrip rejects the default HTTP client user agent (ua_blocked).
    from fastmcp.client.transports import StreamableHttpTransport

    targets: dict[str, Any] = {
        "kiwi": KIWI_URL,
        "hotels": HOTELS_URL,
        "cars": StreamableHttpTransport(
            CARS_URL,
            headers={"User-Agent": "Mozilla/5.0 (compatible; trip-planner/1.0)"},
        ),
        "cars_alt": CARS_ALT_URL,
    }
    key = google_maps_key()
    if key:
        targets["places"] = StreamableHttpTransport(
            PLACES_URL,
            headers={"X-Goog-Api-Key": key},
        )
    return targets


def clear_capture() -> None:
    _LATEST.clear()
    _RUNS.clear()
    _CAPTURE.set(None)
    _RUN_ID.set(None)
    _WINDOW["start"] = None
    _WINDOW["end"] = None


def clear_mcp_cache() -> None:
    global _specialists
    _tool_cache.clear()
    _specialists = None
    clear_capture()


def begin_capture(*, reset: bool = False) -> dict[str, Any]:
    """One mutable record bag for this run. Tool tasks mutate it; they do not replace it."""
    current = _CAPTURE.get()
    run_id = _RUN_ID.get()
    if reset:
        run_id = uuid.uuid4().hex
        current = {"cars": None, "hotels": None, "flights": None}
        _RUN_ID.set(run_id)
        _CAPTURE.set(current)
        _RUNS.clear()
        _RUNS[run_id] = current
        _LATEST.clear()
        _WINDOW["start"] = None
        _WINDOW["end"] = None
        return current
    if current is not None and run_id is not None:
        return current
    # A tool task that did not inherit the caller bag must not wipe the other kind.
    run_id = run_id or uuid.uuid4().hex
    current = {"cars": None, "hotels": None, "flights": None}
    _RUN_ID.set(run_id)
    _CAPTURE.set(current)
    _RUNS[run_id] = current
    return current


def stash_record(kind: str, record: dict[str, Any]) -> None:
    box = _CAPTURE.get()
    run_id = _RUN_ID.get()
    if box is None or run_id is None:
        box = begin_capture()
        run_id = _RUN_ID.get()
    box[kind] = record
    if run_id:
        _RUNS.setdefault(run_id, box)[kind] = record
    _LATEST[kind] = record


def _record_ready(kind: str, record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    if kind == "cars":
        return bool(record.get("offers"))
    if kind == "hotels":
        return bool(record.get("hotels"))
    if kind == "flights":
        return bool(record.get("itineraries"))
    return bool(record)


def stashed_record(kind: str) -> dict[str, Any] | None:
    box = _CAPTURE.get()
    if isinstance(box, dict) and _record_ready(kind, box.get(kind)):
        return box[kind]
    run_id = _RUN_ID.get()
    shared = _RUNS.get(run_id or "", {})
    if isinstance(shared, dict) and _record_ready(kind, shared.get(kind)):
        return shared[kind]
    latest = _LATEST.get(kind)
    if _record_ready(kind, latest):
        return latest
    return None


def bind_trip_window(text: str) -> None:
    """Remember the traveler's start and end dates for flight calls that omit one of them."""
    start, end = requested_trip_dates(text or "")
    if start and end:
        _WINDOW["start"] = start
        _WINDOW["end"] = end


def trip_window() -> tuple[Any, Any]:
    return _WINDOW.get("start"), _WINDOW.get("end")


def _open_capture(state: dict[str, Any], runtime: Any) -> None:
    """Start a fresh bag before the first model call. Later calls keep tool captures."""
    messages = state.get("messages") or []
    seen_tool = any(getattr(message, "type", None) == "tool" or isinstance(message, ToolMessage) for message in messages)
    begin_capture(reset=not seen_tool)
    if not seen_tool:
        bind_trip_window(_human_text(messages))


def _lock() -> asyncio.Lock:
    global _specialist_lock
    if _specialist_lock is None:
        _specialist_lock = asyncio.Lock()
    return _specialist_lock


def _error_code(exc: BaseException) -> int | None:
    error = getattr(exc, "error", None)
    code = getattr(error, "code", None)
    if isinstance(code, int):
        return code
    own = getattr(exc, "code", None)
    return own if isinstance(own, int) else None


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, ToolException):
        return False
    code = _error_code(exc)
    if code is None:
        return True
    return code in RETRYABLE_MCP_CODES


def warn_lastminute_once() -> None:
    """Log the lastminute.com outage once per process. Do not retry it."""
    global _lastminute_warned
    if _lastminute_warned:
        return
    _lastminute_warned = True
    logger.warning(
        "lastminute.com MCP list_tools fails with JSON-RPC -32603 "
        "(HTTP 403 Cloudflare bot challenge, not a retryable protocol error). "
        "Using Skiplagged and not retrying."
    )


def _target_url(target: Any) -> str:
    return str(getattr(target, "url", target) or "")


def _is_lastminute_server(server_name: str) -> bool:
    if server_name == "lastminute":
        return True
    return "lastminute.com" in _target_url(mcp_targets().get(server_name))


async def call_with_retry(call: Callable[[], Awaitable[Any]]) -> Any:
    """Retry transient MCP failures. A definitive server error is not retried."""
    last: BaseException | None = None
    attempts = _max_retries()
    for attempt in range(attempts):
        try:
            return await call()
        except Exception as exc:
            last = exc
            print(
                f"[MCP retry] {type(exc).__name__} "
                f"(attempt {attempt + 1}/{attempts}): {exc}"
            )
            if not _retryable(exc) or attempt == attempts - 1:
                raise
            await asyncio.sleep(2**attempt)
    assert last is not None
    raise last


async def list_server_tools(server_name: str) -> list[BaseTool]:
    """List tools for one server. Same role as ``get_tools(server_name=...)``."""
    if server_name in _tool_cache:
        return _tool_cache[server_name]
    if _is_lastminute_server(server_name):
        warn_lastminute_once()
        raise MCPError(
            -32603,
            "lastminute.com MCP unavailable (Cloudflare challenge); not retrying",
        )
    targets = mcp_targets()
    if server_name not in targets:
        raise KeyError(f"MCP server {server_name!r} is not configured")
    from langchain.mcp import MCPAdapter

    target = targets[server_name]

    async def _list() -> list[BaseTool]:
        async with MCPAdapter(target) as adapter:
            return await asyncio.wait_for(
                adapter.list_tools(cache_mode="bypass"),
                timeout=_timeout_seconds(),
            )

    tools = await call_with_retry(_list)
    _tool_cache[server_name] = list(tools)
    return _tool_cache[server_name]


async def load_with_fallback(
    primary: str,
    alt: str | None = None,
) -> tuple[list[BaseTool], str | None, list[str]]:
    """Use the primary server. On error or timeout, use the alt server."""
    errors: list[str] = []
    try:
        tools = await list_server_tools(primary)
        if tools:
            return list(tools), primary, errors
        errors.append(f"{primary}: server returned no tools")
    except Exception as exc:
        errors.append(f"{primary}: {type(exc).__name__}: {exc}")
    if alt:
        try:
            tools = await list_server_tools(alt)
            if tools:
                return list(tools), alt, errors
            errors.append(f"{alt}: server returned no tools")
        except Exception as exc:
            errors.append(f"{alt}: {type(exc).__name__}: {exc}")
    return [], None, errors


class EmptySource(Exception):
    """A tool call succeeded but returned no usable rows."""

    def __init__(self, result: Any):
        self.result = result
        super().__init__("no_results")


def result_is_empty(value: Any) -> bool:
    """True when a supplier returned no rows, including an explicit no_results."""
    if value is None:
        return True
    if isinstance(value, tuple) and value:
        return result_is_empty(value[0])
    if isinstance(value, list):
        if not value:
            return True
        if all(isinstance(item, dict) and "text" in item for item in value):
            return all(result_is_empty(item.get("text")) for item in value)
        return False
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in {"", "null", "none", "[]", "{}", "no results", "no_results"}:
            return True
        if re.search(r"\bno[_\s-]?results\b", text, re.I):
            return True
        try:
            return result_is_empty(json.loads(text))
        except json.JSONDecodeError:
            return False
    if isinstance(value, dict):
        if not value:
            return True
        if value.get("no_results") is True:
            return True
        status = str(value.get("status") or value.get("error") or "").lower()
        if status in {"no_results", "empty", "not_found"}:
            return True
        for key in ("results", "offers", "cars", "vehicles", "itineraries", "data"):
            if key in value and value[key] in (None, [], {}):
                return True
        if value.get("count") == 0:
            return True
    return False


def _tool_result(tool: BaseTool, value: Any) -> Any:
    if getattr(tool, "response_format", "content") != "content_and_artifact":
        return value
    if isinstance(value, tuple) and len(value) == 2:
        return value
    return (value, None)


def _tool_by_name(tools: list[BaseTool], name: str) -> BaseTool | None:
    return next((candidate for candidate in tools if candidate.name == name), None)


def _accepted_args(tool: BaseTool, raw: dict[str, Any]) -> dict[str, Any]:
    schema = getattr(tool, "args_schema", None)
    props: dict[str, Any] = {}
    if schema is not None and hasattr(schema, "model_json_schema"):
        props = schema.model_json_schema().get("properties") or {}
    elif isinstance(schema, dict):
        props = schema.get("properties") or {}
    filled = {key: value for key, value in raw.items() if value not in ("", None)}
    if not props:
        return filled
    return {key: value for key, value in filled.items() if key in props}


ASSUMED_DRIVER_AGE = "assumed driver age 30 — tell me yours if different"


def driver_age_note(age: Any) -> str:
    """Use a stated age. Otherwise say that 30 was assumed."""
    if age is None or str(age).strip() == "":
        return ASSUMED_DRIVER_AGE
    return f"Driver age {age} was taken from the request."


def stated_driver_age(text: str) -> str | None:
    match = re.search(r"\b(?:age|aged)\s+(\d{2})\b", text or "", re.I)
    if match:
        return match.group(1)
    match = re.search(r"\b(\d{2})\s*(?:years?\s*old|-year-old)\b", text or "", re.I)
    if match:
        return match.group(1)
    return None


def airport_args_for(args: dict[str, Any]) -> dict[str, Any] | None:
    """Retry a city pickup at that city's airport. Leave an airport query alone."""
    for key in ("location", "pickup", "pickup_location", "pickUpLocation", "city"):
        value = args.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        if re.search(r"\bairport\b", value, re.I):
            return None
        updated = dict(args)
        airport = value.strip().rstrip(".") + " Airport"
        updated[key] = airport
        for drop_key in ("dropoff_location", "dropoff", "dropOffLocation", "dropoff_location"):
            current = updated.get(drop_key)
            if isinstance(current, str) and current.strip() and not re.search(r"\bairport\b", current, re.I):
                updated[drop_key] = airport
        return updated
    return None


async def upfront_search_fallback(alt_tools: list[BaseTool], args: dict[str, Any]) -> str | None:
    """Map an OctoTrip search onto Upfront get_locations + get_offers."""
    locations = _tool_by_name(alt_tools, "get_locations")
    offers = _tool_by_name(alt_tools, "get_offers")
    if locations is None or offers is None:
        return None
    location = (
        args.get("location")
        or args.get("pickup")
        or args.get("pickup_location")
        or args.get("city")
        or ""
    )
    pickup_date = args.get("pickup_date") or args.get("pickupDate") or args.get("from") or ""
    dropoff_date = args.get("dropoff_date") or args.get("dropoffDate") or args.get("to") or ""
    try:
        loc_result = await locations.ainvoke(_accepted_args(locations, {"location": location, "query": location}))
    except Exception as exc:
        loc_result = f"get_locations failed: {exc}"
    offer_raw = {
        "location": location,
        "pickup_location": location,
        "pickup": location,
        "pickup_date": pickup_date,
        "dropoff_date": dropoff_date,
        "pickupDate": pickup_date,
        "dropoffDate": dropoff_date,
    }
    try:
        offer_result = await offers.ainvoke(_accepted_args(offers, offer_raw) or offer_raw)
    except Exception as exc:
        offer_result = f"get_offers failed: {exc}"
    return (
        "OctoTrip search returned no results. Fallback source: Upfront Rentals "
        "(get_locations + get_offers).\n"
        f"locations: {loc_result}\n"
        f"offers: {offer_result}"
    )


async def invoke_with_fallback(
    primary_name: str,
    primary_call: Callable[[dict[str, Any]], Awaitable[Any]],
    alt_tools: list[BaseTool],
    args: dict[str, Any],
) -> Any:
    """Call the primary tool. On error, timeout, or empty/no_results, use the alt."""
    try:
        result = await primary_call(args)
    except Exception as exc:
        failure: BaseException = exc
    else:
        if not result_is_empty(result):
            return result
        failure = EmptySource(result)

    if primary_name == "search":
        mapped = await upfront_search_fallback(alt_tools, args)
        if mapped is not None:
            return mapped

    alt = _tool_by_name(alt_tools, primary_name)
    if alt is None:
        return (
            f"Source unavailable ({primary_name}). "
            f"Primary failed: {failure}. "
            f"No fallback tool named {primary_name}."
        )
    try:
        return await alt.ainvoke(args)
    except Exception as alt_exc:
        return (
            f"Source unavailable ({primary_name}). "
            f"Primary: {failure}. Fallback: {alt_exc}."
        )


def attach_call_fallback(tool: BaseTool, alt_server: str) -> BaseTool:
    """On a failed primary call, try the same tool name on the alt server."""
    if getattr(tool, "_trip_fallback", False):
        return tool
    original = tool.coroutine
    if original is None:
        return tool

    async def wrapped(**kwargs: Any) -> Any:
        async def primary_call(args: dict[str, Any]) -> Any:
            result = await call_with_retry(lambda: original(**args))
            if (
                result_is_empty(result)
                and tool.name == "search"
                and alt_server == "cars_alt"
            ):
                airport_args = airport_args_for(args)
                if airport_args is not None:
                    retried = await call_with_retry(lambda: original(**airport_args))
                    if not result_is_empty(retried):
                        return retried
                    raise EmptySource(retried)
            if result_is_empty(result):
                raise EmptySource(result)
            return result

        try:
            result = await primary_call(kwargs)
        except Exception as exc:
            try:
                alt_tools = await list_server_tools(alt_server)
            except Exception as alt_exc:
                return _tool_result(
                    tool,
                    (
                        f"Source unavailable ({tool.name}). Primary failed: {exc}. "
                        f"Fallback server {alt_server} unavailable: {alt_exc}."
                    ),
                )

            async def already_failed(_args: dict[str, Any]) -> Any:
                raise exc

            fallback_args = kwargs
            if tool.name == "search" and alt_server == "cars_alt" and isinstance(exc, EmptySource):
                fallback_args = airport_args_for(kwargs) or kwargs
            result = await invoke_with_fallback(
                tool.name, already_failed, alt_tools, fallback_args
            )
        if tool.name == "search" and alt_server == "cars_alt":
            # OctoTrip defaults age to 30. That default is not the traveler's age.
            raw_age = kwargs.get("age", kwargs.get("driver_age"))
            stated = None if raw_age in (None, "", 30, "30", "30+", 30.0) else raw_age
            note = driver_age_note(stated)
            # Parse the supplier payload here, before any specialist prose exists.
            sealed = seal_car_tool_result(result, kwargs)
            return _tool_result(tool, f"{sealed}\n{note}")
        if isinstance(result, tuple):
            return result
        return _tool_result(tool, result)

    tool.coroutine = wrapped
    setattr(tool, "_trip_fallback", True)
    return tool


def _source_note(label: str, source: str | None, errors: list[str]) -> str:
    if source == label:
        return ""
    detail = "; ".join(errors) if errors else "no details"
    if source is None:
        return (
            f"\n\nLive {label} data is unavailable ({detail}). "
            "Say that the source is unavailable. Do not invent venues, ratings, "
            "hours, prices, or availability."
        )
    return (
        f"\n\nPrimary {label} server failed ({detail}). "
        f"You are using fallback server {source}. Name that source in the answer."
    )


@tool
async def web_search(query: str, search_number: int, max_search_number: int) -> dict[str, Any]:
    """Search the web for information. You must track your search count by providing
    search_number (starting at 1) and max_search_number on every call.
    Queries must use only plain text characters. Do not use accented or special characters
    (e.g., use 'capacite' instead of 'capacité').

    Use this for gaps the travel MCP servers do not cover (events, attractions,
    advisories, opening hours, fees). Do not use it in place of live MCP prices.
    """
    if search_number > max_search_number:
        return {
            "message": "Search limit reached. Please summarize your findings and provide your final answer."
        }
    if not os.getenv("TAVILY_API_KEY", "").strip():
        return {"error": "Tavily unavailable: TAVILY_API_KEY is not set"}
    query = strip_trip_dates(query)
    try:
        search = TavilySearch(max_results=5)
        result = await search.ainvoke({"query": query})
        return {"results": result}
    except Exception as exc:
        return {"error": str(exc)}


FLIGHT_PROMPT = """
You are a travel agent. Search for flights to the desired destination.
You are not allowed to ask any more follow up questions, you must find the best flight options based on the following criteria:
- Price (lowest, economy class)
- Duration (shortest)
- Date (time of year which you believe is best, unless the task already names dates)
When the trip implies a return (a stay, a round trip, or an end date), search the outbound flight on the trip start date and the return flight on the trip end date. Search one-way only when the traveler asked for one-way. Do not search other date ranges. A result for different dates is an alternate: label it as different dates and do not present it as the trip. A return-direction flight on the trip end date is the inbound leg, not a missing return.
Each Kiwi itinerary block is one result. Copy its price, legs, and booking link together. Never attach one itinerary's booking link or price to another itinerary's legs. Two searches that return the same flight numbers and departure and arrival times are one itinerary: keep the cheaper booking link and do not list it twice.
If a block warns that arrival is a day or more after the trip start, or that the trip is mostly eaten by travel, repeat that warning. Show the currency Kiwi returned (for example EUR). Do not relabel it as dollars. If a return arrives on a later calendar day, label it next-day arrival.
You may need to make multiple searches to iteratively find the best options.
You will be given the origin and destination. It is your job to think critically about the best options.
If an MCP tool fails, returns malformed output, or does not give you usable flight results, try the tool again.
Use web_search only for gaps the flight tools do not cover, such as visa rules or airport transfer time.
Do not replace live flight prices with web search. If the flight source is unavailable, say so and do not invent prices.
Once you have found the best options, return a shortlist.
""".strip()

HOTEL_PROMPT = """
You are the hotel specialist in a multi-agent trip planning system. A supervisor
agent sends you one lodging task at a time and merges your answer with results
from flight, rental car, restaurant, and bar specialists. Your reader is the
supervisor, not the traveler, so return facts it can combine, not a sales pitch.

Use the hotel search tools to find live availability and prices. Use web search
only when the hotel tools return nothing useful or lack a detail the task needs
(for example resort fees, parking cost, or neighborhood safety), and say which
facts came from web search because they may be stale. Do not replace live hotel
rates with web search.

Before searching, check that you have destination, check-in and check-out dates,
number of guests, and number of rooms. If one is missing and you cannot infer it
from the task, return a short message naming what is missing instead of guessing.
Budget, neighborhood, star level, and amenities are optional; apply them when given.

Return 3 to 5 options, ranked by fit to the stated preferences. For each one give:
name, neighborhood, nightly rate and total for the stay with currency, whether
taxes and fees are included, rating, cancellation policy if shown, two or three
relevant amenities, and the booking link. Add one line on the tradeoff of each
option (for example cheaper but 25 minutes from the venue).
The hotel name is the property name in the tool result (the Skiplagged Hotel
column). Neighborhood, Nightly rate, Cancellation, and Tradeoff are field
labels, not hotels.

Report only prices and availability that a tool returned in this conversation.
If results are thin or a constraint could not be met, say so plainly and suggest
the nearest alternative. You search and compare; you do not book.
""".strip()

RENTAL_PROMPT = """
You are the rental car specialist in a multi-agent trip planning system. A supervisor
agent sends you one car task at a time and merges your answer with results from
flight, hotel, restaurant, and bar specialists. Your reader is the supervisor, not
the traveler, so return facts it can combine, not a sales pitch.

Use the rental car search tools to find live availability and prices. Use web search
only when those tools return nothing useful or lack a detail the task needs (for
example insurance excess, young-driver fees, or cross-border rules), and say which
facts came from web search because they may be stale. Do not replace live rates
with web search.

For a fly-in trip, default pickup and drop-off to the arrival airport unless the
task names a different place. If a city search returns no_results, search that
city's airport before giving up.
Do not guess a driver age. If the traveler stated an age, use that age. If not,
pass 30 to the tool and write exactly: assumed driver age 30 — tell me yours if different.
Do not write "driver age 30+" as if the traveler said it. Keep each vehicle's
name, price, vendor, and booking link together. Vehicle class, transmission, and budget are optional; apply them
when given.

Return 3 to 5 options, ranked by fit. For each one give: supplier, vehicle class,
pickup location, total price and currency, whether taxes and insurance are included,
mileage rules, cancellation policy if shown, and the booking link. Add one line on
the tradeoff of each option.

Report only prices and availability that a tool returned in this conversation.
If results are thin or a constraint could not be met, say so plainly and suggest
the nearest alternative. You search and compare; you do not book.
""".strip()

RESTAURANT_PROMPT = """
You are the restaurant specialist in a multi-agent trip planning system. A
supervisor agent sends you one dining task at a time and merges your answer
into an overall itinerary. Your reader is the supervisor, so return compact
facts it can slot into a schedule.

Use the places search tools to find restaurants, ratings, hours, and locations.
Use web search to fill gaps those tools leave, such as current menus, tasting
menu prices, reservation difficulty, or recent closures, and flag facts that
came from web search. Do not replace a places result with a web-only listing
when the places tools already answered the question.
Do not put trip dates into places or web search queries.

Name a venue only when a places tool returned it as a place, or when a Tavily
result has a clearly named venue heading or entry. Never use a person, byline,
or photo credit as a restaurant. If places and web search both have no venue
data, say the source is unavailable and name nothing.

Work from the location, meal, and party size in the task. If the task
includes a hotel or activity location, prefer places within a short walk or
drive of it and state the distance. Treat dietary restrictions and allergies as
hard constraints: only include a place when you have evidence it can accommodate
them, and say what that evidence is. Cuisine, budget, and atmosphere are
preferences; honor them where you can. Do not relabel cuisine beyond the source.
If the source lists shrimp stew, do not call the place vegetarian. Do not
recommend a meal the source did not mention: if the source says lunch, do not
recommend the place for dinner.

Return 3 to 5 options per meal requested, with some variety in price and style
unless the task is narrow. For each give: name, cuisine as the source states it,
neighborhood and distance from the reference point, price level, rating, and
hours only when a tool stated them. Say whether reservations are needed and how
to book, and one sentence on why it fits this traveler.

Do not say hours or open status are confirmed for the trip dates unless a tool
result states that. Otherwise say check hours before going. Report only details
a tool returned. You recommend; you do not make reservations.
""".strip()

BAR_PROMPT = """
You are the bar and nightlife specialist in a multi-agent trip planning system.
A supervisor agent sends you one task at a time and merges your answer into an
overall itinerary. Your reader is the supervisor, so return compact facts it
can slot into an evening plan.

Use the places search tools to find bars, ratings, hours, and locations. Bar
details are often thin in structured data, so use web search for what is
missing: drink focus, atmosphere, cover charges, dress codes, live music
schedules, and whether a place has closed recently. Flag facts that came from
web search. Do not replace a places result with web search when the places
tools already answered the question.
Do not put trip dates into places or web search queries.
Name a venue only when a places tool returned it as a place, or when a Tavily
result has a clearly named venue heading or entry. Never use a person, byline,
or photo credit as a bar. Do not recommend pub crawls, walking tours, bus
tours, or other activities that are not a bar. If places and web search both
have no venue data, say the source is unavailable and name nothing.

Work from the location and time window in the task. Match the type of
venue to what was asked (cocktail bar, brewery or taproom, wine bar, rooftop,
dive, live music, sports bar) and do not substitute a different type without
saying so. If the task mentions a dinner spot or hotel, prefer venues within
walking distance of it and state the distance, since travelers who are drinking
should not need to drive. Mention late-night transport options when a venue is
far from the hotel. Do not relabel the venue beyond the source, and do not
move it to a meal or night the source did not mention.

Return 3 to 5 options. For each give: name, type, neighborhood and distance
from the reference point, price level, rating, hours for the requested night,
any cover, reservation, or dress code requirement, and one sentence on the
atmosphere and what it is known for.

Assume all travelers are of legal drinking age for the destination unless the
task says otherwise; if the party includes minors, only include venues that
admit them and say so. Do not say hours or open status are confirmed for the
trip dates unless a tool result states that. Otherwise say check hours before
going. Report only details a tool returned. You recommend; you do not book.
""".strip()


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("text"):
                parts.append(str(block["text"]))
        return "\n".join(parts)
    return str(content or "")


def final_text(result: dict[str, Any]) -> str:
    messages = result.get("messages") or []
    for message in reversed(messages):
        content = getattr(message, "content", None)
        tool_calls = getattr(message, "tool_calls", None)
        if content and not tool_calls:
            text = _message_text(content).strip()
            if text:
                return text
    return "Specialist returned no text."


def _system_text(messages: list[BaseMessage]) -> str:
    chunks = [
        _message_text(message.content)
        for message in messages
        if getattr(message, "type", None) == "system"
    ]
    return "\n".join(chunks).lower()


def _human_text(messages: list[BaseMessage]) -> str:
    chunks = [
        _message_text(message.content)
        for message in messages
        if getattr(message, "type", None) == "human"
    ]
    return "\n".join(chunks)


def _tool_names(tools: list[Any]) -> set[str]:
    names: set[str] = set()
    for item in tools:
        name = getattr(item, "name", None)
        if name is None and isinstance(item, dict):
            name = item.get("name")
        if name:
            names.add(name)
    return names


def _called_tools(messages: list[BaseMessage]) -> set[str]:
    id_to_name: dict[str, str] = {}
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls or []:
                if call.get("id") and call.get("name"):
                    id_to_name[call["id"]] = call["name"]
    called: set[str] = set()
    for message in messages:
        if isinstance(message, ToolMessage):
            name = getattr(message, "name", None) or id_to_name.get(message.tool_call_id or "")
            if name:
                called.add(name)
    return called


def _infer_destination(text: str) -> str:
    match = re.search(r"\bin\s+([A-Z][A-Za-z]+)", text)
    return match.group(1) if match else "unspecified"


def _infer_origin(text: str) -> str:
    match = re.search(r"\bfrom\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)", text)
    return match.group(1) if match else "unspecified"


def _infer_guests(text: str) -> str:
    match = re.search(r"\b(\d+)\s+(?:people|guests|travelers|travellers)\b", text, re.I)
    return match.group(1) if match else "1"


class FakeTripModel(BaseChatModel):
    """Deterministic stand-in used when TRIP_PLANNER_FAKE_LLM=1.

    The coordinator still delegates to the specialist tools. Specialists answer
    directly so a missing API key does not block a local run.
    """

    bound_tools: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "trip-planner-fake"

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> FakeTripModel:
        return self.model_copy(update={"bound_tools": list(tools)})

    def _respond(self, messages: list[BaseMessage]) -> AIMessage:
        system = _system_text(messages)
        names = _tool_names(self.bound_tools)
        user = _human_text(messages)
        coordinator = "trip planner coordinator" in system or (
            "update_state" in names and "search_flights" in names
        )
        if coordinator:
            return self._coordinate(messages, names, user)
        role = "specialist"
        for label in (
            "rental car specialist",
            "hotel specialist",
            "restaurant specialist",
            "bar and nightlife specialist",
            "travel agent",
        ):
            if label in system:
                role = label
                break
        return AIMessage(
            content=(
                f"{role} notes for this request: {user}\n"
                "Offline stand-in model: supplier tools were not called. "
                "Confirm live prices when a hosted model and API keys are configured."
            )
        )

    def _coordinate(self, messages: list[BaseMessage], names: set[str], user: str) -> AIMessage:
        called = _called_tools(messages)
        destination = _infer_destination(user)
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
                args: dict[str, Any] = {
                    "origin": _infer_origin(user),
                    "destination": destination,
                    "guest_count": _infer_guests(user),
                    "genre": "city break",
                }
            else:
                args = {"request": user or f"Plan the trip to {destination}."}
            return AIMessage(
                content="",
                tool_calls=[{"name": name, "args": args, "id": f"call_{name}", "type": "tool_call"}],
            )
        sections = []
        for message in messages:
            if isinstance(message, ToolMessage):
                title = getattr(message, "name", None) or "tool"
                sections.append(f"{title}: {_message_text(message.content)}")
        body = "\n\n".join(sections) if sections else "No specialist results."
        return AIMessage(content=f"Trip plan for {destination}\n\n{body}")

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._respond(list(messages)))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


class ResolvingModel(BaseChatModel):
    """Pick the hosted model, or the offline stand-in, at call time.

    Constructing the graph does not require an API key. Production on Railway
    sets OPENAI_API_KEY and leaves TRIP_PLANNER_FAKE_LLM unset.
    """

    bound_tools: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "trip-planner-resolving"

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> ResolvingModel:
        return self.model_copy(update={"bound_tools": list(tools)})

    def _resolve(self) -> Any:
        if _fake_llm_enabled():
            model: Any = FakeTripModel()
        else:
            model = init_chat_model(resolve_model_name())
        if self.bound_tools:
            model = model.bind_tools(self.bound_tools)
        return model

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = self._resolve().invoke(list(messages))
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = await self._resolve().ainvoke(list(messages))
        return ChatResult(generations=[ChatGeneration(message=message)])


def _agent_model(model: BaseChatModel | None) -> BaseChatModel:
    if model is not None:
        return model
    return ResolvingModel()


async def build_specialists(model: BaseChatModel | None = None) -> dict[str, Any]:
    """Build the five specialists from their MCP servers and Tavily web search.

    Mirrors the requested wiring: one catalog, per-server tool lists, then
    ``create_agent(model, tools=server_tools + [web_search], system_prompt=...)``.
    Skiplagged is the hotel source. lastminute.com is not called. Upfront is the
    car fallback. Google Maps is registered only when ``GOOGLE_MAPS_KEY`` or
    ``GOOGLE_MAPS_API_KEY`` is set.
    """
    flight_tools, flight_source, flight_errors = await load_with_fallback("kiwi")
    hotel_tools, hotel_source, hotel_errors = await load_with_fallback("hotels")
    car_tools, car_source, car_errors = await load_with_fallback("cars", "cars_alt")
    if "places" in mcp_targets():
        place_tools, place_source, place_errors = await load_with_fallback("places")
    else:
        place_tools, place_source, place_errors = (
            [],
            None,
            ["places: skipped, set GOOGLE_MAPS_KEY or GOOGLE_MAPS_API_KEY"],
        )

    flight_tools = [attach_flight_binding(item) for item in flight_tools]
    place_tools = [attach_date_strip(item) for item in place_tools]
    hotel_tools = [
        attach_hotel_capture(item)
        for item in hotel_tools
        if not _is_flight_tool_name(item.name)
    ]
    if car_source == "cars":
        car_tools = [attach_call_fallback(item, "cars_alt") for item in car_tools]
    else:
        car_tools = [attach_car_capture(item) for item in car_tools]

    shared = [web_search]
    flight_agent = create_agent(
        _agent_model(model),
        tools=flight_tools + shared,
        system_prompt=FLIGHT_PROMPT + _source_note("kiwi", flight_source, flight_errors),
        middleware=[retry_mcp_middleware],
    )
    hotel_agent = create_agent(
        _agent_model(model),
        tools=hotel_tools + shared,
        system_prompt=HOTEL_PROMPT + _source_note("hotels", hotel_source, hotel_errors),
        middleware=[retry_mcp_middleware],
    )
    rental_car_agent = create_agent(
        _agent_model(model),
        tools=car_tools + shared,
        system_prompt=RENTAL_PROMPT + _source_note("cars", car_source, car_errors),
        middleware=[retry_mcp_middleware],
    )
    place_note = _source_note("places", place_source, place_errors)
    restaurant_agent = create_agent(
        _agent_model(model),
        tools=place_tools + shared,
        system_prompt=RESTAURANT_PROMPT + place_note,
        middleware=[retry_mcp_middleware],
    )
    bar_agent = create_agent(
        _agent_model(model),
        tools=place_tools + shared,
        system_prompt=BAR_PROMPT + place_note,
        middleware=[retry_mcp_middleware],
    )
    return {
        "flight": flight_agent,
        "hotel": hotel_agent,
        "rental_car": rental_car_agent,
        "restaurant": restaurant_agent,
        "bar": bar_agent,
    }


async def get_specialists() -> dict[str, Any]:
    global _specialists
    async with _lock():
        if _specialists is None:
            _specialists = await build_specialists()
        return _specialists


class TripState(AgentState):
    origin: NotRequired[str]
    destination: NotRequired[str]
    guest_count: NotRequired[str]
    genre: NotRequired[str]


_DATE_PATTERNS = (
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b",
    r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|"
    r"apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:\s+\d{4})?\b",
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\s+\d{4}\b",
)

VENUE_UNAVAILABLE = (
    "Source unavailable. Places and web search returned no venues. "
    "No restaurants or bars are named."
)

_CREDIT_RE = re.compile(
    r"(?:photo|image|photograph|photographer|credit|credits|byline|illustration)"
    r"(?:\s+by|\s*:\s*|\s+)"
    r"([A-Z][\w'’.-]+(?:\s+[A-Z][\w'’.-]+){0,3})",
    re.I,
)
_PLACE_HINTS = {
    "formattedAddress",
    "formatted_address",
    "rating",
    "types",
    "primaryType",
    "googleMapsUri",
    "priceLevel",
    "nationalPhoneNumber",
    "location",
    "userRatingCount",
    "regularOpeningHours",
    "businessStatus",
}
_MEALS = ("breakfast", "brunch", "lunch", "dinner")
_CUISINE_CONTRADICTION = r"shrimp|prawn|meat|fish|chicken|pork|beef|seafood|steak"


def strip_trip_dates(text: str) -> str:
    """Remove calendar dates from a places or Tavily query."""
    cleaned = text
    for pattern in _DATE_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.I)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\b(?:on|for|during|dated)\s*$", "", cleaned, flags=re.I)
    return cleaned.strip(" ,;-")


_MAPS_SUFFIX = re.compile(r"\s+-\s+Google Maps\s*$", re.I)


def looks_like_venue_label(label: str) -> bool:
    label = re.sub(r"\s+", " ", label).strip(" .-*#")
    if not label or len(label) > 80:
        return False
    if re.search(r"\b(photo|credit|image|byline|photographer)\b", label, re.I):
        return False
    if re.search(r"\b(best|top|guide|essential|ultimate|review|where to)\b", label, re.I):
        return False
    if re.match(r"^(?:EUR(?:[_\s-]?\d+)?|€+|\$+)\b", label, re.I):
        return False
    if not re.search(r"[A-Za-z]", label):
        return False
    words = label.split()
    if not 1 <= len(words) <= 14:
        return False
    if re.search(r"\b(restaurants|bars|nightlife|hotels)\b", label, re.I) and len(words) > 3:
        return False
    if not (label[0].isupper() or label[0].isdigit()):
        return False
    return True


def _clean_place_title(title: str) -> str:
    return _MAPS_SUFFIX.sub("", title).strip()


def credit_names(text: str) -> set[str]:
    return {match.group(1).strip() for match in _CREDIT_RE.finditer(text or "")}


def _heading_names(text: str) -> list[str]:
    scrubbed = "\n".join(
        line
        for line in (text or "").splitlines()
        if not re.search(r"\b(photo|credit|byline|photographer)\b", line, re.I)
    )
    found: list[str] = []
    found.extend(match.group(1).strip() for match in re.finditer(r"(?m)^#{1,3}\s+(.+)$", scrubbed))
    found.extend(
        match.group(1).strip()
        for match in re.finditer(r"(?m)^(?:[-*]|\d+[.)])\s+\*\*([^*]+)\*\*", scrubbed)
    )
    found.extend(
        match.group(1).strip()
        for match in re.finditer(
            r"(?m)^(?:name|venue|restaurant|bar)\s*:\s*(.+)$", scrubbed, re.I
        )
    )
    # Grounding Lite summaries name venues as **Name** headings, not only markdown #.
    found.extend(match.group(1).strip() for match in re.finditer(r"\*\*([^*]{2,80})\*\*", scrubbed))
    return found


def _attribution_title(obj: dict[str, Any]) -> str | None:
    """Grounding Lite stores the visible name at place.attribution.title."""
    containers = [obj.get("place"), obj]
    for container in containers:
        if not isinstance(container, dict):
            continue
        attribution = container.get("attribution")
        if isinstance(attribution, dict) and isinstance(attribution.get("title"), str):
            title = _clean_place_title(attribution["title"])
            if title:
                return title
    return None


def _place_name(obj: dict[str, Any]) -> str | None:
    titled = _attribution_title(obj)
    if titled:
        return titled
    if not any(key in obj for key in _PLACE_HINTS):
        return None
    display = obj.get("displayName") or obj.get("display_name")
    if isinstance(display, dict):
        display = display.get("text")
    name = display or obj.get("name") or obj.get("title")
    if not isinstance(name, str):
        return None
    if name.startswith("places/"):
        return display.strip() if isinstance(display, str) else None
    return _clean_place_title(name.strip())


def _walk_place_names(node: Any) -> list[str]:
    names: list[str] = []
    if isinstance(node, dict):
        place = _place_name(node)
        if place:
            names.append(place)
        for key, value in node.items():
            if key in {"content", "snippet", "raw_content", "summary"} and isinstance(value, str):
                names.extend(_heading_names(value))
            else:
                names.extend(_walk_place_names(value))
    elif isinstance(node, list):
        for item in node:
            names.extend(_walk_place_names(item))
    return names


def _venues_from_json(text: str) -> list[str]:
    names: list[str] = []
    candidates = [text]
    if len(text) <= 20000:
        candidates.extend(re.findall(r"(\{.*\}|\[.*\])", text, re.S)[:3])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        names.extend(_walk_place_names(parsed))
    return names


def extract_grounded_venues(text: str) -> set[str]:
    """Places, plus Tavily headings that name a venue. Credits are not venues."""
    if not text:
        return set()
    credits = {name.casefold() for name in credit_names(text)}
    found = _venues_from_json(text) + _heading_names(text)
    cleaned: set[str] = set()
    for label in found:
        label = re.sub(r"\s+", " ", label).strip(" *-")
        if not looks_like_venue_label(label) or label.casefold() in credits:
            continue
        cleaned.add(label)
    return cleaned


HOTEL_FIELD_LABELS = {
    "neighborhood",
    "neighbourhood",
    "nightly rate",
    "cancellation",
    "cancellation policy",
    "booking link",
    "tradeoff",
    "trade-off",
    "trade off",
    "hotel",
    "rating",
    "price/night",
    "price per night",
    "total",
    "amenities",
    "booking",
    "name",
    "address",
    "check-in",
    "check-out",
    "taxes",
    "fees",
}


def _is_hotel_label(name: str) -> bool:
    return name.strip().strip(":").casefold() in HOTEL_FIELD_LABELS


def extract_skiplagged_hotels(text: str) -> list[str]:
    """Property names from a Skiplagged markdown table, never the column headers."""
    names: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"(?m)^\|\s*\*\*([^*|\n]+)\*\*", text or ""):
        name = re.sub(r"\s+", " ", match.group(1)).strip()
        key = name.casefold()
        if not name or _is_hotel_label(name) or key in seen:
            continue
        seen.add(key)
        names.append(name)
    return names


def extract_hotel_names(text: str) -> list[str]:
    """Property names from a captured Skiplagged payload. Field labels are not hotels."""
    if not text:
        return []
    record = read_structured(text, "HOTELS")
    if record:
        return [
            str(hotel.get("name")).strip()
            for hotel in record.get("hotels") or []
            if isinstance(hotel, dict)
            and str(hotel.get("name") or "").strip()
            and not _is_hotel_label(str(hotel.get("name")))
        ]
    names: list[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        name = re.sub(r"\s+", " ", name).strip(" -*")
        key = name.casefold()
        if not name or _is_hotel_label(name) or key in seen:
            return
        if not looks_like_venue_label(name):
            return
        seen.add(key)
        names.append(name)

    hotels_block = re.search(r"(?m)^HOTELS:\n(.*?)(?:\n[A-Z][A-Z ]+:|\Z)", text, re.S)
    if hotels_block:
        for line in hotels_block.group(1).splitlines():
            match = re.match(r"^-\s+(.+)", line.strip())
            if match:
                add(match.group(1).split("|")[0].strip())
        if names:
            return names
    for name in extract_skiplagged_hotels(text):
        add(name)
    if names:
        return names
    for name in extract_listed_names(text):
        add(name)
    return names


def extract_listed_names(text: str) -> list[str]:
    """Hotel or venue names from bold labels and list entries."""
    names: list[str] = []
    for match in re.finditer(r"\*\*([^*]{2,80})\*\*", text or ""):
        names.append(match.group(1).strip())
    for match in re.finditer(r"(?m)^(?:[-*]|\d+[.)])\s+([^:\n]{2,80})", text or ""):
        raw = re.split(r"\s+[—–-]\s+", match.group(1).strip(), maxsplit=1)[0].strip()
        names.append(raw)
    unique: list[str] = []
    seen: set[str] = set()
    for name in names:
        if not looks_like_venue_label(name):
            continue
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(name)
    return unique


def _norm_name(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _name_allowed(name: str, allowed: set[str] | list[str]) -> bool:
    norm = _norm_name(name)
    for item in allowed:
        other = _norm_name(item)
        if not other:
            continue
        if norm == other or other in norm or (norm in other and len(norm) >= 8):
            return True
    return False


_CLAIM_RES = (
    re.compile(r"\*\*([^*]{2,80})\*\*"),
    re.compile(r"(?m)^#{1,3}\s+(.{2,80})$"),
    re.compile(r"(?m)^(?:[-*]|\d+[.)])\s+([^:\n—–-]{2,80})"),
    re.compile(
        r"\b(?:dinner|lunch|breakfast|brunch|drinks?|restaurant|bar)\s+at\s+([A-Z][^.,;\n]{2,60})",
        re.I,
    ),
)


def _claim_names(line: str) -> list[str]:
    found: list[str] = []
    for pattern in _CLAIM_RES:
        found.extend(match.group(1).strip(" .") for match in pattern.finditer(line))
    return [name for name in found if looks_like_venue_label(name)]


def drop_ungrounded_claims(
    text: str,
    allowed: set[str],
    keep: list[str] | None = None,
) -> str:
    """Drop lines that name a venue the tool output did not return."""
    kept: list[str] = []
    safe = keep or []
    for line in text.splitlines():
        claims = _claim_names(line)
        if any(not _name_allowed(claim, allowed) and not _name_allowed(claim, safe) for claim in claims):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def parse_venues_line(text: str) -> set[str] | None:
    match = re.search(r"(?m)^VENUES:\s*(.*)$", text or "")
    if not match:
        return None
    body = match.group(1).strip()
    if body.lower() in {"(none)", "none", ""}:
        return set()
    return {part.strip() for part in body.split("|") if part.strip() and part.strip().lower() != "(none)"}


def evidence_confirms_hours(evidence: str) -> bool:
    """A tool states hours for a date, not a generic open/closed guess."""
    if not evidence:
        return False
    has_date = bool(
        re.search(
            r"\b\d{4}-\d{2}-\d{2}\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b",
            evidence,
            re.I,
        )
    )
    states_hours = bool(re.search(r"\b(open|hours)\b", evidence, re.I))
    explicit = bool(re.search(r"\b(confirmed open|hours confirmed|open on)\b", evidence, re.I))
    return bool(has_date and states_hours and explicit)


def scrub_unconfirmed_hours(answer: str, evidence: str) -> str:
    if evidence_confirms_hours(evidence):
        return answer
    sentences = re.split(r"(?<=[.!\n])\s+", answer)
    rewritten: list[str] = []
    for sentence in sentences:
        if re.search(r"\bconfirmed\b", sentence, re.I) and re.search(r"\b(?:hours?|open)\b", sentence, re.I):
            rewritten.append("Check hours before going.")
        else:
            rewritten.append(sentence)
    return " ".join(part for part in rewritten if part).strip()


def _evidence_window(evidence: str, venue: str, radius: int = 400) -> str:
    index = evidence.casefold().find(venue.casefold())
    if index < 0:
        return evidence
    return evidence[max(0, index - radius) : index + len(venue) + radius]


def scrub_overclaims(answer: str, evidence: str, venues: set[str]) -> str:
    """Don't relabel cuisine or move a place to a meal the source did not state."""
    if not answer or not venues:
        return answer
    lines: list[str] = []
    for line in answer.splitlines():
        updated = line
        for venue in venues:
            if venue.casefold() not in updated.casefold():
                continue
            window = _evidence_window(evidence, venue)
            if re.search(r"\bvegetarian\b", updated, re.I) and re.search(
                _CUISINE_CONTRADICTION, window, re.I
            ) and not re.search(r"\bvegetarian\b", window, re.I):
                updated = re.sub(r"\bvegetarian\b", "the source's menu", updated, flags=re.I)
                updated += " Cuisine was not relabeled beyond the source."
            for meal in _MEALS:
                if not re.search(rf"\b{meal}\b", updated, re.I):
                    continue
                if re.search(rf"\b{meal}\b", window, re.I):
                    continue
                supported = [item for item in _MEALS if re.search(rf"\b{item}\b", window, re.I)]
                if supported:
                    updated = re.sub(rf"\b{meal}\b", supported[0], updated, count=1, flags=re.I)
                    updated += f" The source says {supported[0]}, not {meal}."
        lines.append(updated)
    return "\n".join(lines)


def render_grounded_specialist(prose: str, allowed: set[str], evidence: str, tools_called: bool) -> str:
    """Restaurant and bar answers may name only venues a tool returned."""
    if not tools_called:
        return prose.rstrip() + "\nVENUES: (none)"
    prose = scrub_unconfirmed_hours(prose, evidence)
    prose = scrub_overclaims(prose, evidence, allowed)
    if not allowed:
        return VENUE_UNAVAILABLE + "\nVENUES: (none)"
    prose = drop_ungrounded_claims(prose, allowed)
    if not prose.strip():
        prose = "No grounded venues to recommend."
    listed = " | ".join(sorted(allowed, key=str.lower))
    return prose.strip() + f"\nVENUES: {listed}"


_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
_MONTH_RE = (
    r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)


def _parse_date(value: Any) -> date | None:
    text = str(value or "")
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            return None
    match = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", text)
    if match:
        try:
            return date(int(match.group(3)), int(match.group(2)), int(match.group(1)))
        except ValueError:
            return None
    match = re.search(rf"\b{_MONTH_RE}\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?", text, re.I)
    if match and match.group(3):
        try:
            return date(int(match.group(3)), _MONTHS[match.group(1)[:3].lower()], int(match.group(2)))
        except ValueError:
            return None
    match = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{_MONTH_RE}(?:\s+(\d{{4}}))?", text, re.I)
    if match and match.group(3):
        try:
            return date(int(match.group(3)), _MONTHS[match.group(2)[:3].lower()], int(match.group(1)))
        except ValueError:
            return None
    return None


def iter_dates(text: str) -> list[date]:
    """Dates in reading order. ISO, dd/mm/yyyy, and named months with a year."""
    found: list[tuple[int, date]] = []
    patterns = (
        r"\d{4}-\d{2}-\d{2}",
        r"\b\d{1,2}/\d{1,2}/\d{4}\b",
        rf"\b{_MONTH_RE}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})",
        rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH_RE}(?:\s+\d{{4}})",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text or "", re.I):
            parsed = _parse_date(match.group(0))
            if parsed:
                found.append((match.start(), parsed))
    found.sort(key=lambda item: item[0])
    ordered: list[date] = []
    seen: set[date] = set()
    for _, parsed in found:
        if parsed not in seen:
            seen.add(parsed)
            ordered.append(parsed)
    return ordered


def requested_trip_dates(text: str) -> tuple[date | None, date | None]:
    """Earliest and latest concrete dates in a trip request."""
    dates = iter_dates(text)
    if len(dates) >= 2:
        ordered = sorted(dates)
        return ordered[0], ordered[-1]
    if len(dates) == 1:
        return dates[0], None
    return None, None


def _looks_like_itinerary(node: dict[str, Any]) -> bool:
    has_price = any(key in node for key in ("price", "priceEur", "fare"))
    has_link = any(
        key in node
        for key in ("bookingUrl", "booking_url", "deep_link", "deepLink", "bookingLink")
    )
    has_legs = any(key in node for key in ("route", "legs", "segments", "outbound"))
    return has_price and (has_link or has_legs)


def _walk_itineraries(node: Any) -> list[dict[str, Any]]:
    if isinstance(node, list):
        found: list[dict[str, Any]] = []
        for item in node:
            found.extend(_walk_itineraries(item))
        return found
    if isinstance(node, dict):
        if _looks_like_itinerary(node):
            return [node]
        found = []
        for key in ("itineraries", "data", "results", "flights"):
            if key in node:
                found.extend(_walk_itineraries(node[key]))
        if found:
            return found
        for value in node.values():
            if isinstance(value, (list, dict)):
                found.extend(_walk_itineraries(value))
        return found
    if isinstance(node, str):
        try:
            return _walk_itineraries(json.loads(node))
        except json.JSONDecodeError:
            return []
    return []


def _itinerary_price(node: dict[str, Any], fallback_currency: str = "") -> tuple[str, str]:
    formatted = str(node.get("priceFormatted") or "")
    price = node.get("price", node.get("priceEur", node.get("fare")))
    currency = str(node.get("currency") or node.get("currencyCode") or fallback_currency or "")
    if isinstance(price, dict):
        currency = str(price.get("currency") or currency)
        price = price.get("amount") or price.get("value")
    if not currency and formatted:
        found = re.search(r"\b([A-Z]{3})\b", formatted)
        if found:
            currency = found.group(1)
    if price in (None, "") and formatted:
        amount = re.search(r"([0-9]+(?:\.[0-9]+)?)", formatted)
        if amount:
            price = amount.group(1)
    return str(price), currency


def _itinerary_link(node: dict[str, Any]) -> str:
    for key in ("bookingUrl", "booking_url", "bookingLink", "deep_link", "deepLink", "bookUrl"):
        if node.get(key):
            return str(node[key])
    return ""


def _segment_legs(direction: dict[str, Any]) -> list[dict[str, Any]]:
    """Kiwi puts airport codes in route and the timed hops in segments."""
    segments = direction.get("segments")
    if isinstance(segments, list):
        hops = [leg for leg in segments if isinstance(leg, dict)]
        if hops:
            return hops
    if direction.get("from") or direction.get("flyFrom"):
        return [direction]
    return []


def _itinerary_legs(node: dict[str, Any]) -> list[dict[str, Any]]:
    legs: list[dict[str, Any]] = []
    for key in ("outbound", "inbound", "return"):
        part = node.get(key)
        if isinstance(part, dict):
            legs.extend(_segment_legs(part))
        elif isinstance(part, list):
            for item in part:
                if isinstance(item, dict):
                    legs.extend(_segment_legs(item))
    if legs:
        return legs
    for key in ("segments", "legs", "route"):
        part = node.get(key)
        if isinstance(part, list):
            hops = [leg for leg in part if isinstance(leg, dict)]
            if hops:
                return hops
    return []


def _leg_line(leg: dict[str, Any]) -> str:
    origin = leg.get("from") or leg.get("flyFrom") or leg.get("origin") or "?"
    destination = leg.get("to") or leg.get("flyTo") or leg.get("destination") or "?"
    depart = leg.get("departureTime") or leg.get("local_departure") or leg.get("departure") or ""
    arrive = leg.get("arrivalTime") or leg.get("local_arrival") or leg.get("arrival") or ""
    carrier = leg.get("carrierName") or leg.get("carrier") or ""
    number = leg.get("flightNumber") or ""
    flight = " ".join(part for part in (str(carrier), str(number)) if part)
    line = f"{origin} -> {destination} depart {depart} arrive {arrive}"
    if flight:
        line += f" {flight}"
    return line.strip()


def _outbound_arrival(item: dict[str, Any]) -> date | None:
    """Arrival at the destination, not a connection and not the flight home."""
    outbound = item.get("outbound")
    if isinstance(outbound, dict):
        parsed = _parse_date(
            outbound.get("arrivalTime") or outbound.get("local_arrival") or outbound.get("arrival")
        )
        if parsed:
            return parsed
    legs = _itinerary_legs(item)
    if not legs:
        return None
    last = legs[-1]
    return _parse_date(last.get("arrivalTime") or last.get("local_arrival") or last.get("arrival"))


def _itinerary_search_dates(
    item: dict[str, Any], args: dict[str, Any]
) -> tuple[date | None, date | None]:
    outbound = item.get("outbound") if isinstance(item.get("outbound"), dict) else {}
    inbound = item.get("inbound") if isinstance(item.get("inbound"), dict) else {}
    if not inbound and isinstance(item.get("return"), dict):
        inbound = item["return"]
    depart = _parse_date(outbound.get("departureTime") or outbound.get("local_departure")) or _parse_date(
        args.get("departureDate") or args.get("departure_date") or args.get("date_from")
    )
    ret = _parse_date(inbound.get("departureTime") or inbound.get("local_departure")) or _parse_date(
        args.get("returnDate") or args.get("return_date") or args.get("date_to")
    )
    return depart, ret


def travel_eaten_warning(items: list[dict[str, Any]], args: dict[str, Any]) -> str | None:
    start = _parse_date(args.get("departureDate") or args.get("departure_date") or args.get("date_from"))
    end = _parse_date(args.get("returnDate") or args.get("return_date") or args.get("date_to"))
    if items and start is None:
        start, inferred_end = _itinerary_search_dates(items[0], {})
        end = end or inferred_end
    notes: list[str] = []
    for item in items:
        arrival = _outbound_arrival(item)
        if start and arrival and arrival >= start + timedelta(days=1):
            notes.append(
                f"Arrival {arrival.isoformat()} is a day or more after the trip start "
                f"{start.isoformat()}. Travel eats into the trip."
            )
            break
    if start and end:
        span = (end - start).days
        for item in items:
            arrival = _outbound_arrival(item)
            if arrival is None:
                continue
            ground = (end - arrival).days
            if span >= 2 and ground * 2 < span:
                notes.append("The trip is mostly eaten by travel.")
                break
    if not notes:
        return None
    return " ".join(dict.fromkeys(notes))


def _arrival_note(direction: dict[str, Any]) -> str:
    """Label a hop whose arrival is on a later calendar day."""
    if not isinstance(direction, dict):
        return ""
    depart = _parse_date(
        direction.get("departureTime") or direction.get("local_departure") or direction.get("departure")
    )
    arrive = _parse_date(
        direction.get("arrivalTime") or direction.get("local_arrival") or direction.get("arrival")
    )
    if depart and arrive and arrive > depart:
        return f"next-day arrival ({arrive.day} {arrive.strftime('%b')})"
    return ""


def _direction_notes(item: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    outbound = item.get("outbound")
    inbound = item.get("inbound") if isinstance(item.get("inbound"), dict) else item.get("return")
    if isinstance(outbound, dict):
        note = _arrival_note(outbound)
        if note:
            notes.append(f"outbound: {note}")
    if isinstance(inbound, dict):
        note = _arrival_note(inbound)
        if note:
            notes.append(f"return: {note}")
    if notes:
        return notes
    for leg in _itinerary_legs(item):
        note = _arrival_note(leg)
        if note:
            return [note]
    return []


def render_kiwi_itineraries(payload: Any, args: dict[str, Any] | None = None) -> str:
    """One block per Kiwi result so price, legs, and booking link stay together."""
    args = args or {}
    items = _walk_itineraries(payload)
    if not items:
        data = payload
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                data = None
        if isinstance(data, dict):
            if _flight_payload_is_outage(data):
                return FLIGHT_SOURCE_UNAVAILABLE
            if any(key in data and not data.get(key) for key in ("itineraries", "flights", "results")):
                return NO_FLIGHTS_FOUND
        return payload if isinstance(payload, str) else json.dumps(payload, default=str)
    envelope_currency = str(payload.get("currency") or "") if isinstance(payload, dict) else ""
    blocks: list[str] = []
    for index, item in enumerate(items, 1):
        price, currency = _itinerary_price(item, envelope_currency)
        link = _itinerary_link(item)
        legs = "\n".join(f"  - {_leg_line(leg)}" for leg in _itinerary_legs(item)) or "  - (no legs)"
        outbound_date, return_date = _itinerary_search_dates(item, args)
        warning = travel_eaten_warning([item], args)
        arrival_notes = _direction_notes(item)
        blocks.append(
            f"ITINERARY {index}\n"
            f"price: {price} {currency}".rstrip()
            + "\n"
            + f"booking_link: {link}\n"
            + f"outbound_date: {outbound_date.isoformat() if outbound_date else 'unspecified'}\n"
            + f"return_date: {return_date.isoformat() if return_date else 'unspecified'}\n"
            + f"legs:\n{legs}\n"
            + (("\n".join(arrival_notes) + "\n") if arrival_notes else "")
            + "These price, legs, and booking_link values are one Kiwi result. "
            + "Do not combine them with another itinerary."
            + (f"\nWARNING: {warning}" if warning else "")
        )
    return (
        "Each block is one Kiwi itinerary. Keep its price, legs, and booking link together.\n\n"
        + "\n\n".join(blocks)
    )


def limit_itineraries_to_requested_dates(
    text: str,
    outbound: date | None,
    returning: date | None,
) -> str:
    """Keep searches for the requested dates. Other searches stay, labeled as alternates."""
    if outbound is None or "ITINERARY" not in text:
        return text
    blocks = re.findall(r"ITINERARY \d+\b.*?(?=\nITINERARY \d+\b|\Z)", text, re.S)
    if not blocks:
        return text
    matching: list[str] = []
    alternate: list[str] = []
    for block in blocks:
        out_match = re.search(r"(?m)^outbound_date:\s*(\S+)", block)
        ret_match = re.search(r"(?m)^return_date:\s*(\S+)", block)
        out_date = _parse_date(out_match.group(1) if out_match else "")
        ret_date = _parse_date(ret_match.group(1) if ret_match else "")
        same_outbound = out_date == outbound
        same_return = returning is None or ret_date == returning
        if same_outbound and same_return:
            matching.append(block.strip())
        else:
            alternate.append(block.strip())
    lines = [
        f"Requested trip dates: outbound {outbound.isoformat()}"
        + (f", return {returning.isoformat()}" if returning else "")
        + ". Use only the itineraries in this section for the trip.",
        "",
    ]
    if matching:
        lines.append("\n\n".join(matching))
    else:
        lines.append("No itineraries matched the requested dates.")
    if alternate:
        lines.append(
            "\nALTERNATE DATES — these searches used different dates. "
            "Do not present them as the trip.\n"
        )
        lines.append("\n\n".join(alternate))
    lines.insert(
        0,
        "Each block is one Kiwi itinerary. Keep its price, legs, and booking link together.",
    )
    return "\n".join(lines).strip() + "\n"


def _format_like(sample: str, value: date) -> str:
    if re.search(r"\d{4}-\d{2}-\d{2}", sample or ""):
        return value.isoformat()
    if re.search(r"\d{1,2}/\d{1,2}/\d{4}", sample or ""):
        return f"{value.day:02d}/{value.month:02d}/{value.year}"
    if re.search(r"[A-Za-z]", sample or ""):
        return f"{value.day} {value.strftime('%b')} {value.year}"
    return value.isoformat()


def _flight_departure(args: dict[str, Any]) -> date | None:
    return _parse_date(
        _first_arg(args, ("departureDate", "departure_date", "date_from", "dateFrom", "outboundDate"))
    )


def _flight_return_arg(args: dict[str, Any]) -> date | None:
    return _parse_date(
        _first_arg(args, ("returnDate", "return_date", "date_to", "inboundDate", "returnDate"))
    )


def complete_flight_args(args: dict[str, Any] | None) -> dict[str, Any]:
    """A round trip with an end date must not go out as a one-way on the start date."""
    prepared = dict(args or {})
    start, end = trip_window()
    if not start or not end or _flight_return_arg(prepared):
        return prepared
    depart = _flight_departure(prepared)
    # A reverse one-way that already leaves on the end date is the inbound leg.
    if depart == end and depart != start:
        return prepared
    sample = _first_arg(
        prepared, ("departureDate", "departure_date", "date_from", "dateFrom", "outboundDate")
    )
    if depart is None:
        outbound = _format_like(sample, start)
        if "departure_date" in prepared:
            prepared["departure_date"] = outbound
        elif "date_from" in prepared:
            prepared["date_from"] = outbound
        else:
            prepared["departureDate"] = outbound
            sample = outbound
    prepared["returnDate"] = _format_like(sample, end)
    return prepared


def _has_return_leg(item: dict[str, Any]) -> bool:
    inbound = item.get("inbound") if isinstance(item.get("inbound"), dict) else None
    if inbound is None and isinstance(item.get("return"), dict):
        inbound = item["return"]
    return bool(isinstance(inbound, dict) and _segment_legs(inbound))


def _leg_departure(item: dict[str, Any], args: dict[str, Any]) -> date | None:
    outbound = item.get("outbound") if isinstance(item.get("outbound"), dict) else {}
    parsed = _parse_date(outbound.get("departureTime") or outbound.get("local_departure"))
    if parsed:
        return parsed
    legs = _itinerary_legs(item)
    if legs:
        parsed = _parse_date(legs[0].get("departureTime") or legs[0].get("local_departure") or legs[0].get("departure"))
        if parsed:
            return parsed
    return _flight_departure(args)


def _return_departure(item: dict[str, Any]) -> date | None:
    inbound = item.get("inbound") if isinstance(item.get("inbound"), dict) else None
    if inbound is None and isinstance(item.get("return"), dict):
        inbound = item["return"]
    if not isinstance(inbound, dict):
        return None
    parsed = _parse_date(inbound.get("departureTime") or inbound.get("local_departure"))
    if parsed:
        return parsed
    legs = _segment_legs(inbound)
    if not legs:
        return None
    return _parse_date(legs[0].get("departureTime") or legs[0].get("local_departure") or legs[0].get("departure"))


def _fare_amount(value: Any) -> float:
    text = str(value or "").strip()
    if not text:
        return float("inf")
    try:
        return float(text.split()[0].replace(",", ""))
    except (ValueError, IndexError):
        return float("inf")


def _normalize_itinerary(
    item: dict[str, Any],
    args: dict[str, Any],
    *,
    envelope_currency: str = "",
    has_return: bool,
) -> dict[str, Any]:
    price, currency = _itinerary_price(item, envelope_currency)
    if currency == "€":
        currency = "EUR"
    return {
        "price": price,
        "currency": currency or "EUR",
        "booking_url": _itinerary_link(item),
        "return_url": "",
        "return_price": "",
        "legs": [_leg_line(leg) for leg in _itinerary_legs(item)],
        "notes": _direction_notes(item),
        "has_return": has_return,
    }


def _flight_code(carrier: str, number: str) -> str:
    """Carrier plus flight number, with spaces removed. ``TP 238`` and ``U2 8123`` collapse."""
    number_compact = re.sub(r"[^A-Za-z0-9]", "", number or "").upper()
    carrier_compact = re.sub(r"[^A-Za-z0-9]", "", carrier or "").upper()
    if not number_compact:
        return ""
    if re.fullmatch(r"\d{2,4}", number_compact) and re.fullmatch(r"[A-Z0-9]{2}", carrier_compact):
        return carrier_compact + number_compact
    if re.search(r"\d", number_compact):
        return number_compact
    return ""


def _flight_code_from_tail(tail: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", tail or "")
    if not tokens:
        return ""
    if (
        len(tokens) >= 2
        and re.fullmatch(r"\d{2,4}", tokens[-1])
        and re.fullmatch(r"[A-Za-z0-9]{2}", tokens[-2])
    ):
        return _flight_code(tokens[-2], tokens[-1])
    return _flight_code("", tokens[-1])


def _leg_carrier_code(leg: dict[str, Any]) -> str:
    for key in ("carrierCode", "airlineCode", "carrier"):
        raw = str(leg.get(key) or "").strip()
        if re.fullmatch(r"[A-Za-z0-9]{2}", raw):
            return raw
    return ""


def _leg_identity_from_dict(leg: dict[str, Any]) -> tuple[str, str, str] | None:
    number = _flight_code(_leg_carrier_code(leg), str(leg.get("flightNumber") or ""))
    depart = str(
        leg.get("departureTime") or leg.get("local_departure") or leg.get("departure") or ""
    ).strip()
    arrive = str(leg.get("arrivalTime") or leg.get("local_arrival") or leg.get("arrival") or "").strip()
    if not number or not depart or not arrive:
        return None
    return (number, depart, arrive)


_LEG_WHEN = (
    r"(?:"
    r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?"
    r"|"
    r"\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
    r")"
)


def _leg_identity_from_line(line: str) -> tuple[str, str, str] | None:
    match = re.search(
        rf"depart\s+({_LEG_WHEN})\s+arrive\s+({_LEG_WHEN})(?:\s+(.*?))?\s*$",
        line.strip(),
        re.I,
    )
    if not match:
        return None
    number = _flight_code_from_tail(match.group(3) or "")
    if not number:
        return None
    return (number, match.group(1), match.group(2))


def itinerary_leg_key(item: dict[str, Any]) -> tuple[tuple[str, str, str], ...] | None:
    """Flight numbers plus departure and arrival times. Missing either means no key."""
    legs = item.get("legs")
    if isinstance(legs, list) and legs and all(isinstance(leg, str) for leg in legs):
        keys = [_leg_identity_from_line(leg) for leg in legs]
    else:
        keys = [_leg_identity_from_dict(leg) for leg in _itinerary_legs(item)]
    if not keys or any(key is None for key in keys):
        return None
    return tuple(keys)  # type: ignore[arg-type]


def _itinerary_total(item: dict[str, Any]) -> float:
    total = _fare_amount(item.get("price"))
    if item.get("return_price"):
        total += _fare_amount(item.get("return_price"))
    return total


def dedupe_itineraries_by_legs(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse itineraries with the same legs. Keep the cheapest booking link."""
    best_index: dict[tuple[tuple[str, str, str], ...], int] = {}
    chosen: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        key = itinerary_leg_key(item)
        if key is None:
            chosen.append(item)
            continue
        index = best_index.get(key)
        if index is None:
            best_index[key] = len(chosen)
            chosen.append(item)
            continue
        if _itinerary_total(item) < _itinerary_total(chosen[index]):
            chosen[index] = item
    return chosen


def assemble_flight_record(
    searches: list[dict[str, Any]],
    start: date | None,
    end: date | None,
) -> dict[str, Any]:
    """Round trips for the requested dates, or outbound and return one-ways paired together."""
    rounds: list[dict[str, Any]] = []
    outbound: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    inbound: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    for search in searches:
        args = search.get("args") or {}
        currency = str(search.get("currency") or "")
        for item in search.get("itineraries") or []:
            if not isinstance(item, dict):
                continue
            depart = _leg_departure(item, args)
            if _has_return_leg(item):
                ret = _return_departure(item)
                if start and depart and depart != start:
                    continue
                if end and ret and ret != end:
                    continue
                rounds.append(_normalize_itinerary(item, args, envelope_currency=currency, has_return=True))
                continue
            if end and depart == end and depart != start:
                inbound.append((item, args, currency))
            elif start is None or depart == start or depart is None:
                outbound.append((item, args, currency))
    chosen: list[dict[str, Any]] = []
    if rounds:
        chosen = sorted(dedupe_itineraries_by_legs(rounds), key=_itinerary_total)[:3]
    elif outbound and inbound and start and end:
        pairs: list[tuple[float, dict[str, Any]]] = []
        for out_item, out_args, out_currency in outbound:
            for in_item, in_args, in_currency in inbound:
                out_norm = _normalize_itinerary(out_item, out_args, envelope_currency=out_currency, has_return=False)
                in_norm = _normalize_itinerary(in_item, in_args, envelope_currency=in_currency, has_return=False)
                total = _fare_amount(out_norm.get("price")) + _fare_amount(in_norm.get("price"))
                pairs.append(
                    (
                        total,
                        {
                            "price": out_norm.get("price") or "",
                            "return_price": in_norm.get("price") or "",
                            "currency": out_norm.get("currency") or in_norm.get("currency") or "EUR",
                            "booking_url": out_norm.get("booking_url") or "",
                            "return_url": in_norm.get("booking_url") or "",
                            "legs": list(out_norm.get("legs") or []) + list(in_norm.get("legs") or []),
                            "notes": list(out_norm.get("notes") or []) + list(in_norm.get("notes") or []),
                            "has_return": True,
                        },
                    )
                )
        pairs.sort(key=lambda item: item[0])
        chosen = dedupe_itineraries_by_legs([item for _, item in pairs])[:3]
    elif outbound and not (start and end):
        chosen = sorted(
            dedupe_itineraries_by_legs(
                [
                    _normalize_itinerary(item, args, envelope_currency=currency, has_return=False)
                    for item, args, currency in outbound
                ]
            ),
            key=_itinerary_total,
        )[:3]
    return {
        "outbound_date": start.isoformat() if start else "",
        "return_date": end.isoformat() if end else "",
        "itineraries": chosen,
    }


def stash_flight_search(search: dict[str, Any]) -> None:
    bag = _LATEST.get("_flight_searches")
    searches = list(bag) if isinstance(bag, list) else []
    searches.append(search)
    _LATEST["_flight_searches"] = searches
    start, end = trip_window()
    stash_record("flights", assemble_flight_record(searches, start, end))


NO_FLIGHTS_FOUND = "No flights found for these dates."
FLIGHT_SOURCE_UNAVAILABLE = "The flight source is unavailable."
_FLIGHT_OUTAGE = re.compile(r"unavailable|time-?d?\s*out|rate[\s-]*limit", re.I)


def _flight_payload_is_outage(data: dict[str, Any]) -> bool:
    """An ``error`` key, or a message that the source failed. ``No results`` is an empty search."""
    if data.get("error"):
        return True
    message = data.get("message")
    return isinstance(message, str) and bool(_FLIGHT_OUTAGE.search(message))


def _flight_result_texts(messages: list[Any]) -> list[str]:
    return [
        _message_text(message.content)
        for message in messages or []
        if isinstance(message, ToolMessage) and getattr(message, "name", None) == "search_flights"
    ]


def _json_object(text: str) -> dict[str, Any] | None:
    stripped = (text or "").strip()
    if not stripped.startswith("{"):
        return None
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _result_found_flights(text: str) -> bool:
    """A search that names a fare, even when it is not a structured itinerary block."""
    if "ITINERARY" in (text or "") and re.search(r"(?m)^(?:price|booking_link):", text):
        return True
    saved = read_structured(text or "", "FLIGHTS")
    if saved and saved.get("itineraries"):
        return True
    if re.search(r"kiwi\.com", text or "", re.I):
        return True
    return bool(_FLIGHT_PRICE.search(text or "") and re.search(r"\b[A-Z]{1,3}\d{2,4}\b", text or ""))


def _result_reports_no_flights(text: str) -> bool:
    if _result_found_flights(text):
        return False
    data = _json_object(text)
    if data is not None:
        if _flight_payload_is_outage(data):
            return False
        return any(key in data and not data.get(key) for key in ("itineraries", "flights", "results"))
    return NO_FLIGHTS_FOUND in (text or "")


def _searches_found_no_flights(messages: list[Any]) -> bool:
    """True only when no search returned a flight and the latest one came back empty."""
    results = _flight_result_texts(messages)
    if not results or any(_result_found_flights(item) for item in results):
        return False
    return _result_reports_no_flights(results[-1])


def render_flight_section(record: dict[str, Any]) -> str:
    """Top 3 itineraries. The heading names the trip dates when they are known."""
    itineraries = [item for item in (record.get("itineraries") or []) if isinstance(item, dict)]
    if not itineraries:
        return NO_FLIGHTS_FOUND
    heading = (
        "Flights"
        + _when_span(str(record.get("outbound_date") or ""), str(record.get("return_date") or ""))
        + ":"
    )
    lines = [heading]
    for item in itineraries[:3]:
        if not isinstance(item, dict):
            continue
        currency = item.get("currency") or "EUR"
        raw_price = item.get("price")
        if raw_price is None or str(raw_price).strip() == "":
            fare = "price unavailable"
        elif item.get("return_price"):
            fare = f"{raw_price} {currency} + {item.get('return_price')} {currency}".strip()
        else:
            fare = f"{raw_price} {currency}".strip()
        links = [str(item.get("booking_url") or ""), str(item.get("return_url") or "")]
        lines.append("- " + " | ".join([fare, *[link for link in links if link]]))
        for leg in item.get("legs") or []:
            lines.append(f"  - {leg}")
        for note in item.get("notes") or []:
            lines.append(f"  - {note}")
    return "\n".join(lines)


def flight_record_from_messages(messages: list[Any]) -> dict[str, Any] | None:
    """Recover a compact flight record from tool text when the stash is empty."""
    text = "\n".join(_message_payload_text(item) for item in messages or [] if _toolish(item))
    saved = read_structured(text, "FLIGHTS")
    if saved and saved.get("itineraries"):
        items = dedupe_itineraries_by_legs(
            [item for item in saved["itineraries"] if isinstance(item, dict)]
        )
        items.sort(key=_itinerary_total)
        saved = dict(saved)
        saved["itineraries"] = items[:3]
        return saved
    primary = text.split("ALTERNATE DATES")[0]
    itineraries: list[dict[str, Any]] = []
    outbound = ""
    returning = ""
    for block in _itinerary_blocks(primary):
        price_match = re.search(r"(?m)^price:\s*([0-9]+(?:\.[0-9]+)?)(?:\s*([A-Za-z]{3}|€))?", block)
        if not price_match:
            continue
        currency = (price_match.group(2) or "EUR").upper()
        if currency == "€":
            currency = "EUR"
        out = _block_value(block, "outbound_date")
        ret = _block_value(block, "return_date")
        if out and out != "unspecified":
            outbound = outbound or out
        if ret and ret != "unspecified":
            returning = returning or ret
        legs = [line.strip() for line in re.findall(r"(?m)^\s+-\s+(.+)$", block)]
        hop_lines = [line for line in legs if "->" in line]
        notes = [line for line in legs if line not in hop_lines]
        itineraries.append(
            {
                "price": price_match.group(1),
                "currency": currency,
                "booking_url": _block_value(block, "booking_link"),
                "return_url": "",
                "return_price": "",
                "legs": hop_lines,
                "notes": notes,
                "has_return": bool(ret and ret != "unspecified"),
            }
        )
    if not itineraries:
        return None
    itineraries = dedupe_itineraries_by_legs(itineraries)
    itineraries.sort(key=_itinerary_total)
    return {
        "outbound_date": outbound,
        "return_date": returning,
        "itineraries": itineraries[:3],
    }


def active_flight_record(messages: list[Any], *, allow_stash: bool = False) -> dict[str, Any] | None:
    if allow_stash or _tool_blob(messages, "search_flights").strip():
        stashed = stashed_record("flights")
        if stashed and stashed.get("itineraries"):
            return stashed
    return flight_record_from_messages(messages)


_NO_RETURN = re.compile(
    r"(?im)^.*\b(?:no return flight|there is no return|without a return flight)\b.*$"
)


def scrub_false_no_return(answer: str, record: dict[str, Any]) -> str:
    """A return-direction search on the end date means the return exists."""
    if not any(isinstance(item, dict) and item.get("has_return") for item in record.get("itineraries") or []):
        return answer
    return _NO_RETURN.sub("", answer or "").strip()


def _record_flight_numbers(record: dict[str, Any]) -> set[str]:
    numbers: set[str] = set()
    for item in record.get("itineraries") or []:
        if not isinstance(item, dict):
            continue
        for leg in item.get("legs") or []:
            numbers.update(re.findall(r"\b([A-Z]{1,3}\d{2,4})\b", str(leg)))
    return numbers


_FLIGHT_PRICE = re.compile(
    r"(?:[$€£]\s*\d|\b\d+(?:[.,]\d+)?\s*(?:EUR|USD|GBP)\b|\b(?:EUR|USD|GBP)\s*\d)",
    re.I,
)


_FLIGHT_FIELD = re.compile(
    r"^\s*(?:[-*•]\s*)?(?:\*\*)?(price|outbound|return)(?:\*\*)?\s*:\s*\S",
    re.I,
)


def _line_restates_flight_fields(line: str, record: dict[str, Any]) -> bool:
    """A short Price / Outbound / Return line that repeats the first rendered itinerary."""
    if not _FLIGHT_FIELD.match(line) or len(line.strip()) > 160:
        return False
    itineraries = [item for item in record.get("itineraries") or [] if isinstance(item, dict)]
    if not itineraries:
        return False
    first = itineraries[0]
    parts = str(first.get("price") or "").split()
    price = parts[0] if parts else ""
    if price and re.search(rf"\b{re.escape(price)}\b", line):
        return True
    numbers = _record_flight_numbers({"itineraries": [first]})
    return any(re.search(rf"\b{re.escape(number)}\b", line) for number in numbers)


def _line_mentions_flight_number(line: str, numbers: set[str]) -> bool:
    return any(re.search(rf"\b{re.escape(number)}\b", line) for number in numbers)


def _line_restates_itinerary(line: str, urls: set[str], numbers: set[str]) -> bool:
    """A restatement pairs a rendered flight number with a price or booking link.

    A day-plan bullet that only names the flight (``- Day 1: Fly TP238 overnight``)
    is the itinerary, not a second copy of the Kiwi block.
    """
    if any(url and url in line for url in urls):
        return True
    if not _line_mentions_flight_number(line, numbers):
        return False
    return bool(_FLIGHT_PRICE.search(line) or re.search(r"https?://", line))


def _split_around(text: str, section: str) -> tuple[list[str], list[str]] | None:
    index = text.find(section)
    if index < 0:
        return None
    return text[:index].splitlines(), text[index + len(section) :].splitlines()


def replace_flight_section(answer: str, record: dict[str, Any]) -> str:
    section = render_flight_section(record)
    text = replace_named_section(answer, "flights", section)
    kept = {
        str(item.get(key))
        for item in record.get("itineraries") or []
        if isinstance(item, dict)
        for key in ("booking_url", "return_url")
        if item.get(key)
    }
    numbers = _record_flight_numbers(record)

    def keep(line: str) -> bool:
        links = [link.rstrip(".,)") for link in re.findall(r"https?://\S+", line)]
        flight_links = [link for link in links if "kiwi.com" in link]
        if flight_links and any(link not in kept for link in flight_links):
            return False
        if _line_restates_itinerary(line, kept, numbers):
            return False
        if _line_restates_flight_fields(line, record):
            return False
        return True

    def outside(chunk: list[str]) -> list[str]:
        return [line for line in _under_named_heading(chunk, "flights") if keep(line)]

    parts = _split_around(text, section)
    if parts is None:
        section_lines = set(section.splitlines())
        body = [line for line in text.splitlines() if line not in section_lines]
        lines = [*outside(body), *section.splitlines()]
        return scrub_false_no_return("\n".join(lines).strip(), record)
    before, after = parts
    fares = _record_fare_tokens(record)
    after = _drop_structured_repeats(
        after,
        lambda line: _line_matches_structured_flight(line, numbers, fares),
    )
    lines = outside(before)
    lines.extend(section.splitlines())
    lines.extend(outside(after))
    return _collapse_blank_lines(scrub_false_no_return("\n".join(lines), record))


def _coerce_payload(content: Any) -> Any:
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("text"):
                texts.append(str(block["text"]))
            elif isinstance(block, str):
                texts.append(block)
        joined = "\n".join(texts)
        try:
            return json.loads(joined)
        except json.JSONDecodeError:
            return joined
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content
    return content


def _flight_tool_results(messages: list[Any]) -> list[tuple[dict[str, Any], str]]:
    calls: dict[str, dict[str, Any]] = {}
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls or []:
                if call.get("id"):
                    args = call.get("args") or {}
                    calls[call["id"]] = args if isinstance(args, dict) else {}
    results: list[tuple[dict[str, Any], str]] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        name = getattr(message, "name", None) or ""
        if "flight" not in name and "ITINERARY" not in _message_text(message.content):
            continue
        args = calls.get(message.tool_call_id or "", {})
        results.append((args, _message_text(message.content)))
    return results


def flight_specialist_text(result: dict[str, Any]) -> str | None:
    """Locked Kiwi blocks. Only the requested dates are the trip; other searches are alternates."""
    messages = result.get("messages") or []
    pieces: list[str] = []
    for args, content in _flight_tool_results(messages):
        if "ITINERARY" in content and "outbound_date:" in content:
            piece = content
        else:
            piece = render_kiwi_itineraries(_coerce_payload(content), args)
        piece = re.sub(
            r"^Each block is one Kiwi itinerary\.[^\n]*\n+",
            "",
            piece.strip(),
            count=1,
        )
        if piece:
            pieces.append(piece)
    rendered = "\n\n".join(pieces)
    if "ITINERARY" not in rendered:
        return None
    human = "\n".join(
        _message_text(message.content)
        for message in messages
        if getattr(message, "type", None) == "human"
    )
    outbound, returning = requested_trip_dates(human)
    return limit_itineraries_to_requested_dates(rendered, outbound, returning)


def repair_flight_pairing(answer: str, flight_text: str) -> str:
    """Replace a summary that pairs one itinerary's price with another's link."""
    if "ITINERARY" not in (flight_text or ""):
        return answer
    blocks: list[tuple[str, str]] = []
    for chunk in re.split(r"\n(?=ITINERARY \d+)", flight_text):
        price_match = re.search(r"(?m)^price:\s*([0-9]+(?:\.[0-9]+)?)", chunk)
        link_match = re.search(r"(?m)^booking_link:\s*(\S+)", chunk)
        if price_match and link_match:
            blocks.append((price_match.group(1), link_match.group(1)))
    if len(blocks) < 2:
        return answer
    for price, link in blocks:
        if link not in answer:
            continue
        index = answer.find(link)
        window = answer[max(0, index - 160) : index + len(link) + 160]
        for other_price, _other_link in blocks:
            if other_price == price:
                continue
            if re.search(rf"\b{re.escape(other_price)}\b", window):
                return (
                    answer.rstrip()
                    + "\n\nFlight fields were mixed across Kiwi results. "
                    + "Use these itineraries intact:\n\n"
                    + flight_text
                )
    return answer


def _tool_blob(messages: list[Any], name: str) -> str:
    parts = [
        _message_text(message.content)
        for message in messages
        if isinstance(message, ToolMessage) and getattr(message, "name", None) == name
    ]
    return "\n".join(parts)


class CarOffer(NamedTuple):
    name: str
    price: str
    vendor: str
    link: str


def _car_offer_from_node(node: dict[str, Any]) -> CarOffer | None:
    """One OctoTrip or Upfront offer. Ignore envelopes and field objects."""
    name = node.get("name") or node.get("vehicle") or node.get("car") or node.get("model")
    if not isinstance(name, str) or not name.strip():
        return None
    price = node.get("price")
    if price in (None, "", []) or isinstance(price, (dict, list)):
        price = node.get("total_price") or node.get("totalPrice")
    if price in (None, "", []) or isinstance(price, (dict, list)):
        return None
    vendor = str(node.get("vendor") or node.get("supplier") or node.get("company") or "").strip()
    link = str(
        node.get("booking_url")
        or node.get("bookingUrl")
        or node.get("deep_link")
        or node.get("url")
        or ""
    ).strip()
    if not any(
        key in node
        for key in ("vehicle", "vendor", "supplier", "booking_url", "bookingUrl", "category", "sipp", "car")
    ):
        return None
    currency = str(node.get("currency") or "").strip()
    if isinstance(price, float):
        amount = f"{price:.2f}"
    else:
        amount = str(price)
    return CarOffer(name.strip(), f"{amount} {currency}".strip(), vendor, link)


def _walk_car_offers(node: Any, found: list[CarOffer]) -> None:
    if isinstance(node, dict):
        offer = _car_offer_from_node(node)
        if offer is not None:
            found.append(offer)
        for value in node.values():
            _walk_car_offers(value, found)
    elif isinstance(node, list):
        for item in node:
            _walk_car_offers(item, found)


def _json_values(text: str) -> list[Any]:
    """Decode every JSON value in text, including payloads larger than one regex match."""
    if not text:
        return []
    values: list[Any] = []
    stripped = text.strip()
    try:
        values.append(json.loads(stripped))
        return values
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    index = 0
    while index < len(text):
        start = text.find("{", index)
        bracket = text.find("[", index)
        if start < 0 and bracket < 0:
            break
        if start < 0 or (bracket >= 0 and bracket < start):
            start = bracket
        try:
            value, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            index = start + 1
            continue
        values.append(value)
        index = max(end, start + 1)
    return values


def format_structured(kind: str, record: dict[str, Any]) -> str:
    return (
        f"<<<STRUCTURED_{kind}>>>\n"
        + json.dumps(record, ensure_ascii=False)
        + f"\n<<<END_STRUCTURED_{kind}>>>"
    )


def read_structured(text: str, kind: str) -> dict[str, Any] | None:
    """Read a record this code wrote at tool-call time. Specialist prose is ignored."""
    start = f"<<<STRUCTURED_{kind}>>>"
    end = f"<<<END_STRUCTURED_{kind}>>>"
    blob = text or ""
    begin = blob.find(start)
    if begin < 0:
        return None
    open_brace = blob.find("{", begin + len(start))
    finish = blob.find(end, open_brace if open_brace >= 0 else begin)
    if open_brace < 0 or finish < 0:
        return None
    try:
        data = json.loads(blob[open_brace:finish])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _payload_data(value: Any) -> Any:
    if isinstance(value, tuple) and value:
        return _payload_data(value[0])
    if isinstance(value, list) and value and all(
        isinstance(item, dict) and "text" in item for item in value
    ):
        return _coerce_payload(value)
    if isinstance(value, str):
        return _coerce_payload(value)
    return value


def _first_arg(args: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = args.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def car_record_from_payload(payload: Any, args: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """OctoTrip results[].name/price/currency/vendor/booking_url, captured from the tool payload."""
    args = args or {}
    data = _payload_data(payload)
    found: list[CarOffer] = []
    if isinstance(data, str):
        for value in _json_values(data):
            _walk_car_offers(value, found)
    else:
        _walk_car_offers(data, found)
    if not found:
        return None
    pickup = _first_arg(args, ("location", "pickup", "pickup_location", "pickUpLocation", "city"))
    if not pickup and isinstance(data, dict) and isinstance(data.get("pickup_location_resolved"), str):
        pickup = data["pickup_location_resolved"].strip()
    offers: list[dict[str, str]] = []
    for offer in found:
        parts = offer.price.split()
        amount = parts[0] if parts else offer.price
        currency = parts[1] if len(parts) > 1 else ""
        offers.append(
            {
                "name": offer.name,
                "price": amount,
                "currency": currency,
                "vendor": offer.vendor,
                "booking_url": offer.link,
            }
        )
    return {
        "pickup": pickup,
        "dropoff": _first_arg(args, ("dropoff_location", "dropoff", "dropOffLocation")),
        "pickup_date": _first_arg(args, ("pickup_date", "pickupDate", "from", "start_date")),
        "dropoff_date": _first_arg(args, ("dropoff_date", "dropoffDate", "to", "end_date")),
        "offers": offers,
    }


def seal_car_tool_result(result: Any, args: dict[str, Any] | None = None) -> str:
    """Stash the OctoTrip record and lead with it. A trailer after the raw dump gets cut off."""
    payload = result[0] if isinstance(result, tuple) and result else result
    record = car_record_from_payload(payload, args)
    if not record or not record.get("offers"):
        if isinstance(payload, (dict, list)):
            return json.dumps(payload)
        return _message_text(payload)
    stash_record("cars", record)
    return format_structured("CARS", record) + "\n" + render_car_section(record)


def _is_flight_tool_name(name: str) -> bool:
    folded = (name or "").casefold().replace("-", " ").replace("_", " ")
    return "flight" in folded


def _looks_like_flight_payload(payload: Any) -> bool:
    """Skiplagged flight rows are not hotels, even when they arrive as a markdown table."""
    if isinstance(payload, dict):
        if (payload.get("itineraries") or payload.get("flights")) and not payload.get("hotels"):
            return True
    text = payload if isinstance(payload, str) else _payload_text(payload)
    if re.search(r"virtual interline", text, re.I):
        return True
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip().casefold() for cell in line.strip().strip("|").split("|")]
        joined = " ".join(cells)
        if "hotel" in joined or "night" in joined:
            return False
        if re.search(r"\b(?:flight|stops?|airline)\b", joined):
            return True
        return False
    return False


def seal_hotel_tool_result(
    result: Any,
    args: dict[str, Any] | None = None,
    tool_name: str = "",
) -> str:
    """Stash Skiplagged hotel rows. A flight search must not become the hotel record."""
    payload = result[0] if isinstance(result, tuple) and result else result
    text = _payload_text(payload)
    if _is_flight_tool_name(tool_name) or _looks_like_flight_payload(payload):
        return text
    record = hotel_record_from_payload(text, args)
    if not record or not record.get("hotels"):
        return text
    stash_record("hotels", record)
    return format_structured("HOTELS", record) + "\n" + render_hotel_section(record)


def _price_key(offer: dict[str, Any]) -> float:
    try:
        return float(str(offer.get("price") or "").split()[0])
    except ValueError:
        return 1e18


def _usable_date(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not re.search(r"\d", text):
        return ""
    return text


def _when_span(start: str, end: str) -> str:
    start, end = _usable_date(start), _usable_date(end)
    if start and end:
        return f" ({start} to {end})"
    if start or end:
        return f" ({start or end})"
    return ""


def render_car_section(record: dict[str, Any]) -> str:
    """Top 5 offers by price. The heading always names the pickup and any dates."""
    place = (record.get("pickup") or "the pickup location").strip() or "the pickup location"
    heading = (
        f"Rental Cars in {place}"
        + _when_span(str(record.get("pickup_date") or ""), str(record.get("dropoff_date") or ""))
        + ":"
    )
    ranked = sorted(record.get("offers") or [], key=_price_key)[:5]
    lines = [heading]
    for offer in ranked:
        if not isinstance(offer, dict):
            continue
        price = f"{offer.get('price') or ''} {offer.get('currency') or ''}".strip()
        bits = [str(offer.get("name") or "Vehicle"), price or "price unavailable"]
        if offer.get("vendor"):
            bits.append(str(offer["vendor"]))
        if offer.get("booking_url"):
            bits.append(str(offer["booking_url"]))
        lines.append("- " + " | ".join(bits))
    return "\n".join(lines)


def _payload_text(payload: Any) -> str:
    data = _payload_data(payload)
    if isinstance(data, str):
        return data
    if isinstance(data, (dict, list)):
        return json.dumps(data)
    return _message_text(data)


def _hotel_cell(cells: list[str], index: int) -> str:
    if index < 0 or index >= len(cells):
        return ""
    return re.sub(r"\s+", " ", cells[index]).strip()


def hotel_record_from_payload(payload: Any, args: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Skiplagged hotel rows captured from the tool payload, not from specialist prose."""
    args = args or {}
    text = payload if isinstance(payload, str) else _payload_text(payload)
    hotels: list[dict[str, str]] = []
    header: list[str] | None = None
    nightly_index = 2
    total_index = 3
    for line in text.splitlines():
        if not line.startswith("|") or re.match(r"^\|\s*-+", line):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells:
            continue
        if header is None and not re.search(r"\*\*", cells[0]) and re.search(r"hotel|name", cells[0], re.I):
            header = [cell.casefold() for cell in cells]
            for index, cell in enumerate(header):
                if "total" in cell:
                    total_index = index
                elif "night" in cell or cell.strip() in {"price", "rate"}:
                    nightly_index = index
            continue
        name_match = re.search(r"\*\*([^*|\n]+)\*\*", cells[0])
        if name_match:
            name = name_match.group(1)
        else:
            name = re.sub(r"<[^>]+>", " ", cells[0])
        name = re.sub(r"\s+", " ", name).strip()
        if not name or _is_hotel_label(name) or name.casefold() in {"hotel", "name"}:
            continue
        link_match = re.search(r"\((https?://[^)\s]+)\)", " ".join(cells))
        if not link_match:
            link_match = re.search(r"https?://\S+", " ".join(cells))
        hotels.append(
            {
                "name": name,
                "nightly": _hotel_cell(cells, nightly_index),
                "total": _hotel_cell(cells, total_index),
                "link": (link_match.group(1) if link_match and link_match.lastindex else link_match.group(0) if link_match else ""),
            }
        )
    if not hotels:
        return None
    checkin = _first_arg(args, ("checkin", "check_in", "checkIn"))
    checkout = _first_arg(args, ("checkout", "check_out", "checkOut"))
    if not checkin:
        found = re.search(r"Check-in:\s*(\d{4}-\d{2}-\d{2})", text)
        checkin = found.group(1) if found else ""
    if not checkout:
        found = re.search(r"Check-out:\s*(\d{4}-\d{2}-\d{2})", text)
        checkout = found.group(1) if found else ""
    return {
        "city": _first_arg(args, ("city", "destination", "location")),
        "checkin": checkin,
        "checkout": checkout,
        "hotels": hotels,
    }


def render_hotel_section(record: dict[str, Any]) -> str:
    city = (record.get("city") or "the destination").strip() or "the destination"
    heading = (
        f"Hotels in {city}"
        + _when_span(str(record.get("checkin") or ""), str(record.get("checkout") or ""))
        + ":"
    )
    lines = [heading]
    for hotel in record.get("hotels") or []:
        if not isinstance(hotel, dict):
            continue
        name = str(hotel.get("name") or "").strip()
        if not name or _is_hotel_label(name):
            continue
        bits = [name]
        if hotel.get("nightly"):
            bits.append(f"{hotel['nightly']} per night")
        if hotel.get("total"):
            bits.append(f"{hotel['total']} total")
        if hotel.get("link"):
            bits.append(str(hotel["link"]))
        lines.append("- " + " | ".join(bits))
    return "\n".join(lines)


def _is_section_heading(line: str) -> bool:
    """A heading ends with ':' or starts with '#'. A sentence is not a heading."""
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("#"):
        return True
    bare = re.sub(r"^[#>*\-\s]+", "", stripped)
    bare = re.sub(r"[*_`]", "", bare).strip()
    return bare.endswith(":")


def _marked_heading(line: str) -> bool:
    """A heading by markup: a # line, a line wrapped in ``**...**``, or one ending in ':'."""
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("#") or stripped.endswith(":"):
        return True
    wrapped = re.fullmatch(r"\*\*(.+)\*\*", stripped)
    if not wrapped:
        return False
    return not _is_prose_sentence(wrapped.group(1).strip())


def _is_prose_sentence(text: str) -> bool:
    """True when punctuation ends a sentence, not an abbreviation such as D.C. or St."""
    cleaned = re.sub(r"\b(?:St|Ft|Mt|avg)\.", "", text, flags=re.I)
    cleaned = re.sub(r"(?<![A-Za-z])[A-Z]\.", "", cleaned)
    if re.search(r"[!?]\s+\S", cleaned) or re.search(r"\.\s+\S", cleaned):
        return True
    return cleaned.rstrip().endswith((".", "!", "?"))


def _section_kind(line: str) -> str | None:
    raw = line.strip()
    # The car intro can lead with the assumed age, so it does not start with "Rental".
    if re.search(r"assumed driver age\s*30", raw, re.I) and re.search(r"\brental cars?\b", raw, re.I):
        return "cars"
    stripped = re.sub(r"^[#>*\-\s]+", "", raw).strip()
    bare = re.sub(r"[*_`]", "", stripped).strip()
    if not _marked_heading(raw) and _is_prose_sentence(bare):
        return None
    match = re.match(
        r"(flights?|hotels?|rental cars?|car rental|restaurants?|dining|bars?)\b",
        stripped,
        re.I,
    )
    if not match:
        return None
    if len(stripped) > 90 and not stripped.endswith(":"):
        return None
    word = match.group(1).lower()
    if word.startswith("flight"):
        return "flights"
    if word.startswith("hotel"):
        return "hotels"
    if word.startswith("rental") or word.startswith("car"):
        return "cars"
    if word.startswith("rest") or word.startswith("din"):
        return "dining"
    if word.startswith("bar"):
        return "bars"
    return None


def _label_only_line(line: str) -> bool:
    cleaned = re.sub(r"[*_`#]", "", line)
    cleaned = re.sub(r"^[\-\d.)\s]+", "", cleaned).strip().strip(":").casefold()
    return bool(cleaned) and cleaned in HOTEL_FIELD_LABELS


def _skippable_section_line(kind: str, line: str) -> bool:
    """Lines that belong to the coordinator's old cars block.

    Hotels and flights are not guessed from markup. Those repeats are removed
    only when the line names a structured result.
    """
    stripped = line.strip()
    if not stripped:
        return kind not in {"hotels", "flights"}
    if _section_kind(line):
        return False
    if _label_only_line(line):
        return True
    if kind == "cars" and (
        stripped.startswith(("-", "*", "•"))
        or re.search(r"https?://", stripped)
        or re.fullmatch(r"[\s():]*", stripped)
    ):
        return True
    return False


def _collapse_blank_lines(text: str) -> str:
    """A skip can leave the gap on both sides of a removed block."""
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def replace_named_section(answer: str, kind: str, new_section: str) -> str:
    """Replace one plan section. Prose that follows it stays."""
    lines = (answer or "").splitlines()
    out: list[str] = []
    skipping = False
    inserted = False
    for line in lines:
        if _section_kind(line) == kind:
            if not inserted:
                out.extend(new_section.splitlines())
                inserted = True
            skipping = True
            continue
        if skipping and _skippable_section_line(kind, line):
            continue
        skipping = False
        out.append(line)
    text = _collapse_blank_lines("\n".join(out))
    if not inserted:
        text = _collapse_blank_lines(text + "\n\n" + new_section)
    return text


def replace_car_section(answer: str, record: dict[str, Any]) -> str:
    section = render_car_section(record)
    text = replace_named_section(answer, "cars", section)
    urls = {
        str(offer.get("booking_url"))
        for offer in record.get("offers") or []
        if isinstance(offer, dict) and offer.get("booking_url")
    }
    heading = section.splitlines()[0]
    kept: list[str] = []
    in_section = False
    for line in text.splitlines():
        if line == heading:
            in_section = True
        if not in_section and any(url and url in line for url in urls):
            continue
        if _empty_car_heading(line):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def _empty_car_heading(line: str) -> bool:
    """Drop 'Rental Cars in Lisbon (' and a following '):' with no dates."""
    stripped = line.strip()
    if re.fullmatch(r"[\s():]*", stripped):
        return True
    if not re.search(r"rental cars?\b", stripped, re.I):
        return False
    if re.search(r"\d", stripped):
        return False
    return bool(re.search(r"\(\s*\)", stripped) or re.search(r"\(\s*$", stripped))


def _hotel_names(record: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for hotel in record.get("hotels") or []:
        if not isinstance(hotel, dict):
            continue
        name = str(hotel.get("name") or "").strip()
        if name and not _is_hotel_label(name):
            names.append(name)
    return names


def _list_item_body(line: str) -> str | None:
    match = re.match(r"^\s*(?:[-*•]|\d+[.)])\s+(.*)$", line)
    if not match:
        return None
    return re.sub(r"[*_`]", "", match.group(1)).strip()


def _booking_line_repeats_hotel(line: str, links: set[str]) -> bool:
    """A 'Booking:' line whose only job is to repeat a URL already in the hotel block."""
    if not links or not re.match(r"^\s*(?:[-*•]\s*)?(?:\*\*)?booking(?:\*\*)?\s*:", line, re.I):
        return False
    return any(link in line for link in links)


_STRUCTURED_FIELD_BULLET = re.compile(
    r"^\s*(?:[-*•]|\d+[.)])\s*(?:\*\*)?"
    r"(?:nightly rate|price(?:\s*/\s*night)?|rating|booking|fare|outbound|return)"
    r"(?:\*\*)?\s*:",
    re.I,
)
_DAY_LINE = re.compile(r"\bday\s+\d+\b", re.I)
_GROUND_LEG = re.compile(r"\b(?:train|metro|bus|tram|taxi|ferry)\b", re.I)
_FLIGHT_DESIGNATOR = re.compile(r"\b[A-Z]{1,3}\d{2,4}\b")


def _line_names_structured_hotel(line: str, names: list[str]) -> bool:
    """The line is a copy of a hotel already stored on the hotel record."""
    plain = re.sub(r"[*_`#]", "", line).strip()
    plain = re.sub(r"^(?:[-*•]|\d+[.)])\s+", "", plain).strip()
    folded = plain.casefold()
    return any(name and folded.startswith(name.casefold()) for name in names)


def _record_fare_tokens(record: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for item in record.get("itineraries") or []:
        if not isinstance(item, dict):
            continue
        for key in ("price", "return_price"):
            raw = item.get(key)
            if raw is None or str(raw).strip() == "":
                continue
            token = str(raw).split()[0].lstrip("$€£")
            if token:
                tokens.add(token)
    return tokens


def _line_matches_structured_flight(line: str, numbers: set[str], fares: set[str]) -> bool:
    """A copy of a stored flight number or fare, or an offer heading that names a flight."""
    if _labelled_alternate_flight(line) or _DAY_LINE.search(line) or _GROUND_LEG.search(line):
        return False
    if _line_mentions_flight_number(line, numbers):
        return True
    if any(fare and re.search(rf"\b{re.escape(fare)}\b", line) for fare in fares):
        return True
    if not _FLIGHT_DESIGNATOR.search(line):
        return False
    stripped = line.strip()
    if re.fullmatch(r"\*\*.+\*\*", stripped) or re.match(r"\d+[.)]\s+\S", stripped):
        return True
    return False


def _drop_structured_repeats(lines: list[str], matches) -> list[str]:
    """Drop structured repeats and the field bullets that sit directly under them."""
    kept: list[str] = []
    run = False
    for line in lines:
        if not line.strip():
            run = False
            kept.append(line)
            continue
        if matches(line):
            run = True
            continue
        if run and _STRUCTURED_FIELD_BULLET.match(line):
            continue
        run = False
        kept.append(line)
    return kept


def _line_restates_hotel(line: str, names: list[str]) -> bool:
    """A list item whose text starts with a hotel already in the rendered block."""
    body = _list_item_body(line)
    if body is None or not names:
        return False
    folded = body.casefold()
    return any(folded.startswith(name.casefold()) for name in names)


def _under_named_heading(lines: list[str], kind: str) -> list[str]:
    """Drop a real heading of ``kind`` and the list that sits directly under it."""
    kept: list[str] = []
    under = False
    for line in lines:
        if _is_section_heading(line):
            under = _section_kind(line) == kind
            if under:
                continue
            kept.append(line)
            continue
        if under:
            if not line.strip() or _list_item_body(line) is not None:
                continue
            under = False
        kept.append(line)
    return kept


def replace_hotel_section(answer: str, record: dict[str, Any]) -> str:
    section = render_hotel_section(record)
    text = replace_named_section(answer, "hotels", section)
    names = _hotel_names(record)
    links = {
        str(hotel.get("link"))
        for hotel in record.get("hotels") or []
        if isinstance(hotel, dict) and hotel.get("link")
    }

    def keep(line: str) -> bool:
        if _label_only_line(line):
            return False
        if _line_restates_hotel(line, names):
            return False
        if _booking_line_repeats_hotel(line, links):
            return False
        return True

    parts = _split_around(text, section)
    if parts is None:
        outside = [line for line in text.splitlines() if line not in section.splitlines()]
        kept = [line for line in _under_named_heading(outside, "hotels") if keep(line)]
        return "\n".join([*kept, *section.splitlines()]).strip()
    before, after = parts
    after = _drop_structured_repeats(after, lambda line: _line_names_structured_hotel(line, names))
    lines = [line for line in _under_named_heading(before, "hotels") if keep(line)]
    lines.extend(section.splitlines())
    lines.extend(line for line in _under_named_heading(after, "hotels") if keep(line))
    return _collapse_blank_lines("\n".join(lines))


def separate_alternate_flights(answer: str, flight_text: str) -> str:
    """Drop booking links from date searches that were not the requested trip."""
    if "ALTERNATE DATES" not in (flight_text or ""):
        return answer
    primary, _, alternate = flight_text.partition("ALTERNATE DATES")
    primary_links = set(re.findall(r"booking_link:\s*(\S+)", primary))
    alt_links = [link for link in re.findall(r"booking_link:\s*(\S+)", alternate) if link not in primary_links]
    if not alt_links:
        return answer
    kept: list[str] = []
    removed = False
    for line in answer.splitlines():
        hidden = any(link in line for link in alt_links)
        labeled = bool(re.search(r"alternate|different date", line, re.I))
        if hidden and not labeled:
            removed = True
            continue
        kept.append(line)
    text = "\n".join(kept).strip()
    if removed:
        text += "\n\nItineraries for other dates were left out of the plan."
    return text


def _itinerary_blocks(text: str) -> list[str]:
    return [
        chunk.strip()
        for chunk in re.split(r"\n(?=ITINERARY \d+)", text or "")
        if chunk.strip().startswith("ITINERARY")
    ]


def _block_value(block: str, field: str) -> str:
    match = re.search(rf"(?m)^{field}:\s*(\S+)", block)
    return match.group(1) if match else ""


def ensure_flight_legs(answer: str, flight_text: str) -> str:
    """Keep legs for the itineraries the plan presents, or the top 3."""
    if "legs:" not in (flight_text or ""):
        return answer
    primary = flight_text.split("ALTERNATE DATES")[0]
    blocks = _itinerary_blocks(primary)
    if not blocks:
        return answer
    presented = [
        block for block in blocks if (link := _block_value(block, "booking_link")) and link in answer
    ]
    if not presented:
        presented = [
            block
            for block in blocks
            if (price := _block_value(block, "price")) and price in answer
        ]
    chosen = presented or blocks[:3]
    missing: list[str] = []
    for block in chosen:
        stamps = re.findall(r"depart (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?)", block)
        if stamps and all(stamp in answer for stamp in stamps):
            continue
        missing.append(block)
    if not missing:
        return answer
    return (
        answer.rstrip()
        + "\n\nFlight legs for the requested dates:\n\n"
        + "\n\n".join(missing)
    )


def _block_price_text(block: str) -> str:
    match = re.search(
        r"(?m)^price:\s*([0-9]+(?:\.[0-9]+)?)(?:\s*([A-Za-z]{3}|€))?",
        block or "",
    )
    if not match:
        return ""
    amount, currency = match.group(1), (match.group(2) or "").upper()
    if amount.endswith(".0"):
        amount = amount[:-2]
    if currency == "€":
        currency = "EUR"
    return f"{amount} {currency}".strip()


def ensure_flight_prices(answer: str, flight_text: str) -> str:
    """Put each presented itinerary's price on that itinerary, in Kiwi's currency."""
    if "ITINERARY" not in (flight_text or "") or not answer:
        return answer
    primary = flight_text.split("ALTERNATE DATES")[0]
    blocks = _itinerary_blocks(primary)
    presented = [
        block for block in blocks if (link := _block_value(block, "booking_link")) and link in answer
    ]
    chosen = presented or blocks[:3]
    for block in chosen:
        price = _block_price_text(block)
        link = _block_value(block, "booking_link")
        if not price:
            continue
        amount = price.split()[0]
        if link and link in answer:
            index = answer.find(link)
            window = answer[max(0, index - 80) : index + len(link)]
            if amount in window and ("EUR" not in price or "EUR" in window or "€" in window):
                continue
            answer = answer.replace(link, f"{price} {link}", 1)
        elif not re.search(rf"\b{re.escape(amount)}\b", answer):
            answer = answer.rstrip() + f"\n- {price}" + (f" {link}" if link else "")
    return answer


def lock_flight_currency(answer: str, flight_text: str) -> str:
    """Keep the currency Kiwi returned. Do not leave a euro fare labeled as dollars."""
    if not answer or not flight_text:
        return answer
    for match in re.finditer(
        r"(?m)^price:\s*([0-9]+(?:\.[0-9]+)?)(?:\s*([A-Za-z]{3}|€))?",
        flight_text,
    ):
        amount, currency = match.group(1), (match.group(2) or "").upper()
        if currency not in {"EUR", "€"}:
            continue
        forms = {amount}
        if amount.endswith(".0"):
            forms.add(amount[:-2])
        for form in forms:
            answer = re.sub(rf"\${re.escape(form)}\b", f"{form} EUR", answer)
            answer = re.sub(
                rf"\b{re.escape(form)}\s*(?:USD|US\$)\b",
                f"{form} EUR",
                answer,
                flags=re.I,
            )
    return answer


_INVENTED_DRIVER_AGE = re.compile(
    r"\b(?:driver(?:'s)?\s+age|drivers?\s+aged|age)\s*30\s*\+?\.?",
    re.I,
)


def _human_text(messages: list[Any]) -> str:
    parts: list[str] = []
    for message in messages:
        if getattr(message, "type", None) == "human" or isinstance(message, HumanMessage):
            parts.append(_message_text(message.content))
    return "\n".join(parts)


def enforce_assumed_driver_age(answer: str, messages: list[Any], car_text: str) -> str:
    """A tool default of 30 is an assumption. Do not present it as the traveler's age."""
    answer = re.sub(
        r"assumed driver age 30\s*[—–-]\s*tell me yours if different",
        ASSUMED_DRIVER_AGE,
        answer or "",
        flags=re.I,
    )
    stated = stated_driver_age(_human_text(messages))
    if stated:
        if ASSUMED_DRIVER_AGE in answer:
            return answer.replace(
                ASSUMED_DRIVER_AGE,
                f"Driver age {stated} was taken from the request.",
            )
        return answer
    mentions_cars = bool(car_text.strip()) or bool(
        re.search(r"\brental\b|\bdriver age\b", answer or "", re.I)
    )
    if not mentions_cars and ASSUMED_DRIVER_AGE not in (answer or ""):
        return answer
    chunks = (answer or "").split(ASSUMED_DRIVER_AGE)
    scrubbed = [_INVENTED_DRIVER_AGE.sub("", chunk) for chunk in chunks]
    body = "\n\n".join(re.sub(r"\n{3,}", "\n\n", chunk).strip() for chunk in scrubbed if chunk.strip())
    if ASSUMED_DRIVER_AGE in body:
        return body
    return (body.rstrip() + "\n\n" + ASSUMED_DRIVER_AGE).strip()


_NOT_BAR = re.compile(
    r"\b(pub crawls?|bar crawls?|walking tours?|food tours?|city tours?|bike tours?|bus tours?|tuk-?tuks?|free tours?|sightseeing tours?)\b",
    re.I,
)


def is_bar_venue(name: str) -> bool:
    return bool(name) and _NOT_BAR.search(name) is None


def _toolish(message: Any) -> bool:
    return isinstance(message, ToolMessage) or getattr(message, "type", None) == "tool"


def _message_payload_text(message: Any) -> str:
    parts = [_message_text(getattr(message, "content", ""))]
    artifact = getattr(message, "artifact", None)
    if isinstance(artifact, (dict, list)):
        parts.append(json.dumps(artifact, default=str))
    elif artifact not in (None, ""):
        parts.append(_message_text(artifact))
    return "\n".join(part for part in parts if part)


def car_record_from_messages(messages: list[Any]) -> dict[str, Any] | None:
    """Read a car record from every tool message, including content the model never wrote."""
    text = "\n".join(_message_payload_text(item) for item in messages or [] if _toolish(item))
    record = read_structured(text, "CARS")
    if record and record.get("offers"):
        return record
    parsed = car_record_from_payload(text, {})
    if parsed and parsed.get("offers"):
        return parsed
    return None


def hotel_record_from_messages(messages: list[Any]) -> dict[str, Any] | None:
    text = "\n".join(_message_payload_text(item) for item in messages or [] if _toolish(item))
    record = read_structured(text, "HOTELS")
    if record and record.get("hotels"):
        return record
    parsed = hotel_record_from_payload(text, {})
    if parsed and parsed.get("hotels"):
        return parsed
    return None


def active_car_record(messages: list[Any], *, allow_stash: bool = False) -> dict[str, Any] | None:
    found = car_record_from_messages(messages)
    if found and found.get("offers"):
        return found
    called = allow_stash or bool(_tool_blob(messages, "suggest_rental_cars").strip())
    if not called:
        return None
    return stashed_record("cars")


def _hotels_priced(record: dict[str, Any] | None) -> bool:
    if not record:
        return False
    return any(
        isinstance(hotel, dict) and (hotel.get("total") or hotel.get("nightly"))
        for hotel in record.get("hotels") or []
    )


def active_hotel_record(messages: list[Any], *, allow_stash: bool = False) -> dict[str, Any] | None:
    found = hotel_record_from_messages(messages)
    stashed = None
    if allow_stash or _tool_blob(messages, "search_hotels").strip():
        stashed = stashed_record("hotels")
    if stashed and _hotels_priced(stashed):
        return stashed
    if found and found.get("hotels"):
        return found
    return stashed


def _drop_named_lines(answer: str, names: set[str]) -> str:
    if not names:
        return answer
    kept = [
        line
        for line in (answer or "").splitlines()
        if not any(name.casefold() in line.casefold() for name in names)
    ]
    return "\n".join(kept).strip()


def _labelled_alternate_flight(line: str) -> bool:
    """A coordinator alternate for other dates, which the prompt allows beside the search."""
    return bool(re.search(r"\balternate\b", line, re.I) and re.search(r"different dates", line, re.I))


def _stale_flight_offer_line(line: str) -> bool:
    """A leftover fare under an empty Flights heading: a Kiwi link, or a number beside a price."""
    if _labelled_alternate_flight(line) or _DAY_LINE.search(line) or _GROUND_LEG.search(line):
        return False
    if re.search(r"kiwi\.com", line, re.I):
        return True
    return bool(_FLIGHT_PRICE.search(line) and re.search(r"\b[A-Z]{1,3}\d{2,4}\b", line))


def _replace_empty_flight_section(answer: str) -> str:
    """Say that nothing was found, and drop the priced lines that sat under the old heading."""
    text = replace_named_section(answer, "flights", NO_FLIGHTS_FOUND)
    if NO_FLIGHTS_FOUND not in text:
        text = (text.rstrip() + "\n\n" + NO_FLIGHTS_FOUND).strip()
    kept: list[str] = []
    under = False
    for line in text.splitlines():
        if line.strip() == NO_FLIGHTS_FOUND:
            under = True
            kept.append(line)
            continue
        if under and _is_section_heading(line) and _section_kind(line) != "flights":
            under = False
        if under and _stale_flight_offer_line(line):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def apply_coordinator_checks(answer: str, messages: list[Any]) -> str:
    """Drop ungrounded venues, unconfirmed hours, mixed flight fields, and lost hotels."""
    restaurants = _tool_blob(messages, "suggest_restaurants")
    bars = _tool_blob(messages, "suggest_bars")
    hotels = _tool_blob(messages, "search_hotels")
    flights = _tool_blob(messages, "search_flights")
    cars = _tool_blob(messages, "suggest_rental_cars")
    allowed: set[str] = set()
    explicit_none = False
    dining_called = bool(restaurants.strip() or bars.strip())
    excluded_bars: set[str] = set()
    for blob, bar_blob in ((restaurants, False), (bars, True)):
        parsed = parse_venues_line(blob)
        names = extract_grounded_venues(blob) if parsed is None else parsed
        if bar_blob:
            excluded_bars |= {name for name in names if not is_bar_venue(name)}
            names = {name for name in names if is_bar_venue(name)}
            if parsed is not None and not parsed:
                explicit_none = True
        elif parsed is not None and not parsed:
            explicit_none = True
        allowed |= names
    hotel_names = extract_hotel_names(hotels)
    before = answer
    if excluded_bars:
        answer = _drop_named_lines(answer, excluded_bars)
    if dining_called:
        answer = drop_ungrounded_claims(answer, allowed, hotel_names)
        dropped = answer != before
        answer = scrub_unconfirmed_hours(answer, restaurants + "\n" + bars)
        answer = scrub_overclaims(answer, restaurants + "\n" + bars, allowed)
        if dropped and (explicit_none or not allowed) and "unavailable" not in answer.lower():
            answer = answer.rstrip() + "\n\n" + VENUE_UNAVAILABLE
    hotel_record = active_hotel_record(messages)
    if hotel_record:
        answer = replace_hotel_section(answer, hotel_record)
    car_record = active_car_record(messages)
    if car_record:
        answer = replace_car_section(answer, car_record)
    answer = enforce_assumed_driver_age(answer, messages, cars)
    flight_record = active_flight_record(messages)
    if flight_record and flight_record.get("itineraries"):
        answer = replace_flight_section(answer, flight_record)
    elif _searches_found_no_flights(messages):
        answer = _replace_empty_flight_section(answer)
    else:
        answer = repair_flight_pairing(answer, flights)
        answer = separate_alternate_flights(answer, flights)
        answer = ensure_flight_legs(answer, flights)
        answer = ensure_flight_prices(answer, flights)
    answer = lock_flight_currency(answer, flights)
    return answer


def coordinator_postcheck(state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
    messages = list(state.get("messages") or [])
    if not messages:
        return None
    last = messages[-1]
    if not isinstance(last, AIMessage) or last.tool_calls or not last.content:
        return None
    text = _message_text(last.content)
    updated = apply_coordinator_checks(text, messages)
    if updated == text:
        return None
    if last.id is None:
        last.id = str(uuid.uuid4())
    return {"messages": [last.model_copy(update={"content": updated})]}


coordinator_postcheck_middleware = after_model(coordinator_postcheck)
capture_middleware = before_model(_open_capture)


def _seal_tool_text(tool: BaseTool, content: Any, artifact: Any, sealed: str) -> Any:
    if getattr(tool, "response_format", "content") == "content_and_artifact":
        return (sealed, artifact)
    return sealed


def attach_car_capture(tool: BaseTool) -> BaseTool:
    """Record OctoTrip or Upfront offers at tool-call time when the fallback wrapper is absent."""
    if tool.coroutine is None or getattr(tool, "_trip_car_capture", False):
        return tool
    original = tool.coroutine

    async def wrapped(**kwargs: Any) -> Any:
        value = await original(**kwargs)
        content, artifact = value if isinstance(value, tuple) and len(value) == 2 else (value, None)
        return _seal_tool_text(tool, content, artifact, seal_car_tool_result(content, kwargs))

    tool.coroutine = wrapped
    setattr(tool, "_trip_car_capture", True)
    return tool


def attach_hotel_capture(tool: BaseTool) -> BaseTool:
    """Record Skiplagged hotel rows at tool-call time."""
    if tool.coroutine is None or getattr(tool, "_trip_hotel_capture", False):
        return tool
    original = tool.coroutine

    async def wrapped(**kwargs: Any) -> Any:
        value = await original(**kwargs)
        content, artifact = value if isinstance(value, tuple) and len(value) == 2 else (value, None)
        return _seal_tool_text(
            tool,
            content,
            artifact,
            seal_hotel_tool_result(content, kwargs, tool_name=tool.name),
        )

    tool.coroutine = wrapped
    setattr(tool, "_trip_hotel_capture", True)
    return tool


def attach_flight_binding(tool: BaseTool) -> BaseTool:
    """Replace a Kiwi flight payload with one locked block per itinerary."""
    if "flight" not in tool.name or tool.coroutine is None:
        return tool
    original = tool.coroutine

    async def wrapped(**kwargs: Any) -> Any:
        prepared = complete_flight_args(kwargs)
        value = await original(**prepared)
        content, artifact = value if isinstance(value, tuple) and len(value) == 2 else (value, None)
        payload = _coerce_payload(content)
        items = _walk_itineraries(payload)
        if not items and artifact is not None:
            payload = _coerce_payload(artifact)
            items = _walk_itineraries(payload)
        if items:
            envelope = str(payload.get("currency") or "") if isinstance(payload, dict) else ""
            stash_flight_search({"args": prepared, "itineraries": items, "currency": envelope})
        rendered = render_kiwi_itineraries(payload if items else _coerce_payload(content), prepared)
        if getattr(tool, "response_format", "content") == "content_and_artifact":
            return (rendered, artifact)
        return rendered

    tool.coroutine = wrapped
    return tool


def attach_date_strip(tool: BaseTool) -> BaseTool:
    """Places queries should not carry the trip dates."""
    if tool.coroutine is None or getattr(tool, "_trip_dates_stripped", False):
        return tool
    original = tool.coroutine

    async def wrapped(**kwargs: Any) -> Any:
        cleaned = {
            key: strip_trip_dates(value) if isinstance(value, str) else value
            for key, value in kwargs.items()
        }
        return await original(**cleaned)

    tool.coroutine = wrapped
    setattr(tool, "_trip_dates_stripped", True)
    return tool


async def _delegate(key: str, request: str, runtime: ToolRuntime) -> str:
    try:
        specialists = await get_specialists()
        agent = specialists[key]
        state = runtime.state or {}
        message = (
            f"Origin: {state.get('origin') or 'unspecified'}. "
            f"Destination: {state.get('destination') or 'unspecified'}. "
            f"Guests: {state.get('guest_count') or 'unspecified'}. "
            f"Trip type: {state.get('genre') or 'unspecified'}. "
            f"Task: {request or 'Use the trip details above.'}"
        )
        bind_trip_window(_human_text(list(state.get("messages") or [])) + "\n" + message)
        begin_capture()
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=message)]},
            config={"recursion_limit": 40},
        )
        if key == "flight":
            messages = result.get("messages") or []
            record = active_flight_record(messages, allow_stash=True)
            if record and record.get("itineraries"):
                return render_flight_section(record) + "\n" + format_structured("FLIGHTS", record)
            locked = flight_specialist_text(result)
            if locked:
                return locked
        if key == "rental_car":
            messages = result.get("messages") or []
            record = active_car_record(messages, allow_stash=True)
            user_text = _human_text(list(state.get("messages") or []))
            note = driver_age_note(stated_driver_age(user_text))
            if record:
                # The car section is this record, not whatever the specialist model wrote.
                return (
                    render_car_section(record)
                    + "\n\n"
                    + note
                    + "\n"
                    + format_structured("CARS", record)
                )
            return (final_text(result).rstrip() + "\n" + note).strip()
        if key == "hotel":
            messages = result.get("messages") or []
            record = active_hotel_record(messages, allow_stash=True)
            if record:
                return render_hotel_section(record) + "\n" + format_structured("HOTELS", record)
            return final_text(result)
        if key in {"restaurant", "bar"}:
            messages = result.get("messages") or []
            tool_messages = [item for item in messages if _toolish(item)]
            evidence = "\n".join(_message_payload_text(item) for item in tool_messages)
            allowed = extract_grounded_venues(evidence)
            if key == "bar":
                allowed = {name for name in allowed if is_bar_venue(name)}
            return render_grounded_specialist(
                final_text(result),
                allowed,
                evidence,
                tools_called=bool(tool_messages),
            )
        return final_text(result)
    except Exception as exc:
        return f"{key} specialist failed: {exc}"


@tool
async def search_flights(request: str, runtime: ToolRuntime) -> str:
    """Ask the flight specialist for economy options. Pass origin, destination, and dates. Include a return date when the trip has one. The result is one rendered Flights block; do not restate an itinerary already in it."""
    return await _delegate("flight", request, runtime)


@tool
async def search_hotels(request: str, runtime: ToolRuntime) -> str:
    """Ask the hotel specialist for lodging. Pass destination, dates, guests, rooms, and preferences in request. The result is one rendered Hotels block; do not add a second list of those hotels."""
    return await _delegate("hotel", request, runtime)


@tool
async def suggest_rental_cars(request: str, runtime: ToolRuntime) -> str:
    """Ask the rental car specialist. Pass pickup location, dates, and driver constraints in request."""
    return await _delegate("rental_car", request, runtime)


@tool
async def suggest_restaurants(request: str, runtime: ToolRuntime) -> str:
    """Ask the restaurant specialist. Pass location, meal, date, party size, and dietary needs in request."""
    return await _delegate("restaurant", request, runtime)


@tool
async def suggest_bars(request: str, runtime: ToolRuntime) -> str:
    """Ask the bar specialist. Pass neighborhood, night, and the kind of venue in request."""
    return await _delegate("bar", request, runtime)


@tool
def update_state(
    origin: str,
    destination: str,
    guest_count: str,
    genre: str,
    runtime: ToolRuntime,
) -> Command:
    """Update the state when you know all of the values: origin, destination, guest_count, genre.
    This tool must be called alone, without any other tool calls. It must complete and return to make
    the information available to other tools."""
    return Command(
        update={
            "origin": origin,
            "destination": destination,
            "guest_count": guest_count,
            "genre": genre,
            "messages": [
                ToolMessage(
                    "Successfully updated state",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


COORDINATOR_TOOLS = [
    update_state,
    search_flights,
    search_hotels,
    suggest_rental_cars,
    suggest_restaurants,
    suggest_bars,
    web_search,
]

COORDINATOR_PROMPT = """
You are a trip planner coordinator.
First find origin, destination, guest count, and the kind of trip (genre). When you have them, call update_state by itself and wait for it to return.
Then delegate with search_flights, search_hotels, suggest_rental_cars, suggest_restaurants, and suggest_bars. Independent specialists may be called in parallel; that is faster than waiting for one before the next. Pass each specialist a self-contained request with dates, party size, and preferences from the conversation.
Use web_search only for gaps the specialists do not cover: events, attractions, travel advisories, and opening hours. Do not put trip dates into web search queries. Do not invent prices, venues, ratings, or hours.
Keep every hotel the hotel specialist returned. Drop one only when you explain why. Hotel names are property names from the hotel tool. Do not list field labels such as Neighborhood, Nightly rate, Cancellation, or Tradeoff as hotels.
The hotel specialist returns one rendered Hotels block. That block is the only hotel section. Do not append another list, numbered or bulleted, of hotels already in that block.
For flights, the specialist returns one rendered Flights block. Do not restate an itinerary whose flight numbers and times are already in that block. Keep each itinerary you present (or the top 3) with its price, legs, and booking link. Show the currency Kiwi returned, such as EUR, and do not relabel it as dollars. If a return arrives on a later day, label it next-day arrival. Repeat a warning that arrival is late or that travel eats the trip.
Name a restaurant or bar only when that specialist's tool output names it as a place. Drop photo credits, bylines, and any other name that is not a returned venue. When a specialist says the source is unavailable, repeat that and name no venues.
Do not say hours or open status are confirmed for the trip dates unless a tool result states it. Otherwise say check hours before going. Do not relabel cuisine or move a place to a meal the source did not support.
Do not invent a driver age. When the traveler did not give one, the car search uses 30 and the plan must say exactly: assumed driver age 30 — tell me yours if different. Keep each rental's name, price, vendor, and booking link.
When the specialists have answered, combine their facts into one itinerary. Name any source that was unavailable. You plan and compare; you do not book.
""".strip()


def _is_mcp_request(request: Any) -> bool:
    tool_obj = getattr(request, "tool", None)
    metadata = getattr(tool_obj, "metadata", None) or {}
    return isinstance(metadata, dict) and "mcp" in metadata


def _failure_message(request: Any, text: str) -> ToolMessage:
    call = getattr(request, "tool_call", None) or {}
    return ToolMessage(
        content=text,
        tool_call_id=call.get("id"),
        name=call.get("name"),
        status="error",
    )


async def retry_mcp_tool(request: Any, handler: Callable[[Any], Awaitable[Any]]) -> Any:
    """Retry transient MCP tool failures and return an error message instead of crashing."""
    if not _is_mcp_request(request):
        return await handler(request)
    name = (getattr(request, "tool_call", None) or {}).get("name", "unknown")
    last: BaseException | None = None
    attempts = _max_retries()
    for attempt in range(attempts):
        try:
            return await handler(request)
        except MCPError as exc:
            last = exc
            print(
                f"[MCP interceptor] {type(exc).__name__} on {name} "
                f"(code {exc.error.code}, attempt {attempt + 1}/{attempts}): {exc}"
            )
            if exc.error.code not in RETRYABLE_MCP_CODES:
                return _failure_message(request, f"Tool call failed (non-retryable): {exc}")
        except Exception as exc:
            last = exc
            print(
                f"[MCP interceptor] {type(exc).__name__} on {name} "
                f"(attempt {attempt + 1}/{attempts}): {exc}"
            )
            if isinstance(exc, ToolException):
                return _failure_message(request, f"Tool call failed (non-retryable): {exc}")
        if attempt < attempts - 1:
            await asyncio.sleep(2**attempt)
    print(f"[MCP interceptor] all {attempts} retries exhausted for {name}")
    return _failure_message(
        request,
        f"Tool call failed after {attempts} attempts: {last}",
    )


retry_mcp_middleware = wrap_tool_call(retry_mcp_tool)


def build_graph():
    return create_agent(
        ResolvingModel(),
        tools=COORDINATOR_TOOLS,
        system_prompt=COORDINATOR_PROMPT,
        state_schema=TripState,
        middleware=[capture_middleware, coordinator_postcheck_middleware],
        name="trip_planner",
    )


graph = build_graph()


async def run_query(query: str) -> str:
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"recursion_limit": 50},
    )
    return final_text(result)


def main() -> None:
    import sys

    query = " ".join(sys.argv[1:]).strip() or "plan 3 days in Lisbon in March"
    print(asyncio.run(run_query(query)))


if __name__ == "__main__":
    main()
