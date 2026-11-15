"""Trip planner: a coordinator and five specialist agents.

``langchain-mcp-adapters`` (``MultiServerMCPClient``) is archived. MCP support
lives in ``langchain.mcp.MCPAdapter``, which is what this module uses. Each
specialist still loads only its own server, the same split as
``client.get_tools(server_name=...)``, then ``create_agent`` with that
server's tools plus Tavily ``web_search``.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any, Awaitable, Callable

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import wrap_tool_call
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

RETRYABLE_MCP_CODES = {-32603}
DEFAULT_MODEL = "openai:gpt-5-nano"

KIWI_URL = "https://mcp.kiwi.com"
HOTELS_URL = "https://mcp.lastminute.com/mcp"
HOTELS_ALT_URL = "https://mcp.skiplagged.com/mcp"
CARS_URL = "https://mcp.octotrip.app/rental-cars/mcp"
CARS_ALT_URL = "https://www.upfront-rentals.com/MCP/Streamable"
PLACES_URL = "https://mapstools.googleapis.com/mcp"

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
    targets: dict[str, Any] = {
        "kiwi": KIWI_URL,
        "hotels": HOTELS_URL,
        "hotels_alt": HOTELS_ALT_URL,
        "cars": CARS_URL,
        "cars_alt": CARS_ALT_URL,
    }
    key = google_maps_key()
    if key:
        from fastmcp.client.transports import StreamableHttpTransport

        targets["places"] = StreamableHttpTransport(
            PLACES_URL,
            headers={"X-Goog-Api-Key": key},
        )
    return targets


def clear_mcp_cache() -> None:
    global _specialists
    _tool_cache.clear()
    _specialists = None


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
    targets = mcp_targets()
    if server_name not in targets:
        raise KeyError(f"MCP server {server_name!r} is not configured")
    from langchain.mcp import MCPAdapter

    target = targets[server_name]
    async with MCPAdapter(target) as adapter:
        tools = await asyncio.wait_for(
            adapter.list_tools(cache_mode="bypass"),
            timeout=_timeout_seconds(),
        )
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


def _tool_result(tool: BaseTool, value: Any) -> Any:
    if getattr(tool, "response_format", "content") != "content_and_artifact":
        return value
    if isinstance(value, tuple) and len(value) == 2:
        return value
    return (value, None)


async def invoke_with_fallback(
    primary_name: str,
    primary_call: Callable[[dict[str, Any]], Awaitable[Any]],
    alt_tools: list[BaseTool],
    args: dict[str, Any],
) -> Any:
    """Call the primary tool. If it fails, call the alt tool of the same name."""
    try:
        return await primary_call(args)
    except Exception as exc:
        alt = next((candidate for candidate in alt_tools if candidate.name == primary_name), None)
        if alt is None:
            return (
                f"Source unavailable ({primary_name}). "
                f"Primary failed: {exc}. "
                f"No fallback tool named {primary_name}."
            )
        try:
            return await alt.ainvoke(args)
        except Exception as alt_exc:
            return (
                f"Source unavailable ({primary_name}). "
                f"Primary: {exc}. Fallback: {alt_exc}."
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
            return await call_with_retry(lambda: original(**args))

        try:
            return await primary_call(kwargs)
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

            result = await invoke_with_fallback(tool.name, already_failed, alt_tools, kwargs)
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
            "Say that clearly. Do not invent prices or availability."
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
To make things easy, only look for one ticket, one way.
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

Before searching, check that you have pickup location, pickup and drop-off dates
and times, and driver age if the task depends on it. If one is missing and you
cannot infer it from the task, return a short message naming what is missing
instead of guessing. Vehicle class, transmission, and budget are optional; apply
them when given.

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

Work from the location, date, meal, and party size in the task. If the task
includes a hotel or activity location, prefer places within a short walk or
drive of it and state the distance. Treat dietary restrictions and allergies as
hard constraints: only include a place when you have evidence it can accommodate
them, and say what that evidence is. Cuisine, budget, and atmosphere are
preferences; honor them where you can.

Return 3 to 5 options per meal requested, with some variety in price and style
unless the task is narrow. For each give: name, cuisine, neighborhood and
distance from the reference point, price level, rating, hours for the requested
day, whether reservations are needed and how to book, and one sentence on why
it fits this traveler.

Confirm each place is open on the requested day before including it. Report
only details a tool returned; if you could not verify hours or a dietary need,
say so. You recommend; you do not make reservations.
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

Work from the location, date, and time window in the task. Match the type of
venue to what was asked (cocktail bar, brewery or taproom, wine bar, rooftop,
dive, live music, sports bar) and do not substitute a different type without
saying so. If the task mentions a dinner spot or hotel, prefer venues within
walking distance of it and state the distance, since travelers who are drinking
should not need to drive. Mention late-night transport options when a venue is
far from the hotel.

Return 3 to 5 options. For each give: name, type, neighborhood and distance
from the reference point, price level, rating, hours for the requested night,
any cover, reservation, or dress code requirement, and one sentence on the
atmosphere and what it is known for.

Assume all travelers are of legal drinking age for the destination unless the
task says otherwise; if the party includes minors, only include venues that
admit them and say so. Confirm each place is open on the requested night.
Report only details a tool returned. You recommend; you do not book.
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
    Skiplagged and Upfront are fallbacks. Google Maps is registered only when
    ``GOOGLE_MAPS_KEY`` or ``GOOGLE_MAPS_API_KEY`` is set.
    """
    flight_tools, flight_source, flight_errors = await load_with_fallback("kiwi")
    hotel_tools, hotel_source, hotel_errors = await load_with_fallback("hotels", "hotels_alt")
    car_tools, car_source, car_errors = await load_with_fallback("cars", "cars_alt")
    if "places" in mcp_targets():
        place_tools, place_source, place_errors = await load_with_fallback("places")
    else:
        place_tools, place_source, place_errors = (
            [],
            None,
            ["places: skipped, set GOOGLE_MAPS_KEY or GOOGLE_MAPS_API_KEY"],
        )

    if hotel_source == "hotels":
        hotel_tools = [attach_call_fallback(item, "hotels_alt") for item in hotel_tools]
    if car_source == "cars":
        car_tools = [attach_call_fallback(item, "cars_alt") for item in car_tools]

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
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=message)]},
            config={"recursion_limit": 40},
        )
        return final_text(result)
    except Exception as exc:
        return f"{key} specialist failed: {exc}"


@tool
async def search_flights(request: str, runtime: ToolRuntime) -> str:
    """Ask the flight specialist for one-way economy options. Pass origin, destination, and dates in request."""
    return await _delegate("flight", request, runtime)


@tool
async def search_hotels(request: str, runtime: ToolRuntime) -> str:
    """Ask the hotel specialist for lodging. Pass destination, dates, guests, rooms, and preferences in request."""
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
Then delegate, one specialist at a time, using search_flights, search_hotels, suggest_rental_cars, suggest_restaurants, and suggest_bars. Pass each specialist a self-contained request with dates, party size, and preferences from the conversation.
Use web_search only for gaps the specialists do not cover: events, attractions, travel advisories, and opening hours. Do not invent prices.
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
