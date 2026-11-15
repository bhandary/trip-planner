import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, TextContent

RETRYABLE_MCP_CODES = {-32603}

class RetryMCPInterceptor:
    """Intercept MCP tool calls: retry transient failures, surface all errors gracefully.

    - Retryable McpError codes (e.g. -32603): retry with exponential backoff.
    - Non-retryable McpError codes (e.g. -32602): return error message immediately.
    - Any other exception (fetch failed, network errors, etc.): retry then return error message.
    """

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    async def __call__(self, request, handler):
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return await handler(request)
            except McpError as exc:
                last_error = exc
                print(f"[MCP interceptor] {type(exc).__name__} on {request.name} "
                      f"(code {exc.error.code}, attempt {attempt+1}/{self.max_retries}): {exc}")
                if exc.error.code not in RETRYABLE_MCP_CODES:
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"Tool call failed (non-retryable): {exc}")],
                        isError=False,
                    )
            except Exception as exc:
                last_error = exc
                print(f"[MCP interceptor] {type(exc).__name__} on {request.name} "
                      f"(attempt {attempt+1}/{self.max_retries}): {exc}")

            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)

        print(f"[MCP interceptor] all {self.max_retries} retries exhausted for {request.name}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Tool call failed after {self.max_retries} attempts: {last_error}")],
            isError=False,
        )

client = MultiServerMCPClient(
    {
        "travel_server": {
                "transport": "streamable_http",
                "url": "https://mcp.kiwi.com"
            }
    },
    tool_interceptors=[RetryMCPInterceptor()],
)

tools = await client.get_tools()



from typing import Dict, Any
from tavily import TavilyClient
from langchain.tools import tool

tavily_client = TavilyClient()

@tool
def web_search(query: str, search_number: int, max_search_number: int) -> Dict[str, Any]:
    """Search the web for information. You must track your search count by providing
    search_number (starting at 1) and max_search_number on every call.
    Queries must use only plain text characters. Do not use accented or special characters     
      (e.g., use 'capacite' instead of 'capacité').
    """
    if search_number > max_search_number:
        return {"message": "Search limit reached. Please summarize your findings and provide your final answer."}
    try:
        return tavily_client.search(query)
    except Exception as e:
        return {"error": str(e)}


from langchain.agents import AgentState

class TripState(AgentState):
    origin: str
    destination: str



from langchain.agents import create_agent

# Travel agent
flight_agent = create_agent(
    model="gpt-5-nano",
    tools=tools,
    system_prompt="""
    You are a travel agent. Search for flights to the desired destination wedding location.
    You are not allowed to ask any more follow up questions, you must find the best flight options based on the following criteria:
    - Price (lowest, economy class)
    - Duration (shortest)
    - Date (time of year which you believe is best for a wedding at this location)
    To make things easy, only look for one ticket, one way.
    You may need to make multiple searches to iteratively find the best options.
    You will be given no extra information, only the origin and destination. It is your job to think critically about the best options.
    If the MCP tool fails, returns malformed output, or does not give you usable flight results, try the tool again.
    Once you have found the best options, let the user know your shortlist of options.
    """
)


# Hotel agent
hotel_agent = create_agent(
    model="gpt-5-nano",
    tools=tools,
    system_prompt="""
  You are the hotel specialist in a multi-agent trip planning system. A supervisor
agent sends you one lodging task at a time and merges your answer with results
from flight, rental car, restaurant, and bar specialists. Your reader is the
supervisor, not the traveler, so return facts it can combine, not a sales pitch.

Use the hotel search tools to find live availability and prices. Use web search
only when the hotel tools return nothing useful or lack a detail the task needs
(for example resort fees, parking cost, or neighborhood safety), and say which
facts came from web search because they may be stale.

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
    """
)

# Rental car agent
rental_car_agent = create_agent(
    model="gpt-5-nano",
    tools=tools,
    system_prompt="""
  You are the hotel specialist in a multi-agent trip planning system. A supervisor
agent sends you one lodging task at a time and merges your answer with results
from flight, rental car, restaurant, and bar specialists. Your reader is the
supervisor, not the traveler, so return facts it can combine, not a sales pitch.

Use the hotel search tools to find live availability and prices. Use web search
only when the hotel tools return nothing useful or lack a detail the task needs
(for example resort fees, parking cost, or neighborhood safety), and say which
facts came from web search because they may be stale.

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
    """
)

# Restaurant agent
restaurant_agent = create_agent(
    model="gpt-5-nano",
    tools=tools,
    system_prompt="""
You are the restaurant specialist in a multi-agent trip planning system. A
supervisor agent sends you one dining task at a time and merges your answer
into an overall itinerary. Your reader is the supervisor, so return compact
facts it can slot into a schedule.

Use the places search tools to find restaurants, ratings, hours, and locations.
Use web search to fill gaps those tools leave, such as current menus, tasting
menu prices, reservation difficulty, or recent closures, and flag facts that
came from web search.

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
    """
)


# Bar agent
bar_agent = create_agent(
    model="gpt-5-nano",
    tools=tools,
    system_prompt="""
You are the bar and nightlife specialist in a multi-agent trip planning system.
A supervisor agent sends you one task at a time and merges your answer into an
overall itinerary. Your reader is the supervisor, so return compact facts it
can slot into an evening plan.

Use the places search tools to find bars, ratings, hours, and locations. Bar
details are often thin in structured data, so use web search for what is
missing: drink focus, atmosphere, cover charges, dress codes, live music
schedules, and whether a place has closed recently. Flag facts that came from
web search.

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
    """
)


from langchain.tools import ToolRuntime
from langchain.messages import HumanMessage, ToolMessage
from langgraph.types import Command

@tool
async def search_flights(runtime: ToolRuntime) -> str:
    """Travel agent searches for flights to the desired destination wedding location."""
    origin = runtime.state["origin"]
    destination = runtime.state["destination"]
    response = await travel_agent.ainvoke({"messages": [HumanMessage(content=f"Find flights from {origin} to {destination}")]})
    return response['messages'][-1].content

@tool
def search_hotels(runtime: ToolRuntime) -> str:
    return response['messages'][-1].content

@tool
def suggest_rental_cars(runtime: ToolRuntime) -> str:
    return response['messages'][-1].content

@tool
def suggest_restaurants(runtime: ToolRuntime) -> str:
    return response['messages'][-1].content

@tool
def suggest_bars(runtime: ToolRuntime) -> str:
    return response['messages'][-1].content

@tool
def update_state(origin: str, destination: str, guest_count: str, genre: str, runtime: ToolRuntime) -> str:
    """Update the state when you know all of the values: origin, destination, guest_count, genre. 
    This tool must be called alone, without any other tool calls. It must complete and return to make,
    the information available to other tools."""
    return Command(update={
        "origin": origin, 
        "destination": destination, 
        "guest_count": guest_count, 
        "genre": genre, 
        "messages": [ToolMessage("Successfully updated state", tool_call_id=runtime.tool_call_id)]}
        )



from langchain.agents import create_agent

coordinator = create_agent(
    model="gpt-5-nano",
    tools=[search_flights, search_venues, suggest_playlist, update_state],
    state_schema=WeddingState,
    system_prompt="""
    You are a trip planner coordinator. 
    First find all the information you need to update the state. When you have the information, update the state.
    Once that has completed and returned, you can delegate the tasks 
    to your specialists for flights, hotels, rental cars, restaurants and bars.
    Once you have received their answers, coordinate the perfect trip for me.
    """
)
