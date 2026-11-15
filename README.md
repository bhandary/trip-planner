# Trip planner

A coordinator agent plans a trip and delegates to flight, hotel, rental car, restaurant, and bar specialists. The coordinator is a LangGraph graph (`trip_planner`) so [Agent Chat UI](ui/) can talk to it.

## What you need

- Python 3.11+
- Node 22 and pnpm 10 (for the UI)
- `OPENAI_API_KEY` for a real hosted model. Without it, set `TRIP_PLANNER_FAKE_LLM=1` to run the same graph with an offline stand-in. Do not set that flag on Railway.
- Optional: `TAVILY_API_KEY` for web search, and `GOOGLE_MAPS_KEY` (or `GOOGLE_MAPS_API_KEY`) for restaurant and bar places.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

The default model is OpenAI `gpt-4.1-mini` (`openai:gpt-4.1-mini`). Change it with `OPENAI_MODEL` or `TRIP_PLANNER_MODEL` (for example `openai:gpt-4.1-mini`). Production uses that hosted model. There is no local model server.

## Run the backend and the UI

```bash
make dev
```

That starts the LangGraph server on port 2024 and the UI on port 3000. Or run them separately:

```bash
make backend   # http://127.0.0.1:2024  (health: /health)
make ui        # http://127.0.0.1:3000
```

The UI defaults to `NEXT_PUBLIC_API_URL=http://localhost:2024` and `NEXT_PUBLIC_ASSISTANT_ID=trip_planner`. A message such as "plan 3 days in Lisbon in March" runs the coordinator and streams the reply.

The CLI still works:

```bash
python agent.py "plan 3 days in Lisbon in March"
```

## Tests and live MCP checks

```bash
make test
make probe
```

`make probe` connects to each configured MCP server, lists tools, and attempts one non-booking call.

## MCP servers

| Agent | Primary | Fallback |
| --- | --- | --- |
| Flights | [Kiwi](https://mcp.kiwi.com) | none |
| Hotels | [lastminute.com](https://mcp.lastminute.com/mcp) | [Skiplagged](https://mcp.skiplagged.com/mcp) |
| Rental cars | [OctoTrip](https://mcp.octotrip.app/rental-cars/mcp) | [Upfront Rentals](https://www.upfront-rentals.com/MCP/Streamable) |
| Restaurants and bars | [Google Maps Grounding Lite](https://mapstools.googleapis.com/mcp) | none |

Kiwi, lastminute.com, Skiplagged, OctoTrip, and Upfront Rentals are called with no API key. OctoTrip blocks the default HTTP client user agent, so the car agent sends `User-Agent: Mozilla/5.0 (compatible; trip-planner/1.0)`. Google Maps is added only when `GOOGLE_MAPS_KEY` or `GOOGLE_MAPS_API_KEY` is set, and the header is `X-Goog-Api-Key`. If that key is missing, the restaurant and bar agents skip Maps and say so.

Tavily (`langchain-tavily`) is a general web search tool for gaps: events, attractions, advisories, hours, and fees. It is not a substitute for live MCP prices. Without `TAVILY_API_KEY` the tool returns an error payload instead of raising.

If a primary server errors or times out, the hotel and car agents use the fallback server. If both fail, the specialist says the source is unavailable and does not crash the run.

`langchain-mcp-adapters` is archived. Tool loading uses `langchain.mcp.MCPAdapter` (`list_tools` on one server), which replaces `MultiServerMCPClient.get_tools(server_name=...)`.

## Deploy to Railway

Do this as two services from this repo. No LangSmith license is required.

The backend is `langgraph dev` (`langgraph-cli[inmem]`). It speaks the Agent Server protocol the chat UI uses (threads, runs, streaming) and does not need `LANGGRAPH_CLOUD_LICENSE_KEY`. `langgraph up` is the licensed deployment path, so it is not used here. The tradeoff is that threads stay on the instance disk and disappear when the container is replaced.

### Backend service

- Root directory: repository root
- Config: `Dockerfile` and `railway.toml`
- Listens on `0.0.0.0:$PORT`
- Health check: `GET /health`
- Variables: `OPENAI_API_KEY`, and any of `OPENAI_MODEL`, `TRIP_PLANNER_MODEL`, `TAVILY_API_KEY`, `GOOGLE_MAPS_KEY`, `MCP_TIMEOUT_SECONDS`. Leave `TRIP_PLANNER_FAKE_LLM` unset.

Generate a public domain. The examples below assume the service is named `trip-planner`.

Every MCP server in this project is remote streamable HTTP. The backend image does not install Node, and it does not spawn `npx` or stdio MCP processes. A server that is down is reported as unavailable.

### Frontend service

- Root directory: `ui`
- Config: `ui/Dockerfile` and `ui/railway.toml`
- Listens on `0.0.0.0:$PORT`
- Health check: `GET /health`
- Build-time variables (Next.js inlines `NEXT_PUBLIC_*` during `pnpm build`):

```
NEXT_PUBLIC_API_URL=https://${{trip-planner.RAILWAY_PUBLIC_DOMAIN}}
NEXT_PUBLIC_ASSISTANT_ID=trip_planner
```

`NEXT_PUBLIC_API_URL` is a Railway reference variable. It must be set on the frontend service before the Docker build, because the Dockerfile declares matching `ARG`s and the browser bundle bakes in the value. Redeploy the frontend after the backend domain exists.

The browser calls the backend directly, so the backend URL has to be reachable from the user's browser.

## Layout

- `agent.py` exports the `trip_planner` graph
- `http_app.py` is the `/health` route
- `langgraph.json` points the LangGraph server at the graph
- `ui/` is the vendored Agent Chat UI (see `ui/VENDOR.md`)
