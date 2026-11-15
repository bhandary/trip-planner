# Trip planner

A coordinator agent plans a trip and delegates to flight, hotel, rental car, restaurant, and bar specialists. The coordinator is a LangGraph graph (`trip_planner`) so [Agent Chat UI](ui/) can talk to it.

## What you need

- Python 3.11+
- Node 22 and pnpm 10 (for the UI)

### Required environment variables

Copy `.env.example` to `.env`. Do not commit real keys.

| Variable | Role |
| --- | --- |
| `OPENAI_API_KEY` | Hosted chat model. Required for a real plan. Without it, set `TRIP_PLANNER_FAKE_LLM=1` to run the same graph offline. Do not set that flag on Railway. |
| `TAVILY_API_KEY` | Web search for gaps the MCP servers do not cover. |
| `GOOGLE_MAPS_API_KEY` | Google Maps Grounding Lite for restaurants and bars. `GOOGLE_MAPS_KEY` is an accepted alias. |

The process can boot and serve `GET /health` without these keys. A live trip plan needs all three.

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
| Hotels | [Skiplagged](https://mcp.skiplagged.com/mcp) | none (lastminute.com is not called) |
| Rental cars | [OctoTrip](https://mcp.octotrip.app/rental-cars/mcp) | [Upfront Rentals](https://www.upfront-rentals.com/MCP/Streamable) |
| Restaurants and bars | [Google Maps Grounding Lite](https://mapstools.googleapis.com/mcp) | none |

Kiwi, Skiplagged, OctoTrip, and Upfront Rentals are called with no API key. OctoTrip blocks the default HTTP client user agent, so the car agent sends `User-Agent: Mozilla/5.0 (compatible; trip-planner/1.0)`. Google Maps is added only when `GOOGLE_MAPS_KEY` or `GOOGLE_MAPS_API_KEY` is set, and the header is `X-Goog-Api-Key`. If that key is missing, the restaurant and bar agents skip Maps and say so.

lastminute.com documents a streamable-HTTP server at `https://mcp.lastminute.com/mcp`. From this environment every probe (browser and client user agents, `Accept: application/json` and `Accept: application/json, text/event-stream`, GET and POST, `/mcp`, `/sse`, and `/`) returns HTTP 403 `cf-mitigated: challenge` HTML, with no `mcp-session-id`. The MCP streamable-HTTP client maps that non-JSON error to JSON-RPC `-32603` (`Server returned an error response`). Headers, Accept, transport, path, and session handling do not get past the Cloudflare challenge, so the hotel agent does not call lastminute.com and does not retry it. The process logs one warning.

Tavily (`langchain-tavily`) is a general web search tool for gaps: events, attractions, advisories, hours, and fees. It is not a substitute for live MCP prices. Without `TAVILY_API_KEY` the tool returns an error payload instead of raising.

If the car server errors or times out, the car agent uses Upfront Rentals. If a source fails, the specialist says it is unavailable and does not crash the run.

`langchain-mcp-adapters` is archived. Tool loading uses `langchain.mcp.MCPAdapter` (`list_tools` on one server), which replaces `MultiServerMCPClient.get_tools(server_name=...)`.

## Deploy to Railway

Two services from this repo. Do not deploy `langgraph up` for this first cut. That command builds the licensed Agent Server.

### Backend service

- Root directory: repository root
- Config: `Dockerfile`, `railway.toml`, and `Procfile`
- Build: the Dockerfile installs the package and runs `scripts/start-backend.sh`
- Start command: `sh scripts/start-backend.sh`
- Bind: `langgraph dev --host 0.0.0.0 --port $PORT` (`PORT` defaults to 2024)
- Health check: `GET /health`
- Variables: `OPENAI_API_KEY`, `TAVILY_API_KEY`, `GOOGLE_MAPS_API_KEY`, and `PORT` (Railway sets `PORT`). Also set `CORS_ALLOW_ORIGINS` to the UI origin (below). Optional: `OPENAI_MODEL`, `TRIP_PLANNER_MODEL`, `GOOGLE_MAPS_KEY`, `MCP_TIMEOUT_SECONDS`. Leave `TRIP_PLANNER_FAKE_LLM` unset.

Generate a public domain. The frontend example assumes this service is named `trip-planner`.

The browser calls this API from the UI's origin. LangGraph reads `CORS_ALLOW_ORIGINS` (comma-separated). Its own default is `*`. The start script does not use that: when the variable is unset it allows `http://localhost:3000` and `http://127.0.0.1:3000` only. On Railway set the UI's public origin, for example `https://trip-planner-ui.up.railway.app`.

Every MCP server in this project is remote streamable HTTP. The backend image does not install Node, and it does not spawn `npx` or stdio MCP processes. A server that is down is reported as unavailable.

### UI service

- Root directory: `ui`
- Config: `ui/Dockerfile` and `ui/railway.toml` (same shape as the hike-ui service: Dockerfile builder, health check `/health`, no start-command override)
- Build command: `pnpm build` inside the image. `NEXT_PUBLIC_API_URL` and `NEXT_PUBLIC_ASSISTANT_ID` are build args.
- Start command: `node server.js` (the image CMD). Locally, after `pnpm build`, `pnpm start` is `next start -H 0.0.0.0 -p $PORT`.
- Bind: `0.0.0.0` and `$PORT` (`PORT` defaults to 3000)
- Health check: `GET /health`
- Backend URL: `NEXT_PUBLIC_API_URL`. The UI reads that name (see `ui/next.config.mjs` and `ui/src/providers/Stream.tsx`). If it is unset at build time, the default is `http://localhost:2024`.

`NEXT_PUBLIC_*` is inlined by `next build`. Set it on the UI service before the Docker build, then redeploy when it changes. A new value in the running container does nothing until the next build.

```
NEXT_PUBLIC_API_URL=https://${{trip-planner.RAILWAY_PUBLIC_DOMAIN}}
NEXT_PUBLIC_ASSISTANT_ID=trip_planner
```

Copy `ui/.env.example` to `ui/.env` for local `pnpm dev`. Do not put API keys in it.

### Production upgrade

This first deploy uses `langgraph dev` (in-memory, dev server). For production move to `langgraph build` / `langgraph up` or the LangGraph platform image with Postgres and Redis. `langgraph up` needs `LANGGRAPH_CLOUD_LICENSE_KEY`. Threads from `langgraph dev` disappear when the container is replaced.

## Layout

- `agent.py` exports the `trip_planner` graph
- `http_app.py` is the `/health` route
- `langgraph.json` points the LangGraph server at the graph
- `ui/` is the vendored Agent Chat UI (see `ui/VENDOR.md`)
