# Vendored Agent Chat UI

Source: https://github.com/langchain-ai/agent-chat-ui
Commit: `0d21f7ff4c45287797e876dc6d8b150b6f277b1e`
License: MIT (see `LICENSE`). Copyright (c) 2025 Brace Sproul.

This snapshot is vendored in the repository instead of a git submodule. Railway
and local Docker builds can then install and build the UI without a second
clone, and the snapshot stays pinned. The upstream repository is archived.

Integration changes on top of that commit:

- `next.config.mjs` defaults `NEXT_PUBLIC_API_URL` to `http://localhost:2024` and `NEXT_PUBLIC_ASSISTANT_ID` to `trip_planner`
- `.env.example` uses the same assistant id
- `src/providers/Stream.tsx` defaults the setup form's assistant id to `trip_planner`
- `src/app/health/route.ts` adds a health check for Railway
- `package.json` `start` binds Next.js to `0.0.0.0` and uses Railway's `PORT`
- `Dockerfile`, `railway.toml`, and `.dockerignore` run the Next.js app on `0.0.0.0:$PORT`
