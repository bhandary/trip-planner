# Next.js chat UI. NEXT_PUBLIC_* is inlined by `next build`, so pass the
# backend URL as a build arg. On Railway, set the service variable before
# the build:
#   NEXT_PUBLIC_API_URL=https://${{trip-planner.RAILWAY_PUBLIC_DOMAIN}}
#   NEXT_PUBLIC_ASSISTANT_ID=trip_planner
# The runner listens on 0.0.0.0:$PORT (node server.js).

FROM node:22-bookworm-slim AS build

WORKDIR /app
RUN corepack enable && corepack prepare pnpm@10.5.1 --activate

COPY package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile

COPY . .

ARG NEXT_PUBLIC_API_URL=http://localhost:2024
ARG NEXT_PUBLIC_ASSISTANT_ID=trip_planner
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL \
    NEXT_PUBLIC_ASSISTANT_ID=$NEXT_PUBLIC_ASSISTANT_ID \
    NEXT_TELEMETRY_DISABLED=1

RUN pnpm build

FROM node:22-bookworm-slim AS runner

WORKDIR /app
ENV NODE_ENV=production \
    HOSTNAME=0.0.0.0 \
    PORT=3000 \
    NEXT_TELEMETRY_DISABLED=1

COPY --from=build /app/public ./public
COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static

EXPOSE 3000

CMD ["node", "server.js"]
