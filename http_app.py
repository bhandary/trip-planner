"""Extra HTTP routes mounted by the LangGraph server."""

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route


async def health(_request):
    return JSONResponse({"status": "ok"})


app = Starlette(routes=[Route("/health", health, methods=["GET"])])
