import asyncio
import os
from typing import Any

import jwt
import uvicorn
from mcp.server import MCPServer
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.routes import build_resource_metadata_url
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from starlette.datastructures import Headers
from starlette.middleware import Middleware
from starlette.responses import JSONResponse

from support_agent import tools
from support_agent.auth import ISSUER, JWKS_URL, Verifier

HOST = "0.0.0.0"
PORT = 8000
RESOURCE_URL = "http://127.0.0.1:8000/mcp"
READ_SCOPE = "orders:read"
WRITE_SCOPE = "refunds:write"
TOOL_SCOPES = {"issue_refund": WRITE_SCOPE}
METADATA_URL = str(build_resource_metadata_url(RESOURCE_URL))
SECURITY = TransportSecuritySettings(
    allowed_hosts=["127.0.0.1:*", "localhost:*"],
    allowed_origins=["http://127.0.0.1:*", "http://localhost:*"],
)

verifier = Verifier(os.environ.get("OAUTH_JWKS_URL", JWKS_URL), RESOURCE_URL)


class TokenVerifier:
    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            claims = await asyncio.to_thread(verifier.verify, token)
        except jwt.PyJWTError:
            return None
        return AccessToken(
            token=token,
            client_id=claims.get("azp", ""),
            scopes=claims.get("scope", "").split(),
            expires_at=claims["exp"],
            resource=RESOURCE_URL,
            subject=claims["sub"],
        )


mcp = MCPServer(
    "support-tools",
    token_verifier=TokenVerifier(),
    auth=AuthSettings(
        issuer_url=ISSUER,
        resource_server_url=RESOURCE_URL,
        required_scopes=[READ_SCOPE],
        validate_token_resource=False,
    ),
)


def require_scope(scope: str) -> None:
    token = get_access_token()
    if token is None or scope not in token.scopes:
        raise ToolError(f"Token lacks the {scope} scope.")


@mcp.tool()
async def get_order(customer_id: int, order_id: int) -> dict[str, Any] | None:
    """Look up one of the customer's orders by its order number."""
    return await tools.get_order(customer_id, order_id)


@mcp.tool()
async def list_orders(customer_id: int) -> list[dict[str, Any]]:
    """List all of the customer's orders, newest first."""
    return await tools.list_orders(customer_id)


@mcp.tool()
async def issue_refund(customer_id: int, order_id: int, reason: str) -> dict[str, Any]:
    """Refund the full total of one of the customer's delivered orders."""
    require_scope(WRITE_SCOPE)
    return await tools.issue_refund(customer_id, order_id, reason)


class ToolScopeGate:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        needed = TOOL_SCOPES.get(headers.get("mcp-name", ""))
        user = scope.get("user")
        if (
            needed
            and headers.get("mcp-method") == "tools/call"
            and user is not None
            and user.is_authenticated
            and needed not in scope["auth"].scopes
        ):
            challenge = (
                f'Bearer error="insufficient_scope", scope="{needed}", '
                f'resource_metadata="{METADATA_URL}"'
            )
            response = JSONResponse(
                {"error": "insufficient_scope", "error_description": f"Needs {needed}"},
                status_code=403,
                headers={"WWW-Authenticate": challenge},
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


def http_app():
    app = mcp.streamable_http_app(
        json_response=True, transport_security=SECURITY, host=HOST
    )
    app.user_middleware.append(Middleware(ToolScopeGate))
    return app


if __name__ == "__main__":
    uvicorn.run(http_app(), host=HOST, port=PORT)
