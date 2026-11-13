from typing import Any

import uvicorn
from mcp.server import MCPServer
from mcp.server.transport_security import TransportSecuritySettings

from support_agent import tools

HOST = "0.0.0.0"
PORT = 8000
SECURITY = TransportSecuritySettings(
    allowed_hosts=["127.0.0.1:*", "localhost:*"],
    allowed_origins=["http://127.0.0.1:*", "http://localhost:*"],
)

mcp = MCPServer("support-tools")


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
    return await tools.issue_refund(customer_id, order_id, reason)


def http_app():
    return mcp.streamable_http_app(
        json_response=True, transport_security=SECURITY, host=HOST
    )


if __name__ == "__main__":
    uvicorn.run(http_app(), host=HOST, port=PORT)
