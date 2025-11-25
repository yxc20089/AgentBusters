"""
MCP Servers for AgentBusters

Actual MCP server implementations using FastMCP that provide
financial data to Purple Agents via the Model Context Protocol.

Servers:
- sec_edgar: SEC EDGAR filings and XBRL data
- yahoo_finance: Market data and statistics
- sandbox: Python code execution
"""

from mcp_servers.sec_edgar import create_edgar_server
from mcp_servers.yahoo_finance import create_yahoo_finance_server
from mcp_servers.sandbox import create_sandbox_server

__all__ = [
    "create_edgar_server",
    "create_yahoo_finance_server",
    "create_sandbox_server",
]
