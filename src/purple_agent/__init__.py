"""
Purple Agent - Finance Analysis Agent for AgentBeats Competition

This agent implements the A2A (Agent2Agent) protocol to receive
evaluation tasks from Green Agents and provide financial analysis responses.

Uses in-process MCP servers (FastMCP) for controlled competition environment.
"""

from purple_agent.agent import FinanceAnalysisAgent, create_agent
from purple_agent.executor import FinanceAgentExecutor
from purple_agent.card import get_agent_card
from purple_agent.mcp_toolkit import MCPToolkit

__all__ = [
    "FinanceAnalysisAgent",
    "FinanceAgentExecutor",
    "MCPToolkit",
    "create_agent",
    "get_agent_card",
]
