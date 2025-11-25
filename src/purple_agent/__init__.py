"""
Purple Agent - Finance Analysis Agent for AgentBeats Competition

This agent implements the A2A (Agent2Agent) protocol to receive
evaluation tasks from Green Agents and provide financial analysis responses.
"""

from purple_agent.agent import FinanceAnalysisAgent
from purple_agent.executor import FinanceAgentExecutor
from purple_agent.card import get_agent_card

__all__ = [
    "FinanceAnalysisAgent",
    "FinanceAgentExecutor",
    "get_agent_card",
]
