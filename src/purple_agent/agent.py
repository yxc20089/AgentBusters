"""
Finance Analysis Agent - Main Agent Class

Provides a high-level interface for the Purple Agent that combines
the A2A protocol implementation with finance analysis capabilities.

Uses in-process MCP servers for controlled competition environment.
"""

import asyncio
from datetime import datetime
from typing import Any

from purple_agent.card import get_agent_card
from purple_agent.executor import FinanceAgentExecutor
from purple_agent.mcp_toolkit import MCPToolkit


class FinanceAnalysisAgent:
    """
    Finance Analysis Agent for the AgentBeats competition.

    This agent implements the A2A protocol and provides comprehensive
    financial analysis capabilities including:
    - Earnings analysis (beat/miss determination)
    - SEC filing analysis
    - Financial ratio calculations
    - Investment recommendations
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8101,
        llm_client: Any = None,
        model: str = "gpt-4o",
        simulation_date: datetime | None = None,
    ):
        """
        Initialize the Finance Analysis Agent.

        Args:
            host: Hostname for the A2A server
            port: Port for the A2A server
            llm_client: LLM client (OpenAI or Anthropic)
            model: Model identifier
            simulation_date: Optional date for temporal locking
        """
        self.host = host
        self.port = port
        self.llm_client = llm_client
        self.model = model
        self.simulation_date = simulation_date

        # Initialize components
        self.card = get_agent_card(host, port)
        self.executor = FinanceAgentExecutor(
            llm_client=llm_client,
            model=model,
            simulation_date=simulation_date,
        )

        # Always use in-process MCP servers (FastMCP) for controlled environment
        self.toolkit = MCPToolkit(simulation_date=simulation_date)

    async def analyze(self, question: str, ticker: str | None = None) -> str:
        """
        Perform financial analysis on a question.

        This is a simplified interface for direct analysis without
        going through the A2A protocol.

        Args:
            question: The analysis question
            ticker: Optional ticker to focus on

        Returns:
            Analysis response string
        """
        # Parse task info (LLM-based classification)
        task_info = await self.executor._parse_task(question)

        # Override ticker if provided
        if ticker and ticker not in task_info["tickers"]:
            task_info["tickers"] = [ticker] + task_info["tickers"]

        # Gather data
        financial_data = await self.executor._gather_data(task_info)

        # Generate analysis
        analysis = await self.executor._generate_analysis(
            user_input=question,
            task_info=task_info,
            financial_data=financial_data,
        )

        return analysis

    async def get_stock_data(self, ticker: str) -> dict[str, Any]:
        """
        Get comprehensive stock data.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with stock info, financials, and filings
        """
        return await self.toolkit.get_comprehensive_analysis(ticker)

    async def check_earnings_beat(
        self,
        ticker: str,
        actual_revenue: float | None = None,
        actual_eps: float | None = None,
        expected_revenue: float | None = None,
        expected_eps: float | None = None,
    ) -> dict[str, Any]:
        """
        Check if earnings beat or missed expectations.

        Args:
            ticker: Stock ticker symbol
            actual_revenue: Actual reported revenue
            actual_eps: Actual reported EPS
            expected_revenue: Analyst consensus revenue estimate
            expected_eps: Analyst consensus EPS estimate

        Returns:
            Dictionary with beat/miss analysis
        """
        result = {
            "ticker": ticker,
            "revenue_beat": None,
            "eps_beat": None,
            "overall_assessment": None,
        }

        if actual_revenue is not None and expected_revenue is not None:
            revenue_diff = actual_revenue - expected_revenue
            revenue_pct = (revenue_diff / expected_revenue) * 100
            result["revenue_beat"] = {
                "actual": actual_revenue,
                "expected": expected_revenue,
                "difference": revenue_diff,
                "difference_pct": revenue_pct,
                "beat": actual_revenue > expected_revenue,
            }

        if actual_eps is not None and expected_eps is not None:
            eps_diff = actual_eps - expected_eps
            eps_pct = (eps_diff / expected_eps) * 100 if expected_eps != 0 else 0
            result["eps_beat"] = {
                "actual": actual_eps,
                "expected": expected_eps,
                "difference": eps_diff,
                "difference_pct": eps_pct,
                "beat": actual_eps > expected_eps,
            }

        # Overall assessment
        revenue_beat = result["revenue_beat"]["beat"] if result["revenue_beat"] else None
        eps_beat = result["eps_beat"]["beat"] if result["eps_beat"] else None

        if revenue_beat is True and eps_beat is True:
            result["overall_assessment"] = "Beat"
        elif revenue_beat is False and eps_beat is False:
            result["overall_assessment"] = "Miss"
        elif revenue_beat is True or eps_beat is True:
            result["overall_assessment"] = "Mixed"
        else:
            result["overall_assessment"] = "Insufficient data"

        return result

    def get_card(self) -> dict[str, Any]:
        """
        Get the Agent Card as a dictionary.

        Returns:
            Agent Card data for A2A protocol
        """
        return self.card.model_dump(exclude_none=True)


async def create_agent(
    host: str = "localhost",
    port: int = 8101,
    openai_api_key: str | None = None,
    anthropic_api_key: str | None = None,
    model: str | None = None,
    simulation_date: datetime | None = None,
) -> FinanceAnalysisAgent:
    """
    Factory function to create a Finance Analysis Agent.

    Args:
        host: Hostname for A2A server
        port: Port for A2A server
        openai_api_key: OpenAI API key (optional)
        anthropic_api_key: Anthropic API key (optional)
        model: Model to use (defaults based on available API key)
        simulation_date: Optional date for temporal locking

    Returns:
        Configured FinanceAnalysisAgent instance
    """
    llm_client = None
    default_model = model

    # Try to initialize LLM client
    if openai_api_key:
        try:
            from openai import OpenAI
            import os
            base_url = os.environ.get("OPENAI_API_BASE")
            llm_client = OpenAI(
                api_key=openai_api_key,
                base_url=base_url  # Supports local vLLM
            )
            default_model = model or os.environ.get("LLM_MODEL", "gpt-4o")
        except ImportError:
            pass

    elif anthropic_api_key:
        try:
            from anthropic import Anthropic
            llm_client = Anthropic(api_key=anthropic_api_key)
            default_model = model or "claude-sonnet-4-20250514"
        except ImportError:
            pass

    return FinanceAnalysisAgent(
        host=host,
        port=port,
        llm_client=llm_client,
        model=default_model or "gpt-4o",
        simulation_date=simulation_date,
    )
