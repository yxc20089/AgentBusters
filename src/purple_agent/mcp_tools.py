"""
MCP-based Finance Analysis Tools for Purple Agent

Uses the MCP servers (SEC EDGAR, Yahoo Finance, Sandbox) for real
financial data access with temporal locking and cost tracking.
"""

import asyncio
from datetime import datetime
from typing import Any
from dataclasses import dataclass, field

from mcp_clients.base import MCPConfig
from mcp_clients.edgar import MeteredEDGARClient
from mcp_clients.yahoo_finance import TimeMachineYFinanceClient
from mcp_clients.sandbox import QuantSandboxClient


@dataclass
class MCPFinancialMetrics:
    """Core financial metrics extracted from MCP sources."""

    ticker: str
    period: str
    revenue: float | None = None
    net_income: float | None = None
    gross_profit: float | None = None
    gross_margin: float | None = None
    operating_income: float | None = None
    operating_margin: float | None = None
    eps: float | None = None
    eps_diluted: float | None = None
    total_assets: float | None = None
    total_liabilities: float | None = None
    total_equity: float | None = None
    cash: float | None = None
    debt: float | None = None
    free_cash_flow: float | None = None

    # Market data
    market_cap: float | None = None
    price: float | None = None
    pe_ratio: float | None = None
    pb_ratio: float | None = None

    # Growth metrics
    revenue_growth_yoy: float | None = None
    net_income_growth_yoy: float | None = None

    # Additional context
    extra: dict = field(default_factory=dict)


class MCPFinanceToolkit:
    """
    Finance toolkit using MCP servers for data access.

    This toolkit connects to the MCP servers defined in the architecture:
    - SEC EDGAR MCP: For SEC filings and XBRL data
    - Yahoo Finance MCP: For market data and statistics
    - Sandbox MCP: For code execution
    """

    def __init__(
        self,
        edgar_url: str = "http://localhost:8001",
        yfinance_url: str = "http://localhost:8002",
        sandbox_url: str = "http://localhost:8003",
        simulation_date: datetime | None = None,
    ):
        """
        Initialize the MCP Finance Toolkit.

        Args:
            edgar_url: URL for SEC EDGAR MCP server
            yfinance_url: URL for Yahoo Finance MCP server
            sandbox_url: URL for Sandbox MCP server
            simulation_date: Optional date for temporal locking
        """
        self.simulation_date = simulation_date

        # Initialize MCP clients
        self.edgar = MeteredEDGARClient(
            config=MCPConfig(base_url=edgar_url),
            simulation_date=simulation_date,
            temporal_lock_enabled=True,
        )

        self.yfinance = TimeMachineYFinanceClient(
            config=MCPConfig(base_url=yfinance_url),
            simulation_date=simulation_date,
        )

        self.sandbox = QuantSandboxClient(
            config=MCPConfig(base_url=sandbox_url),
        )

    async def close(self):
        """Close all MCP client connections."""
        await asyncio.gather(
            self.edgar.close(),
            self.yfinance.close(),
            self.sandbox.close(),
        )

    async def get_stock_info(self, ticker: str) -> dict[str, Any]:
        """
        Get basic stock information from Yahoo Finance MCP.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with stock info
        """
        try:
            # Get company info
            info = await self.yfinance.get_company_info(ticker)

            # Get key statistics
            stats = await self.yfinance.get_key_statistics(ticker)

            return {
                "ticker": ticker,
                "name": info.get("name", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": stats.market_cap,
                "price": info.get("current_price"),
                "pe_ratio": stats.pe_ratio,
                "forward_pe": stats.forward_pe,
                "dividend_yield": stats.dividend_yield,
                "52_week_high": stats.fifty_two_week_high,
                "52_week_low": stats.fifty_two_week_low,
            }
        except Exception as e:
            return {"ticker": ticker, "error": str(e)}

    async def get_financials(
        self,
        ticker: str,
        fiscal_year: int | None = None,
    ) -> MCPFinancialMetrics:
        """
        Get financial data from SEC EDGAR MCP (XBRL data).

        Args:
            ticker: Stock ticker symbol
            fiscal_year: Specific fiscal year to retrieve

        Returns:
            MCPFinancialMetrics with financial data
        """
        try:
            # Get income statement data
            income_data = await self.edgar.parse_xbrl_financials(
                ticker=ticker,
                statement_type="IS",
                fiscal_year=fiscal_year,
            )

            # Get balance sheet data
            balance_data = await self.edgar.parse_xbrl_financials(
                ticker=ticker,
                statement_type="BS",
                fiscal_year=fiscal_year,
            )

            # Extract metrics from XBRL data
            is_data = income_data.data
            bs_data = balance_data.data

            revenue = is_data.get("Revenue") or is_data.get("Revenues") or is_data.get("TotalRevenue")
            net_income = is_data.get("NetIncome") or is_data.get("NetIncomeLoss")
            gross_profit = is_data.get("GrossProfit")
            operating_income = is_data.get("OperatingIncome") or is_data.get("OperatingIncomeLoss")
            eps = is_data.get("EarningsPerShareBasic") or is_data.get("EPS")
            eps_diluted = is_data.get("EarningsPerShareDiluted")

            total_assets = bs_data.get("TotalAssets") or bs_data.get("Assets")
            total_liabilities = bs_data.get("TotalLiabilities") or bs_data.get("Liabilities")
            total_equity = bs_data.get("StockholdersEquity") or bs_data.get("TotalEquity")
            cash = bs_data.get("CashAndCashEquivalents") or bs_data.get("Cash")
            debt = bs_data.get("TotalDebt") or bs_data.get("LongTermDebt")

            # Calculate margins
            gross_margin = None
            if revenue and gross_profit:
                gross_margin = gross_profit / revenue

            operating_margin = None
            if revenue and operating_income:
                operating_margin = operating_income / revenue

            return MCPFinancialMetrics(
                ticker=ticker,
                period=income_data.fiscal_period,
                revenue=revenue,
                net_income=net_income,
                gross_profit=gross_profit,
                gross_margin=gross_margin,
                operating_income=operating_income,
                operating_margin=operating_margin,
                eps=eps,
                eps_diluted=eps_diluted,
                total_assets=total_assets,
                total_liabilities=total_liabilities,
                total_equity=total_equity,
                cash=cash,
                debt=debt,
            )

        except Exception as e:
            return MCPFinancialMetrics(
                ticker=ticker,
                period="error",
                extra={"error": str(e)},
            )

    async def get_sec_filing(
        self,
        ticker: str,
        form_type: str,
        fiscal_year: int | None = None,
    ) -> dict[str, Any]:
        """
        Get SEC filing from EDGAR MCP.

        Args:
            ticker: Stock ticker symbol
            form_type: Filing type (10-K, 10-Q, 8-K)
            fiscal_year: Specific fiscal year

        Returns:
            Filing data dictionary
        """
        try:
            filing = await self.edgar.get_filing(
                ticker=ticker,
                form_type=form_type,
                fiscal_year=fiscal_year,
            )
            return filing
        except Exception as e:
            return {"ticker": ticker, "form_type": form_type, "error": str(e)}

    async def get_filing_section(
        self,
        ticker: str,
        form_type: str,
        section_name: str,
        fiscal_year: int | None = None,
    ) -> dict[str, Any]:
        """
        Get a specific section from an SEC filing.

        Args:
            ticker: Stock ticker symbol
            form_type: Filing type
            section_name: Section to extract (e.g., "Item 1A", "MD&A")
            fiscal_year: Specific fiscal year

        Returns:
            Section content
        """
        try:
            section = await self.edgar.get_filing_section(
                ticker=ticker,
                form_type=form_type,
                section_name=section_name,
                fiscal_year=fiscal_year,
            )
            return {
                "ticker": section.ticker,
                "form_type": section.form_type,
                "section": section.section_name,
                "content": section.content,
                "filing_date": section.filing_date,
            }
        except Exception as e:
            return {"ticker": ticker, "error": str(e)}

    async def get_historical_prices(
        self,
        ticker: str,
        period: str = "1y",
    ) -> list[dict]:
        """
        Get historical price data from Yahoo Finance MCP.

        Args:
            ticker: Stock ticker symbol
            period: Time period (1mo, 3mo, 6mo, 1y, 2y, 5y)

        Returns:
            List of price data dictionaries
        """
        try:
            history = await self.yfinance.get_historical(
                ticker=ticker,
                period=period,
            )
            return history.data
        except Exception as e:
            return [{"error": str(e)}]

    async def execute_code(
        self,
        code: str,
        timeout: int = 30,
    ) -> dict[str, Any]:
        """
        Execute Python code in the sandbox MCP.

        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds

        Returns:
            Execution result with output and any errors
        """
        try:
            result = await self.sandbox.execute_code(
                code=code,
                timeout_seconds=timeout,
            )
            return {
                "success": result.success,
                "output": result.stdout,
                "error": result.stderr or result.error_message,
                "return_value": result.return_value,
                "execution_time": result.execution_time_ms,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_comprehensive_analysis(self, ticker: str) -> dict[str, Any]:
        """
        Get comprehensive financial analysis combining all MCP sources.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with comprehensive financial data
        """
        # Fetch data from multiple MCP sources concurrently
        stock_info_task = self.get_stock_info(ticker)
        financials_task = self.get_financials(ticker)
        filing_task = self.get_sec_filing(ticker, "10-K")

        stock_info, financials, filing = await asyncio.gather(
            stock_info_task,
            financials_task,
            filing_task,
            return_exceptions=True,
        )

        # Handle exceptions
        if isinstance(stock_info, Exception):
            stock_info = {"error": str(stock_info)}
        if isinstance(financials, Exception):
            financials = MCPFinancialMetrics(ticker=ticker, period="error")
        if isinstance(filing, Exception):
            filing = {"error": str(filing)}

        return {
            "ticker": ticker,
            "simulation_date": self.simulation_date.isoformat() if self.simulation_date else None,
            "stock_info": stock_info,
            "financials": {
                "revenue": financials.revenue,
                "net_income": financials.net_income,
                "gross_margin": financials.gross_margin,
                "operating_margin": financials.operating_margin,
                "eps": financials.eps,
                "eps_diluted": financials.eps_diluted,
                "period": financials.period,
            },
            "balance_sheet": {
                "total_assets": financials.total_assets,
                "total_liabilities": financials.total_liabilities,
                "total_equity": financials.total_equity,
                "cash": financials.cash,
                "debt": financials.debt,
            },
            "recent_filing": filing,
            "mcp_metrics": self.get_metrics(),
        }

    def get_metrics(self) -> dict[str, Any]:
        """Get metrics from all MCP clients."""
        return {
            "edgar": self.edgar.get_metrics(),
            "yfinance": self.yfinance.get_metrics(),
            "sandbox": self.sandbox.get_metrics(),
            "temporal_violations": len(self.edgar.temporal_violations),
            "lookahead_penalty": self.edgar.calculate_lookahead_penalty(),
        }

    def get_tool_calls(self) -> list:
        """Get all tool calls from MCP clients."""
        calls = []
        calls.extend(self.edgar.call_log)
        calls.extend(self.yfinance.call_log)
        calls.extend(self.sandbox.call_log)
        return calls
