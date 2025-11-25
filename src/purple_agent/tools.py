"""
Finance Analysis Tools for Purple Agent

Provides tools for retrieving and analyzing financial data from
SEC EDGAR, Yahoo Finance, and other data sources.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any
from dataclasses import dataclass, field
import yfinance as yf
from pydantic import BaseModel


@dataclass
class FinancialMetrics:
    """Core financial metrics extracted from data sources."""

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


class YahooFinanceTool:
    """
    Tool for retrieving financial data from Yahoo Finance.

    Provides access to stock prices, financial statements,
    and market data with simulation date support for temporal locking.
    """

    def __init__(self, simulation_date: datetime | None = None):
        """
        Initialize the Yahoo Finance tool.

        Args:
            simulation_date: Optional date for temporal locking.
                            If set, data after this date will be filtered.
        """
        self.simulation_date = simulation_date or datetime.now()

    async def get_stock_info(self, ticker: str) -> dict[str, Any]:
        """
        Get basic stock information.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with stock info (name, sector, industry, etc.)
        """
        def _fetch():
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                "ticker": ticker,
                "name": info.get("longName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap"),
                "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "pb_ratio": info.get("priceToBook"),
                "dividend_yield": info.get("dividendYield"),
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "52_week_low": info.get("fiftyTwoWeekLow"),
                "analyst_rating": info.get("recommendationKey"),
                "target_price": info.get("targetMeanPrice"),
            }

        return await asyncio.to_thread(_fetch)

    async def get_financials(
        self,
        ticker: str,
        period: str = "quarterly"
    ) -> FinancialMetrics:
        """
        Get financial statements data.

        Args:
            ticker: Stock ticker symbol
            period: "quarterly" or "annual"

        Returns:
            FinancialMetrics with income statement, balance sheet data
        """
        def _fetch():
            stock = yf.Ticker(ticker)

            # Get financial statements
            if period == "quarterly":
                income_stmt = stock.quarterly_income_stmt
                balance_sheet = stock.quarterly_balance_sheet
                cash_flow = stock.quarterly_cashflow
            else:
                income_stmt = stock.income_stmt
                balance_sheet = stock.balance_sheet
                cash_flow = stock.cashflow

            # Get the most recent period
            if income_stmt.empty:
                return FinancialMetrics(ticker=ticker, period=period)

            latest_col = income_stmt.columns[0]
            period_str = latest_col.strftime("%Y-%m-%d") if hasattr(latest_col, "strftime") else str(latest_col)

            def safe_get(df, key):
                """Safely get a value from dataframe."""
                if df.empty or key not in df.index:
                    return None
                val = df.loc[key, latest_col] if latest_col in df.columns else None
                return float(val) if val is not None and not (isinstance(val, float) and val != val) else None

            revenue = safe_get(income_stmt, "Total Revenue")
            net_income = safe_get(income_stmt, "Net Income")
            gross_profit = safe_get(income_stmt, "Gross Profit")
            operating_income = safe_get(income_stmt, "Operating Income")

            # Calculate margins
            gross_margin = None
            if revenue and gross_profit:
                gross_margin = gross_profit / revenue

            operating_margin = None
            if revenue and operating_income:
                operating_margin = operating_income / revenue

            # Balance sheet items
            total_assets = safe_get(balance_sheet, "Total Assets")
            total_liabilities = safe_get(balance_sheet, "Total Liabilities Net Minority Interest")
            total_equity = safe_get(balance_sheet, "Total Equity Gross Minority Interest")
            cash = safe_get(balance_sheet, "Cash And Cash Equivalents")
            debt = safe_get(balance_sheet, "Total Debt")

            # Cash flow
            free_cash_flow = safe_get(cash_flow, "Free Cash Flow")

            # EPS
            eps = safe_get(income_stmt, "Basic EPS")
            eps_diluted = safe_get(income_stmt, "Diluted EPS")

            # YoY growth (compare to same quarter last year if available)
            revenue_growth = None
            net_income_growth = None
            if len(income_stmt.columns) >= 5:  # At least 5 quarters of data
                prev_col = income_stmt.columns[4]  # Same quarter last year
                prev_revenue = safe_get(income_stmt.loc[:, [prev_col]], "Total Revenue") if "Total Revenue" in income_stmt.index else None
                prev_net_income = safe_get(income_stmt.loc[:, [prev_col]], "Net Income") if "Net Income" in income_stmt.index else None

                if prev_revenue and revenue:
                    revenue_growth = (revenue - prev_revenue) / abs(prev_revenue)
                if prev_net_income and net_income:
                    net_income_growth = (net_income - prev_net_income) / abs(prev_net_income)

            return FinancialMetrics(
                ticker=ticker,
                period=period_str,
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
                free_cash_flow=free_cash_flow,
                revenue_growth_yoy=revenue_growth,
                net_income_growth_yoy=net_income_growth,
            )

        return await asyncio.to_thread(_fetch)

    async def get_historical_prices(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> list[dict]:
        """
        Get historical price data.

        Args:
            ticker: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
            interval: Data interval (1m, 5m, 15m, 1h, 1d, 1wk, 1mo)

        Returns:
            List of price data dictionaries
        """
        def _fetch():
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval=interval)

            # Filter by simulation date (handle timezone-aware index)
            if self.simulation_date:
                # Convert simulation date to timezone-aware if index is tz-aware
                if hist.index.tz is not None:
                    import pandas as pd
                    sim_date_aware = pd.Timestamp(self.simulation_date).tz_localize(hist.index.tz)
                    hist = hist[hist.index <= sim_date_aware]
                else:
                    hist = hist[hist.index <= self.simulation_date]

            return [
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "open": row["Open"],
                    "high": row["High"],
                    "low": row["Low"],
                    "close": row["Close"],
                    "volume": row["Volume"],
                }
                for idx, row in hist.iterrows()
            ]

        return await asyncio.to_thread(_fetch)

    async def get_analyst_recommendations(self, ticker: str) -> list[dict]:
        """
        Get analyst recommendations.

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of analyst recommendation dictionaries
        """
        def _fetch():
            stock = yf.Ticker(ticker)
            recs = stock.recommendations

            if recs is None or recs.empty:
                return []

            return [
                {
                    "date": idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx),
                    "firm": row.get("Firm", ""),
                    "grade": row.get("To Grade", row.get("toGrade", "")),
                    "action": row.get("Action", row.get("action", "")),
                }
                for idx, row in recs.head(10).iterrows()
            ]

        return await asyncio.to_thread(_fetch)


class SECEdgarTool:
    """
    Tool for retrieving SEC EDGAR filings.

    Provides access to 10-K, 10-Q, 8-K, and other SEC filings
    with simulation date support for temporal locking.
    """

    def __init__(self, simulation_date: datetime | None = None):
        """
        Initialize the SEC EDGAR tool.

        Args:
            simulation_date: Optional date for temporal locking.
        """
        self.simulation_date = simulation_date or datetime.now()
        self._company_cache: dict[str, Any] = {}

    async def get_company_info(self, ticker: str) -> dict[str, Any]:
        """
        Get company information from SEC EDGAR.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with company info
        """
        try:
            from edgartools import Company

            def _fetch():
                company = Company(ticker)
                return {
                    "cik": company.cik,
                    "name": company.name,
                    "ticker": ticker,
                    "sic": getattr(company, "sic", None),
                    "sic_description": getattr(company, "sic_description", None),
                    "fiscal_year_end": getattr(company, "fiscal_year_end", None),
                }

            return await asyncio.to_thread(_fetch)
        except ImportError:
            return {"error": "edgartools not installed", "ticker": ticker}
        except Exception as e:
            return {"error": str(e), "ticker": ticker}

    async def get_recent_filings(
        self,
        ticker: str,
        form_type: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """
        Get recent SEC filings for a company.

        Args:
            ticker: Stock ticker symbol
            form_type: Filter by form type (10-K, 10-Q, 8-K, etc.)
            limit: Maximum number of filings to return

        Returns:
            List of filing metadata dictionaries
        """
        try:
            from edgartools import Company

            def _fetch():
                company = Company(ticker)
                filings = company.get_filings(form=form_type) if form_type else company.get_filings()

                results = []
                for filing in filings[:limit]:
                    filing_date = filing.filing_date

                    # Apply temporal locking
                    if self.simulation_date and filing_date > self.simulation_date.date():
                        continue

                    results.append({
                        "form": filing.form,
                        "filing_date": str(filing_date),
                        "accession_number": filing.accession_number,
                        "description": getattr(filing, "description", ""),
                    })

                return results[:limit]

            return await asyncio.to_thread(_fetch)
        except ImportError:
            return [{"error": "edgartools not installed"}]
        except Exception as e:
            return [{"error": str(e)}]

    async def get_10k_summary(self, ticker: str, fiscal_year: int | None = None) -> dict:
        """
        Get summary of 10-K filing.

        Args:
            ticker: Stock ticker symbol
            fiscal_year: Specific fiscal year to retrieve

        Returns:
            Dictionary with 10-K summary data
        """
        try:
            from edgartools import Company

            def _fetch():
                company = Company(ticker)
                filings = company.get_filings(form="10-K")

                # Find the appropriate filing
                for filing in filings:
                    filing_date = filing.filing_date

                    # Apply temporal locking
                    if self.simulation_date and filing_date > self.simulation_date.date():
                        continue

                    # If fiscal year specified, try to match
                    if fiscal_year:
                        # 10-K is typically filed 60-90 days after fiscal year end
                        if filing_date.year not in [fiscal_year, fiscal_year + 1]:
                            continue

                    return {
                        "form": "10-K",
                        "ticker": ticker,
                        "filing_date": str(filing_date),
                        "fiscal_year": fiscal_year or filing_date.year - 1,
                        "accession_number": filing.accession_number,
                        # Note: Full text extraction would require additional processing
                    }

                return {"error": "No matching 10-K found", "ticker": ticker}

            return await asyncio.to_thread(_fetch)
        except ImportError:
            return {"error": "edgartools not installed", "ticker": ticker}
        except Exception as e:
            return {"error": str(e), "ticker": ticker}

    async def get_10q_summary(
        self,
        ticker: str,
        fiscal_year: int | None = None,
        quarter: int | None = None,
    ) -> dict:
        """
        Get summary of 10-Q filing.

        Args:
            ticker: Stock ticker symbol
            fiscal_year: Specific fiscal year
            quarter: Specific quarter (1, 2, or 3)

        Returns:
            Dictionary with 10-Q summary data
        """
        try:
            from edgartools import Company

            def _fetch():
                company = Company(ticker)
                filings = company.get_filings(form="10-Q")

                for filing in filings:
                    filing_date = filing.filing_date

                    # Apply temporal locking
                    if self.simulation_date and filing_date > self.simulation_date.date():
                        continue

                    return {
                        "form": "10-Q",
                        "ticker": ticker,
                        "filing_date": str(filing_date),
                        "accession_number": filing.accession_number,
                    }

                return {"error": "No matching 10-Q found", "ticker": ticker}

            return await asyncio.to_thread(_fetch)
        except ImportError:
            return {"error": "edgartools not installed", "ticker": ticker}
        except Exception as e:
            return {"error": str(e), "ticker": ticker}


class FinanceToolkit:
    """
    Unified toolkit combining all finance analysis tools.
    """

    def __init__(self, simulation_date: datetime | None = None):
        """
        Initialize the finance toolkit.

        Args:
            simulation_date: Optional date for temporal locking across all tools.
        """
        self.simulation_date = simulation_date
        self.yahoo = YahooFinanceTool(simulation_date)
        self.edgar = SECEdgarTool(simulation_date)

    async def get_comprehensive_analysis(self, ticker: str) -> dict[str, Any]:
        """
        Get comprehensive financial analysis for a ticker.

        Combines data from multiple sources into a unified analysis.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with comprehensive financial data
        """
        # Fetch data from multiple sources concurrently
        stock_info, financials, filings = await asyncio.gather(
            self.yahoo.get_stock_info(ticker),
            self.yahoo.get_financials(ticker, "quarterly"),
            self.edgar.get_recent_filings(ticker, limit=5),
            return_exceptions=True,
        )

        # Handle exceptions
        if isinstance(stock_info, Exception):
            stock_info = {"error": str(stock_info)}
        if isinstance(financials, Exception):
            financials = FinancialMetrics(ticker=ticker, period="error")
        if isinstance(filings, Exception):
            filings = [{"error": str(filings)}]

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
                "revenue_growth_yoy": financials.revenue_growth_yoy,
                "net_income_growth_yoy": financials.net_income_growth_yoy,
                "period": financials.period,
            },
            "balance_sheet": {
                "total_assets": financials.total_assets,
                "total_liabilities": financials.total_liabilities,
                "total_equity": financials.total_equity,
                "cash": financials.cash,
                "debt": financials.debt,
            },
            "recent_filings": filings,
        }
