"""
Yahoo Finance MCP Server

A FastMCP server that provides access to market data and financial statistics.
Implements temporal locking (Time Machine) to prevent look-ahead bias.
"""

import os
from datetime import datetime
from typing import Any

from fastmcp import FastMCP
from pydantic import BaseModel, Field
import yfinance as yf


class StockQuote(BaseModel):
    """Current stock quote data."""
    ticker: str
    name: str = ""
    sector: str = ""
    industry: str = ""
    current_price: float | None = None
    market_cap: float | None = None
    pe_ratio: float | None = None
    forward_pe: float | None = None
    dividend_yield: float | None = None
    fifty_two_week_high: float | None = None
    fifty_two_week_low: float | None = None
    analyst_rating: str = ""


class FinancialStatements(BaseModel):
    """Financial statement data."""
    ticker: str
    period: str
    statement_type: str
    data: dict[str, float | None] = Field(default_factory=dict)


class HistoricalPrice(BaseModel):
    """Historical price data point."""
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    adj_close: float | None = None


def create_yahoo_finance_server(
    simulation_date: datetime | None = None,
    name: str = "yahoo-finance-mcp",
) -> FastMCP:
    """
    Create the Yahoo Finance MCP server.

    Args:
        simulation_date: Optional date for temporal locking (Time Machine)
        name: Server name

    Returns:
        Configured FastMCP server
    """
    mcp = FastMCP(name)

    _simulation_date = simulation_date

    def filter_by_simulation_date(df, date_column=None):
        """Filter dataframe to only include data up to simulation date."""
        if _simulation_date is None or df.empty:
            return df

        if date_column:
            return df[df[date_column] <= _simulation_date]
        elif hasattr(df.index, 'tz'):
            # Handle timezone-aware index
            import pandas as pd
            sim_date_aware = pd.Timestamp(_simulation_date)
            if df.index.tz is not None:
                sim_date_aware = sim_date_aware.tz_localize(df.index.tz)
            return df[df.index <= sim_date_aware]
        else:
            return df[df.index <= _simulation_date]

    @mcp.tool
    def get_quote(ticker: str) -> dict[str, Any]:
        """
        Get current stock quote and basic info.

        Args:
            ticker: Stock ticker symbol (e.g., "NVDA", "AAPL", "MSFT")

        Returns:
            Stock quote with price, market cap, ratios, and company info
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            return StockQuote(
                ticker=ticker,
                name=info.get("longName", ""),
                sector=info.get("sector", ""),
                industry=info.get("industry", ""),
                current_price=info.get("currentPrice") or info.get("regularMarketPrice"),
                market_cap=info.get("marketCap"),
                pe_ratio=info.get("trailingPE"),
                forward_pe=info.get("forwardPE"),
                dividend_yield=info.get("dividendYield"),
                fifty_two_week_high=info.get("fiftyTwoWeekHigh"),
                fifty_two_week_low=info.get("fiftyTwoWeekLow"),
                analyst_rating=info.get("recommendationKey", ""),
            ).model_dump()

        except Exception as e:
            return {"error": str(e), "ticker": ticker}

    @mcp.tool
    def get_historical_prices(
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> list[dict[str, Any]]:
        """
        Get historical price data.

        Args:
            ticker: Stock ticker symbol
            period: Time period - "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"
            interval: Data interval - "1d", "1wk", "1mo"

        Returns:
            List of historical price data points (OHLCV)
        """
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval=interval)

            # Apply temporal lock
            hist = filter_by_simulation_date(hist)

            return [
                HistoricalPrice(
                    date=idx.strftime("%Y-%m-%d"),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=int(row["Volume"]),
                    adj_close=float(row.get("Adj Close", row["Close"])),
                ).model_dump()
                for idx, row in hist.iterrows()
            ]

        except Exception as e:
            return [{"error": str(e), "ticker": ticker}]

    @mcp.tool
    def get_financials(
        ticker: str,
        statement_type: str = "income",
        period: str = "quarterly",
    ) -> dict[str, Any]:
        """
        Get financial statement data.

        Args:
            ticker: Stock ticker symbol
            statement_type: "income", "balance", or "cashflow"
            period: "quarterly" or "annual"

        Returns:
            Financial statement data with key metrics
        """
        try:
            stock = yf.Ticker(ticker)

            if period == "quarterly":
                if statement_type == "income":
                    stmt = stock.quarterly_income_stmt
                elif statement_type == "balance":
                    stmt = stock.quarterly_balance_sheet
                else:
                    stmt = stock.quarterly_cashflow
            else:
                if statement_type == "income":
                    stmt = stock.income_stmt
                elif statement_type == "balance":
                    stmt = stock.balance_sheet
                else:
                    stmt = stock.cashflow

            if stmt.empty:
                return {"error": f"No {statement_type} data for {ticker}", "ticker": ticker}

            # Get most recent column
            latest_col = stmt.columns[0]
            period_str = latest_col.strftime("%Y-%m-%d") if hasattr(latest_col, "strftime") else str(latest_col)

            # Extract key metrics
            data = {}
            for idx in stmt.index:
                val = stmt.loc[idx, latest_col]
                if val is not None and not (isinstance(val, float) and val != val):
                    data[str(idx)] = float(val)

            return FinancialStatements(
                ticker=ticker,
                period=period_str,
                statement_type=statement_type,
                data=data,
            ).model_dump()

        except Exception as e:
            return {"error": str(e), "ticker": ticker}

    @mcp.tool
    def get_key_statistics(ticker: str) -> dict[str, Any]:
        """
        Get key statistics and valuation metrics.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Key statistics including P/E, P/B, beta, and other metrics
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            return {
                "ticker": ticker,
                "market_cap": info.get("marketCap"),
                "enterprise_value": info.get("enterpriseValue"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "price_to_sales": info.get("priceToSalesTrailing12Months"),
                "ev_to_ebitda": info.get("enterpriseToEbitda"),
                "ev_to_revenue": info.get("enterpriseToRevenue"),
                "beta": info.get("beta"),
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
                "fifty_day_average": info.get("fiftyDayAverage"),
                "two_hundred_day_average": info.get("twoHundredDayAverage"),
                "dividend_yield": info.get("dividendYield"),
                "dividend_rate": info.get("dividendRate"),
                "payout_ratio": info.get("payoutRatio"),
            }

        except Exception as e:
            return {"error": str(e), "ticker": ticker}

    @mcp.tool
    def get_analyst_estimates(ticker: str) -> dict[str, Any]:
        """
        Get analyst estimates and recommendations.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Analyst price targets, recommendations, and estimates
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Get recommendations
            recs = stock.recommendations
            recent_recs = []
            if recs is not None and not recs.empty:
                for idx, row in recs.head(5).iterrows():
                    recent_recs.append({
                        "date": idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx),
                        "firm": row.get("Firm", ""),
                        "grade": row.get("To Grade", row.get("toGrade", "")),
                    })

            return {
                "ticker": ticker,
                "target_mean_price": info.get("targetMeanPrice"),
                "target_high_price": info.get("targetHighPrice"),
                "target_low_price": info.get("targetLowPrice"),
                "target_median_price": info.get("targetMedianPrice"),
                "recommendation_mean": info.get("recommendationMean"),
                "recommendation_key": info.get("recommendationKey"),
                "number_of_analyst_opinions": info.get("numberOfAnalystOpinions"),
                "recent_recommendations": recent_recs,
            }

        except Exception as e:
            return {"error": str(e), "ticker": ticker}

    @mcp.tool
    def get_earnings(ticker: str) -> dict[str, Any]:
        """
        Get earnings history and estimates.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Historical earnings and upcoming estimates
        """
        try:
            stock = yf.Ticker(ticker)

            # Get earnings history
            earnings_hist = []
            if hasattr(stock, 'earnings_history') and stock.earnings_history is not None:
                eh = stock.earnings_history
                if not eh.empty:
                    for idx, row in eh.iterrows():
                        earnings_hist.append({
                            "date": str(idx),
                            "eps_actual": row.get("epsActual"),
                            "eps_estimate": row.get("epsEstimate"),
                            "surprise": row.get("surprise"),
                            "surprise_pct": row.get("surprisePercent"),
                        })

            # Get earnings dates
            earnings_dates = []
            if hasattr(stock, 'earnings_dates') and stock.earnings_dates is not None:
                ed = stock.earnings_dates
                if not ed.empty:
                    for idx, row in ed.head(4).iterrows():
                        earnings_dates.append({
                            "date": idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx),
                            "eps_estimate": row.get("EPS Estimate"),
                            "reported_eps": row.get("Reported EPS"),
                            "surprise_pct": row.get("Surprise(%)"),
                        })

            return {
                "ticker": ticker,
                "earnings_history": earnings_hist,
                "upcoming_earnings": earnings_dates,
            }

        except Exception as e:
            return {"error": str(e), "ticker": ticker}

    @mcp.tool
    def compare_stocks(tickers: list[str], metric: str = "pe_ratio") -> list[dict[str, Any]]:
        """
        Compare multiple stocks on a specific metric.

        Args:
            tickers: List of stock ticker symbols
            metric: Metric to compare - "pe_ratio", "market_cap", "dividend_yield", "beta"

        Returns:
            Comparison data for each ticker
        """
        results = []
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info

                metric_map = {
                    "pe_ratio": "trailingPE",
                    "market_cap": "marketCap",
                    "dividend_yield": "dividendYield",
                    "beta": "beta",
                    "price_to_book": "priceToBook",
                    "forward_pe": "forwardPE",
                }

                yf_metric = metric_map.get(metric, metric)
                value = info.get(yf_metric)

                results.append({
                    "ticker": ticker,
                    "name": info.get("longName", ""),
                    "metric": metric,
                    "value": value,
                })

            except Exception as e:
                results.append({
                    "ticker": ticker,
                    "error": str(e),
                })

        return results

    @mcp.resource("yahoo://quote/{ticker}")
    def quote_resource(ticker: str) -> str:
        """
        Get a formatted stock quote summary.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Human-readable stock quote summary
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            price = info.get("currentPrice") or info.get("regularMarketPrice", "N/A")
            market_cap = info.get("marketCap", 0)
            market_cap_str = f"${market_cap/1e9:.1f}B" if market_cap else "N/A"

            return f"""
Stock Quote: {ticker}
Name: {info.get('longName', 'N/A')}
Sector: {info.get('sector', 'N/A')}
Industry: {info.get('industry', 'N/A')}

Price: ${price}
Market Cap: {market_cap_str}
P/E Ratio: {info.get('trailingPE', 'N/A')}
Forward P/E: {info.get('forwardPE', 'N/A')}

52-Week High: ${info.get('fiftyTwoWeekHigh', 'N/A')}
52-Week Low: ${info.get('fiftyTwoWeekLow', 'N/A')}

Analyst Rating: {info.get('recommendationKey', 'N/A')}
Target Price: ${info.get('targetMeanPrice', 'N/A')}
""".strip()

        except Exception as e:
            return f"Error fetching quote for {ticker}: {e}"

    return mcp


# CLI entry point
if __name__ == "__main__":
    import sys

    simulation_date = None
    if len(sys.argv) > 1:
        try:
            simulation_date = datetime.fromisoformat(sys.argv[1])
        except ValueError:
            pass

    server = create_yahoo_finance_server(simulation_date=simulation_date)
    server.run()
