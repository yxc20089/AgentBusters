"""
SEC EDGAR MCP Server

A FastMCP server that provides access to SEC EDGAR filings and XBRL data.
Implements temporal locking to prevent look-ahead bias in evaluations.
"""

import os
from datetime import datetime
from typing import Any

from fastmcp import FastMCP
from pydantic import BaseModel, Field

# Try to import edgar (edgartools package) for real data
try:
    from edgar import Company, set_identity
    EDGARTOOLS_AVAILABLE = True
    # Set identity for SEC EDGAR API
    set_identity("AgentBusters research@agentbusters.ai")
except ImportError:
    EDGARTOOLS_AVAILABLE = False


class FilingResult(BaseModel):
    """Result from a filing query."""
    ticker: str
    form_type: str
    filing_date: str
    accession_number: str = ""
    fiscal_year: int | None = None
    fiscal_period: str = ""
    content: str = ""
    sections: dict[str, str] = Field(default_factory=dict)


class XBRLData(BaseModel):
    """Parsed XBRL financial data."""
    ticker: str
    fiscal_year: int
    fiscal_period: str
    statement_type: str
    filing_date: str
    data: dict[str, float | None] = Field(default_factory=dict)


def create_edgar_server(
    simulation_date: datetime | None = None,
    name: str = "sec-edgar-mcp",
) -> FastMCP:
    """
    Create the SEC EDGAR MCP server.

    Args:
        simulation_date: Optional date for temporal locking
        name: Server name

    Returns:
        Configured FastMCP server
    """
    mcp = FastMCP(name)

    # Store simulation date in server context
    _simulation_date = simulation_date

    def check_temporal_lock(filing_date_str: str) -> bool:
        """Check if filing date violates temporal constraints."""
        if not _simulation_date:
            return True

        try:
            filing_date = datetime.fromisoformat(filing_date_str.replace("Z", "+00:00"))
            if filing_date.tzinfo:
                filing_date = filing_date.replace(tzinfo=None)
            return filing_date <= _simulation_date
        except ValueError:
            return True

    @mcp.tool
    def get_filing(
        ticker: str,
        form_type: str,
        fiscal_year: int | None = None,
    ) -> dict[str, Any]:
        """
        Get an SEC filing for a company.

        Args:
            ticker: Stock ticker symbol (e.g., "NVDA", "AAPL")
            form_type: Filing type (e.g., "10-K", "10-Q", "8-K")
            fiscal_year: Optional fiscal year to filter by

        Returns:
            Filing data including content and metadata
        """
        if not EDGARTOOLS_AVAILABLE:
            return {
                "error": "edgartools not installed",
                "ticker": ticker,
                "form_type": form_type,
            }

        try:
            company = Company(ticker)
            filings = company.get_filings(form=form_type)

            for filing in filings:
                filing_date = str(filing.filing_date)

                # Check temporal lock
                if not check_temporal_lock(filing_date):
                    continue

                # Check fiscal year if specified
                if fiscal_year:
                    # 10-K is typically filed 60-90 days after fiscal year end
                    if filing.filing_date.year not in [fiscal_year, fiscal_year + 1]:
                        continue

                return FilingResult(
                    ticker=ticker,
                    form_type=form_type,
                    filing_date=filing_date,
                    accession_number=filing.accession_number,
                    fiscal_year=fiscal_year,
                    content=f"Filing {form_type} for {ticker} filed on {filing_date}",
                ).model_dump()

            return {"error": f"No {form_type} filing found for {ticker}", "ticker": ticker}

        except Exception as e:
            return {"error": str(e), "ticker": ticker, "form_type": form_type}

    @mcp.tool
    def get_filing_section(
        ticker: str,
        form_type: str,
        section_name: str,
        fiscal_year: int | None = None,
    ) -> dict[str, Any]:
        """
        Get a specific section from an SEC filing.

        Args:
            ticker: Stock ticker symbol
            form_type: Filing type (e.g., "10-K", "10-Q")
            section_name: Section to extract (e.g., "Item 1A", "Item 7", "MD&A")
            fiscal_year: Optional fiscal year

        Returns:
            Section content and metadata
        """
        if not EDGARTOOLS_AVAILABLE:
            return {"error": "edgartools not installed", "ticker": ticker}

        try:
            company = Company(ticker)
            filings = company.get_filings(form=form_type)

            for filing in filings:
                filing_date = str(filing.filing_date)

                if not check_temporal_lock(filing_date):
                    continue

                # For now, return placeholder - full section extraction requires more work
                return {
                    "ticker": ticker,
                    "form_type": form_type,
                    "section_name": section_name,
                    "filing_date": filing_date,
                    "content": f"Section {section_name} from {form_type} filing",
                }

            return {"error": f"No filing found for {ticker}", "ticker": ticker}

        except Exception as e:
            return {"error": str(e), "ticker": ticker}

    @mcp.tool
    def get_xbrl_financials(
        ticker: str,
        statement_type: str = "IS",
        fiscal_year: int | None = None,
    ) -> dict[str, Any]:
        """
        Get parsed XBRL financial data from SEC filings.

        Args:
            ticker: Stock ticker symbol
            statement_type: Statement type - "IS" (Income Statement), "BS" (Balance Sheet), "CF" (Cash Flow)
            fiscal_year: Optional fiscal year

        Returns:
            Parsed financial data with key metrics
        """
        if not EDGARTOOLS_AVAILABLE:
            return {"error": "edgartools not installed", "ticker": ticker}

        try:
            company = Company(ticker)
            filings = company.get_filings(form="10-K")

            for filing in filings:
                filing_date = str(filing.filing_date)

                if not check_temporal_lock(filing_date):
                    continue

                # Get the filing object and try to extract financials
                # This is a simplified version - real implementation would parse XBRL
                data: dict[str, float | None] = {}

                if statement_type == "IS":
                    # Income statement fields
                    data = {
                        "Revenue": None,
                        "NetIncome": None,
                        "GrossProfit": None,
                        "OperatingIncome": None,
                        "EarningsPerShareBasic": None,
                        "EarningsPerShareDiluted": None,
                    }
                elif statement_type == "BS":
                    # Balance sheet fields
                    data = {
                        "TotalAssets": None,
                        "TotalLiabilities": None,
                        "StockholdersEquity": None,
                        "CashAndCashEquivalents": None,
                        "TotalDebt": None,
                    }
                elif statement_type == "CF":
                    # Cash flow fields
                    data = {
                        "OperatingCashFlow": None,
                        "CapitalExpenditures": None,
                        "FreeCashFlow": None,
                    }

                return XBRLData(
                    ticker=ticker,
                    fiscal_year=fiscal_year or filing.filing_date.year,
                    fiscal_period="FY",
                    statement_type=statement_type,
                    filing_date=filing_date,
                    data=data,
                ).model_dump()

            return {"error": f"No 10-K filing found for {ticker}", "ticker": ticker}

        except Exception as e:
            return {"error": str(e), "ticker": ticker}

    @mcp.tool
    def search_filings(
        ticker: str,
        form_type: str,
        keywords: list[str],
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Search SEC filings by keyword.

        Args:
            ticker: Stock ticker symbol
            form_type: Filing type to search
            keywords: Keywords to search for
            limit: Maximum number of results

        Returns:
            List of matching filing excerpts
        """
        if not EDGARTOOLS_AVAILABLE:
            return [{"error": "edgartools not installed"}]

        try:
            company = Company(ticker)
            filings = company.get_filings(form=form_type)

            results = []
            for filing in filings[:limit]:
                filing_date = str(filing.filing_date)

                if not check_temporal_lock(filing_date):
                    continue

                results.append({
                    "ticker": ticker,
                    "form_type": form_type,
                    "filing_date": filing_date,
                    "accession_number": filing.accession_number,
                    "keywords_matched": keywords,  # Simplified - would actually search
                })

            return results

        except Exception as e:
            return [{"error": str(e)}]

    @mcp.tool
    def get_company_info(ticker: str) -> dict[str, Any]:
        """
        Get basic company information from SEC EDGAR.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Company information including CIK, name, SIC code
        """
        if not EDGARTOOLS_AVAILABLE:
            return {"error": "edgartools not installed", "ticker": ticker}

        try:
            company = Company(ticker)
            return {
                "ticker": ticker,
                "cik": company.cik,
                "name": company.name,
                "sic": getattr(company, "sic", None),
                "sic_description": getattr(company, "sic_description", None),
            }
        except Exception as e:
            return {"error": str(e), "ticker": ticker}

    @mcp.resource("edgar://filings/{ticker}")
    def list_filings_resource(ticker: str) -> str:
        """
        List recent filings for a company.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Summary of recent filings
        """
        if not EDGARTOOLS_AVAILABLE:
            return f"edgartools not installed - cannot fetch filings for {ticker}"

        try:
            company = Company(ticker)
            filings = company.get_filings()

            lines = [f"Recent SEC filings for {ticker}:", ""]
            for filing in filings[:10]:
                filing_date = str(filing.filing_date)
                if check_temporal_lock(filing_date):
                    lines.append(f"- {filing.form} ({filing_date})")

            return "\n".join(lines)
        except Exception as e:
            return f"Error fetching filings for {ticker}: {e}"

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

    server = create_edgar_server(simulation_date=simulation_date)
    server.run()
