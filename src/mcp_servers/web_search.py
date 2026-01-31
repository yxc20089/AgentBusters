"""
Web Search MCP Server

A FastMCP server that provides web search capabilities using Tavily API.
Enables agents to search for recent financial news, earnings calls, and market data.
"""

import os
from datetime import datetime
from typing import Any

from fastmcp import FastMCP
from pydantic import BaseModel, Field

# Try to import tavily
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False


class SearchResult(BaseModel):
    """A single search result."""
    title: str
    url: str
    content: str
    score: float = 0.0
    published_date: str = ""


class SearchResponse(BaseModel):
    """Response from a web search."""
    query: str
    results: list[SearchResult] = Field(default_factory=list)
    answer: str = ""
    search_depth: str = "basic"
    total_results: int = 0


def create_web_search_server(
    api_key: str | None = None,
    name: str = "web-search-mcp",
) -> FastMCP:
    """
    Create the Web Search MCP server.

    Args:
        api_key: Tavily API key (or use TAVILY_API_KEY env var)
        name: Server name

    Returns:
        Configured FastMCP server
    """
    mcp = FastMCP(name)

    # Get API key from env if not provided
    _api_key = api_key or os.environ.get("TAVILY_API_KEY")
    _client = None

    if TAVILY_AVAILABLE and _api_key:
        _client = TavilyClient(api_key=_api_key)

    @mcp.tool
    def web_search(
        query: str,
        search_depth: str = "basic",
        max_results: int = 5,
        include_answer: bool = True,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> SearchResponse:
        """
        Search the web for information.

        Args:
            query: Search query string
            search_depth: "basic" for fast results, "advanced" for comprehensive
            max_results: Maximum number of results to return (1-10)
            include_answer: Whether to include AI-generated answer summary
            include_domains: Only include results from these domains
            exclude_domains: Exclude results from these domains

        Returns:
            SearchResponse with results and optional answer
        """
        if not TAVILY_AVAILABLE:
            return SearchResponse(
                query=query,
                results=[],
                answer="Tavily library not installed. Install with: pip install tavily-python",
                total_results=0,
            )

        if not _client:
            return SearchResponse(
                query=query,
                results=[],
                answer="TAVILY_API_KEY not configured",
                total_results=0,
            )

        try:
            response = _client.search(
                query=query,
                search_depth=search_depth,
                max_results=min(max_results, 10),
                include_answer=include_answer,
                include_domains=include_domains or [],
                exclude_domains=exclude_domains or [],
            )

            results = [
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    content=r.get("content", ""),
                    score=r.get("score", 0.0),
                    published_date=r.get("published_date", ""),
                )
                for r in response.get("results", [])
            ]

            return SearchResponse(
                query=query,
                results=results,
                answer=response.get("answer", ""),
                search_depth=search_depth,
                total_results=len(results),
            )

        except Exception as e:
            return SearchResponse(
                query=query,
                results=[],
                answer=f"Search error: {str(e)}",
                total_results=0,
            )

    @mcp.tool
    def search_financial_news(
        company: str,
        topic: str = "",
        days_back: int = 30,
        max_results: int = 5,
    ) -> SearchResponse:
        """
        Search for recent financial news about a company.

        Args:
            company: Company name or ticker symbol
            topic: Specific topic (e.g., "earnings", "guidance", "merger")
            days_back: How many days back to search
            max_results: Maximum number of results

        Returns:
            SearchResponse with financial news results
        """
        # Build search query optimized for financial news
        query_parts = [company]
        if topic:
            query_parts.append(topic)
        query_parts.append("financial news")

        query = " ".join(query_parts)

        # Use domains known for financial news
        finance_domains = [
            "reuters.com",
            "bloomberg.com",
            "wsj.com",
            "cnbc.com",
            "seekingalpha.com",
            "fool.com",
            "marketwatch.com",
            "finance.yahoo.com",
            "investors.com",
        ]

        return web_search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=True,
            include_domains=finance_domains,
        )

    @mcp.tool
    def search_earnings_info(
        ticker: str,
        quarter: str = "",
        year: int | None = None,
    ) -> SearchResponse:
        """
        Search for earnings call information and guidance.

        Args:
            ticker: Stock ticker symbol
            quarter: Quarter (e.g., "Q1", "Q2", "Q3", "Q4")
            year: Fiscal year

        Returns:
            SearchResponse with earnings information
        """
        query_parts = [ticker, "earnings"]
        if quarter:
            query_parts.append(quarter)
        if year:
            query_parts.append(str(year))
        query_parts.extend(["call", "guidance", "results"])

        query = " ".join(query_parts)

        return web_search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_answer=True,
        )

    @mcp.tool
    def search_sec_filings_news(
        ticker: str,
        filing_type: str = "",
    ) -> SearchResponse:
        """
        Search for news and analysis about SEC filings.

        Args:
            ticker: Stock ticker symbol
            filing_type: Type of filing (e.g., "10-K", "10-Q", "8-K")

        Returns:
            SearchResponse with filing-related news
        """
        query_parts = [ticker]
        if filing_type:
            query_parts.append(filing_type)
        query_parts.extend(["SEC filing", "analysis"])

        query = " ".join(query_parts)

        return web_search(
            query=query,
            search_depth="basic",
            max_results=5,
            include_answer=True,
        )

    return mcp


# Main entry point for standalone server
if __name__ == "__main__":
    import uvicorn

    server = create_web_search_server()
    uvicorn.run(server, host="0.0.0.0", port=8107)
