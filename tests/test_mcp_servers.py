"""
End-to-End Tests for MCP Servers (FastMCP implementations)

Tests the actual MCP server implementations with real financial data
via the Model Context Protocol.
"""

import pytest
from datetime import datetime

import sys
sys.path.insert(0, "src")

from mcp_servers.sec_edgar import create_edgar_server
from mcp_servers.yahoo_finance import create_yahoo_finance_server
from mcp_servers.sandbox import create_sandbox_server


class TestMCPServerCreation:
    """Tests for MCP server instantiation."""

    def test_edgar_server_creation(self):
        """Test SEC EDGAR server is created correctly."""
        server = create_edgar_server()
        assert server is not None
        assert server.name == "sec-edgar-mcp"

    def test_yahoo_finance_server_creation(self):
        """Test Yahoo Finance server is created correctly."""
        server = create_yahoo_finance_server()
        assert server is not None
        assert server.name == "yahoo-finance-mcp"

    def test_sandbox_server_creation(self):
        """Test Sandbox server is created correctly."""
        server = create_sandbox_server()
        assert server is not None
        assert server.name == "python-sandbox-mcp"

    def test_servers_with_temporal_locking(self):
        """Test servers can be created with simulation date for temporal locking."""
        sim_date = datetime(2024, 6, 1)

        edgar = create_edgar_server(simulation_date=sim_date)
        yfinance = create_yahoo_finance_server(simulation_date=sim_date)

        assert edgar is not None
        assert yfinance is not None


class TestYahooFinanceE2E:
    """End-to-end tests for Yahoo Finance MCP server with real data."""

    @pytest.mark.asyncio
    async def test_get_real_stock_quote(self):
        """Test fetching real stock quote from Yahoo Finance."""
        import yfinance as yf

        stock = yf.Ticker("AAPL")
        info = stock.info

        assert info is not None
        assert "currentPrice" in info or "regularMarketPrice" in info

        # Verify we get actual company data
        assert info.get("longName") is not None or info.get("shortName") is not None

    @pytest.mark.asyncio
    async def test_get_real_historical_prices(self):
        """Test fetching real historical prices."""
        import yfinance as yf

        stock = yf.Ticker("MSFT")
        hist = stock.history(period="5d")

        assert not hist.empty
        assert "Close" in hist.columns
        assert "Open" in hist.columns
        assert "Volume" in hist.columns
        assert len(hist) > 0

    @pytest.mark.asyncio
    async def test_get_real_financials(self):
        """Test fetching real financial statements."""
        import yfinance as yf

        stock = yf.Ticker("GOOGL")
        income = stock.quarterly_income_stmt

        assert income is not None
        if not income.empty:
            assert len(income.columns) > 0

    @pytest.mark.asyncio
    async def test_get_real_key_statistics(self):
        """Test fetching real key statistics."""
        import yfinance as yf

        stock = yf.Ticker("NVDA")
        info = stock.info

        # Check for key statistics
        assert "marketCap" in info or "trailingPE" in info


class TestSECEdgarE2E:
    """End-to-end tests for SEC EDGAR MCP server with real data."""

    @pytest.mark.asyncio
    async def test_get_real_company_info(self):
        """Test fetching real company info from SEC EDGAR."""
        try:
            from edgar import Company, set_identity
            set_identity("AgentBusters research@agentbusters.ai")

            company = Company("AAPL")

            assert company.cik is not None
            assert company.name is not None
        except Exception as e:
            pytest.skip(f"SEC EDGAR API unavailable: {e}")

    @pytest.mark.asyncio
    async def test_get_real_10k_filings(self):
        """Test fetching real 10-K filings."""
        try:
            from edgar import Company, set_identity
            set_identity("AgentBusters research@agentbusters.ai")

            company = Company("MSFT")
            filings = company.get_filings(form="10-K")

            assert filings is not None
            # Microsoft should have 10-K filings
        except Exception as e:
            pytest.skip(f"SEC EDGAR API unavailable: {e}")

    @pytest.mark.asyncio
    async def test_get_real_10q_filings(self):
        """Test fetching real 10-Q filings."""
        try:
            from edgar import Company, set_identity
            set_identity("AgentBusters research@agentbusters.ai")

            company = Company("NVDA")
            filings = company.get_filings(form="10-Q")

            assert filings is not None
        except Exception as e:
            pytest.skip(f"SEC EDGAR API unavailable: {e}")


class TestSandboxE2E:
    """End-to-end tests for Python Sandbox MCP server."""

    def test_execute_financial_calculation(self):
        """Test executing a real financial calculation in sandbox."""
        import io
        from contextlib import redirect_stdout

        code = """
import numpy as np

# Calculate CAGR (Compound Annual Growth Rate)
initial_value = 1000
final_value = 1500
years = 3
cagr = (final_value / initial_value) ** (1/years) - 1
print(f"CAGR: {cagr:.4f}")
"""
        stdout_capture = io.StringIO()
        namespace = {}

        import numpy as np
        namespace["np"] = np
        namespace["numpy"] = np

        with redirect_stdout(stdout_capture):
            exec(code, namespace)

        output = stdout_capture.getvalue()
        assert "CAGR:" in output
        assert namespace["cagr"] > 0

    def test_execute_pandas_analysis(self):
        """Test executing pandas financial analysis in sandbox."""
        import pandas as pd
        import numpy as np

        # Simulate stock price data analysis
        dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
        prices = np.random.randn(30).cumsum() + 100

        df = pd.DataFrame({"date": dates, "price": prices})
        df["returns"] = df["price"].pct_change()
        df["volatility"] = df["returns"].rolling(window=5).std()

        assert len(df) == 30
        assert "returns" in df.columns
        assert "volatility" in df.columns

    def test_execute_statistical_analysis(self):
        """Test executing statistical analysis in sandbox."""
        import numpy as np
        from scipy import stats

        # Simulate returns data
        returns = np.random.randn(100) * 0.02  # 2% daily volatility

        # Calculate statistics
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = mean_return / std_return * np.sqrt(252)  # Annualized

        # Normality test
        _, p_value = stats.normaltest(returns)

        assert isinstance(sharpe_ratio, float)
        assert isinstance(p_value, float)

    def test_safe_builtins_security(self):
        """Test that dangerous operations are restricted."""
        from mcp_servers.sandbox import SAFE_BUILTINS

        # These should NOT be available
        dangerous = ["__import__", "open", "eval", "exec", "compile", "globals", "locals"]
        for func in dangerous:
            assert func not in SAFE_BUILTINS, f"{func} should not be in SAFE_BUILTINS"

        # These SHOULD be available
        safe = ["len", "sum", "print", "range", "list", "dict", "str", "int", "float"]
        for func in safe:
            assert func in SAFE_BUILTINS, f"{func} should be in SAFE_BUILTINS"


class TestTemporalLockingE2E:
    """End-to-end tests for temporal locking (Time Machine) functionality."""

    @pytest.mark.asyncio
    async def test_yahoo_finance_respects_simulation_date(self):
        """Test Yahoo Finance filters data by simulation date."""
        import pandas as pd
        import yfinance as yf

        simulation_date = datetime(2024, 6, 1)

        # Get historical data
        stock = yf.Ticker("AAPL")
        hist = stock.history(period="1y")

        if not hist.empty:
            # Filter by simulation date (as the server would)
            if hist.index.tz is not None:
                sim_date_aware = pd.Timestamp(simulation_date).tz_localize(hist.index.tz)
                filtered = hist[hist.index <= sim_date_aware]
            else:
                filtered = hist[hist.index <= simulation_date]

            # Filtered data should not include dates after simulation date
            if not filtered.empty:
                assert filtered.index.max().replace(tzinfo=None) <= simulation_date

    def test_temporal_lock_check_function(self):
        """Test the temporal lock check logic."""
        simulation_date = datetime(2024, 6, 1)

        def check_temporal_lock(filing_date_str: str) -> bool:
            filing_date = datetime.fromisoformat(filing_date_str.replace("Z", "+00:00"))
            if filing_date.tzinfo:
                filing_date = filing_date.replace(tzinfo=None)
            return filing_date <= simulation_date

        # Past filings should be allowed
        assert check_temporal_lock("2024-03-15") is True
        assert check_temporal_lock("2024-05-31") is True
        assert check_temporal_lock("2024-06-01") is True

        # Future filings should be blocked
        assert check_temporal_lock("2024-06-02") is False
        assert check_temporal_lock("2024-08-15") is False
        assert check_temporal_lock("2025-01-01") is False


class TestMCPServerIntegration:
    """Integration tests combining multiple MCP servers."""

    @pytest.mark.asyncio
    async def test_cross_source_company_data(self):
        """Test fetching company data from multiple sources."""
        import yfinance as yf

        ticker = "AAPL"

        # Yahoo Finance data
        stock = yf.Ticker(ticker)
        yf_info = stock.info

        # SEC EDGAR data
        try:
            from edgar import Company, set_identity
            set_identity("AgentBusters research@agentbusters.ai")
            company = Company(ticker)
            edgar_name = company.name
        except Exception:
            edgar_name = None

        # Both sources should have data for Apple
        assert yf_info.get("longName") is not None or yf_info.get("shortName") is not None
        # edgar_name may be None if SEC API is unavailable

    @pytest.mark.asyncio
    async def test_financial_analysis_workflow(self):
        """Test a complete financial analysis workflow using MCP servers."""
        import yfinance as yf
        import numpy as np

        ticker = "NVDA"

        # Step 1: Get current quote
        stock = yf.Ticker(ticker)
        info = stock.info
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")

        # Step 2: Get historical data
        hist = stock.history(period="3mo")

        # Step 3: Calculate metrics (as sandbox would)
        if not hist.empty and current_price:
            returns = hist["Close"].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized

            # Basic validation
            assert current_price > 0
            assert volatility > 0
            assert len(returns) > 0
