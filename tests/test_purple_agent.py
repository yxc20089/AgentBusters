"""
Tests for Purple Agent - Finance Analysis Agent

Tests the A2A protocol implementation, MCP-based finance tools,
and agent execution capabilities.

All tests use in-process MCP servers for controlled environment.
"""

import os
import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

import sys
sys.path.insert(0, "src")

from purple_agent.card import get_agent_card
from purple_agent.mcp_toolkit import MCPToolkit
from purple_agent.executor import FinanceAgentExecutor
from purple_agent.agent import FinanceAnalysisAgent, create_agent


class TestAgentCard:
    """Tests for Agent Card generation."""

    def test_agent_card_creation(self):
        """Test basic agent card creation."""
        card = get_agent_card()

        assert card.name == "Purple Finance Agent"
        assert card.version == "1.0.0"
        assert card.url == "http://localhost:8101/"

    def test_agent_card_custom_host_port(self):
        """Test agent card with custom host and port."""
        card = get_agent_card(host="example.com", port=9000)

        assert card.url == "http://example.com:9000/"

    def test_agent_card_has_skills(self):
        """Test agent card has required skills."""
        card = get_agent_card()

        assert len(card.skills) >= 5

        skill_ids = [s.id for s in card.skills]
        assert "earnings_analysis" in skill_ids
        assert "sec_filing_analysis" in skill_ids
        assert "financial_ratio_calculation" in skill_ids

    def test_agent_card_capabilities(self):
        """Test agent card capabilities."""
        card = get_agent_card()

        assert card.capabilities.streaming is True
        assert card.capabilities.state_transition_history is True

    def test_agent_card_provider(self):
        """Test agent card provider info."""
        card = get_agent_card()

        assert card.provider.organization == "AgentBusters Team"
        assert "github.com" in card.provider.url


class TestMCPToolkit:
    """Tests for the MCP-based finance toolkit."""

    @pytest.fixture
    def toolkit(self):
        """Create MCP toolkit (in-process MCP servers)."""
        return MCPToolkit()

    @pytest.fixture
    def toolkit_with_simulation(self):
        """Create MCP toolkit with simulation date."""
        return MCPToolkit(simulation_date=datetime(2024, 6, 1))

    @pytest.mark.asyncio
    async def test_get_quote(self, toolkit):
        """Test getting stock quote via MCP."""
        quote = await toolkit.get_quote("AAPL")

        assert quote is not None
        # Should have price data
        assert "current_price" in quote or "error" not in quote

    @pytest.mark.asyncio
    async def test_get_financials(self, toolkit):
        """Test getting financial data via MCP."""
        financials = await toolkit.get_financials("AAPL", "income", "quarterly")

        assert financials is not None
        assert "ticker" in financials

    @pytest.mark.asyncio
    async def test_get_key_statistics(self, toolkit):
        """Test getting key statistics via MCP."""
        stats = await toolkit.get_key_statistics("MSFT")

        assert stats is not None

    @pytest.mark.asyncio
    async def test_comprehensive_analysis(self, toolkit):
        """Test comprehensive analysis via MCP."""
        data = await toolkit.get_comprehensive_analysis("NVDA")

        assert data["ticker"] == "NVDA"
        assert "quote" in data
        assert "statistics" in data
        assert "company_info" in data

    @pytest.mark.asyncio
    async def test_execute_python_in_sandbox(self, toolkit):
        """Test Python code execution via sandbox MCP."""
        result = await toolkit.execute_python("""
import numpy as np
values = [100, 110, 121, 133.1]
growth_rate = (values[-1] / values[0]) ** (1/3) - 1
print(f"CAGR: {growth_rate:.2%}")
""")

        assert result["success"] is True
        assert "CAGR" in result["stdout"]

    @pytest.mark.asyncio
    async def test_calculate_financial_metric(self, toolkit):
        """Test financial metric calculation via sandbox MCP."""
        result = await toolkit.calculate_financial_metric(
            metric="gross_margin",
            values={"revenue": 100_000_000, "cogs": 30_000_000}
        )

        assert "value" in result
        assert result["value"] == 0.7  # 70% gross margin

    @pytest.mark.asyncio
    async def test_analyze_time_series(self, toolkit):
        """Test time series analysis via sandbox MCP."""
        result = await toolkit.analyze_time_series(
            data=[100, 105, 103, 110, 115, 112, 120],
            operations=["mean", "std", "trend"]
        )

        assert "mean" in result
        assert "std" in result
        assert "trend" in result

    @pytest.mark.asyncio
    async def test_calculate_option_price_local_computation(self, toolkit):
        """Test that calculate_option_price uses local Black-Scholes computation."""
        # Test basic call option
        result = await toolkit.calculate_option_price(
            spot_price=450.0,
            strike_price=460.0,
            days_to_expiry=45,
            volatility=0.45,
            risk_free_rate=0.0525,
            option_type="call",
        )

        # Should return valid result without MCP server errors
        assert "error" not in result
        assert "price" in result
        assert "delta" in result
        assert "gamma" in result
        assert "theta" in result
        assert "vega" in result
        assert "rho" in result
        
        # Verify price is reasonable (OTM call with high IV)
        assert result["price"] > 0
        assert result["spot_price"] == 450.0
        assert result["strike_price"] == 460.0
        assert result["days_to_expiry"] == 45
        
        # Delta should be between 0 and 1 for call
        assert 0 <= result["delta"] <= 1
        
        # Gamma should be positive
        assert result["gamma"] > 0

    @pytest.mark.asyncio
    async def test_calculate_option_price_put(self, toolkit):
        """Test put option pricing."""
        result = await toolkit.calculate_option_price(
            spot_price=100.0,
            strike_price=105.0,
            days_to_expiry=30,
            volatility=0.25,
            risk_free_rate=0.05,
            option_type="put",
        )

        assert "error" not in result
        assert "price" in result
        
        # Delta should be between -1 and 0 for put
        assert -1 <= result["delta"] <= 0
        
        # ITM put should have significant value
        assert result["price"] > 0

    @pytest.mark.asyncio
    async def test_calculate_option_price_at_expiration(self, toolkit):
        """Test option pricing at or past expiration (T=0)."""
        # ITM call at expiration
        result = await toolkit.calculate_option_price(
            spot_price=110.0,
            strike_price=100.0,
            days_to_expiry=0,
            volatility=0.20,
            risk_free_rate=0.05,
            option_type="call",
        )

        assert "error" not in result
        # Should equal intrinsic value
        assert result["price"] == 10.0  # 110 - 100
        assert result["gamma"] == 0.0
        assert result["vega"] == 0.0

    @pytest.mark.asyncio
    async def test_calculate_option_price_with_dividend(self, toolkit):
        """Test option pricing with dividend yield."""
        result = await toolkit.calculate_option_price(
            spot_price=150.0,
            strike_price=150.0,
            days_to_expiry=90,
            volatility=0.30,
            risk_free_rate=0.05,
            option_type="call",
            dividend_yield=0.02,
        )

        assert "error" not in result
        assert "price" in result
        
        # ATM call with dividend should have reasonable delta
        # With 2% dividend yield, delta will be slightly > 0.5 for ATM call
        assert 0.4 < result["delta"] < 0.7


class TestFinanceAgentExecutorTemperature:
    """
    Tests for FinanceAgentExecutor temperature handling.

    Addresses Copilot review: Ensure temperature parameter with fallback to
    PURPLE_LLM_TEMPERATURE env var and default 0.0 is correctly exercised.
    """

    def test_temperature_default_is_zero(self):
        """Verify default temperature is 0.0 when neither param nor env var is set."""
        # Ensure env var is not set
        with patch.dict(os.environ, {}, clear=True):
            # Remove PURPLE_LLM_TEMPERATURE if present
            os.environ.pop("PURPLE_LLM_TEMPERATURE", None)
            executor = FinanceAgentExecutor()
            assert executor.temperature == 0.0

    def test_temperature_explicit_parameter(self):
        """Verify explicit temperature parameter takes precedence."""
        executor = FinanceAgentExecutor(temperature=0.7)
        assert executor.temperature == 0.7

    def test_temperature_from_env_var(self):
        """Verify PURPLE_LLM_TEMPERATURE env var is parsed and applied."""
        with patch.dict(os.environ, {"PURPLE_LLM_TEMPERATURE": "0.5"}):
            executor = FinanceAgentExecutor()
            assert executor.temperature == 0.5

    def test_temperature_explicit_overrides_env_var(self):
        """Verify explicit parameter takes precedence over env var."""
        with patch.dict(os.environ, {"PURPLE_LLM_TEMPERATURE": "0.5"}):
            executor = FinanceAgentExecutor(temperature=0.3)
            assert executor.temperature == 0.3

    def test_temperature_invalid_env_var_fallback(self):
        """Verify invalid env var value falls back to 0.0."""
        with patch.dict(os.environ, {"PURPLE_LLM_TEMPERATURE": "invalid"}):
            executor = FinanceAgentExecutor()
            assert executor.temperature == 0.0

    def test_temperature_empty_env_var_fallback(self):
        """Verify empty env var value falls back to 0.0."""
        with patch.dict(os.environ, {"PURPLE_LLM_TEMPERATURE": ""}):
            executor = FinanceAgentExecutor()
            # Empty string will cause ValueError, should fall back to 0.0
            assert executor.temperature == 0.0

    def test_temperature_zero_reproducibility(self):
        """Verify default temperature of 0.0 for reproducible benchmarks."""
        executor = FinanceAgentExecutor()
        assert executor.temperature == 0.0, "Default temperature must be 0.0 for reproducibility"


class TestFinanceAgentExecutor:
    """Tests for the agent executor (uses in-process MCP servers)."""

    @pytest.fixture
    def executor(self):
        """Create executor without LLM (uses in-process MCP servers)."""
        return FinanceAgentExecutor()

    @pytest.mark.asyncio
    async def test_parse_task_ticker_extraction(self, executor):
        """Test ticker extraction from question."""
        task_info = await executor._parse_task(
            "Did NVDA beat or miss Q3 FY2026 expectations?"
        )

        assert "NVDA" in task_info["tickers"]

    @pytest.mark.asyncio
    async def test_parse_task_type_detection(self, executor):
        """Test task type detection."""
        # Beat/miss task
        task_info = await executor._parse_task(
            "Did AAPL beat or miss earnings expectations?"
        )
        assert task_info["task_type"] == "beat_or_miss"

        # SEC filing task
        task_info = await executor._parse_task(
            "Summarize MSFT's latest 10-K filing"
        )
        assert task_info["task_type"] == "sec_filing"

        # Ratio calculation
        task_info = await executor._parse_task(
            "What is NVDA's P/E ratio?"
        )
        assert task_info["task_type"] == "ratio_calculation"

    @pytest.mark.asyncio
    async def test_parse_task_fiscal_year_extraction(self, executor):
        """Test fiscal year extraction."""
        task_info = await executor._parse_task(
            "Analyze Q3 FY2026 results"
        )
        assert task_info["fiscal_year"] == 2026

    @pytest.mark.asyncio
    async def test_parse_task_quarter_extraction(self, executor):
        """Test quarter extraction."""
        task_info = await executor._parse_task(
            "Analyze Q3 earnings report"
        )
        assert task_info["quarter"] == 3

    @pytest.mark.asyncio
    async def test_gather_data(self, executor):
        """Test data gathering."""
        task_info = {"tickers": ["AAPL"], "task_type": "general"}
        data = await executor._gather_data(task_info)

        assert "AAPL" in data["tickers"]
        assert "error" not in data["tickers"]["AAPL"] or data["tickers"]["AAPL"].get("stock_info")

    @pytest.mark.asyncio
    async def test_generate_fallback_response(self, executor):
        """Test fallback response generation without LLM."""
        task_info = {"tickers": ["AAPL"], "task_type": "financial_metrics"}
        financial_data = {
            "tickers": {
                "AAPL": {
                    "stock_info": {
                        "name": "Apple Inc.",
                        "sector": "Technology",
                        "price": 150.0,
                    },
                    "financials": {
                        "revenue": 100_000_000_000,
                        "net_income": 20_000_000_000,
                    },
                }
            }
        }

        response = executor._generate_fallback_response(task_info, financial_data)

        assert "Apple Inc." in response
        assert "Revenue" in response
        assert "100,000,000,000" in response


class TestFinanceAnalysisAgent:
    """Tests for the main agent class (uses in-process MCP servers)."""

    @pytest.fixture
    def agent(self):
        """Create agent without LLM (uses in-process MCP servers)."""
        return FinanceAnalysisAgent()

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.host == "localhost"
        assert agent.port == 8101
        assert agent.card is not None
        assert agent.executor is not None

    def test_get_card(self, agent):
        """Test getting card as dictionary."""
        card_dict = agent.get_card()

        assert card_dict["name"] == "Purple Finance Agent"
        assert "skills" in card_dict
        assert "capabilities" in card_dict

    @pytest.mark.asyncio
    async def test_analyze(self, agent):
        """Test direct analysis method."""
        result = await agent.analyze(
            "What are Apple's key financial metrics?",
            ticker="AAPL"
        )

        assert len(result) > 0
        # Should have some content about the company
        assert "AAPL" in result or "Apple" in result

    @pytest.mark.asyncio
    async def test_get_stock_data(self, agent):
        """Test getting stock data via MCP servers."""
        data = await agent.get_stock_data("MSFT")

        assert data["ticker"] == "MSFT"
        # MCP toolkit returns quote, statistics, company_info structure
        assert "quote" in data or "statistics" in data

    def test_check_earnings_beat(self, agent):
        """Test earnings beat/miss calculation."""
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            agent.check_earnings_beat(
                ticker="NVDA",
                actual_revenue=57_000_000_000,
                actual_eps=1.30,
                expected_revenue=54_920_000_000,
                expected_eps=1.25,
            )
        )

        assert result["ticker"] == "NVDA"
        assert result["revenue_beat"]["beat"] is True
        assert result["eps_beat"]["beat"] is True
        assert result["overall_assessment"] == "Beat"

    def test_check_earnings_miss(self, agent):
        """Test earnings miss calculation."""
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            agent.check_earnings_beat(
                ticker="TEST",
                actual_revenue=50_000_000_000,
                actual_eps=1.00,
                expected_revenue=55_000_000_000,
                expected_eps=1.20,
            )
        )

        assert result["revenue_beat"]["beat"] is False
        assert result["eps_beat"]["beat"] is False
        assert result["overall_assessment"] == "Miss"

    def test_check_earnings_mixed(self, agent):
        """Test mixed earnings result."""
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            agent.check_earnings_beat(
                ticker="TEST",
                actual_revenue=57_000_000_000,  # Beat
                actual_eps=1.00,  # Miss
                expected_revenue=55_000_000_000,
                expected_eps=1.20,
            )
        )

        assert result["revenue_beat"]["beat"] is True
        assert result["eps_beat"]["beat"] is False
        assert result["overall_assessment"] == "Mixed"


class TestAgentFactory:
    """Tests for agent factory function (uses in-process MCP servers)."""

    @pytest.mark.asyncio
    async def test_create_agent_no_api_keys(self):
        """Test creating agent without API keys."""
        agent = await create_agent()

        assert agent is not None
        assert agent.llm_client is None

    @pytest.mark.asyncio
    async def test_create_agent_with_simulation_date(self):
        """Test creating agent with simulation date."""
        sim_date = datetime(2025, 11, 20)
        agent = await create_agent(simulation_date=sim_date)

        assert agent.simulation_date == sim_date
        assert agent.toolkit.simulation_date == sim_date


class TestA2AIntegration:
    """Integration tests for A2A protocol."""

    @pytest.mark.asyncio
    async def test_server_creation(self):
        """Test A2A server can be created."""
        from purple_agent.server import create_app

        app = create_app()

        assert app is not None
        # Check routes exist
        routes = [route.path for route in app.routes]
        assert "/health" in routes
        # A2A agent card endpoint (may be agent.json or agent-card.json)
        assert any("agent" in r and ".json" in r for r in routes)
        assert "/analyze" in routes


class TestNVIDIAScenario:
    """
    End-to-end test scenario using NVIDIA Q3 FY2026 data.

    This tests the Purple Agent's ability to analyze earnings
    with real financial data.
    """

    # Real NVIDIA Q3 FY2026 data (quarter ended October 26, 2025)
    NVIDIA_Q3_FY2026 = {
        "ticker": "NVDA",
        "fiscal_year": 2026,
        "quarter": 3,
        "actual_revenue": 57_000_000_000,
        "actual_eps": 1.30,
        "expected_revenue": 54_920_000_000,
        "expected_eps": 1.25,
        "data_center_revenue": 51_200_000_000,
        "gross_margin": 0.734,
    }

    @pytest.fixture
    def agent(self):
        """Create agent for testing (uses in-process MCP servers)."""
        return FinanceAnalysisAgent(
            simulation_date=datetime(2025, 11, 20),
        )

    @pytest.mark.asyncio
    async def test_nvidia_beat_detection(self, agent):
        """Test NVIDIA Q3 FY2026 beat detection."""
        result = await agent.check_earnings_beat(
            ticker=self.NVIDIA_Q3_FY2026["ticker"],
            actual_revenue=self.NVIDIA_Q3_FY2026["actual_revenue"],
            actual_eps=self.NVIDIA_Q3_FY2026["actual_eps"],
            expected_revenue=self.NVIDIA_Q3_FY2026["expected_revenue"],
            expected_eps=self.NVIDIA_Q3_FY2026["expected_eps"],
        )

        # NVIDIA beat on both metrics
        assert result["overall_assessment"] == "Beat"
        assert result["revenue_beat"]["beat"] is True
        assert result["eps_beat"]["beat"] is True

        # Check magnitude of beat
        revenue_beat_pct = result["revenue_beat"]["difference_pct"]
        assert revenue_beat_pct > 0  # Should be positive (beat)

    @pytest.mark.asyncio
    async def test_nvidia_analysis_question(self, agent):
        """Test analyzing NVIDIA earnings question."""
        question = (
            "Did NVIDIA beat or miss analyst expectations in Q3 FY2026? "
            "What were the key drivers of performance?"
        )

        analysis = await agent.analyze(question, ticker="NVDA")

        # Should contain relevant information
        assert len(analysis) > 100  # Substantive response
        # Should mention NVIDIA or the ticker
        assert "NVDA" in analysis or "NVIDIA" in analysis.upper()


class TestToolCallLogging:
    """Tests for tool call logging functionality in executor."""

    def test_init_tool_call_log(self):
        """Test that _init_tool_call_log creates empty list."""
        executor = FinanceAgentExecutor(llm_client=None, model="test")
        executor._init_tool_call_log()
        
        assert hasattr(executor, '_tool_call_log')
        assert executor._tool_call_log == []

    def test_log_tool_call_basic(self):
        """Test logging a basic tool call."""
        executor = FinanceAgentExecutor(llm_client=None, model="test")
        executor._init_tool_call_log()
        
        executor._log_tool_call(
            tool_name="get_quote",
            args={"ticker": "AAPL"},
            result={"price": 185.50, "volume": 1000000},
            elapsed_ms=150
        )
        
        log = executor.get_tool_call_log()
        assert len(log) == 1
        assert log[0]["tool"] == "get_quote"
        assert log[0]["params"] == {"ticker": "AAPL"}
        assert "185.5" in log[0]["result"]  # JSON serializes 185.50 as 185.5
        assert log[0]["is_error"] is False
        assert log[0]["elapsed_ms"] == 150
        assert "timestamp" in log[0]

    def test_log_tool_call_with_error(self):
        """Test logging a tool call that returned an error."""
        executor = FinanceAgentExecutor(llm_client=None, model="test")
        executor._init_tool_call_log()
        
        executor._log_tool_call(
            tool_name="execute_python",
            args={"code": "print(x)"},
            result={"success": False, "error": "NameError: name 'x' is not defined"},
            elapsed_ms=5
        )
        
        log = executor.get_tool_call_log()
        assert len(log) == 1
        assert log[0]["is_error"] is True

    def test_log_tool_call_result_truncation(self):
        """Test that large results are truncated to 3000 chars."""
        executor = FinanceAgentExecutor(llm_client=None, model="test")
        executor._init_tool_call_log()
        
        # Create a large result (> 3000 chars)
        large_result = {"data": "x" * 5000}
        
        executor._log_tool_call(
            tool_name="get_financials",
            args={"ticker": "AAPL"},
            result=large_result,
            elapsed_ms=200
        )
        
        log = executor.get_tool_call_log()
        assert len(log[0]["result"]) <= 3020  # 3000 + "[truncated]"
        assert log[0]["result"].endswith("...[truncated]")

    def test_log_multiple_tool_calls(self):
        """Test logging multiple tool calls."""
        executor = FinanceAgentExecutor(llm_client=None, model="test")
        executor._init_tool_call_log()
        
        executor._log_tool_call("get_quote", {"ticker": "AAPL"}, {"price": 185}, 100)
        executor._log_tool_call("get_quote", {"ticker": "MSFT"}, {"price": 420}, 110)
        executor._log_tool_call("execute_python", {"code": "1+1"}, {"result": 2}, 5)
        
        log = executor.get_tool_call_log()
        assert len(log) == 3
        assert log[0]["tool"] == "get_quote"
        assert log[1]["params"]["ticker"] == "MSFT"
        assert log[2]["tool"] == "execute_python"

    def test_get_tool_call_log_empty(self):
        """Test get_tool_call_log returns empty list when not initialized."""
        executor = FinanceAgentExecutor(llm_client=None, model="test")
        
        # Should return empty list even if _init not called
        log = executor.get_tool_call_log()
        assert log == []


class TestToolCallModel:
    """Tests for ToolCall model with result field."""

    def test_tool_call_with_result(self):
        """Test ToolCall model includes result field."""
        from cio_agent.models import ToolCall
        
        tc = ToolCall(
            tool_name="get_quote",
            params={"ticker": "AAPL"},
            timestamp=datetime.now(),
            duration_ms=150,
            success=True,
            result='{"price": 185.50}'
        )
        
        assert tc.result == '{"price": 185.50}'
        assert tc.tool_name == "get_quote"
        assert tc.success is True

    def test_tool_call_result_optional(self):
        """Test ToolCall result field is optional."""
        from cio_agent.models import ToolCall
        
        tc = ToolCall(
            tool_name="get_quote",
            params={"ticker": "AAPL"},
            timestamp=datetime.now(),
        )
        
        assert tc.result is None


class TestMessengerToolCalls:
    """Tests for messenger tool calls extraction."""

    def test_get_last_tool_calls_empty(self):
        """Test get_last_tool_calls returns empty list when no calls made."""
        from cio_agent.messenger import Messenger
        
        messenger = Messenger()
        tool_calls = messenger.get_last_tool_calls()
        
        assert tool_calls == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
