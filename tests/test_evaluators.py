"""
Unit tests for dataset-specific evaluators.
"""

import pytest
from evaluators.base import EvalResult
from evaluators.bizfinbench_evaluator import BizFinBenchEvaluator
from evaluators.prbench_evaluator import PRBenchEvaluator
from evaluators.options import OptionsEvaluator, OptionsScore


class TestOptionsEvaluatorImport:
    """Test OptionsEvaluator import and basic functionality.
    
    This tests the fix for: name 're' is not defined
    The import re was moved to the top of the file.
    """

    def test_options_evaluator_import(self):
        """Test that OptionsEvaluator can be imported without errors."""
        # This import should not raise "name 're' is not defined"
        from evaluators.options import OptionsEvaluator
        assert OptionsEvaluator is not None

    def test_options_evaluator_extract_numbers(self):
        """Test that _extract_numbers_from_text works (uses re module)."""
        from evaluators.options import OptionsEvaluator
        from cio_agent.models import Task, GroundTruth, TaskCategory, TaskRubric
        from datetime import datetime, timezone
        
        # Create a minimal task for the evaluator
        task = Task(
            question_id="test_001",
            category=TaskCategory.OPTIONS_PRICING,
            question="Test question",
            ticker="AAPL",
            fiscal_year=2024,
            simulation_date=datetime.now(timezone.utc),
            ground_truth=GroundTruth(macro_thesis="test"),
            rubric=TaskRubric(),
        )
        
        evaluator = OptionsEvaluator(task=task)
        
        # Test that _extract_numbers_from_text works (uses re.findall)
        numbers = evaluator._extract_numbers_from_text("Price is $150.50 with delta 0.45")
        assert 150.50 in numbers
        assert 0.45 in numbers


class TestEvalResult:
    """Test EvalResult dataclass."""

    def test_percentage(self):
        result = EvalResult(score=0.8, max_score=1.0)
        assert result.percentage == 80.0

    def test_is_correct_true(self):
        result = EvalResult(score=1.0, max_score=1.0)
        assert result.is_correct is True

    def test_is_correct_false(self):
        result = EvalResult(score=0.5, max_score=1.0)
        assert result.is_correct is False


class TestBizFinBenchEvaluator:
    """Test BizFinBenchEvaluator."""

    @pytest.fixture
    def evaluator(self):
        return BizFinBenchEvaluator()

    # Numerical tests
    def test_numerical_exact_match(self, evaluator):
        result = evaluator.evaluate("1.2532", "1.2532", task_type="financial_quantitative_computation")
        assert result.score == 1.0
        assert result.is_correct

    def test_numerical_within_tolerance(self, evaluator):
        # 1% tolerance: 1.2532 * 0.01 = 0.0125, so 1.26 should pass
        result = evaluator.evaluate("1.26", "1.2532", task_type="financial_quantitative_computation")
        assert result.score == 1.0

    def test_numerical_outside_tolerance(self, evaluator):
        # 1.30 is > 1% away from 1.2532
        result = evaluator.evaluate("1.30", "1.2532", task_type="financial_quantitative_computation")
        assert result.score == 0.0

    def test_numerical_with_text(self, evaluator):
        result = evaluator.evaluate("The answer is 1.2532", "1.2532", task_type="financial_quantitative_computation")
        assert result.score == 1.0

    # Sequence tests
    def test_sequence_exact_match(self, evaluator):
        result = evaluator.evaluate("2,1,4,3", "2,1,4,3", task_type="event_logic_reasoning")
        assert result.score == 1.0
        assert result.is_correct

    def test_sequence_wrong_order(self, evaluator):
        result = evaluator.evaluate("1,2,3,4", "2,1,4,3", task_type="event_logic_reasoning")
        assert result.score == 0.0

    def test_sequence_with_spaces(self, evaluator):
        result = evaluator.evaluate("2, 1, 4, 3", "2,1,4,3", task_type="event_logic_reasoning")
        assert result.score == 1.0

    # Classification tests
    def test_classification_match(self, evaluator):
        result = evaluator.evaluate("positive", "positive", task_type="user_sentiment_analysis")
        assert result.score == 1.0

    def test_classification_case_insensitive(self, evaluator):
        result = evaluator.evaluate("POSITIVE", "positive", task_type="user_sentiment_analysis")
        assert result.score == 1.0

    def test_classification_mismatch(self, evaluator):
        result = evaluator.evaluate("negative", "positive", task_type="user_sentiment_analysis")
        assert result.score == 0.0

    # JSON list match tests (financial_data_description task type)
    def test_json_list_match_exact(self, evaluator):
        """Test exact match of ID lists."""
        result = evaluator.evaluate('{"answer": [1, 3, 5]}', '{"answer": [1, 3, 5]}', task_type="financial_data_description")
        assert result.score == 1.0
        assert result.is_correct

    def test_json_list_match_different_order(self, evaluator):
        """Test ID lists match regardless of order (set comparison)."""
        result = evaluator.evaluate('{"answer": [5, 1, 3]}', '{"answer": [1, 3, 5]}', task_type="financial_data_description")
        assert result.score == 1.0

    def test_json_list_match_bare_list(self, evaluator):
        """Test bare JSON list format."""
        result = evaluator.evaluate('[1, 3, 5]', '[1, 3, 5]', task_type="financial_data_description")
        assert result.score == 1.0

    def test_json_list_match_mismatch(self, evaluator):
        """Test mismatched ID lists."""
        result = evaluator.evaluate('{"answer": [1, 3]}', '{"answer": [1, 3, 5]}', task_type="financial_data_description")
        assert result.score == 0.0

    def test_json_list_match_embedded_in_text(self, evaluator):
        """Test extraction of JSON from longer text."""
        predicted = 'Based on my analysis, the errors are: {"answer": [2, 4, 6]}'
        expected = '{"answer": [2, 4, 6]}'
        result = evaluator.evaluate(predicted, expected, task_type="financial_data_description")
        assert result.score == 1.0

    def test_json_list_match_with_invalid_values(self, evaluator):
        """Test handling of non-integer values in list (should be skipped)."""
        # Partial extraction - only valid integers should be extracted
        result = evaluator.evaluate('[1, "invalid", 3]', '[1, 3]', task_type="financial_data_description")
        assert result.score == 1.0

    def test_json_list_match_empty_lists(self, evaluator):
        """Test empty ID lists."""
        result = evaluator.evaluate('[]', '[]', task_type="financial_data_description")
        assert result.score == 1.0

    def test_json_list_match_bracket_fallback(self, evaluator):
        """Test fallback to bracket extraction."""
        result = evaluator.evaluate('The IDs are [1, 2, 3]', '[1, 2, 3]', task_type="financial_data_description")
        assert result.score == 1.0

    # Edge cases
    def test_empty_prediction(self, evaluator):
        result = evaluator.evaluate("", "1.2532", task_type="financial_quantitative_computation")
        assert result.score == 0.0

    def test_empty_expected(self, evaluator):
        result = evaluator.evaluate("1.2532", "", task_type="financial_quantitative_computation")
        assert result.score == 0.0


class TestPRBenchEvaluator:
    """Test PRBenchEvaluator."""

    @pytest.fixture
    def evaluator(self):
        return PRBenchEvaluator(use_llm=False)

    def test_scratchpad_exact_match(self, evaluator):
        """Test fallback to scratchpad comparison when no rubric."""
        result = evaluator.evaluate(
            "The answer is Q3 2024 revenue",
            "The answer is Q3 2024 revenue",
            rubric=None
        )
        assert result.score == 1.0

    def test_scratchpad_partial_match(self, evaluator):
        """Test partial scratchpad match."""
        result = evaluator.evaluate(
            "Revenue was strong",
            "Revenue was strong in Q3",
            rubric=None
        )
        # Partial match gives partial credit
        assert 0 < result.score <= 1.0

    def test_scratchpad_no_match(self, evaluator):
        """Test no scratchpad match."""
        result = evaluator.evaluate(
            "Something completely different",
            "Expected answer about revenue",
            rubric=None
        )
        assert result.score < 1.0

    def test_empty_rubric(self, evaluator):
        result = evaluator.evaluate("Some answer", "Expected", rubric=None)
        # Falls back to simple comparison
        assert result.score >= 0.0

    def test_empty_prediction(self, evaluator):
        """Test empty prediction returns zero score."""
        result = evaluator.evaluate("", "Expected answer", rubric=None)
        assert result.score == 0.0


class TestBizFinBenchEvaluatorBatch:
    """Test batch evaluation."""

    def test_batch_evaluate(self):
        evaluator = BizFinBenchEvaluator()
        predictions = ["1.0", "2.0", "3.0"]
        expected = ["1.0", "2.0", "4.0"]
        
        results = evaluator.evaluate_batch(
            predictions, expected,
            task_type="financial_quantitative_computation"
        )
        
        assert len(results) == 3
        assert results[0].score == 1.0
        assert results[1].score == 1.0
        assert results[2].score == 0.0

    def test_aggregate_results(self):
        evaluator = BizFinBenchEvaluator()
        results = [
            EvalResult(score=1.0, max_score=1.0),
            EvalResult(score=1.0, max_score=1.0),
            EvalResult(score=0.0, max_score=1.0),
        ]
        
        agg = evaluator.aggregate_results(results)
        
        assert agg["count"] == 3
        assert agg["correct_count"] == 2
        assert agg["accuracy"] == 2/3


class TestBizFinBenchEvaluatorEdgeCases:
    """Test BizFinBench error cases."""

    @pytest.fixture
    def evaluator(self):
        return BizFinBenchEvaluator()

    def test_no_extractable_number(self, evaluator):
        """Test prediction with no extractable number."""
        result = evaluator.evaluate("no numbers here", "1.2532", task_type="financial_quantitative_computation")
        assert result.score == 0.0
        assert "Could not extract" in result.feedback

    def test_malformed_sequence(self, evaluator):
        """Test malformed sequence input."""
        result = evaluator.evaluate("abc", "2,1,4,3", task_type="event_logic_reasoning")
        assert result.score == 0.0

    def test_whitespace_only_prediction(self, evaluator):
        """Test whitespace-only prediction."""
        result = evaluator.evaluate("   ", "answer", task_type="user_sentiment_analysis")
        assert result.score == 0.0


class TestPRBenchEvaluatorEdgeCases:
    """Test PRBench edge cases."""

    @pytest.fixture
    def evaluator(self):
        return PRBenchEvaluator(use_llm=False)

    def test_case_insensitive_match(self, evaluator):
        """Test that matching is case insensitive."""
        result = evaluator.evaluate(
            "REVENUE GROWTH WAS STRONG",
            "Revenue growth was strong",
            rubric=None
        )
        assert result.score > 0.5

    def test_whitespace_handling(self, evaluator):
        """Test handling of whitespace variations."""
        result = evaluator.evaluate(
            "  The   answer  is  here  ",
            "The answer is here",
            rubric=None
        )
        assert result.score > 0.0

    def test_empty_expected(self, evaluator):
        """Test empty expected value."""
        result = evaluator.evaluate("Some answer", "", rubric=None)
        assert result.score >= 0.0


class TestOptionsGreeksExtraction:
    """Test Greeks extraction from various response formats."""

    @pytest.fixture
    def evaluator(self):
        """Create an OptionsEvaluator instance for testing."""
        from cio_agent.models import Task, GroundTruth, TaskCategory, TaskRubric
        from datetime import datetime, timezone
        
        task = Task(
            question_id="test_greeks",
            category=TaskCategory.GREEKS_ANALYSIS,
            question="Calculate Greeks",
            ticker="SPY",
            fiscal_year=2024,
            simulation_date=datetime.now(timezone.utc),
            ground_truth=GroundTruth(macro_thesis="test"),
            rubric=TaskRubric(),
        )
        return OptionsEvaluator(task=task)

    def test_extract_delta_colon_format(self, evaluator):
        """Test extraction of Delta: 0.42 format."""
        text = "The option has Delta: 0.42 and Gamma: 0.025"
        assert evaluator._extract_greek_value(text, "delta") == 0.42
        assert evaluator._extract_greek_value(text, "gamma") == 0.025

    def test_extract_delta_equals_format(self, evaluator):
        """Test extraction of delta = 0.42 format."""
        text = "delta = 0.42, gamma = 0.025, theta = -0.35"
        assert evaluator._extract_greek_value(text, "delta") == 0.42
        assert evaluator._extract_greek_value(text, "gamma") == 0.025
        assert evaluator._extract_greek_value(text, "theta") == -0.35

    def test_extract_delta_of_format(self, evaluator):
        """Test extraction of 'delta of 0.5' format (common in LLM responses)."""
        text = "A delta of 0.5 means that for every $1 increase in price"
        assert evaluator._extract_greek_value(text, "delta") == 0.5

    def test_extract_delta_is_format(self, evaluator):
        """Test extraction of 'delta is 0.42' format."""
        text = "The calculated delta is 0.42 for this option"
        assert evaluator._extract_greek_value(text, "delta") == 0.42

    def test_extract_markdown_bold_format(self, evaluator):
        """Test extraction from **Delta**: 0.42 markdown format."""
        text = """
        ### Greeks:
        - **Delta**: 0.45
        - **Gamma**: 0.03
        - **Theta**: -0.25
        - **Vega**: 0.18
        """
        assert evaluator._extract_greek_value(text, "delta") == 0.45
        assert evaluator._extract_greek_value(text, "gamma") == 0.03
        assert evaluator._extract_greek_value(text, "theta") == -0.25
        assert evaluator._extract_greek_value(text, "vega") == 0.18

    def test_extract_negative_theta(self, evaluator):
        """Test extraction of negative theta values."""
        text = "theta: -0.35 per day"
        assert evaluator._extract_greek_value(text, "theta") == -0.35

    def test_extract_bullet_list_format(self, evaluator):
        """Test extraction from bullet list format."""
        text = """
        Option Greeks:
        - Delta: 0.55
        • Gamma: 0.04
        * Theta: -0.30
        """
        assert evaluator._extract_greek_value(text, "delta") == 0.55
        assert evaluator._extract_greek_value(text, "gamma") == 0.04
        assert evaluator._extract_greek_value(text, "theta") == -0.30

    def test_extract_approximately_format(self, evaluator):
        """Test extraction with 'approximately' keyword."""
        text = "The delta is approximately 0.48"
        assert evaluator._extract_greek_value(text, "delta") == 0.48

    def test_extract_around_format(self, evaluator):
        """Test extraction with 'around' keyword."""
        text = "gamma around 0.025 indicates sensitivity"
        assert evaluator._extract_greek_value(text, "gamma") == 0.025

    def test_extract_case_insensitive(self, evaluator):
        """Test case-insensitive extraction."""
        text1 = "DELTA: 0.42"
        text2 = "Delta: 0.42"
        text3 = "delta: 0.42"
        assert evaluator._extract_greek_value(text1, "delta") == 0.42
        assert evaluator._extract_greek_value(text2, "delta") == 0.42
        assert evaluator._extract_greek_value(text3, "delta") == 0.42

    def test_extract_real_llm_response(self, evaluator):
        """Test extraction from a real LLM response format."""
        text = """
        ### Greeks Explanation:

        1. **Delta**:
           - **Interpretation**: A delta of 0.5 means that for every $1 increase 
             in TSLA's stock price, the call option's price is expected to increase 
             by $0.50.

        2. **Gamma**:
           - **Definition**: Gamma measures the rate of change of delta.
           - The gamma is approximately 0.025 for this option.

        3. **Theta**:
           - theta: -0.35 per day

        4. **Vega**:
           - vega = 0.28
        """
        assert evaluator._extract_greek_value(text, "delta") == 0.5
        assert evaluator._extract_greek_value(text, "gamma") == 0.025
        assert evaluator._extract_greek_value(text, "theta") == -0.35
        assert evaluator._extract_greek_value(text, "vega") == 0.28

    def test_extract_semicolon_separated(self, evaluator):
        """Test extraction from semicolon-separated format (like expected values)."""
        text = "delta: 0.42; gamma: 0.025; theta: -0.35; vega: 0.28"
        assert evaluator._extract_greek_value(text, "delta") == 0.42
        assert evaluator._extract_greek_value(text, "gamma") == 0.025
        assert evaluator._extract_greek_value(text, "theta") == -0.35
        assert evaluator._extract_greek_value(text, "vega") == 0.28

    def test_no_greek_found(self, evaluator):
        """Test that None is returned when Greek is not found."""
        text = "This text has no Greeks values"
        assert evaluator._extract_greek_value(text, "delta") is None
        assert evaluator._extract_greek_value(text, "gamma") is None

    @pytest.mark.asyncio
    async def test_extract_options_data_integration(self, evaluator):
        """Test _extract_options_data method integrates Greek extraction correctly."""
        from cio_agent.models import AgentResponse
        
        response = AgentResponse(
            agent_id="test_agent",
            task_id="test_task",
            analysis="""
            For the TSLA $250 call option:
            
            **Greeks:**
            - Delta: 0.42
            - Gamma: 0.025
            - Theta: -0.35
            - Vega: 0.28
            
            The iron condor strategy provides limited risk.
            Max profit: $500
            Max loss: $1,000
            """,
            recommendation="Buy the option",
            confidence=0.8,
        )
        
        # Use regex-only mode to avoid LLM call in tests
        extracted = await evaluator._extract_options_data(response, use_llm_for_greeks=False)
        
        assert extracted.delta == 0.42
        assert extracted.gamma == 0.025
        assert extracted.theta == -0.35
        assert extracted.vega == 0.28
        assert extracted.strategy_name == "iron condor"
        assert extracted.max_profit == 500.0
        assert extracted.max_loss == 1000.0


class TestOptionsLLMGreeksExtraction:
    """Tests for LLM-based Greeks extraction."""
    
    @pytest.fixture
    def evaluator(self):
        """Create an OptionsEvaluator instance for testing."""
        from cio_agent.models import Task, GroundTruth, TaskCategory, TaskRubric
        from datetime import datetime, timezone
        
        task = Task(
            question_id="test_llm_greeks",
            category=TaskCategory.GREEKS_ANALYSIS,
            question="Test LLM Greeks extraction",
            ticker="AAPL",
            fiscal_year=2024,
            simulation_date=datetime.now(timezone.utc),
            ground_truth=GroundTruth(macro_thesis="test"),
            rubric=TaskRubric(),
        )
        return OptionsEvaluator(task=task)
    
    @pytest.mark.asyncio
    async def test_llm_extract_greeks_prompt_format(self, evaluator):
        """Test that _llm_extract_greeks returns expected format when client is None."""
        # Explicitly ensure no LLM client is available (hermetic test)
        evaluator.llm_client = None
        evaluator._get_llm_client = lambda: None  # Stub to prevent env-based client creation
        
        result, error = await evaluator._llm_extract_greeks("Delta: 0.5")
        assert result == {}
        assert error == "llm_client_unavailable"
    
    @pytest.mark.asyncio
    async def test_extract_options_data_with_llm_disabled(self, evaluator):
        """Test extraction with LLM disabled falls back to regex."""
        from cio_agent.models import AgentResponse
        
        response = AgentResponse(
            agent_id="test_agent",
            task_id="test_task",
            analysis="Delta: 0.55, Gamma: 0.03, Theta: -0.25, Vega: 0.15",
            recommendation="Hold",
            confidence=0.7,
        )
        
        # Disable LLM extraction
        evaluator._use_llm_extraction = False
        extracted = await evaluator._extract_options_data(response)
        
        # Should still extract using regex
        assert extracted.delta == 0.55
        assert extracted.gamma == 0.03
        assert extracted.theta == -0.25
        assert extracted.vega == 0.15
    
    @pytest.mark.asyncio
    async def test_extract_options_data_llm_fallback_to_regex(self, evaluator):
        """Test that when LLM fails, it falls back to regex extraction."""
        from cio_agent.models import AgentResponse
        
        response = AgentResponse(
            agent_id="test_agent",
            task_id="test_task",
            analysis="Delta = 0.42; Gamma = 0.025",
            recommendation="Buy",
            confidence=0.8,
        )
        
        # Stub _get_llm_client to ensure no env-based client is created (hermetic test)
        evaluator._use_llm_extraction = True
        evaluator.llm_client = None
        evaluator._get_llm_client = lambda: None
        extracted = await evaluator._extract_options_data(response)
        
        # Should extract via regex fallback
        assert extracted.delta == 0.42
        assert extracted.gamma == 0.025
