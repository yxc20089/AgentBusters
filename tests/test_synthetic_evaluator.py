"""
Tests for SyntheticEvaluator.

Tests cover:
- Rubric-based evaluation with LLM
- Fallback evaluation without LLM
- Numerical matching with tolerance
- Component scoring
- Edge cases (empty response, missing rubric, etc.)
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from evaluators.synthetic_evaluator import SyntheticEvaluator
from evaluators.base import EvalResult


class TestSyntheticEvaluatorInit:
    """Test SyntheticEvaluator initialization."""

    def test_init_default(self):
        """Test default initialization."""
        evaluator = SyntheticEvaluator()
        assert evaluator.name == "synthetic"
        assert evaluator.use_llm is True
        assert evaluator.llm_client is None

    def test_init_no_llm(self):
        """Test initialization without LLM."""
        evaluator = SyntheticEvaluator(use_llm=False)
        assert evaluator.use_llm is False

    def test_init_with_llm_client(self):
        """Test initialization with LLM client."""
        mock_client = Mock()
        evaluator = SyntheticEvaluator(use_llm=True, llm_client=mock_client)
        assert evaluator.llm_client is mock_client


class TestSyntheticEvaluatorFallback:
    """Test fallback evaluation (without LLM)."""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator without LLM."""
        return SyntheticEvaluator(use_llm=False)

    def test_empty_response(self, evaluator):
        """Test evaluation with empty response."""
        result = evaluator.evaluate(predicted="", expected="20%")
        assert result.score == 0.0
        assert "empty" in result.feedback.lower() or "no meaningful" in result.feedback.lower()
        assert result.details.get("llm_used") is False

    def test_short_response(self, evaluator):
        """Test evaluation with very short response."""
        result = evaluator.evaluate(predicted="Hi", expected="20%")
        assert result.score == 0.0

    def test_exact_match(self, evaluator):
        """Test exact string match."""
        result = evaluator.evaluate(
            predicted="The answer is 20%",
            expected="20%"
        )
        assert result.score == 1.0
        assert "expected answer" in result.feedback.lower() or "found" in result.feedback.lower()

    def test_numerical_match_percentage(self, evaluator):
        """Test numerical matching with percentages."""
        result = evaluator.evaluate(
            predicted="The discount rate is 19.5%",
            expected="20%"
        )
        # 19.5 is within 10% of 20
        assert result.score >= 0.8

    def test_numerical_match_currency(self, evaluator):
        """Test numerical matching with currency."""
        result = evaluator.evaluate(
            predicted="The profit is $1,250",
            expected="$1,300"
        )
        # 1250 is within 10% of 1300
        assert result.score >= 0.8

    def test_numerical_no_match(self, evaluator):
        """Test when numbers don't match."""
        result = evaluator.evaluate(
            predicted="The answer is $500",
            expected="$1,000"
        )
        # 500 is NOT within 10% of 1000
        assert result.score < 0.8

    def test_ground_truth_value_dict(self, evaluator):
        """Test with structured ground truth value."""
        result = evaluator.evaluate(
            predicted="Yes, there is an arbitrage opportunity. Profit is $1,250.",
            expected="Yes, profit ~$1,273",
            ground_truth_value={"opportunity": True, "profit": 1272.73}
        )
        # Should match "yes" and approximate profit
        assert result.score >= 0.5

    def test_ground_truth_value_numeric(self, evaluator):
        """Test with numeric ground truth value."""
        result = evaluator.evaluate(
            predicted="The value is approximately 7,200",
            expected="$7,273",
            ground_truth_value=7272.73
        )
        assert result.score >= 0.8

    def test_no_ground_truth(self, evaluator):
        """Test with no ground truth."""
        result = evaluator.evaluate(
            predicted="Some response here that is long enough",
            expected="",
            ground_truth_value=None
        )
        assert result.score == 0.5  # Neutral score
        assert "no ground truth" in result.feedback.lower()


class TestSyntheticEvaluatorRubric:
    """Test rubric-based evaluation."""

    @pytest.fixture
    def evaluator_no_llm(self):
        """Create evaluator without LLM."""
        return SyntheticEvaluator(use_llm=False)

    @pytest.fixture
    def sample_rubric(self):
        """Sample rubric for testing."""
        return {
            "components": [
                {
                    "name": "methodology",
                    "description": "Set up NPV equality equation correctly",
                    "weight": 0.3
                },
                {
                    "name": "algebra",
                    "description": "Correctly solved for discount rate",
                    "weight": 0.3
                },
                {
                    "name": "answer",
                    "description": "Final answer is 20%",
                    "expected_value": "20%",
                    "weight": 0.4
                }
            ],
            "max_score": 100
        }

    def test_rubric_fallback_to_expected(self, evaluator_no_llm, sample_rubric):
        """Test that rubric evaluation falls back when no LLM."""
        result = evaluator_no_llm.evaluate(
            predicted="The crossover rate is 20%",
            expected="20%",
            rubric=sample_rubric,
            question="At what discount rate are the two projects equal?"
        )
        # Should use fallback and find 20%
        assert result.score >= 0.8
        assert result.details.get("llm_used") is False

    def test_empty_rubric_components(self, evaluator_no_llm):
        """Test with empty rubric components."""
        result = evaluator_no_llm.evaluate(
            predicted="The answer is 20%",
            expected="20%",
            rubric={"components": [], "max_score": 100}
        )
        # Should fall back to expected comparison
        assert result.score == 1.0


class TestSyntheticEvaluatorLLM:
    """Test LLM-based evaluation."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        return Mock()

    @pytest.fixture
    def evaluator_with_llm(self, mock_llm_client):
        """Create evaluator with mock LLM."""
        return SyntheticEvaluator(
            use_llm=True,
            llm_client=mock_llm_client,
            llm_model="gpt-4o-mini",
            llm_temperature=0.0
        )

    def test_llm_evaluation_success(self, evaluator_with_llm, mock_llm_client):
        """Test successful LLM evaluation."""
        rubric = {
            "components": [
                {"name": "formula", "description": "Correct formula", "weight": 0.5},
                {"name": "calculation", "description": "Correct calculation", "weight": 0.5}
            ]
        }

        # Mock LLM response
        llm_response = json.dumps({
            "component_scores": {
                "formula": {"score": 1.0, "reasoning": "Correct formula used"},
                "calculation": {"score": 0.8, "reasoning": "Minor rounding error"}
            },
            "overall_feedback": "Good response with minor issues"
        })

        with patch("evaluators.synthetic_evaluator.call_llm", return_value=llm_response):
            result = evaluator_with_llm.evaluate(
                predicted="F = S * (1 + r_d) / (1 + r_f) = 1.1214",
                expected="1.1214",
                question="Calculate the forward rate",
                rubric=rubric
            )

        assert result.score == pytest.approx(0.9, abs=0.01)  # (1.0 * 0.5 + 0.8 * 0.5)
        assert result.details.get("llm_used") is True
        assert "component_scores" in result.details
        assert result.details["component_scores"]["formula"]["score"] == 1.0

    def test_llm_evaluation_failure_fallback(self, evaluator_with_llm):
        """Test fallback when LLM fails."""
        rubric = {
            "components": [
                {"name": "answer", "description": "Correct answer", "weight": 1.0}
            ]
        }

        with patch("evaluators.synthetic_evaluator.call_llm", return_value=None):
            result = evaluator_with_llm.evaluate(
                predicted="The answer is 20%",
                expected="20%",
                rubric=rubric
            )

        # Should fall back to heuristic
        assert result.score >= 0.8
        assert result.details.get("llm_used") is False

    def test_llm_invalid_json_fallback(self, evaluator_with_llm):
        """Test fallback when LLM returns invalid JSON."""
        rubric = {
            "components": [
                {"name": "answer", "description": "Correct answer", "weight": 1.0}
            ]
        }

        with patch("evaluators.synthetic_evaluator.call_llm", return_value="Invalid response"):
            result = evaluator_with_llm.evaluate(
                predicted="The answer is 20%",
                expected="20%",
                rubric=rubric
            )

        # Should fall back to heuristic
        assert result.details.get("llm_used") is False


class TestNumberExtraction:
    """Test number extraction utilities."""

    @pytest.fixture
    def evaluator(self):
        return SyntheticEvaluator(use_llm=False)

    def test_extract_percentage(self, evaluator):
        """Test extracting percentages."""
        nums = evaluator._extract_numbers("The rate is 20%")
        assert 0.20 in nums  # Converted to decimal

    def test_extract_currency(self, evaluator):
        """Test extracting currency values."""
        nums = evaluator._extract_numbers("Profit is $1,234.56")
        assert 1234.56 in nums

    def test_extract_large_number(self, evaluator):
        """Test extracting large numbers with commas."""
        nums = evaluator._extract_numbers("Revenue: $1,234,567.89")
        assert 1234567.89 in nums

    def test_extract_multiple_numbers(self, evaluator):
        """Test extracting multiple numbers."""
        nums = evaluator._extract_numbers("NPV is $100 at 5% and $200 at 10%")
        assert 100.0 in nums
        assert 200.0 in nums
        assert 0.05 in nums
        assert 0.10 in nums

    def test_extract_scientific_notation(self, evaluator):
        """Test extracting scientific notation."""
        nums = evaluator._extract_numbers("Value is 1.5e6")
        assert 1500000.0 in nums


class TestNumberComparison:
    """Test number comparison utilities."""

    @pytest.fixture
    def evaluator(self):
        return SyntheticEvaluator(use_llm=False)

    def test_numbers_close_exact(self, evaluator):
        """Test exact match."""
        assert evaluator._numbers_close(100, 100) is True

    def test_numbers_close_within_tolerance(self, evaluator):
        """Test within tolerance."""
        assert evaluator._numbers_close(100, 105, tolerance=0.10) is True
        assert evaluator._numbers_close(100, 95, tolerance=0.10) is True

    def test_numbers_close_outside_tolerance(self, evaluator):
        """Test outside tolerance."""
        assert evaluator._numbers_close(100, 120, tolerance=0.10) is False

    def test_numbers_close_zero(self, evaluator):
        """Test with zero values."""
        assert evaluator._numbers_close(0, 0) is True
        assert evaluator._numbers_close(0, 0.01, tolerance=0.05) is True


class TestCoveredInterestArbitrage:
    """Test the specific SYN_LOGIC_0025 case that was failing."""

    @pytest.fixture
    def evaluator(self):
        return SyntheticEvaluator(use_llm=False)

    def test_arbitrage_calculation_correct(self, evaluator):
        """Test correct arbitrage calculation is recognized."""
        predicted = """
        **Covered Interest Rate Parity Check:**
        - Theoretical forward rate = Spot × (1 + r_USD) / (1 + r_EUR) = 1.10 × 1.05 / 1.03 = **1.12136**
        - Market forward rate = **1.12**
        
        **Arbitrage Opportunity:**
        - Market forward (1.12) < Theoretical forward (1.12136)
        - EUR is **undervalued in the forward market**
        
        **Arbitrage Strategy:**
        1. **Borrow EUR** at 3% interest rate
        2. **Convert to USD** at spot rate 1.10
        3. **Invest USD** at 5% for 1 year
        4. **Enter forward contract** to buy EUR at 1.12
        
        **Profit Calculation per $1 million:**
        - **Absolute profit:** $1,250 (or €1,136.36)
        """

        result = evaluator.evaluate(
            predicted=predicted,
            expected="Yes, profit of approximately $7,273 per $1M",
            ground_truth_value={"opportunity": True, "profit": 7272.73}
        )

        # Should get a reasonable score even without exact match
        # The response shows correct methodology even if exact profit differs
        assert result.score >= 0.5
        assert result.details.get("llm_used") is False

    def test_arbitrage_with_rubric_llm(self):
        """Test arbitrage with rubric using LLM."""
        rubric = {
            "components": [
                {"name": "cip_formula", "description": "Correctly stated covered interest parity formula", "weight": 0.2},
                {"name": "fair_forward", "description": "Calculated fair forward rate ~1.1214", "weight": 0.2},
                {"name": "arbitrage_direction", "description": "Identified correct direction of arbitrage", "weight": 0.3},
                {"name": "profit_calc", "description": "Calculated profit (varies based on exact calculation)", "expected_value": "$1K-$10K range", "weight": 0.3}
            ],
            "max_score": 100
        }

        predicted = """
        Theoretical forward = 1.10 * 1.05 / 1.03 = 1.12136
        Since market forward (1.12) < theoretical (1.12136), EUR is undervalued forward.
        Strategy: Borrow EUR, convert to USD, invest at 5%, buy EUR forward.
        Profit ≈ $1,250 per $1M
        """

        llm_response = json.dumps({
            "component_scores": {
                "cip_formula": {"score": 1.0, "reasoning": "Correct formula"},
                "fair_forward": {"score": 1.0, "reasoning": "Calculated 1.12136 correctly"},
                "arbitrage_direction": {"score": 1.0, "reasoning": "Correct direction identified"},
                "profit_calc": {"score": 0.8, "reasoning": "Profit in reasonable range but differs from ground truth"}
            },
            "overall_feedback": "Excellent analysis with correct methodology"
        })

        evaluator = SyntheticEvaluator(use_llm=True, llm_client=Mock())

        with patch("evaluators.synthetic_evaluator.call_llm", return_value=llm_response):
            result = evaluator.evaluate(
                predicted=predicted,
                expected="Yes, profit of approximately $7,273 per $1M",
                question="Is there a covered interest arbitrage opportunity?",
                rubric=rubric,
                ground_truth_value={"opportunity": True, "profit": 7272.73}
            )

        # Weighted score: 1.0*0.2 + 1.0*0.2 + 1.0*0.3 + 0.8*0.3 = 0.2 + 0.2 + 0.3 + 0.24 = 0.94
        assert result.score == pytest.approx(0.94, abs=0.01)
        assert result.score >= 0.7  # Threshold for "correct" in GreenAgent
        assert result.details.get("llm_used") is True


class TestIntegrationWithGreenAgent:
    """Test integration patterns used by GreenAgent."""

    @pytest.fixture
    def evaluator(self):
        return SyntheticEvaluator(use_llm=False)

    def test_metadata_extraction(self, evaluator):
        """Test evaluation with metadata similar to GreenAgent."""
        # Simulate how GreenAgent calls the evaluator
        metadata = {
            "rubric": {
                "components": [
                    {"name": "answer", "description": "Correct NPV crossover rate", "expected_value": "20%", "weight": 1.0}
                ]
            },
            "ground_truth_value": 0.2,
            "calculation_steps": [
                "Set NPV_A = NPV_B",
                "Solve for r = 20%"
            ],
            "category": "Financial Logic"
        }

        result = evaluator.evaluate(
            predicted="The crossover rate is 20%",
            expected="20%",
            question="At what discount rate are the two projects equally attractive?",
            rubric=metadata.get("rubric"),
            ground_truth_value=metadata.get("ground_truth_value"),
            calculation_steps=metadata.get("calculation_steps"),
            category=metadata.get("category"),
        )

        assert result.score >= 0.8
        assert isinstance(result, EvalResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
