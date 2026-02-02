"""
Unit tests for dataset-specific evaluators.
"""

import pytest
from evaluators.base import EvalResult
from evaluators.bizfinbench_evaluator import BizFinBenchEvaluator
from evaluators.prbench_evaluator import PRBenchEvaluator


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

