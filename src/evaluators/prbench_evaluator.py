"""
PRBench dataset evaluator.

Evaluates predictions against PRBench expert rubrics using weighted criteria.
PRBench rubrics contain 10-30 criteria per task with importance weights:
- critically_important: 3.0
- important: 2.0
- slightly_important: 1.0
- detrimental: -1.0 (penalty for including this)

Evaluation can be:
1. Fast mode: Keyword/substring matching against criteria
2. LLM mode: Use LLM to evaluate each criterion
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from evaluators.base import BaseDatasetEvaluator, EvalResult
from evaluators.llm_utils import (
    build_llm_client_for_evaluator,
    call_llm,
    coerce_bool,
    extract_json,
    get_model_for_evaluator,
    get_temperature_for_evaluator,
    get_max_tokens_for_evaluator,
)

logger = logging.getLogger(__name__)

# Evaluator name for config lookup
EVALUATOR_NAME = "prbench"


class PRBenchEvaluator(BaseDatasetEvaluator):
    """
    Evaluator for Scale AI PRBench dataset.

    Uses weighted rubric criteria for evaluation:
    - critically_important: 3.0 weight
    - important: 2.0 weight
    - slightly_important: 1.0 weight
    - detrimental: -1.0 weight (penalty)

    Supports two evaluation modes:
    1. Fast mode: Keyword matching (default)
    2. LLM mode: LLM judges each criterion (more accurate)
    """

    name = "prbench"

    # Weight class mappings
    WEIGHT_CLASS_MAP = {
        "critically important": 3.0,
        "important": 2.0,
        "slightly important": 1.0,
        "detrimental": -1.0,
    }

    # Default weights if not specified
    DEFAULT_POSITIVE_WEIGHT = 1.0
    DEFAULT_PENALTY_WEIGHT = -0.5

    LLM_MAX_TOKENS = 1000

    def __init__(
        self,
        use_llm: bool = False,
        llm_client: Any = None,
        llm_model: Optional[str] = None,
        llm_temperature: Optional[float] = None,
        llm_max_tokens: Optional[int] = None,
        min_score: float = 0.0,
    ):
        """
        Initialize PRBench evaluator.

        Args:
            use_llm: Whether to use LLM for evaluation (more accurate)
            llm_client: LLM client for LLM-based evaluation
            llm_model: Optional LLM model override
            llm_temperature: Optional temperature override
            llm_max_tokens: Optional max_tokens override
            min_score: Minimum score floor (default 0.0)
        """
        self.use_llm = use_llm
        self.llm_client = llm_client
        self.min_score = min_score

        # Use per-evaluator config with optional overrides
        self.llm_model = llm_model or get_model_for_evaluator(EVALUATOR_NAME)
        self.llm_temperature = (
            llm_temperature if llm_temperature is not None
            else get_temperature_for_evaluator(EVALUATOR_NAME)
        )
        self._llm_max_tokens = (
            llm_max_tokens if llm_max_tokens is not None
            else get_max_tokens_for_evaluator(EVALUATOR_NAME)
        )

    def evaluate(
        self,
        predicted: str,
        expected: str,
        rubric: Any = None,
        rubric_weights: Dict[str, float] = None,
        question: str = None,
        **kwargs
    ) -> EvalResult:
        """
        Evaluate predicted answer against PRBench rubric.

        Args:
            predicted: Model's predicted answer
            expected: Ground truth / scratchpad (reference)
            rubric: TaskRubric with criteria and penalty_conditions
            rubric_weights: Dictionary mapping criteria -> weight
            question: Original question (for LLM context)
            **kwargs: Additional parameters (domain, topic, etc.)

        Returns:
            EvalResult with weighted score
        """
        if not predicted:
            return EvalResult(
                score=0.0,
                feedback="Empty prediction",
                details={"predicted": predicted, "expected": expected}
            )

        # Extract criteria from rubric
        criteria = []
        penalties = []

        if rubric is not None:
            if hasattr(rubric, 'criteria'):
                criteria = rubric.criteria or []
            if hasattr(rubric, 'penalty_conditions'):
                penalties = rubric.penalty_conditions or []

        # If no rubric, fall back to scratchpad comparison
        if not criteria and not penalties:
            return self._eval_scratchpad_match(predicted, expected)

        # Build weighted criteria list
        weighted_criteria = self._build_weighted_criteria(
            criteria, penalties, rubric_weights
        )

        # Evaluate based on mode
        if self.use_llm:
            result = self._llm_evaluate(
                predicted=predicted,
                expected=expected,
                weighted_criteria=weighted_criteria,
                question=question,
                **kwargs
            )
            if result is not None:
                return result

        # Fast mode: keyword matching
        return self._fast_evaluate(
            predicted=predicted,
            weighted_criteria=weighted_criteria,
        )

    def _build_weighted_criteria(
        self,
        criteria: List[str],
        penalties: List[str],
        rubric_weights: Optional[Dict[str, float]],
    ) -> List[Tuple[str, float, bool]]:
        """
        Build list of (criterion, weight, is_penalty) tuples.

        Args:
            criteria: List of positive criteria
            penalties: List of penalty criteria (detrimental)
            rubric_weights: Optional weights from metadata

        Returns:
            List of (criterion, weight, is_penalty) tuples
        """
        weighted = []

        # Add positive criteria
        for crit in criteria:
            weight = self.DEFAULT_POSITIVE_WEIGHT
            if rubric_weights and crit in rubric_weights:
                weight = rubric_weights[crit]
            weighted.append((crit, weight, False))

        # Add penalty criteria
        for crit in penalties:
            weight = self.DEFAULT_PENALTY_WEIGHT
            if rubric_weights and crit in rubric_weights:
                weight = rubric_weights[crit]
            weighted.append((crit, weight, True))

        return weighted

    def _fast_evaluate(
        self,
        predicted: str,
        weighted_criteria: List[Tuple[str, float, bool]],
    ) -> EvalResult:
        """
        Fast evaluation using keyword/substring matching.

        Args:
            predicted: Model's predicted answer
            weighted_criteria: List of (criterion, weight, is_penalty)

        Returns:
            EvalResult with weighted score
        """
        predicted_lower = predicted.lower()

        total_score = 0.0
        max_possible = 0.0
        matched_criteria = []
        triggered_penalties = []

        for criterion, weight, is_penalty in weighted_criteria:
            # Extract key terms from criterion for matching
            key_terms = self._extract_key_terms(criterion)

            if is_penalty:
                # For penalties, matching is bad
                if self._matches_criterion(predicted_lower, key_terms):
                    total_score += weight  # weight is negative
                    triggered_penalties.append(criterion)
            else:
                # For positive criteria, matching is good
                max_possible += weight
                if self._matches_criterion(predicted_lower, key_terms):
                    total_score += weight
                    matched_criteria.append(criterion)

        # Normalize score to 0-1
        if max_possible > 0:
            normalized_score = max(self.min_score, total_score / max_possible)
        else:
            normalized_score = 0.0

        # Clamp to [0, 1]
        normalized_score = max(0.0, min(1.0, normalized_score))

        return EvalResult(
            score=normalized_score,
            correct_count=len(matched_criteria),
            total_count=len([c for c in weighted_criteria if not c[2]]),
            feedback=f"Matched {len(matched_criteria)}/{len(weighted_criteria)} criteria",
            details={
                "matched_criteria": matched_criteria[:5],  # Limit for output
                "triggered_penalties": triggered_penalties,
                "raw_score": total_score,
                "max_possible": max_possible,
                "evaluation_mode": "fast",
            }
        )

    def _extract_key_terms(self, criterion: str) -> List[str]:
        """
        Extract key terms from a criterion for matching.

        Removes common words and extracts meaningful terms.
        """
        # Common words to ignore
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'up', 'about',
            'into', 'over', 'after', 'that', 'this', 'these', 'those',
            'response', 'answer', 'should', 'must', 'mention', 'include',
            'specifically', 'mentions', 'includes', 'addresses', 'discusses',
        }

        # Tokenize and filter
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9-]*\b', criterion.lower())
        key_terms = [w for w in words if w not in stopwords and len(w) > 2]

        return key_terms

    def _matches_criterion(
        self,
        text: str,
        key_terms: List[str],
        threshold: float = 0.5,
    ) -> bool:
        """
        Check if text matches criterion based on key terms.

        Args:
            text: Text to check (lowercase)
            key_terms: Key terms to look for
            threshold: Proportion of terms that must match

        Returns:
            True if enough key terms are present
        """
        if not key_terms:
            return False

        matches = sum(1 for term in key_terms if term in text)
        return (matches / len(key_terms)) >= threshold

    def _eval_scratchpad_match(
        self,
        predicted: str,
        expected: str,
    ) -> EvalResult:
        """
        Fallback evaluation using scratchpad/reference comparison.

        Uses simple text similarity when no rubric is available.
        """
        if not expected:
            return EvalResult(
                score=0.5,  # Neutral if no reference
                feedback="No reference answer available",
                details={"evaluation_mode": "no_reference"}
            )

        # Normalize texts
        pred_norm = self._normalize_text(predicted)
        exp_norm = self._normalize_text(expected)

        # Simple overlap-based scoring
        pred_words = set(pred_norm.split())
        exp_words = set(exp_norm.split())

        if not exp_words:
            return EvalResult(
                score=0.5,
                feedback="Empty reference after normalization",
                details={"evaluation_mode": "scratchpad"}
            )

        overlap = len(pred_words & exp_words)
        precision = overlap / len(pred_words) if pred_words else 0
        recall = overlap / len(exp_words)

        # F1-like score
        if precision + recall > 0:
            score = 2 * precision * recall / (precision + recall)
        else:
            score = 0.0

        return EvalResult(
            score=score,
            feedback=f"Scratchpad overlap: precision={precision:.2f}, recall={recall:.2f}",
            details={
                "evaluation_mode": "scratchpad",
                "precision": precision,
                "recall": recall,
                "overlap_words": overlap,
            }
        )

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[.,;:!?\'"()\[\]{}]', '', text)
        return text

    def _get_llm_client(self) -> Optional[Any]:
        """Get or create LLM client."""
        if self.llm_client is None:
            self.llm_client = build_llm_client_for_evaluator(EVALUATOR_NAME)
        return self.llm_client

    def _llm_evaluate(
        self,
        predicted: str,
        expected: str,
        weighted_criteria: List[Tuple[str, float, bool]],
        question: Optional[str] = None,
        **kwargs
    ) -> Optional[EvalResult]:
        """
        LLM-based evaluation against rubric criteria.

        Args:
            predicted: Model's predicted answer
            expected: Reference/scratchpad
            weighted_criteria: List of (criterion, weight, is_penalty)
            question: Original question for context

        Returns:
            EvalResult or None if LLM evaluation fails
        """
        client = self._get_llm_client()
        if not client:
            logger.warning("LLM client unavailable, falling back to fast evaluation")
            return None

        # Build criteria list for prompt
        criteria_text = []
        for i, (crit, weight, is_penalty) in enumerate(weighted_criteria[:20]):  # Limit
            weight_label = "PENALTY" if is_penalty else f"weight={weight:.1f}"
            criteria_text.append(f"{i+1}. [{weight_label}] {crit}")

        domain = kwargs.get("domain", "Professional")
        topic = kwargs.get("topic", "")

        # Truncate inputs to prevent context overflow
        max_question_chars = 4000
        max_expected_chars = 1500
        max_predicted_chars = 2500
        
        question_truncated = (question or "N/A")[:max_question_chars]
        expected_truncated = (expected or "N/A")[:max_expected_chars]
        predicted_truncated = predicted[:max_predicted_chars]
        
        if len(question or "") > max_question_chars:
            question_truncated += "\n... [truncated]"

        system_prompt = f"You are a strict evaluator for {domain} professional reasoning tasks."
        prompt = f"""DOMAIN: {domain}
TOPIC: {topic}

QUESTION:
{question_truncated}

REFERENCE REASONING (scratchpad):
{expected_truncated}

CANDIDATE ANSWER:
{predicted_truncated}

EVALUATION CRITERIA:
{chr(10).join(criteria_text)}

For each criterion, determine if the candidate answer satisfies it.
For PENALTY criteria, check if the answer inappropriately includes/violates it.

Return JSON only (no markdown, no explanation outside JSON):
{{
    "criteria_met": [1, 3, 5],
    "penalties_triggered": [8],
    "overall_quality": 0.75,
    "reasoning": "brief explanation"
}}
"""

        try:
            raw = call_llm(
                client=client,
                prompt=prompt,
                model=self.llm_model,
                system_prompt=system_prompt,
                temperature=self.llm_temperature,
                max_tokens=self._llm_max_tokens,
            )
            data = extract_json(raw)
            if not data:
                # Retry with simpler prompt if JSON parsing fails
                logger.debug("First attempt failed, retrying with simpler prompt")
                simple_prompt = f"""Evaluate this answer briefly.

QUESTION: {question_truncated[:1000]}
ANSWER: {predicted_truncated[:1000]}

Return ONLY valid JSON:
{{"criteria_met": [], "penalties_triggered": [], "overall_quality": 0.5, "reasoning": "evaluation"}}
"""
                raw = call_llm(
                    client=client,
                    prompt=simple_prompt,
                    model=self.llm_model,
                    system_prompt="Return only valid JSON.",
                    temperature=0.0,
                    max_tokens=300,
                )
                data = extract_json(raw)
                if not data:
                    logger.warning("LLM returned invalid JSON for PRBench evaluation (both attempts)")
                    return None
        except Exception as e:
            logger.warning(f"LLM PRBench evaluation failed: {e}")
            return None

        # Calculate weighted score from LLM response
        criteria_met = data.get("criteria_met", [])
        penalties_triggered = data.get("penalties_triggered", [])
        overall_quality = data.get("overall_quality", 0.5)
        reasoning = data.get("reasoning", "LLM evaluation")

        # Calculate score from met criteria
        total_score = 0.0
        max_possible = 0.0

        for i, (crit, weight, is_penalty) in enumerate(weighted_criteria):
            idx = i + 1  # 1-indexed
            if is_penalty:
                if idx in penalties_triggered:
                    total_score += weight  # Negative
            else:
                max_possible += weight
                if idx in criteria_met:
                    total_score += weight

        # Combine rule-based and overall quality
        if max_possible > 0:
            rule_score = max(0.0, total_score / max_possible)
        else:
            rule_score = overall_quality

        # Blend rule-based score with LLM's overall assessment
        final_score = 0.7 * rule_score + 0.3 * overall_quality
        final_score = max(0.0, min(1.0, final_score))

        return EvalResult(
            score=final_score,
            correct_count=len(criteria_met),
            total_count=len([c for c in weighted_criteria if not c[2]]),
            feedback=reasoning,
            details={
                "criteria_met": criteria_met,
                "penalties_triggered": penalties_triggered,
                "overall_quality": overall_quality,
                "rule_score": rule_score,
                "evaluation_mode": "llm",
                "llm_model": self.llm_model,
                "llm_raw": raw[:500] if raw else None,
            }
        )
