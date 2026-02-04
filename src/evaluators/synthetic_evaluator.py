"""
Synthetic questions evaluator.

Evaluates agent responses against synthetic FAB-style questions with rubric-based
scoring. Each question has components with weights that define the evaluation criteria.

Rubric format (from questions.json):
{
    "rubric": {
        "components": [
            {"name": "methodology", "description": "Set up equation correctly", "weight": 0.3},
            {"name": "calculation", "description": "Correct numerical result", "expected_value": "20%", "weight": 0.4},
            ...
        ],
        "max_score": 100
    }
}

This evaluator uses LLM-as-judge to assess each rubric component.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from evaluators.base import BaseDatasetEvaluator, EvalResult
from evaluators.llm_utils import (
    build_llm_client_for_evaluator,
    call_llm,
    extract_json,
    get_model_for_evaluator,
    get_temperature_for_evaluator,
    get_max_tokens_for_evaluator,
)

logger = logging.getLogger(__name__)

# Evaluator name for config lookup
EVALUATOR_NAME = "synthetic"


class SyntheticEvaluator(BaseDatasetEvaluator):
    """
    Evaluator for synthetic FAB-style questions.

    Uses LLM-as-judge to evaluate responses against rubric components.
    Each component has a weight and optional expected value.
    
    Scoring:
        - Each component is scored 0-1 based on LLM judgment
        - Component scores are weighted by their weight values
        - Final score is normalized to 0-1
    """

    name = "synthetic"

    LLM_MAX_TOKENS = 2000

    def __init__(
        self,
        use_llm: bool = True,
        llm_client: Any = None,
        llm_model: Optional[str] = None,
        llm_temperature: Optional[float] = None,
        llm_max_tokens: Optional[int] = None,
    ):
        """
        Initialize Synthetic evaluator.

        Args:
            use_llm: Whether to use LLM for evaluation
            llm_client: LLM client for evaluation
            llm_model: Optional LLM model override
            llm_temperature: Optional temperature override
            llm_max_tokens: Optional max_tokens override
        """
        self.use_llm = use_llm
        self.llm_client = llm_client

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
        expected: str = "",
        question: str = None,
        rubric: Dict[str, Any] = None,
        ground_truth_value: Any = None,
        calculation_steps: List[str] = None,
        **kwargs
    ) -> EvalResult:
        """
        Evaluate predicted answer against rubric components.

        Args:
            predicted: Agent's response
            expected: Ground truth formatted answer (e.g., "20%" or "$1,250")
            question: The original question text
            rubric: Rubric dict with "components" and "max_score"
            ground_truth_value: Numeric or structured ground truth
            calculation_steps: Reference calculation steps
            **kwargs: Additional parameters (category, difficulty, etc.)

        Returns:
            EvalResult with score and component-level details
        """
        if not predicted or len(predicted.strip()) < 5:
            return EvalResult(
                score=0.0,
                feedback="No meaningful response provided",
                details={"error": "empty_response", "llm_used": False},
            )

        # Extract rubric components
        components = []
        if rubric and isinstance(rubric, dict):
            components = rubric.get("components", [])
        
        # If no rubric or components, fall back to answer comparison
        if not components:
            return self._fallback_evaluate(predicted, expected, ground_truth_value)

        # Use LLM to evaluate each component
        if self.use_llm and self.llm_client:
            result = self._llm_evaluate(
                predicted=predicted,
                expected=expected,
                question=question or "",
                components=components,
                ground_truth_value=ground_truth_value,
                calculation_steps=calculation_steps,
                category=kwargs.get("category", ""),
            )
            if result is not None:
                return result
            # Fall through to fallback if LLM fails

        # Fallback: basic evaluation
        return self._fallback_evaluate(predicted, expected, ground_truth_value)

    def _llm_evaluate(
        self,
        predicted: str,
        expected: str,
        question: str,
        components: List[Dict[str, Any]],
        ground_truth_value: Any = None,
        calculation_steps: List[str] = None,
        category: str = "",
    ) -> Optional[EvalResult]:
        """
        Use LLM to evaluate response against rubric components.

        Returns:
            EvalResult if successful, None if LLM call fails
        """
        # Build component descriptions for prompt
        component_lines = []
        for i, comp in enumerate(components, 1):
            name = comp.get("name", f"component_{i}")
            desc = comp.get("description", "No description")
            expected_val = comp.get("expected_value")
            weight = comp.get("weight", 0.25)
            
            line = f"{i}. {name} (weight: {weight:.0%}): {desc}"
            if expected_val:
                line += f" [Expected: {expected_val}]"
            component_lines.append(line)

        components_text = "\n".join(component_lines)
        
        # Build calculation steps if available
        steps_text = ""
        if calculation_steps:
            steps_text = "\n\nReference calculation steps:\n" + "\n".join(
                f"  - {step}" for step in calculation_steps
            )

        # Build ground truth info
        gt_text = ""
        if expected:
            gt_text = f"\n\nGround truth answer: {expected}"
        if ground_truth_value is not None:
            if isinstance(ground_truth_value, dict):
                gt_text += f"\nGround truth value: {ground_truth_value}"
            else:
                gt_text += f"\nGround truth value: {ground_truth_value}"

        prompt = f"""You are evaluating a finance agent's response to a synthetic benchmark question.

Question:
{question}

Candidate Answer:
{predicted}
{gt_text}
{steps_text}

Evaluation Rubric Components:
{components_text}

Instructions:
1. Evaluate how well the candidate answer satisfies each rubric component
2. For each component, assign a score from 0.0 to 1.0:
   - 1.0 = Fully satisfies the component
   - 0.5-0.9 = Partially satisfies (correct approach but minor errors)
   - 0.1-0.4 = Minimal satisfaction (some relevant content but significant issues)
   - 0.0 = Does not satisfy the component

3. Consider:
   - For numerical answers: Is the value correct within reasonable tolerance (5%)?
   - For methodology: Did they use the correct approach?
   - For explanations: Is the reasoning sound and complete?

Respond with ONLY a JSON object in this exact format:
{{
    "component_scores": {{
        "<component_name>": {{
            "score": <0.0-1.0>,
            "reasoning": "<brief explanation>"
        }},
        ...
    }},
    "overall_feedback": "<2-3 sentence summary of evaluation>"
}}
"""

        try:
            response = call_llm(
                self.llm_client,
                prompt,
                model=self.llm_model,
                temperature=self.llm_temperature,
                max_tokens=self._llm_max_tokens,
            )
            
            if not response:
                logger.warning("Empty LLM response for synthetic evaluation")
                return None

            # Extract JSON from response
            parsed = extract_json(response)
            if not parsed or "component_scores" not in parsed:
                logger.warning(f"Failed to parse LLM response: {response[:200]}")
                return None

            # Calculate weighted score
            component_scores = parsed.get("component_scores", {})
            total_weight = 0.0
            weighted_sum = 0.0
            component_details = {}

            for comp in components:
                name = comp.get("name", "")
                weight = comp.get("weight", 0.25)
                
                if name in component_scores:
                    comp_data = component_scores[name]
                    score = float(comp_data.get("score", 0.0))
                    score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
                    
                    weighted_sum += score * weight
                    total_weight += weight
                    
                    component_details[name] = {
                        "score": score,
                        "weight": weight,
                        "weighted_contribution": score * weight,
                        "reasoning": comp_data.get("reasoning", ""),
                    }

            # Normalize score
            final_score = weighted_sum / total_weight if total_weight > 0 else 0.0

            # Count correct components (score >= 0.7)
            correct_count = sum(
                1 for c in component_details.values() 
                if c.get("score", 0) >= 0.7
            )

            feedback = parsed.get("overall_feedback", "")
            if not feedback:
                feedback = f"Scored {final_score:.1%} across {len(components)} components"

            return EvalResult(
                score=final_score,
                max_score=1.0,
                correct_count=correct_count,
                total_count=len(components),
                feedback=feedback,
                details={
                    "llm_used": True,
                    "component_scores": component_details,
                    "llm_raw_output": response[:1000] if len(response) > 1000 else response,
                },
            )

        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            return None

    def _fallback_evaluate(
        self,
        predicted: str,
        expected: str,
        ground_truth_value: Any = None,
    ) -> EvalResult:
        """
        Fallback evaluation without LLM.
        
        Uses simple heuristics:
        - Check if expected value appears in response
        - Check for numerical matches with tolerance
        """
        if not expected and ground_truth_value is None:
            # No ground truth to compare against
            return EvalResult(
                score=0.5,  # Neutral score
                feedback="No ground truth available for comparison",
                details={"llm_used": False, "method": "no_ground_truth"},
            )

        predicted_lower = predicted.lower()
        score = 0.0
        feedback_parts = []

        # Check for expected string match
        if expected:
            expected_lower = expected.lower()
            # Check exact or near match
            if expected_lower in predicted_lower:
                score = 1.0
                feedback_parts.append(f"Found expected answer: {expected}")
            else:
                # Try to extract numbers and compare
                expected_nums = self._extract_numbers(expected)
                predicted_nums = self._extract_numbers(predicted)
                
                if expected_nums and predicted_nums:
                    # Check if any expected number is close to any predicted number
                    for exp_num in expected_nums:
                        for pred_num in predicted_nums:
                            if self._numbers_close(exp_num, pred_num, tolerance=0.10):
                                score = max(score, 0.8)
                                feedback_parts.append(
                                    f"Found approximate numerical match: {pred_num} ≈ {exp_num}"
                                )
                                break

        # Check ground_truth_value for structured comparison
        if ground_truth_value is not None and score < 0.8:
            if isinstance(ground_truth_value, dict):
                # Check each key-value pair
                matches = 0
                total = len(ground_truth_value)
                for key, val in ground_truth_value.items():
                    if isinstance(val, (int, float)):
                        pred_nums = self._extract_numbers(predicted)
                        for pred_num in pred_nums:
                            if self._numbers_close(val, pred_num, tolerance=0.10):
                                matches += 1
                                break
                    elif isinstance(val, bool):
                        val_str = "yes" if val else "no"
                        if val_str in predicted_lower:
                            matches += 1
                if total > 0:
                    dict_score = matches / total
                    if dict_score > score:
                        score = dict_score
                        feedback_parts.append(
                            f"Matched {matches}/{total} ground truth components"
                        )
            elif isinstance(ground_truth_value, (int, float)):
                pred_nums = self._extract_numbers(predicted)
                for pred_num in pred_nums:
                    if self._numbers_close(ground_truth_value, pred_num, tolerance=0.10):
                        score = max(score, 0.8)
                        feedback_parts.append(
                            f"Found approximate numerical match: {pred_num} ≈ {ground_truth_value}"
                        )
                        break

        if not feedback_parts:
            feedback_parts.append(f"Expected: {expected}, no clear match found")

        return EvalResult(
            score=score,
            feedback="; ".join(feedback_parts),
            details={
                "llm_used": False,
                "method": "fallback_heuristic",
                "expected": expected,
                "ground_truth_value": ground_truth_value,
            },
        )

    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numerical values from text."""
        # Match numbers including percentages, currency, scientific notation
        patterns = [
            r'[-+]?\d*\.?\d+%',  # Percentages
            r'\$[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?',  # Currency with commas
            r'[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?',  # Numbers with commas
            r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?',  # Scientific notation
        ]
        
        numbers = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    # Clean the match
                    clean = match.replace('$', '').replace(',', '').replace('%', '')
                    num = float(clean)
                    # Convert percentage to decimal if needed
                    if '%' in match:
                        num = num / 100
                    numbers.append(num)
                except ValueError:
                    continue
        
        return list(set(numbers))  # Remove duplicates

    def _numbers_close(self, a: float, b: float, tolerance: float = 0.05) -> bool:
        """Check if two numbers are within tolerance of each other."""
        if a == 0 and b == 0:
            return True
        if a == 0 or b == 0:
            return abs(a - b) < tolerance
        return abs(a - b) / max(abs(a), abs(b)) < tolerance
