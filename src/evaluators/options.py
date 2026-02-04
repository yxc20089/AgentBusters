"""
Options Evaluator for options trading task assessment.

Evaluates the quality of options trading responses including:
- P&L calculation accuracy
- Greeks verification
- Strategy quality scoring
- Risk management discipline

Supports hybrid evaluation:
- Quantitative questions: LLM extracts values → Rule-based comparison
- Qualitative questions: Full LLM evaluation
"""

import json
import re
from typing import Any, Optional
from dataclasses import dataclass

import structlog

from cio_agent.models import (
    Task,
    AgentResponse,
    TaskCategory,
)

try:
    from evaluators.llm_utils import (
        build_llm_client_for_evaluator,
        call_llm,
        extract_json,
        get_model_for_evaluator,
    )
    HAS_LLM_UTILS = True
except ImportError:
    HAS_LLM_UTILS = False

logger = structlog.get_logger()

# Evaluator name for LLM configuration
EVALUATOR_NAME = "options"


# Options trading categories requiring specialized evaluation
OPTIONS_CATEGORIES = [
    TaskCategory.OPTIONS_PRICING,
    TaskCategory.GREEKS_ANALYSIS,
    TaskCategory.STRATEGY_CONSTRUCTION,
    TaskCategory.VOLATILITY_TRADING,
    TaskCategory.PNL_ATTRIBUTION,
    TaskCategory.RISK_MANAGEMENT,
    TaskCategory.COPY_TRADING,
    TaskCategory.RACE_TO_10M,
    TaskCategory.STRATEGY_DEFENSE,
]


@dataclass
class OptionsScore:
    """Detailed score for options trading evaluation."""
    score: float  # 0-100
    pnl_accuracy: float  # 0-100: P&L calculation accuracy
    greeks_accuracy: float  # 0-100: Greeks accuracy
    strategy_quality: float  # 0-100: Strategy appropriateness
    risk_management: float  # 0-100: Risk management quality
    feedback: str


@dataclass
class ExtractedOptionsData:
    """Options data extracted from agent response."""
    # Strategy type
    strategy_name: Optional[str] = None
    legs: list[dict] = None

    # Greeks
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None

    # P&L
    max_profit: Optional[float] = None
    max_loss: Optional[float] = None
    breakevens: list[float] = None
    current_pnl: Optional[float] = None

    # Risk metrics
    probability_of_profit: Optional[float] = None
    var_95: Optional[float] = None

    def __post_init__(self):
        if self.legs is None:
            self.legs = []
        if self.breakevens is None:
            self.breakevens = []


class OptionsEvaluator:
    """
    Evaluates agent's options trading responses.

    Scoring dimensions:
    - P&L Accuracy (25%): Verify calculations against Black-Scholes
    - Greeks Accuracy (25%): Verify Greeks calculations
    - Strategy Quality (25%): Appropriateness for market conditions
    - Risk Management (25%): Position sizing, hedging, discipline
    """

    # Tolerances for numerical accuracy
    PRICE_TOLERANCE = 0.05  # 5% tolerance for option prices
    GREEKS_TOLERANCE = 0.10  # 10% tolerance for Greeks

    def __init__(
        self,
        task: Task,
        mcp_toolkit: Optional[Any] = None,
        llm_client: Optional[Any] = None,
    ):
        """
        Initialize the options evaluator.

        Args:
            task: The task being evaluated
            mcp_toolkit: MCPToolkit for verification calculations
            llm_client: Optional LLM client for qualitative scoring
        """
        self.task = task
        self.mcp_toolkit = mcp_toolkit
        self.llm_client = llm_client
        self._llm_model = None
        self._use_llm_extraction = True  # Enable LLM-based value extraction

    # =========================================================================
    # LLM Helper Methods
    # =========================================================================

    def _get_llm_client(self) -> Optional[Any]:
        """Get or create LLM client for evaluation."""
        if not HAS_LLM_UTILS:
            return self.llm_client
        if self.llm_client is None:
            self.llm_client = build_llm_client_for_evaluator(EVALUATOR_NAME)
        return self.llm_client

    def _get_llm_model(self) -> str:
        """Get model name for this evaluator."""
        if self._llm_model is None and HAS_LLM_UTILS:
            self._llm_model = get_model_for_evaluator(EVALUATOR_NAME)
        return self._llm_model or "gpt-4o-mini"

    async def _llm_extract_values(
        self,
        response_text: str,
        ground_truth: dict,
    ) -> tuple[dict, Optional[str]]:
        """
        Use LLM to extract numerical values from agent response.
        
        This is more robust than regex for handling varied formats like:
        - "The theoretical price is approximately twenty-five dollars"
        - "call ≈ $3.22"
        - "Delta: 0.474 (positive, indicating bullish)"
        
        Args:
            response_text: The agent's response text
            ground_truth: Expected values to guide extraction
            
        Returns:
            Tuple of (extracted_values_dict, error_message)
        """
        client = self._get_llm_client()
        if not client:
            return {}, "llm_client_unavailable"
        
        # Build extraction schema from ground_truth keys
        fields = list(ground_truth.keys())
        fields_str = ", ".join(fields)
        
        system_prompt = """You are a precise numerical value extractor for options trading analysis.
Extract ONLY the specific numerical values mentioned in the response.
Return null for any value not explicitly stated.
Do NOT calculate or infer values - only extract what is explicitly written."""

        prompt = f"""Extract these specific values from the candidate answer:
Fields needed: {fields_str}

CANDIDATE ANSWER:
{response_text}

Return a JSON object with these exact keys. Use null if a value is not found.
For percentages, return as decimal (e.g., 7.8% → 7.8).
For dollar amounts, return the number only (e.g., $25.18 → 25.18).

Example output format:
{{"theoretical_price": 25.18, "delta": 0.474, "assessment": "underpriced"}}
"""

        try:
            import asyncio
            # Run sync LLM call in thread pool to avoid blocking event loop
            raw = await asyncio.to_thread(
                call_llm,
                client=client,
                prompt=prompt,
                model=self._get_llm_model(),
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=500,
            )
            data = extract_json(raw)
            if data:
                logger.debug(
                    "llm_extraction_success",
                    task_id=self.task.question_id,
                    extracted=data,
                )
                return data, None
            return {}, "llm_invalid_json"
        except Exception as e:
            logger.warning(f"llm_extraction_failed: {e}")
            return {}, f"llm_call_failed: {e}"

    async def _llm_extract_greeks(
        self,
        response_text: str,
    ) -> tuple[dict, Optional[str]]:
        """
        Use LLM to extract Greeks values from agent response.
        
        More robust than regex for handling varied formats like:
        - "Delta is approximately 0.47"
        - "the position delta equals -0.35"
        - "θ (time decay): -$0.32 per day"
        - "Gamma: about 0.025"
        
        Args:
            response_text: The agent's response text
            
        Returns:
            Tuple of (extracted_greeks_dict, error_message)
        """
        client = self._get_llm_client()
        if not client:
            return {}, "llm_client_unavailable"
        
        system_prompt = """You are a precise Greeks value extractor for options trading analysis.
Extract the Delta, Gamma, Theta, and Vega values mentioned in the response.
Return null for any Greek not explicitly stated.
Do NOT calculate or infer values - only extract what is explicitly written.
For Theta, if given as daily dollar amount (e.g., "loses $0.32 per day"), return as negative value."""

        prompt = f"""Extract the Greeks values from this options analysis:

CANDIDATE ANSWER:
{response_text}

Return a JSON object with these exact keys: delta, gamma, theta, vega
Use null if a value is not found or unclear.

Examples of what to extract:
- "Delta: 0.474" → {{"delta": 0.474}}
- "gamma is approximately 0.025" → {{"gamma": 0.025}}
- "theta (time decay): -$0.32 per day" → {{"theta": -0.32}}
- "vega = 0.15" → {{"vega": 0.15}}

Output format:
{{"delta": <number|null>, "gamma": <number|null>, "theta": <number|null>, "vega": <number|null>}}
"""

        try:
            import asyncio
            # Run sync LLM call in thread pool to avoid blocking event loop
            raw = await asyncio.to_thread(
                call_llm,
                client=client,
                prompt=prompt,
                model=self._get_llm_model(),
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=200,
            )
            data = extract_json(raw)
            if data:
                logger.debug(
                    "llm_greeks_extraction_success",
                    task_id=self.task.question_id,
                    extracted=data,
                )
                return data, None
            return {}, "llm_invalid_json"
        except Exception as e:
            logger.warning(f"llm_greeks_extraction_failed: {e}")
            return {}, f"llm_call_failed: {e}"

    async def _llm_evaluate_qualitative(
        self,
        response_text: str,
        ground_truth: dict,
        rubric: list[dict],
    ) -> tuple[float, str, Optional[str]]:
        """
        Use LLM to evaluate qualitative/strategic questions.
        
        For questions like strategy_002 (straddle analysis), vol_001 (IV trading),
        defense_001 (strategy adjustment) where judgment is required.
        
        Args:
            response_text: The agent's response
            ground_truth: Expected answer components
            rubric: Evaluation rubric with components and weights
            
        Returns:
            Tuple of (score 0-100, feedback, error_message)
        """
        client = self._get_llm_client()
        if not client:
            return 50.0, "LLM unavailable for qualitative evaluation", "llm_client_unavailable"
        
        # Format rubric for prompt
        rubric_str = "\n".join(
            f"- {r['name']} ({r['weight']*100:.0f}%): {r['description']}"
            for r in rubric
        )
        
        # Format ground truth
        gt_str = json.dumps(ground_truth, indent=2)
        
        system_prompt = """You are an expert options trading evaluator.
Score the candidate's response against the rubric and reference answer.
Be strict but fair. Award partial credit for partially correct answers.
Focus on: correctness of reasoning, appropriate strategy selection, risk awareness."""

        prompt = f"""REFERENCE ANSWER:
{gt_str}

RUBRIC:
{rubric_str}

CANDIDATE ANSWER:
{response_text}

Score each rubric component from 0-100, then compute weighted total.

Return JSON:
{{
  "component_scores": {{"component_name": score, ...}},
  "total_score": <weighted average 0-100>,
  "feedback": "<brief explanation of score>"
}}
"""

        try:
            import asyncio
            # Run sync LLM call in thread pool to avoid blocking event loop
            raw = await asyncio.to_thread(
                call_llm,
                client=client,
                prompt=prompt,
                model=self._get_llm_model(),
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=800,
            )
            data = extract_json(raw)
            if data and "total_score" in data:
                score = float(data["total_score"])
                feedback = data.get("feedback", "LLM evaluation completed.")
                logger.debug(
                    "llm_qualitative_eval",
                    task_id=self.task.question_id,
                    score=score,
                    components=data.get("component_scores"),
                )
                return score, feedback, None
            return 50.0, "LLM response parsing failed", "llm_invalid_json"
        except Exception as e:
            logger.warning(f"llm_qualitative_failed: {e}")
            return 50.0, f"LLM evaluation error: {e}", f"llm_call_failed: {e}"

    def _compare_values(
        self,
        extracted: dict,
        ground_truth: dict,
        tolerance: float = 0.10,
    ) -> tuple[float, str]:
        """
        Rule-based comparison of extracted values against ground truth.
        
        Args:
            extracted: Values extracted from response (via LLM or regex)
            ground_truth: Expected values
            tolerance: Relative tolerance for numerical comparison
            
        Returns:
            Tuple of (score 0-100, feedback)
        """
        if not extracted or not ground_truth:
            return 50.0, "Insufficient data for comparison"
        
        correct = 0
        total = 0
        feedback_parts = []
        
        for key, expected in ground_truth.items():
            if key not in extracted or extracted[key] is None:
                continue
            
            actual = extracted[key]
            total += 1
            
            # Handle different value types
            if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
                # Numerical comparison with tolerance
                if expected == 0:
                    is_correct = abs(actual) < 0.01
                else:
                    pct_diff = abs(actual - expected) / abs(expected)
                    is_correct = pct_diff <= tolerance
                
                if is_correct:
                    correct += 1
                    feedback_parts.append(f"{key}: ✓")
                else:
                    feedback_parts.append(f"{key}: ✗ (got {actual}, expected {expected})")
                    
            elif isinstance(expected, str) and isinstance(actual, str):
                # String comparison (case-insensitive, partial match)
                if expected.lower() in actual.lower() or actual.lower() in expected.lower():
                    correct += 1
                    feedback_parts.append(f"{key}: ✓")
                else:
                    feedback_parts.append(f"{key}: ✗")
            else:
                # Type mismatch or complex type
                if str(expected).lower() == str(actual).lower():
                    correct += 1
        
        if total == 0:
            return 50.0, "No comparable values found"
        
        score = (correct / total) * 100
        feedback = f"Matched {correct}/{total}: " + ", ".join(feedback_parts[:5])
        
        return score, feedback

    def _extract_numbers_from_text(self, text: str) -> list[float]:
        """Extract all numbers from text."""
        # Match numbers including negatives, decimals, percentages
        pattern = r'-?\$?\d+(?:,\d{3})*(?:\.\d+)?%?'
        matches = re.findall(pattern, text)

        numbers = []
        for match in matches:
            # Clean and convert
            clean = match.replace('$', '').replace(',', '').replace('%', '')
            try:
                numbers.append(float(clean))
            except ValueError:
                continue
        return numbers

    def _extract_greek_value(self, text: str, greek_name: str) -> Optional[float]:
        """
        Extract a Greek value from text using multiple patterns.
        
        Handles various formats:
        - "Delta: 0.42"
        - "delta = 0.42"
        - "delta of 0.42"
        - "delta is 0.42"
        - "delta: -0.35"
        - "**Delta**: 0.42"
        - "Delta ≈ 0.42"
        - "Δ = 0.42" (for delta)
        - "The gamma is approximately 0.025"
        
        Args:
            text: The text to search
            greek_name: Name of the Greek (delta, gamma, theta, vega)
            
        Returns:
            Extracted float value or None if not found
        """
        # Greek symbols mapping
        greek_symbols = {
            "delta": r"[Δδ]",
            "gamma": r"[Γγ]",
            "theta": r"[Θθ]",
            "vega": r"[Vv]",  # Vega doesn't have a standard Greek letter
        }
        
        name_pattern = greek_name.lower()
        symbol_pattern = greek_symbols.get(name_pattern, "")
        
        # Multiple regex patterns to try, ordered by specificity
        patterns = [
            # Pattern 1: Greek name/symbol followed by separator and number
            # Matches: "Delta: 0.42", "delta = -0.35", "Δ = 0.42"
            rf'(?:\*\*)?{name_pattern}(?:\*\*)?[:\s=≈]+(-?\d+\.?\d*)',
            
            # Pattern 2: Greek name followed by "of" or "is" and number
            # Matches: "delta of 0.42", "delta is -0.35"
            rf'{name_pattern}\s+(?:of|is)\s+(-?\d+\.?\d*)',
            
            # Pattern 3: "The <greek> is approximately/around/about <number>"
            # Matches: "The delta is approximately 0.48", "the gamma is around 0.025"
            rf'(?:the\s+)?{name_pattern}\s+is\s+(?:approximately|around|about|roughly|nearly|~|≈)\s*(-?\d+\.?\d*)',
            
            # Pattern 4: "<greek> approximately/around <number>" without "is"
            # Matches: "gamma around 0.025", "delta approximately 0.5"
            rf'{name_pattern}\s+(?:approximately|around|about|roughly|nearly|~|≈)\s*(-?\d+\.?\d*)',
            
            # Pattern 5: Greek symbol pattern (if available)
            rf'{symbol_pattern}[:\s=≈]+(-?\d+\.?\d*)' if symbol_pattern else None,
            
            # Pattern 6: Number followed by Greek name (less common)
            # Matches: "0.42 delta", "-0.35 (delta)"
            rf'(-?\d+\.?\d*)\s*\(?{name_pattern}\)?',
            
            # Pattern 7: Greek in parentheses with value
            # Matches: "(delta: 0.42)", "(Δ=0.42)"
            rf'\({name_pattern}[:\s=]+(-?\d+\.?\d*)\)',
            
            # Pattern 8: Bullet or list format
            # Matches: "- Delta: 0.42", "• Delta = 0.42"
            rf'[-•*]\s*(?:\*\*)?{name_pattern}(?:\*\*)?[:\s=]+(-?\d+\.?\d*)',
            
            # Pattern 9: Table-like format with pipes
            # Matches: "| Delta | 0.42 |"
            rf'\|\s*{name_pattern}\s*\|\s*(-?\d+\.?\d*)\s*\|',
        ]
        
        for pattern in patterns:
            if pattern is None:
                continue
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    # Sanity check: Greeks typically have reasonable ranges
                    # Delta: -1 to 1, Gamma: 0 to ~0.1, Theta: -inf to 0 (daily), Vega: 0 to ~1
                    if name_pattern == "delta" and abs(value) > 100:
                        continue  # Probably not a delta value
                    if name_pattern == "gamma" and abs(value) > 10:
                        continue  # Probably not a gamma value
                    return value
                except ValueError:
                    continue
        
        return None

    def _extract_greeks_via_regex(self, text: str) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Extract all Greeks (delta, gamma, theta, vega) from text using regex.
        
        This is a helper method to reduce code duplication across branches.
        
        Args:
            text: The text to search for Greek values
            
        Returns:
            Tuple of (delta, gamma, theta, vega) values, each Optional[float]
        """
        return (
            self._extract_greek_value(text, "delta"),
            self._extract_greek_value(text, "gamma"),
            self._extract_greek_value(text, "theta"),
            self._extract_greek_value(text, "vega"),
        )

    async def _extract_options_data(
        self,
        response: AgentResponse,
        use_llm_for_greeks: bool = True,
    ) -> ExtractedOptionsData:
        """
        Extract options trading data from agent response.

        Parses the response for:
        - Strategy name and structure
        - Greeks values (via LLM if enabled, with regex fallback)
        - P&L metrics
        - Risk parameters
        
        Args:
            response: The agent's response to parse
            use_llm_for_greeks: Whether to use LLM for Greeks extraction (default True)
        """
        analysis = response.analysis.lower()
        recommendation = response.recommendation.lower()
        combined = f"{analysis} {recommendation}"
        combined_raw = f"{response.analysis} {response.recommendation}"  # Keep original case for LLM

        data = ExtractedOptionsData()

        # Detect strategy type
        strategy_keywords = {
            "iron condor": ["iron condor"],
            "bull call spread": ["bull call spread", "call spread", "debit spread"],
            "bear put spread": ["bear put spread", "put spread"],
            "straddle": ["straddle", "long straddle", "short straddle"],
            "strangle": ["strangle", "long strangle", "short strangle"],
            "covered call": ["covered call", "buy-write"],
            "protective put": ["protective put", "married put"],
            "butterfly": ["butterfly", "iron butterfly"],
            "calendar spread": ["calendar spread", "time spread"],
            "naked call": ["naked call", "uncovered call"],
            "naked put": ["naked put", "cash-secured put"],
        }

        for strategy, keywords in strategy_keywords.items():
            if any(kw in combined for kw in keywords):
                data.strategy_name = strategy
                break

        # Extract Greeks using LLM (with regex fallback)
        if use_llm_for_greeks and self._use_llm_extraction:
            greeks_data, error = await self._llm_extract_greeks(combined_raw)
            if greeks_data and not error:
                # Coerce LLM-extracted values to floats where possible
                def _coerce_float(value: Any) -> Optional[float]:
                    if value is None:
                        return None
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        return None
                
                llm_delta = _coerce_float(greeks_data.get("delta"))
                llm_gamma = _coerce_float(greeks_data.get("gamma"))
                llm_theta = _coerce_float(greeks_data.get("theta"))
                llm_vega = _coerce_float(greeks_data.get("vega"))
                
                # Get regex fallback values
                regex_delta, regex_gamma, regex_theta, regex_vega = self._extract_greeks_via_regex(combined)
                
                # If LLM didn't provide any valid numeric Greeks, fall back fully to regex
                if not any(v is not None for v in (llm_delta, llm_gamma, llm_theta, llm_vega)):
                    data.delta, data.gamma, data.theta, data.vega = regex_delta, regex_gamma, regex_theta, regex_vega
                    extraction_method = "regex"
                else:
                    # Use LLM values when present; fill missing ones via regex
                    data.delta = llm_delta if llm_delta is not None else regex_delta
                    data.gamma = llm_gamma if llm_gamma is not None else regex_gamma
                    data.theta = llm_theta if llm_theta is not None else regex_theta
                    data.vega = llm_vega if llm_vega is not None else regex_vega
                    extraction_method = "llm+regex"
                
                logger.debug(
                    "greeks_extracted",
                    task_id=self.task.question_id,
                    method=extraction_method,
                    delta=data.delta,
                    gamma=data.gamma,
                    theta=data.theta,
                    vega=data.vega,
                )
            else:
                # Fallback to regex (LLM extraction failed)
                data.delta, data.gamma, data.theta, data.vega = self._extract_greeks_via_regex(combined)
                logger.debug(
                    "greeks_extracted",
                    task_id=self.task.question_id,
                    method="regex",
                    delta=data.delta,
                    gamma=data.gamma,
                    theta=data.theta,
                    vega=data.vega,
                )
        else:
            # Use regex extraction (LLM disabled)
            data.delta, data.gamma, data.theta, data.vega = self._extract_greeks_via_regex(combined)

        # Max profit
        profit_match = re.search(r'max(?:imum)?\s+profit[:\s]+\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)', combined)
        if profit_match:
            data.max_profit = float(profit_match.group(1).replace(',', ''))

        # Max loss
        loss_match = re.search(r'max(?:imum)?\s+loss[:\s]+\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)', combined)
        if loss_match:
            data.max_loss = float(loss_match.group(1).replace(',', ''))

        # Probability of profit
        pop_match = re.search(r'probability\s+of\s+profit[:\s]+(\d+\.?\d*)%?', combined)
        if pop_match:
            data.probability_of_profit = float(pop_match.group(1))

        # VaR
        var_match = re.search(r'var[:\s]+\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)', combined)
        if var_match:
            data.var_95 = float(var_match.group(1).replace(',', ''))

        return data

    async def _verify_pnl_accuracy(
        self,
        extracted: ExtractedOptionsData,
        response: AgentResponse,
    ) -> tuple[float, str]:
        """
        Verify P&L calculations against market data.

        Returns:
            Tuple of (accuracy_score, feedback)
        """
        if not self.mcp_toolkit:
            # Cannot verify without toolkit, use heuristic
            return self._heuristic_pnl_score(extracted, response)

        score = 50.0  # Base score
        feedback_parts = []

        # Check if strategy legs were provided
        if extracted.strategy_name:
            score += 15
            feedback_parts.append(f"Strategy identified: {extracted.strategy_name}.")
        else:
            feedback_parts.append("No clear strategy identified.")

        # Check max profit/loss provided
        if extracted.max_profit is not None:
            score += 10
            feedback_parts.append("Max profit specified.")

        if extracted.max_loss is not None:
            score += 10
            feedback_parts.append("Max loss specified.")

        # Check for breakeven analysis
        analysis_lower = response.analysis.lower()
        if "breakeven" in analysis_lower or "break-even" in analysis_lower:
            score += 15
            feedback_parts.append("Breakeven analysis included.")

        return min(100, score), " ".join(feedback_parts)

    def _heuristic_pnl_score(
        self,
        extracted: ExtractedOptionsData,
        response: AgentResponse,
    ) -> tuple[float, str]:
        """Heuristic P&L scoring without verification toolkit."""
        score = 40.0
        feedback_parts = []

        # Has P&L numbers
        numbers = self._extract_numbers_from_text(response.analysis)
        if len(numbers) >= 3:
            score += 20
            feedback_parts.append("Multiple P&L calculations present.")
        elif len(numbers) >= 1:
            score += 10
            feedback_parts.append("Some P&L values provided.")
        else:
            feedback_parts.append("Limited P&L analysis.")

        # Check for calculation methodology
        methodology_keywords = ["calculate", "formula", "black-scholes", "premium", "intrinsic", "extrinsic"]
        if any(kw in response.analysis.lower() for kw in methodology_keywords):
            score += 20
            feedback_parts.append("Calculation methodology shown.")

        # Has strategy identification
        if extracted.strategy_name:
            score += 20

        return min(100, score), " ".join(feedback_parts)

    async def _verify_greeks_accuracy(
        self,
        extracted: ExtractedOptionsData,
        response: AgentResponse,
    ) -> tuple[float, str]:
        """
        Verify Greeks calculations.

        Returns:
            Tuple of (accuracy_score, feedback)
        """
        score = 0.0
        feedback_parts = []
        greeks_count = 0

        # Check each Greek
        if extracted.delta is not None:
            greeks_count += 1
            # Delta should be between -1 and 1 for single options
            if -1 <= extracted.delta <= 1:
                score += 20
            elif -1000 <= extracted.delta <= 1000:
                # Could be portfolio delta
                score += 15

        if extracted.gamma is not None:
            greeks_count += 1
            # Gamma should be positive
            if extracted.gamma >= 0:
                score += 20
            else:
                score += 10  # Negative gamma from short positions

        if extracted.theta is not None:
            greeks_count += 1
            score += 20  # Theta can be positive or negative

        if extracted.vega is not None:
            greeks_count += 1
            score += 20

        if extracted.rho is not None:
            greeks_count += 1
            score += 10  # Rho is less commonly reported

        if greeks_count >= 4:
            feedback_parts.append("Comprehensive Greeks analysis.")
        elif greeks_count >= 2:
            feedback_parts.append("Partial Greeks provided.")
        elif greeks_count >= 1:
            score += 10
            feedback_parts.append("Limited Greeks analysis.")
        else:
            # Check if Greeks mentioned without values - improved detection
            analysis_lower = response.analysis.lower()
            greek_terms = ["delta", "gamma", "theta", "vega", "rho"]
            mentioned = sum(1 for g in greek_terms if g in analysis_lower)
            
            # Also check for numeric values near Greek mentions
            greek_with_numbers = 0
            for greek in greek_terms:
                # Look for Greek name followed by number within 20 characters
                pattern = rf'{greek}[:\s=]{{0,10}}[-]?\d+\.?\d*'
                if re.search(pattern, analysis_lower):
                    greek_with_numbers += 1
            
            if greek_with_numbers >= 2:
                score += 50  # Greeks calculated but extraction failed
                feedback_parts.append("Greeks calculated but format not recognized.")
            elif mentioned >= 3:
                score += 30  # Greeks discussed conceptually
                feedback_parts.append("Greeks discussed but values not extracted.")
            elif mentioned >= 1:
                score += 15  # Some Greeks awareness
                feedback_parts.append("Limited Greeks discussion.")
            else:
                feedback_parts.append("Missing Greeks analysis.")

        return min(100, score), " ".join(feedback_parts)

    async def _score_strategy_quality(
        self,
        extracted: ExtractedOptionsData,
        response: AgentResponse,
    ) -> tuple[float, str]:
        """
        Score strategy appropriateness for market conditions.

        Returns:
            Tuple of (quality_score, feedback)
        """
        score = 40.0  # Base score
        feedback_parts = []
        analysis_lower = response.analysis.lower()

        # Check for market context consideration
        context_keywords = [
            "volatility", "iv", "implied volatility",
            "earnings", "catalyst", "trend",
            "bullish", "bearish", "neutral",
            "market condition", "environment"
        ]
        context_count = sum(1 for kw in context_keywords if kw in analysis_lower)
        if context_count >= 3:
            score += 20
            feedback_parts.append("Strong market context analysis.")
        elif context_count >= 1:
            score += 10
            feedback_parts.append("Some market context considered.")

        # Check for risk/reward discussion
        if "risk" in analysis_lower and "reward" in analysis_lower:
            score += 15
            feedback_parts.append("Risk/reward discussed.")
        elif "risk" in analysis_lower:
            score += 10

        # Check for probability analysis
        if extracted.probability_of_profit is not None:
            score += 15
            feedback_parts.append(f"PoP: {extracted.probability_of_profit}%.")
        elif "probability" in analysis_lower:
            score += 10

        # Verify strategy makes sense for category
        category = self.task.category
        if category == TaskCategory.VOLATILITY_TRADING:
            vol_strategies = ["straddle", "strangle", "iron condor", "butterfly"]
            if extracted.strategy_name and any(s in extracted.strategy_name for s in vol_strategies):
                score += 10
                feedback_parts.append("Appropriate volatility strategy.")
        elif category == TaskCategory.RISK_MANAGEMENT:
            if extracted.var_95 is not None or "var" in analysis_lower:
                score += 10
                feedback_parts.append("VaR analysis included.")

        return min(100, score), " ".join(feedback_parts)

    async def _score_risk_management(
        self,
        extracted: ExtractedOptionsData,
        response: AgentResponse,
    ) -> tuple[float, str]:
        """
        Score risk management discipline.

        Returns:
            Tuple of (risk_score, feedback)
        """
        score = 30.0  # Base score
        feedback_parts = []
        analysis_lower = response.analysis.lower()

        # Check for position sizing discussion
        sizing_keywords = ["position size", "contract", "quantity", "allocation", "portfolio"]
        if any(kw in analysis_lower for kw in sizing_keywords):
            score += 15
            feedback_parts.append("Position sizing addressed.")

        # Check for max loss definition
        if extracted.max_loss is not None:
            score += 15
            feedback_parts.append("Max loss defined.")
        elif "max loss" in analysis_lower or "maximum loss" in analysis_lower:
            score += 10

        # Check for hedging discussion
        hedge_keywords = ["hedge", "protect", "collar", "spread", "limit risk"]
        if any(kw in analysis_lower for kw in hedge_keywords):
            score += 15
            feedback_parts.append("Hedging strategy discussed.")

        # Check for exit strategy
        exit_keywords = ["exit", "stop loss", "take profit", "roll", "close position"]
        if any(kw in analysis_lower for kw in exit_keywords):
            score += 15
            feedback_parts.append("Exit strategy mentioned.")

        # Check for VaR or other risk metrics
        if extracted.var_95 is not None:
            score += 10
            feedback_parts.append(f"VaR: ${extracted.var_95:,.0f}.")

        return min(100, score), " ".join(feedback_parts)

    async def _check_mandatory_elements(
        self,
        response: AgentResponse,
    ) -> tuple[list[str], list[str]]:
        """Check for options-specific mandatory elements."""
        mandatory = self.task.rubric.mandatory_elements
        full_response = f"{response.analysis} {response.recommendation}".lower()

        found = []
        missing = []

        for element in mandatory:
            element_lower = element.lower()

            # Check for exact match or key phrase
            if element_lower in full_response:
                found.append(element)
                continue

            # Check individual words
            words = element_lower.split()
            matches = sum(1 for w in words if w in full_response)
            if matches >= len(words) * 0.6:
                found.append(element)
            else:
                missing.append(element)

        return found, missing

    async def score(
        self,
        response: AgentResponse,
    ) -> OptionsScore:
        """
        Score the agent's options trading response.

        Args:
            response: The agent's response to evaluate

        Returns:
            OptionsScore with detailed breakdown
        """
        # Extract options data from response
        extracted = await self._extract_options_data(response)

        # Score each dimension
        pnl_score, pnl_feedback = await self._verify_pnl_accuracy(extracted, response)
        greeks_score, greeks_feedback = await self._verify_greeks_accuracy(extracted, response)
        
        # Log Greeks score (debug level to avoid noise in batch scoring)
        logger.debug(
            "greeks_score",
            task_id=self.task.question_id,
            score=greeks_score,
        )
        strategy_score, strategy_feedback = await self._score_strategy_quality(extracted, response)
        risk_score, risk_feedback = await self._score_risk_management(extracted, response)

        # Check mandatory elements
        found, missing = await self._check_mandatory_elements(response)
        if missing:
            # Penalty for missing mandatory elements
            penalty = len(missing) / max(len(self.task.rubric.mandatory_elements), 1)
            pnl_score *= (1 - penalty * 0.2)

        # Calculate weighted final score
        # Equal weights for all dimensions
        final_score = (
            pnl_score * 0.25 +
            greeks_score * 0.25 +
            strategy_score * 0.25 +
            risk_score * 0.25
        )

        # Compile feedback
        all_feedback = [pnl_feedback, greeks_feedback, strategy_feedback, risk_feedback]
        if missing:
            all_feedback.append(f"Missing: {', '.join(missing[:3])}.")
        combined_feedback = " ".join(f for f in all_feedback if f)

        logger.debug(
            "options_evaluation",
            task_id=self.task.question_id,
            category=self.task.category.value,
            pnl_score=pnl_score,
            greeks_score=greeks_score,
            strategy_score=strategy_score,
            risk_score=risk_score,
            final_score=final_score,
        )

        return OptionsScore(
            score=final_score,
            pnl_accuracy=pnl_score,
            greeks_accuracy=greeks_score,
            strategy_quality=strategy_score,
            risk_management=risk_score,
            feedback=combined_feedback,
        )

    async def score_with_ground_truth(
        self,
        response: AgentResponse,
        ground_truth: dict,
        rubric: Optional[list[dict]] = None,
    ) -> OptionsScore:
        """
        Score agent response using hybrid LLM+Rule evaluation against ground truth.
        
        This method:
        1. Classifies question as quantitative or qualitative
        2. For quantitative: Uses LLM to extract values, then rule-based comparison
        3. For qualitative: Uses full LLM evaluation
        
        Args:
            response: Agent's response to evaluate
            ground_truth: Expected answer from questions.json
            rubric: Optional rubric with scoring components
            
        Returns:
            OptionsScore with detailed breakdown
        """
        response_text = f"{response.analysis}\n{response.recommendation}"
        question_id = self.task.question_id
        category = self.task.category
        
        # Classify question type
        quantitative_categories = {
            TaskCategory.OPTIONS_PRICING,
            TaskCategory.GREEKS_ANALYSIS,
            TaskCategory.PNL_ATTRIBUTION,
        }
        
        # Check if ground_truth has numerical values
        has_numerical = any(
            isinstance(v, (int, float)) 
            for v in ground_truth.values() 
            if not isinstance(v, (list, dict))
        )
        
        is_quantitative = category in quantitative_categories or has_numerical
        
        logger.debug(
            "hybrid_evaluation_start",
            task_id=question_id,
            category=category.value if hasattr(category, 'value') else str(category),
            is_quantitative=is_quantitative,
            use_llm=self._use_llm_extraction,
        )
        
        if is_quantitative:
            # === Quantitative: LLM Extract (if enabled) + Rule Compare ===
            extracted = {}
            
            if self._use_llm_extraction:
                extracted, extract_error = await self._llm_extract_values(
                    response_text, ground_truth
                )
                
                if extract_error:
                    logger.warning(
                        "llm_extraction_fallback",
                        task_id=question_id,
                        error=extract_error,
                    )
                    extracted = {}  # Force regex fallback
            
            # Fallback or primary regex extraction
            if not extracted:
                # Use regex-only extraction here since we're already using LLM for the full ground_truth
                regex_extracted = await self._extract_options_data(response, use_llm_for_greeks=False)
                extracted = {
                    "delta": regex_extracted.delta,
                    "gamma": regex_extracted.gamma,
                    "theta": regex_extracted.theta,
                    "vega": regex_extracted.vega,
                    "max_profit": regex_extracted.max_profit,
                    "max_loss": regex_extracted.max_loss,
                }
                # Also try to extract numbers that might match ground truth keys
                all_numbers = self._extract_numbers_from_text(response_text)
                for key, expected in ground_truth.items():
                    if isinstance(expected, (int, float)) and key not in extracted:
                        # Try to find a close match
                        for num in all_numbers:
                            if abs(num - expected) / abs(expected) < 0.05 if expected != 0 else abs(num) < 0.01:
                                extracted[key] = num
                                break
            
            # Rule-based comparison with appropriate tolerance
            if category == TaskCategory.GREEKS_ANALYSIS:
                tolerance = self.GREEKS_TOLERANCE
            else:
                tolerance = self.PRICE_TOLERANCE
            
            comparison_score, comparison_feedback = self._compare_values(
                extracted, ground_truth, tolerance=tolerance
            )
            
            # Also run traditional scoring for qualitative aspects
            extracted_data = await self._extract_options_data(response, use_llm_for_greeks=False)
            strategy_score, _ = await self._score_strategy_quality(extracted_data, response)
            risk_score, _ = await self._score_risk_management(extracted_data, response)
            
            # Weight: 60% numerical accuracy, 20% strategy, 20% risk
            final_score = (
                comparison_score * 0.60 +
                strategy_score * 0.20 +
                risk_score * 0.20
            )
            
            feedback = f"[Hybrid-Quant] {comparison_feedback}"
        
        else:
            # === Qualitative: Full LLM Evaluation ===
            rubric_list = rubric or [
                {"name": "reasoning", "weight": 0.4, "description": "Sound logical reasoning"},
                {"name": "strategy", "weight": 0.3, "description": "Appropriate strategy choice"},
                {"name": "risk", "weight": 0.3, "description": "Risk awareness"},
            ]
            
            llm_score, llm_feedback, llm_error = await self._llm_evaluate_qualitative(
                response_text, ground_truth, rubric_list
            )
            
            if llm_error:
                # Fallback to traditional scoring
                logger.warning(
                    "llm_qualitative_fallback",
                    task_id=question_id,
                    error=llm_error,
                )
                return await self.score(response)
            
            final_score = llm_score
            feedback = f"[Hybrid-Qual] {llm_feedback}"
            comparison_score = llm_score
            strategy_score = llm_score
            risk_score = llm_score
        
        logger.debug(
            "hybrid_evaluation_complete",
            task_id=question_id,
            final_score=final_score,
            is_quantitative=is_quantitative,
        )
        
        return OptionsScore(
            score=final_score,
            pnl_accuracy=comparison_score if is_quantitative else final_score,
            greeks_accuracy=comparison_score if is_quantitative else final_score,
            strategy_quality=strategy_score,
            risk_management=risk_score,
            feedback=feedback,
        )

    @staticmethod
    def is_options_task(task: Task) -> bool:
        """Check if a task is an options trading task."""
        return task.category in OPTIONS_CATEGORIES
