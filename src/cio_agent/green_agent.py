"""
Green Agent Implementation for AgentBeats Platform

This is the core agent logic that orchestrates evaluation of Purple Agents.
It receives an assessment request with participant agent URLs and config,
then runs the FAB++ evaluation pipeline.

Supported modes:
    - config: Use YAML config file for multi-dataset evaluation
    - synthetic: Generated questions from JSON file
    - bizfinbench: HiThink BizFinBench.v2 dataset (single task type)
    - prbench: Scale AI PRBench dataset (professional reasoning)
    - crypto: Crypto trading benchmark scenarios (config mode)
"""

import asyncio
import json
import os
import httpx
import logging
from pathlib import Path
from typing import Any, Optional, List, Union
from pydantic import BaseModel

logger = logging.getLogger("cio_agent.green_agent")
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from cio_agent.messenger import Messenger
from cio_agent.evaluator import ComprehensiveEvaluator, EvaluationReporter
from cio_agent.task_generator import DynamicTaskGenerator
from cio_agent.models import Task as FABTask, TaskCategory, TaskDifficulty, GroundTruth, FinancialData, TaskRubric
from cio_agent.eval_config import (
    EvaluationConfig,
    ConfigurableDatasetLoader,
    LoadedExample,
    create_default_config,
)
from cio_agent.agentbeats_results import format_and_save_results
from cio_agent.unified_scoring import UnifiedScorer, ScoreSection, DATASET_SECTION_MAP
from cio_agent.crypto_benchmark import CryptoTradingEvaluator, stable_seed

# Dataset providers (for legacy single-dataset mode)
from cio_agent.data_providers import BizFinBenchProvider, CsvFinanceDatasetProvider

# Dataset-specific evaluators
from evaluators import BizFinBenchEvaluator, PRBenchEvaluator, OptionsEvaluator, SyntheticEvaluator
from evaluators.gdpval_evaluator import GDPValEvaluator
from evaluators.llm_utils import build_llm_client, should_use_llm


class GreenAgent:
    """
    CIO-Agent Green Agent for FAB++ Finance Agent Benchmark.
    
    This agent evaluates Purple Agents on their financial analysis capabilities
    using the FAB++ evaluation framework.
    
    Initialization modes:
        1. Config-based (recommended): Pass eval_config for multi-dataset support
        2. Legacy single-dataset: Use dataset_type, dataset_path, etc.
        3. Synthetic: Use synthetic_questions list
    
    Required roles:
        - purple_agent: The finance agent being evaluated
        
    Config options:
        - num_tasks: Number of evaluation tasks (default: 1)
        - conduct_debate: Whether to run adversarial debate (default: True)
    """
    
    # Required participant roles
    required_roles: list[str] = ["purple_agent"]
    
    # Required config keys (optional ones will have defaults)
    required_config_keys: list[str] = []

    def validate_request(self, request: "EvalRequest") -> tuple[bool, str]:
        """Validate required roles and config keys for an evaluation request."""
        missing_roles = [role for role in self.required_roles if role not in request.participants]
        if missing_roles:
            return False, f"Missing roles: {', '.join(missing_roles)}"

        missing_keys = [key for key in self.required_config_keys if key not in request.config]
        if missing_keys:
            return False, f"Missing config keys: {', '.join(missing_keys)}"

        return True, "ok"


    def __init__(
        self,
        eval_config: Optional[Union[EvaluationConfig, str, Path]] = None,
        synthetic_questions: Optional[List[dict]] = None,
        dataset_type: str = "synthetic",
        dataset_path: Optional[str] = None,
        task_type: Optional[str] = None,
        language: str = "en",
        limit: Optional[int] = None,
        eval_use_llm: Optional[bool] = None,
        eval_llm_model: Optional[str] = None,
        eval_llm_temperature: Optional[float] = None,
        store_predicted: bool = False,
        truncate_predicted: Optional[bool] = None,
        predicted_max_chars: Optional[int] = None,
        store_question: bool = False,
        truncate_question: Optional[bool] = None,
        question_max_chars: Optional[int] = None,
        store_expected: bool = False,
        truncate_expected: Optional[bool] = None,
        expected_max_chars: Optional[int] = None,
    ):
        """
        Initialize the Green Agent.
        
        Args:
            eval_config: Configuration for multi-dataset evaluation.
                        Can be EvaluationConfig, path to YAML file, or None.
            synthetic_questions: Optional list of synthetic questions to use
                                for evaluation. If provided, these will be used
                                instead of generating new tasks.
            dataset_type: Type of dataset to use ('synthetic', 'bizfinbench', 'prbench')
            dataset_path: Path to dataset directory or file
            task_type: For BizFinBench, the specific task type to evaluate
            language: Language for BizFinBench ('en' or 'cn')
            limit: Optional limit on number of examples
            eval_use_llm: Optional override to enable/disable LLM grading
            eval_llm_model: Optional LLM model override for grading
            eval_llm_temperature: Optional temperature override for grading
            store_predicted: Whether to store predicted outputs in results
            truncate_predicted: Optional override to truncate predicted outputs
            predicted_max_chars: Optional max length for predicted outputs
            store_question: Whether to store full question text in results
            truncate_question: Optional override to truncate question text
            question_max_chars: Optional max length for question text
            store_expected: Whether to store full expected answer in results
            truncate_expected: Optional override to truncate expected answer
            expected_max_chars: Optional max length for expected answer
        """
        self.messenger = Messenger()
        self.evaluator = ComprehensiveEvaluator()
        self.task_generator = DynamicTaskGenerator()
        self.synthetic_questions = synthetic_questions or []
        
        # Legacy dataset configuration
        self.dataset_type = dataset_type
        self.dataset_path = dataset_path
        self.task_type = task_type
        self.language = language
        self.limit = limit
        
        # Config-based multi-dataset support
        self.eval_config: Optional[EvaluationConfig] = None
        self.dataset_loader: Optional[ConfigurableDatasetLoader] = None
        self._loaded_examples: Optional[List[LoadedExample]] = None
        
        # Initialize based on provided mode
        self.dataset_provider = None
        self.dataset_evaluator = None
        self._examples = None  # Legacy cached examples
        
        # Priority: eval_config > single dataset > synthetic
        if eval_config is not None:
            # Config-based multi-dataset mode
            if isinstance(eval_config, (str, Path)):
                self.eval_config = EvaluationConfig.from_yaml(eval_config)
            else:
                self.eval_config = eval_config
            
            self.dataset_loader = ConfigurableDatasetLoader(self.eval_config)
            self._loaded_examples = self.dataset_loader.load()

        config_llm = self.eval_config.llm_eval if self.eval_config else None
        config_use_llm = config_llm.enabled if config_llm and config_llm.enabled is not None else None
        config_llm_model = config_llm.model if config_llm and config_llm.model else None
        config_llm_temp = (
            config_llm.temperature if config_llm and config_llm.temperature is not None else None
        )
        # Extract api_base and api_key for separate evaluator API endpoint
        config_api_base = config_llm.api_base if config_llm and config_llm.api_base else None
        config_api_key = config_llm.api_key if config_llm and config_llm.api_key else None

        if eval_use_llm is not None:
            self.use_llm = eval_use_llm
        elif config_use_llm is not None:
            self.use_llm = config_use_llm
        else:
            self.use_llm = should_use_llm()

        self.llm_model = eval_llm_model or config_llm_model
        self.llm_temperature = (
            eval_llm_temperature if eval_llm_temperature is not None else config_llm_temp
        )
        self.llm_client = build_llm_client(api_base=config_api_base, api_key=config_api_key) if self.use_llm else None
        if self.use_llm and self.llm_client is None:
            self.use_llm = False

        self.store_predicted = store_predicted
        if truncate_predicted is None:
            truncate_predicted = True
        self.truncate_predicted = truncate_predicted

        if predicted_max_chars is None:
            predicted_max_chars = 200
        self.predicted_max_chars = predicted_max_chars
        if self.truncate_predicted and self.predicted_max_chars <= 0:
            self.predicted_max_chars = 200

        # Question storage options
        self.store_question = store_question
        if truncate_question is None:
            truncate_question = True
        self.truncate_question = truncate_question

        if question_max_chars is None:
            question_max_chars = 200
        self.question_max_chars = question_max_chars
        if self.truncate_question and self.question_max_chars <= 0:
            self.question_max_chars = 200

        # Expected answer storage options
        self.store_expected = store_expected
        if truncate_expected is None:
            truncate_expected = True
        self.truncate_expected = truncate_expected

        if expected_max_chars is None:
            expected_max_chars = 100
        self.expected_max_chars = expected_max_chars
        if self.truncate_expected and self.expected_max_chars <= 0:
            self.expected_max_chars = 100

        if self.eval_config is not None:
            # Initialize evaluators for each dataset type present
            self._evaluators = {
                "bizfinbench": BizFinBenchEvaluator(
                    use_llm=self.use_llm,
                    llm_client=self.llm_client,
                    llm_model=self.llm_model,
                    llm_temperature=self.llm_temperature,
                ),
                "prbench": PRBenchEvaluator(
                    use_llm=self.use_llm,
                    llm_client=self.llm_client,
                    llm_model=self.llm_model,
                    llm_temperature=self.llm_temperature,
                ),
                "gdpval": GDPValEvaluator(
                    use_llm=self.use_llm,
                    llm_client=self.llm_client,
                    llm_model=self.llm_model,
                    llm_temperature=self.llm_temperature,
                ),
                "synthetic": SyntheticEvaluator(
                    use_llm=self.use_llm,
                    llm_client=self.llm_client,
                    llm_model=self.llm_model,
                    llm_temperature=self.llm_temperature,
                ),
                "options": None,  # Options use OptionsEvaluator initialized per-task
                "crypto": None,  # Crypto uses CryptoTradingEvaluator initialized per-scenario
            }
            
        elif dataset_type == "bizfinbench" and dataset_path:
            # Legacy single BizFinBench dataset
            self.dataset_provider = BizFinBenchProvider(
                base_path=dataset_path,
                task_type=task_type or "event_logic_reasoning",
                language=language,
                limit=limit,
            )
            self.dataset_evaluator = BizFinBenchEvaluator(
                use_llm=self.use_llm,
                llm_client=self.llm_client,
                llm_model=self.llm_model,
                llm_temperature=self.llm_temperature,
            )
            self._examples = self.dataset_provider.load()
            
        elif dataset_type == "prbench":
            # Legacy single PRBench dataset
            from cio_agent.data_providers import PRBenchProvider
            self.dataset_provider = PRBenchProvider(
                splits=["finance", "legal"],
                limit=limit,
            )
            self.dataset_evaluator = PRBenchEvaluator(
                use_llm=self.use_llm,
                llm_client=self.llm_client,
                llm_model=self.llm_model,
                llm_temperature=self.llm_temperature,
            )
            examples = self.dataset_provider.load()
            self._examples = examples[:limit] if limit else examples

    async def run_eval(self, request: Any, updater: TaskUpdater) -> None:
        """
        Run evaluation with EvalRequest from agentbeats-client.

        Args:
            request: EvalRequest with participants and config
            updater: TaskUpdater for reporting progress and results
        """
        # Get participant URL from request (first participant)
        participants = request.participants
        if not participants:
            raise ValueError("No participants provided in request")

        purple_agent_url = str(list(participants.values())[0])
        config = request.config or {}

        # Get config values with defaults
        num_tasks = int(config.get("num_tasks", os.environ.get("EVAL_NUM_TASKS", "10")))
        conduct_debate = config.get("conduct_debate", os.environ.get("EVAL_CONDUCT_DEBATE", "false"))
        if isinstance(conduct_debate, str):
            conduct_debate = conduct_debate.lower() == "true"

        await self._run_evaluation(
            purple_agent_url=purple_agent_url,
            num_tasks=num_tasks,
            conduct_debate=conduct_debate,
            updater=updater,
        )

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """
        Run the FAB++ evaluation assessment (legacy method).

        Args:
            message: The incoming A2A message (plain text trigger)
            updater: TaskUpdater for reporting progress and results
        """
        input_text = get_message_text(message)

        # Get configuration from environment variables (set by docker-compose/scenario)
        purple_agent_url = os.environ.get("PURPLE_AGENT_URL", "http://purple_agent:9009")
        num_tasks = int(os.environ.get("EVAL_NUM_TASKS", "10"))
        conduct_debate = os.environ.get("EVAL_CONDUCT_DEBATE", "false").lower() == "true"

        await self._run_evaluation(
            purple_agent_url=purple_agent_url,
            num_tasks=num_tasks,
            conduct_debate=conduct_debate,
            updater=updater,
        )

    async def _run_evaluation(
        self,
        purple_agent_url: str,
        num_tasks: int,
        conduct_debate: bool,
        updater: TaskUpdater,
    ) -> None:
        """
        Internal method to run the actual evaluation.

        Args:
            purple_agent_url: URL of the purple agent to evaluate
            num_tasks: Number of tasks to evaluate
            conduct_debate: Whether to conduct adversarial debate
            updater: TaskUpdater for reporting progress and results
        """
        ticker = os.environ.get("EVAL_TICKER", "NVDA")

        # Report starting
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Starting FAB++ evaluation for {ticker}...")
        )

        try:
            # Generate evaluation task(s)
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("Generating evaluation tasks...")
            )
            
            # Get simulation date from environment or use current date
            from datetime import datetime
            simulation_date_str = os.environ.get("SIMULATION_DATE")
            if simulation_date_str:
                simulation_date = datetime.fromisoformat(simulation_date_str)
            else:
                simulation_date = datetime.now()
            
            # Use synthetic questions if available, otherwise generate tasks
            # Priority: config-based > legacy single dataset > synthetic > dynamic generation
            if self._loaded_examples is not None:
                # Config-based multi-dataset evaluation
                summary = self.dataset_loader.summary()
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        f"Using {summary['total']} examples from {len(summary['by_dataset'])} datasets..."
                    )
                )
                all_results = await self._evaluate_with_config(
                    purple_agent_url=purple_agent_url,
                    num_tasks=num_tasks,
                    conduct_debate=conduct_debate,
                    updater=updater,
                )

                # Use unified scoring system
                scorer = UnifiedScorer()
                normalized_results = []

                for r in all_results:
                    if "error" in r:
                        continue

                    dataset_type = r.get("dataset_type", "unknown")
                    raw_score = r.get("score", 0.0)
                    is_correct = r.get("is_correct", False)

                    # Extract sub-scores for options
                    sub_scores = {}
                    if dataset_type == "options":
                        sub_scores = {
                            "pnl_accuracy": r.get("pnl_accuracy", 0),
                            "greeks_accuracy": r.get("greeks_accuracy", 0),
                            "strategy_quality": r.get("strategy_quality", 0),
                            "risk_management": r.get("risk_management", 0),
                        }
                    elif dataset_type == "crypto":
                        sub_scores = {
                            "baseline": r.get("baseline_score", 0),
                            "noisy": r.get("noisy_score", 0),
                            "adversarial": r.get("adversarial_score", 0),
                            "meta": r.get("meta_score", 0),
                        }
                    elif dataset_type == "gdpval":
                        sub_scores = {
                            "completion": r.get("completion", 0),
                            "accuracy": r.get("accuracy", 0),
                            "format": r.get("format", 0),
                            "professionalism": r.get("professionalism", 0),
                        }

                    normalized = scorer.create_normalized_result(
                        task_id=r.get("example_id", ""),
                        dataset_type=dataset_type,
                        raw_score=raw_score,
                        is_correct=is_correct,
                        feedback=r.get("feedback", ""),
                        sub_scores=sub_scores,
                    )
                    if normalized:
                        normalized_results.append(normalized)

                # Compute unified result
                unified_result = scorer.compute_unified_result(
                    task_results=normalized_results,
                    purple_agent_url=purple_agent_url,
                    conduct_debate=conduct_debate,
                )
                if set(summary["by_dataset"].keys()) == {"crypto"}:
                    unified_result.benchmark = "AgentBusters Crypto Trading Benchmark"
                    unified_result.version = "1.0.0"

                # Convert to dict for serialization
                assessment_result = unified_result.to_dict()

                # Add config summary for compatibility
                assessment_result["config_summary"] = summary
                assessment_result["results"] = all_results  # Keep detailed results

                # Save AgentBeats-compliant results
                participant_id = os.environ.get("AGENTBEATS_PURPLE_AGENT_ID", "")
                participant_name = os.environ.get("PARTICIPANT_NAME", "purple_agent")
                scenario_id = os.environ.get("AGENTBEATS_SCENARIO_ID", "")
                green_agent_id = os.environ.get("AGENTBEATS_GREEN_AGENT_ID", "")

                try:
                    results_path, leaderboard_path = format_and_save_results(
                        participant_id=participant_id,
                        participant_name=participant_name,
                        evaluation_results=assessment_result,
                        by_dataset=None,  # Unified result handles this differently
                        scenario_id=scenario_id,
                        green_agent_id=green_agent_id,
                        results_dir="results",
                    )
                    import structlog
                    logger = structlog.get_logger()
                    logger.info("agentbeats_results_saved", results_path=str(results_path), leaderboard_path=str(leaderboard_path))
                except Exception as e:
                    import structlog
                    logger = structlog.get_logger()
                    logger.warning("agentbeats_results_save_failed", error=str(e))

                # Report results as artifact
                overall = unified_result.overall_score
                section_summary = "\n".join(
                    f"  {name}: {ss.score:.1f}/100 (weight: {ss.weight:.0%}, {ss.task_count} tasks)"
                    for name, ss in unified_result.section_scores.items()
                )
                await updater.add_artifact(
                    parts=[
                        Part(root=TextPart(text=f"FAB++ Unified Evaluation Complete\n\nOverall Score: {overall.score:.1f}/100 (Grade: {overall.grade})\n\nSection Scores:\n{section_summary}")),
                        Part(root=DataPart(data=assessment_result)),
                    ],
                    name="evaluation_result",
                )
                return
            
            elif self._examples and self.dataset_type in ("bizfinbench", "prbench"):
                # Legacy: Use single dataset examples directly
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Using {len(self._examples)} {self.dataset_type} examples for evaluation...")
                )
                # For dataset-based evaluation, we'll use a different flow
                all_results = await self._evaluate_with_dataset(
                    purple_agent_url=purple_agent_url,
                    num_tasks=num_tasks,
                    conduct_debate=conduct_debate,
                    updater=updater,
                )
                
                # Calculate aggregate metrics
                valid_results = [r for r in all_results if "error" not in r]
                avg_score = sum(r.get("score", 0) for r in valid_results) / len(valid_results) if valid_results else 0.0
                accuracy = sum(1 for r in valid_results if r.get("is_correct", False)) / len(valid_results) if valid_results else 0.0
                
                # Create assessment result
                assessment_result = {
                    "benchmark": f"FAB++ {self.dataset_type}",
                    "version": "1.1.0",
                    "purple_agent": purple_agent_url,
                    "dataset_type": self.dataset_type,
                    "task_type": self.task_type,
                    "language": self.language,
                    "num_examples": len(self._examples),
                    "num_evaluated": len(all_results),
                    "num_successful": len(valid_results),
                    "average_score": round(avg_score, 4),
                    "accuracy": round(accuracy, 4),
                    "results": all_results,
                }
                
                # Report results as artifact
                await updater.add_artifact(
                    parts=[
                        Part(root=TextPart(text=f"FAB++ {self.dataset_type} Evaluation Complete\\n\\nAverage Score: {avg_score:.4f}\\nAccuracy: {accuracy:.2%}")),
                        Part(root=DataPart(data=assessment_result)),
                    ],
                    name="evaluation_result",
                )
                return
            
            elif self.synthetic_questions:
                tasks = self._convert_synthetic_to_tasks(num_tasks)
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Using {len(tasks)} synthetic questions for evaluation...")
                )
            else:
                tasks = await self.task_generator.generate_task_batch(
                    count=num_tasks,
                    simulation_date=simulation_date,
                )
            
            if not tasks:
                # Create a default task if generation fails
                from datetime import datetime
                from cio_agent.models import GroundTruth, FinancialData, TaskRubric

                # Convert task_category string to enum
                try:
                    default_category = TaskCategory(task_category)
                except ValueError:
                    default_category = TaskCategory.BEAT_OR_MISS

                tasks = [FABTask(
                    question_id=f"fab_{ticker}_eval",
                    category=default_category,
                    question=f"Did {ticker} beat or miss analyst expectations in the most recent quarter?",
                    ticker=ticker,
                    fiscal_year=2026,
                    simulation_date=datetime.now(),
                    ground_truth=GroundTruth(
                        macro_thesis="Evaluate earnings performance",
                        key_themes=["revenue", "earnings", "guidance"],
                        financials=FinancialData(),
                        expected_recommendation="Evaluate",
                    ),
                    difficulty=TaskDifficulty.MEDIUM,
                    rubric=TaskRubric(
                        criteria=["Accuracy", "Analysis depth", "Recommendation quality"],
                        mandatory_elements=["beat/miss determination"],
                    ),
                )]

            all_results = []
            
            for i, task in enumerate(tasks):
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Evaluating task {i+1}/{len(tasks)}: {task.question_id}")
                )
                
                # Send task to Purple Agent
                task_message = json.dumps({
                    "question": task.question,
                    "ticker": task.ticker,
                    "fiscal_year": task.fiscal_year,
                    "category": task.category.value,
                }, ensure_ascii=False)
                
                try:
                    response = await self.messenger.talk_to_agent(
                        message=task_message,
                        url=purple_agent_url,
                        new_conversation=True,
                        timeout=300,
                    )
                    
                    # Parse agent response
                    from cio_agent.models import AgentResponse, FinancialData as FD
                    agent_response = AgentResponse(
                        agent_id="purple_agent",
                        task_id=task.question_id,
                        analysis=response,
                        recommendation=self._extract_recommendation(response),
                        extracted_financials=FD(),  # Would be parsed from response
                        tool_calls=[],
                        code_executions=[],
                        execution_time_seconds=0.0,
                    )
                    
                    # Conduct debate if enabled
                    agent_rebuttal = None
                    if conduct_debate:
                        await updater.update_status(
                            TaskState.working,
                            new_agent_text_message("Conducting adversarial debate...")
                        )
                        
                        counter_arg = f"Challenge: What are the key risks to your {ticker} analysis?"
                        rebuttal_response = await self.messenger.talk_to_agent(
                            message=counter_arg,
                            url=purple_agent_url,
                            new_conversation=False,
                        )
                        
                        from cio_agent.models import DebateRebuttal
                        agent_rebuttal = DebateRebuttal(
                            agent_id="purple_agent",
                            task_id=task.question_id,
                            defense=rebuttal_response,
                        )
                    
                    # Evaluate response
                    result = await self.evaluator.evaluate_response(
                        task=task,
                        agent_response=agent_response,
                        agent_rebuttal=agent_rebuttal,
                    )
                    
                    all_results.append({
                        "task_id": task.question_id,
                        "alpha_score": result.alpha_score.score,
                        "role_score": result.role_score.total,
                        "debate_multiplier": result.debate_result.debate_multiplier,
                    })
                    
                except Exception as e:
                    all_results.append({
                        "task_id": task.question_id,
                        "error": str(e),
                        "alpha_score": 0.0,
                    })

            # Calculate aggregate metrics
            valid_results = [r for r in all_results if "error" not in r]
            avg_alpha = sum(r["alpha_score"] for r in valid_results) / len(valid_results) if valid_results else 0.0
            
            # Create assessment result
            assessment_result = {
                "benchmark": "FAB++ Finance Agent Benchmark",
                "version": "1.0.0",
                "purple_agent": purple_agent_url,
                "ticker": ticker,
                "num_tasks": len(tasks),
                "num_successful": len(valid_results),
                "average_alpha_score": round(avg_alpha, 2),
                "results": all_results,
            }
            
            # Report results as artifact
            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text=f"FAB++ Evaluation Complete\n\nAverage Alpha Score: {avg_alpha:.2f}")),
                    Part(root=DataPart(data=assessment_result)),
                ],
                name="evaluation_result",
            )

        except Exception as e:
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(f"Evaluation failed: {str(e)}")
            )
            raise

    def _extract_recommendation(self, response: str) -> str:
        """Extract recommendation from agent response."""
        response_lower = response.lower()
        if "beat" in response_lower:
            return "Beat"
        elif "miss" in response_lower:
            return "Miss"
        elif "buy" in response_lower:
            return "Buy"
        elif "sell" in response_lower:
            return "Sell"
        elif "hold" in response_lower:
            return "Hold"
        return "Unknown"

    def _format_predicted(self, response: str) -> str:
        if not self.store_predicted:
            return ""
        if not self.truncate_predicted or self.predicted_max_chars <= 0:
            return response
        if len(response) <= self.predicted_max_chars:
            return response
        return response[: self.predicted_max_chars] + "..."

    def _format_question(self, question: str) -> str:
        """Format question text based on storage and truncation settings."""
        if self.store_question:
            # User wants full question stored
            if not self.truncate_question or self.question_max_chars <= 0:
                return question
            if len(question) <= self.question_max_chars:
                return question
            return question[: self.question_max_chars] + "..."
        else:
            # Default behavior: always truncate to 200 chars
            if len(question) <= 200:
                return question
            return question[:200] + "..."

    def _format_expected(self, expected: str) -> str:
        """Format expected answer text based on storage and truncation settings."""
        if not expected:
            return ""
        if self.store_expected:
            # User wants full expected stored
            if not self.truncate_expected or self.expected_max_chars <= 0:
                return expected
            if len(expected) <= self.expected_max_chars:
                return expected
            return expected[: self.expected_max_chars] + "..."
        else:
            # Default behavior: always truncate to 100 chars
            if len(expected) <= 100:
                return expected
            return expected[:100] + "..."

    async def _run_evaluator_async(self, evaluator, **kwargs):
        """Run a synchronous evaluator.evaluate() call in a thread pool.
        
        This prevents blocking the event loop when evaluators call LLM APIs,
        enabling concurrent evaluation of multiple purple agents.
        """
        return await asyncio.to_thread(evaluator.evaluate, **kwargs)

    def _extract_excel_content(self, content: bytes, filename: str) -> str:
        """Extract content from Excel file as formatted text."""
        import io
        try:
            import pandas as pd
            xlsx = pd.ExcelFile(io.BytesIO(content))
            parts = [f"Excel file with {len(xlsx.sheet_names)} sheet(s): {xlsx.sheet_names}\n"]

            for sheet_name in xlsx.sheet_names:
                df = pd.read_excel(xlsx, sheet_name=sheet_name)
                parts.append(f"\n--- Sheet: {sheet_name} ({df.shape[0]} rows × {df.shape[1]} cols) ---\n")
                parts.append(df.to_csv(index=False))

            return "".join(parts)
        except Exception as e:
            logger.warning(f"Failed to extract Excel content from {filename}: {e}")
            return f"[Excel file: {filename} - {len(content)} bytes, extraction failed: {e}]"

    def _extract_pdf_content(self, content: bytes, filename: str) -> str:
        """Extract text content from PDF file."""
        try:
            import pypdf
            import io
            reader = pypdf.PdfReader(io.BytesIO(content))
            parts = [f"PDF file with {len(reader.pages)} page(s)\n"]

            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    parts.append(f"\n--- Page {i+1} ---\n")
                    parts.append(text)

            return "".join(parts)
        except ImportError:
            return f"[PDF file: {filename} - {len(content)} bytes, pypdf not installed]"
        except Exception as e:
            logger.warning(f"Failed to extract PDF content from {filename}: {e}")
            return f"[PDF file: {filename} - {len(content)} bytes, extraction failed: {e}]"

    def _format_reference_files_for_agent(self, metadata: dict) -> str:
        """
        Format reference files as URLs for Purple Agent to fetch on demand.

        Instead of fetching and embedding file contents, this provides the file
        URLs so Purple Agent can use fetch_reference_file tool to retrieve them.

        Args:
            metadata: Example metadata containing reference_file_urls or reference_file_hf_uris

        Returns:
            Formatted string with reference file URLs and instructions
        """
        reference_files = metadata.get("reference_files", [])
        reference_urls = metadata.get("reference_file_urls", [])

        if not reference_files:
            return ""

        lines = []
        lines.append("\n\n--- REFERENCE FILES AVAILABLE ---")
        lines.append("The following reference files are available for this task.")
        lines.append("Use the fetch_reference_file tool to download and read any files you need.\n")

        for i, filename in enumerate(reference_files):
            url = reference_urls[i] if i < len(reference_urls) else None
            # Detect file type from extension
            ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "unknown"
            file_type_map = {
                "pdf": "PDF document",
                "xlsx": "Excel spreadsheet",
                "xls": "Excel spreadsheet",
                "csv": "CSV data file",
                "json": "JSON data file",
                "txt": "Text file",
                "md": "Markdown file",
                "docx": "Word document",
                "png": "Image (PNG)",
                "jpg": "Image (JPEG)",
                "jpeg": "Image (JPEG)",
            }
            file_type = file_type_map.get(ext, f"{ext.upper()} file")

            if url:
                lines.append(f"- {filename} ({file_type})")
                lines.append(f"  URL: {url}")
            else:
                lines.append(f"- {filename} ({file_type}) [URL not available]")

        lines.append("\n--- END REFERENCE FILES ---\n")

        logger.debug(f"Formatted {len(reference_files)} reference file URLs for Purple Agent")
        return "\n".join(lines)

    def _convert_synthetic_to_tasks(self, num_tasks: int) -> list[FABTask]:
        """
        Convert synthetic question dicts to FABTask objects.
        
        Args:
            num_tasks: Maximum number of tasks to return
            
        Returns:
            List of FABTask objects
        """
        from datetime import datetime
        
        tasks = []
        questions_to_use = self.synthetic_questions[:num_tasks]
        
        for sq in questions_to_use:
            # Handle category enum
            category_value = sq.get("category", "Quantitative Retrieval")
            try:
                category = TaskCategory(category_value)
            except ValueError:
                category = TaskCategory.QUANTITATIVE_RETRIEVAL
            
            # Handle difficulty enum
            difficulty_value = sq.get("difficulty", "medium")
            try:
                difficulty = TaskDifficulty(difficulty_value)
            except ValueError:
                difficulty = TaskDifficulty.MEDIUM
            
            # Build ground truth with required fields
            ground_truth = GroundTruth(
                macro_thesis=str(sq.get("ground_truth_formatted", "Evaluate the analysis")),
                key_themes=sq.get("calculation_steps", []),
                expected_recommendation=str(sq.get("ground_truth_formatted", "")),
                financials=FinancialData(),
            )
            
            # Build rubric from components
            rubric_data = sq.get("rubric", {})
            rubric_components = rubric_data.get("components", [])
            rubric = TaskRubric(
                criteria=[c.get("description", "") for c in rubric_components],
                max_score=rubric_data.get("max_score", 100),
            )
            
            task = FABTask(
                question_id=sq.get("question_id", f"SYN_{len(tasks):04d}"),
                category=category,
                difficulty=difficulty,
                question=sq.get("question", ""),
                ticker=sq.get("ticker", "AAPL"),
                fiscal_year=sq.get("fiscal_year", 2024),
                simulation_date=datetime.now(),
                ground_truth=ground_truth,
                rubric=rubric,
                requires_code_execution=sq.get("requires_code_execution", False),
            )
            tasks.append(task)
        
        return tasks

    async def _evaluate_with_dataset(
        self,
        purple_agent_url: str,
        num_tasks: int,
        conduct_debate: bool,
        updater: TaskUpdater,
    ) -> List[dict]:
        """
        Evaluate Purple Agent using dataset examples and dataset-specific evaluator.
        
        Args:
            purple_agent_url: URL of the Purple Agent to evaluate
            num_tasks: Maximum number of examples to evaluate
            conduct_debate: Whether to conduct adversarial debate
            updater: TaskUpdater for progress reporting
            
        Returns:
            List of evaluation results
        """
        all_results = []
        examples_to_eval = self._examples[:num_tasks] if num_tasks else self._examples
        
        for i, example in enumerate(examples_to_eval):
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Evaluating example {i+1}/{len(examples_to_eval)}: {example.example_id}")
            )
            
            try:
                # Send question to Purple Agent
                response = await self.messenger.talk_to_agent(
                    message=example.question,
                    url=purple_agent_url,
                    new_conversation=True,
                    timeout=300,
                )
                predicted_text = self._format_predicted(response)
                
                # Use dataset-specific evaluator
                if self.dataset_type == "bizfinbench":
                    eval_result = await self._run_evaluator_async(
                        self.dataset_evaluator,
                        predicted=response,
                        expected=example.answer,
                        task_type=self.task_type,
                        question=example.question,
                    )
                    result = {
                        "example_id": example.example_id,
                        "task_type": self.task_type,
                        "question": self._format_question(example.question),
                        "expected": self._format_expected(example.answer),
                        "predicted": predicted_text,
                        "score": eval_result.score,
                        "is_correct": eval_result.is_correct,
                        "feedback": eval_result.feedback,
                    }
                    if eval_result.details:
                        for key in ("llm_used", "llm_failure", "llm_raw_output"):
                            if key in eval_result.details:
                                result[key] = eval_result.details.get(key)
                    result["llm_used"] = eval_result.details.get("llm_used", False) if eval_result.details else False
                    result["sub_scores"] = {}
                    
                elif self.dataset_type == "prbench":
                    # PRBench uses weighted rubric criteria
                    rubric = getattr(example, 'rubric', None)
                    rubric_weights = example.metadata.get("rubric_weights", {}) if hasattr(example, 'metadata') else {}

                    eval_result = await self._run_evaluator_async(
                        self.dataset_evaluator,
                        predicted=response,
                        expected=example.answer,
                        rubric=rubric,
                        rubric_weights=rubric_weights,
                        question=example.question,
                        domain=example.metadata.get("domain", "") if hasattr(example, 'metadata') else "",
                        topic=example.metadata.get("topic", "") if hasattr(example, 'metadata') else "",
                    )
                    result = {
                        "example_id": example.example_id,
                        "category": example.category.value if hasattr(example.category, 'value') else str(example.category),
                        "question": self._format_question(example.question),
                        "expected": self._format_expected(example.answer or ""),
                        "predicted": predicted_text,
                        "score": eval_result.score,
                        "is_correct": eval_result.score >= 0.7,  # Threshold for correctness
                        "correct_count": eval_result.correct_count,
                        "total_count": eval_result.total_count,
                        "feedback": eval_result.feedback,
                    }
                    if eval_result.details:
                        for key in (
                            "llm_used",
                            "llm_failure",
                            "llm_raw_output",
                            "matched_criteria",
                            "triggered_penalties",
                        ):
                            if key in eval_result.details:
                                result[key] = eval_result.details.get(key)
                    result["llm_used"] = eval_result.details.get("llm_used", False) if eval_result.details else False
                    result["sub_scores"] = {}
                else:
                    result = {
                        "example_id": example.example_id,
                        "error": f"Unknown dataset type: {self.dataset_type}",
                    }
                
                # Optional debate (simplified for dataset-based evaluation)
                if conduct_debate and eval_result.score > 0:
                    challenge = f"Challenge your analysis. What risks or uncertainties did you consider?"
                    try:
                        rebuttal = await self.messenger.talk_to_agent(
                            message=challenge,
                            url=purple_agent_url,
                            new_conversation=False,
                            timeout=60,
                        )
                        result["rebuttal_received"] = True
                        result["rebuttal_preview"] = rebuttal[:100] + "..." if len(rebuttal) > 100 else rebuttal
                    except Exception:
                        result["rebuttal_received"] = False
                
                all_results.append(result)
                
            except Exception as e:
                all_results.append({
                    "example_id": example.example_id,
                    "error": str(e),
                    "score": 0.0,
                    "is_correct": False,
                })
        
        return all_results



    async def _evaluate_with_config(
        self,
        purple_agent_url: str,
        num_tasks: int,
        conduct_debate: bool,
        updater: TaskUpdater,
    ) -> List[dict]:
        """
        Evaluate Purple Agent using config-based multi-dataset loader.
        
        Args:
            purple_agent_url: URL of the Purple Agent to evaluate
            num_tasks: Maximum number of examples to evaluate
            conduct_debate: Whether to conduct adversarial debate
            updater: TaskUpdater for progress reporting
            
        Returns:
            List of evaluation results
        """
        all_results = []
        crypto_evaluator = None
        examples_to_eval = self._loaded_examples[:num_tasks] if num_tasks else self._loaded_examples
        
        for i, example in enumerate(examples_to_eval):
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"[{i+1}/{len(examples_to_eval)}] Evaluating {example.dataset_type}: {example.example_id}"
                )
            )
            
            try:
                response = ""
                tool_calls_raw = []  # Initialize tool_calls for this example
                if example.dataset_type != "crypto":
                    # Build message with reference file URLs for GDPVal
                    # Purple Agent will use fetch_reference_file tool to download files as needed
                    message = example.question
                    if example.dataset_type == "gdpval" and example.metadata.get("has_reference_files"):
                        try:
                            reference_info = self._format_reference_files_for_agent(example.metadata)
                            if reference_info:
                                message = example.question + reference_info
                                logger.debug(f"Added reference file URLs to GDPVal task (Purple Agent will fetch on demand)")
                        except Exception as e:
                            logger.warning(f"Failed to format reference files for {example.example_id}: {e}")

                    # Send question to Purple Agent
                    response = await self.messenger.talk_to_agent(
                        message=message,
                        url=purple_agent_url,
                        new_conversation=True,
                        timeout=self.eval_config.timeout_seconds if self.eval_config else 300,
                    )
                    # Get tool calls from the last response
                    tool_calls_raw = self.messenger.get_last_tool_calls()
                predicted_text = self._format_predicted(response)
                
                # Get appropriate evaluator (options handled specially below)
                evaluator = self._evaluators.get(example.dataset_type)
                if not evaluator and example.dataset_type not in ("options", "crypto"):
                    all_results.append({
                        "example_id": example.example_id,
                        "dataset_type": example.dataset_type,
                        "error": f"No evaluator for dataset type: {example.dataset_type}",
                        "score": 0.0,
                        "is_correct": False,
                    })
                    continue

                # Evaluate based on dataset type
                if example.dataset_type == "bizfinbench":
                    eval_result = await self._run_evaluator_async(
                        evaluator,
                        predicted=response,
                        expected=example.answer,
                        task_type=example.task_type,
                        question=example.question,
                    )
                    result = {
                        "example_id": example.example_id,
                        "dataset_type": example.dataset_type,
                        "task_type": example.task_type,
                        "language": example.language,
                        "question": self._format_question(example.question),
                        "expected": self._format_expected(example.answer),
                        "predicted": predicted_text,
                        "score": eval_result.score,
                        "is_correct": eval_result.is_correct,
                        "feedback": eval_result.feedback,
                    }
                    if eval_result.details:
                        for key in ("llm_used", "llm_failure", "llm_raw_output"):
                            if key in eval_result.details:
                                result[key] = eval_result.details.get(key)
                    
                elif example.dataset_type == "prbench":
                    # PRBench uses weighted rubric criteria
                    rubric = example.metadata.get("rubric")
                    rubric_weights = example.metadata.get("rubric_weights", {})

                    eval_result = await self._run_evaluator_async(
                        evaluator,
                        predicted=response,
                        expected=example.answer,
                        rubric=rubric,
                        rubric_weights=rubric_weights,
                        question=example.question,
                        domain=example.metadata.get("domain", ""),
                        topic=example.metadata.get("topic", ""),
                    )
                    result = {
                        "example_id": example.example_id,
                        "dataset_type": example.dataset_type,
                        "category": example.category,
                        "question": self._format_question(example.question),
                        "expected": self._format_expected(example.answer or ""),
                        "predicted": predicted_text,
                        "score": eval_result.score,
                        "is_correct": eval_result.score >= 0.7,
                        "correct_count": eval_result.correct_count,
                        "total_count": eval_result.total_count,
                        "feedback": eval_result.feedback,
                    }
                    if eval_result.details:
                        for key in (
                            "llm_used",
                            "llm_failure",
                            "llm_raw_output",
                            "matched_criteria",
                            "triggered_penalties",
                        ):
                            if key in eval_result.details:
                                result[key] = eval_result.details.get(key)
                    
                elif example.dataset_type == "synthetic":
                    # Synthetic questions: Use rubric-based LLM evaluation
                    # Extract rubric and ground truth from metadata
                    rubric = example.metadata.get("rubric") if example.metadata else None
                    ground_truth_value = example.metadata.get("ground_truth_value") if example.metadata else None
                    calculation_steps = example.metadata.get("calculation_steps") if example.metadata else None
                    
                    eval_result = await self._run_evaluator_async(
                        evaluator,
                        predicted=response,
                        expected=example.answer or "",
                        question=example.question,
                        rubric=rubric,
                        ground_truth_value=ground_truth_value,
                        calculation_steps=calculation_steps,
                        category=example.category,
                    )
                    
                    # Build result dict
                    result = {
                        "example_id": example.example_id,
                        "dataset_type": example.dataset_type,
                        "category": example.category,
                        "question": self._format_question(example.question),
                        "expected": self._format_expected(example.answer or ""),
                        "predicted": predicted_text[:200] + "..." if len(predicted_text) > 200 else predicted_text,
                        "predicted_full": predicted_text,
                        "score": eval_result.score,
                        "is_correct": eval_result.score >= 0.7,
                        "feedback": eval_result.feedback,
                    }
                    
                    # Add detailed component scores if available
                    sub_scores: dict[str, float] = {}
                    if eval_result.details:
                        result["llm_used"] = eval_result.details.get("llm_used", False)
                        if "component_scores" in eval_result.details:
                            for comp_name, comp_data in eval_result.details["component_scores"].items():
                                sub_scores[comp_name] = comp_data.get("score", 0.0)
                        if "llm_raw_output" in eval_result.details:
                            result["llm_raw_output"] = eval_result.details["llm_raw_output"]
                    else:
                        result["llm_used"] = False
                    result["sub_scores"] = sub_scores

                elif example.dataset_type == "gdpval":
                    # GDPVal: Open-ended professional tasks (LLM-as-judge)
                    eval_result = await self._run_evaluator_async(
                        evaluator,
                        predicted=response,
                        expected="",  # GDPVal has no ground truth
                        task_prompt=example.question,
                        occupation=example.task_type,  # task_type stores occupation
                        sector=example.category,  # category stores sector
                        reference_files=example.metadata.get("reference_files", []),
                        question=example.question,
                    )
                    result = {
                        "example_id": example.example_id,
                        "dataset_type": example.dataset_type,
                        "occupation": example.task_type,
                        "sector": example.category,
                        "question": self._format_question(example.question),
                        "predicted": predicted_text,
                        "score": eval_result.score,
                        "is_correct": eval_result.score >= 0.7,  # 70% threshold
                        "feedback": eval_result.feedback,
                        "has_reference_files": example.metadata.get("has_reference_files", False),
                    }
                    # Add detailed scores if available
                    sub_scores: dict[str, float] = {}
                    if eval_result.details:
                        for key in ("completion", "accuracy", "format", "professionalism", "llm_used"):
                            if key in eval_result.details:
                                result[key] = eval_result.details[key]
                                if key in ("completion", "accuracy", "format", "professionalism"):
                                    sub_scores[key] = float(eval_result.details[key])
                    result["llm_used"] = eval_result.details.get("llm_used", False) if eval_result.details else False
                    result["sub_scores"] = sub_scores

                elif example.dataset_type == "options":
                    # Options Alpha Challenge evaluation
                    from cio_agent.models import AgentResponse
                    from datetime import datetime, timezone
                    
                    try:
                        # Import OptionsEvaluator explicitly to avoid module resolution issues
                        from evaluators.options import OptionsEvaluator
                        
                        # Map string category to TaskCategory enum
                        category_map = {
                            "Options Pricing": TaskCategory.OPTIONS_PRICING,
                            "Greeks Analysis": TaskCategory.GREEKS_ANALYSIS,
                            "Strategy Construction": TaskCategory.STRATEGY_CONSTRUCTION,
                            "Volatility Trading": TaskCategory.VOLATILITY_TRADING,
                            "P&L Attribution": TaskCategory.PNL_ATTRIBUTION,
                            "Risk Management": TaskCategory.RISK_MANAGEMENT,
                            "Copy Trading": TaskCategory.COPY_TRADING,
                            "Race to 10M": TaskCategory.RACE_TO_10M,
                            "Strategy Defense": TaskCategory.STRATEGY_DEFENSE,
                        }
                        task_category = category_map.get(example.category, TaskCategory.OPTIONS_PRICING)

                        # Extract ticker from metadata if available
                        ticker = example.metadata.get("ticker", "SPY")

                        # Create a FABTask for the evaluator
                        fab_task = FABTask(
                            question_id=example.example_id,
                            category=task_category,
                            difficulty=TaskDifficulty.MEDIUM,
                            question=example.question,
                            ticker=ticker,
                            fiscal_year=2025,
                            simulation_date=datetime.now(timezone.utc),
                            ground_truth=GroundTruth(
                                macro_thesis=example.answer,
                                key_themes=[example.category],
                            ),
                            rubric=TaskRubric(criteria=[], penalty_conditions=[]),
                        )

                        # Create AgentResponse
                        agent_response = AgentResponse(
                            agent_id="purple_agent",
                            task_id=example.example_id,
                            analysis=response,
                            recommendation=self._extract_recommendation(response),
                        )

                        # Initialize OptionsEvaluator with the task and LLM client
                        options_evaluator = OptionsEvaluator(
                            task=fab_task,
                            llm_client=self.llm_client,
                        )
                        options_score = await options_evaluator.score(agent_response)
                    except ImportError as ie:
                        logger.error(f"Failed to import OptionsEvaluator: {ie}")
                        raise RuntimeError(f"OptionsEvaluator import failed: {ie}") from ie
                    except NameError as ne:
                        logger.error(f"NameError in options evaluation: {ne}")
                        # Log detailed error for debugging
                        import traceback
                        logger.error(f"Full traceback:\n{traceback.format_exc()}")
                        raise RuntimeError(f"Module resolution error in options evaluation: {ne}") from ne

                    # Options scores are already on 0-100 scale - don't normalize here
                    # The unified scorer handles 0-100 scale for options
                    result = {
                        "example_id": example.example_id,
                        "dataset_type": example.dataset_type,
                        "category": example.category,
                        "question": self._format_question(example.question),
                        "expected": self._format_expected(example.answer),
                        "predicted": predicted_text,
                        "score": options_score.score,  # Keep as 0-100 scale
                        "is_correct": options_score.score >= 70,  # 70/100 threshold
                        "pnl_accuracy": options_score.pnl_accuracy,
                        "greeks_accuracy": options_score.greeks_accuracy,
                        "strategy_quality": options_score.strategy_quality,
                        "risk_management": options_score.risk_management,
                        "feedback": options_score.feedback,
                    }
                    result["llm_used"] = False
                    result["sub_scores"] = {
                        "pnl_accuracy": options_score.pnl_accuracy,
                        "greeks_accuracy": options_score.greeks_accuracy,
                        "strategy_quality": options_score.strategy_quality,
                        "risk_management": options_score.risk_management,
                    }
                    eval_result = type('obj', (object,), {'score': options_score.score})()

                elif example.dataset_type == "crypto":
                    if crypto_evaluator is None:
                        crypto_evaluator = CryptoTradingEvaluator(
                            messenger=self.messenger,
                            timeout_seconds=self.eval_config.timeout_seconds if self.eval_config else 300,
                        )

                    scenario_meta = example.metadata or {}
                    scenario_seed_base = os.environ.get("AGENTBEATS_PURPLE_AGENT_ID") or purple_agent_url
                    scenario_seed = scenario_meta.get("seed")
                    if scenario_seed is None:
                        scenario_seed = stable_seed(scenario_seed_base, example.example_id)

                    # Check if detailed interaction recording is enabled
                    # Set RECORD_INTERACTIONS=1 in .env to save Green/Purple message exchanges
                    record_interactions = os.environ.get("RECORD_INTERACTIONS", "0").lower() in ("1", "true", "yes")
                    
                    crypto_result = await crypto_evaluator.evaluate_scenario(
                        scenario_meta=scenario_meta,
                        purple_agent_url=purple_agent_url,
                        seed=scenario_seed,
                        record_interactions=record_interactions,
                    )

                    if "error" in crypto_result:
                        result = {
                            "example_id": example.example_id,
                            "dataset_type": example.dataset_type,
                            "error": crypto_result["error"],
                            "score": 0.0,
                            "is_correct": False,
                            "llm_used": False,
                            "sub_scores": {},
                        }
                        eval_result = type('obj', (object,), {'score': 0.0})()
                    else:
                        result = {
                            "example_id": example.example_id,
                            "dataset_type": example.dataset_type,
                            "scenario_id": scenario_meta.get("scenario_id", example.example_id),
                            "scenario_name": scenario_meta.get("name", example.example_id),
                            "score": crypto_result["final_score"],
                            "baseline_score": crypto_result["baseline"]["score"],
                            "noisy_score": crypto_result["noisy"]["score"],
                            "adversarial_score": crypto_result["adversarial"]["score"],
                            "meta_score": crypto_result["meta"]["score"],
                            "grade": crypto_result["grade"],
                            "random_seed": crypto_result["random_seed"],
                            "metrics": {
                                "baseline": crypto_result["baseline"]["metrics"],
                                "noisy": crypto_result["noisy"]["metrics"],
                                "adversarial": crypto_result["adversarial"]["metrics"],
                                "meta": crypto_result["meta"],
                            },
                            "events": crypto_result.get("events", []),
                            "llm_used": False,
                            "sub_scores": {
                                "baseline": crypto_result["baseline"]["score"],
                                "noisy": crypto_result["noisy"]["score"],
                                "adversarial": crypto_result["adversarial"]["score"],
                                "meta": crypto_result["meta"]["score"],
                            },
                        }
                        # Include detailed interactions if recorded
                        if "interactions" in crypto_result.get("baseline", {}):
                            result["interactions"] = crypto_result["baseline"]["interactions"]
                        if "trades" in crypto_result.get("baseline", {}):
                            result["trades"] = crypto_result["baseline"]["trades"]
                        
                        result["is_correct"] = crypto_result["final_score"] >= 70
                        result["feedback"] = (
                            f"Final score {crypto_result['final_score']:.2f} "
                            f"(grade {crypto_result['grade']})"
                        )
                        eval_result = type('obj', (object,), {'score': crypto_result["final_score"]})()

                else:
                    # Generic handling for unknown types
                    result = {
                        "example_id": example.example_id,
                        "dataset_type": example.dataset_type,
                        "question": self._format_question(example.question),
                        "predicted": predicted_text,
                        "score": 0.0,  # No evaluator, no score
                        "is_correct": False,
                        "feedback": "No evaluator configured for this dataset type",
                        "llm_used": False,
                        "sub_scores": {},
                    }
                
                # Optional debate
                if conduct_debate and eval_result.score > 0 and example.dataset_type != "crypto":
                    try:
                        rebuttal = await self.messenger.talk_to_agent(
                            message="Challenge your analysis. What risks or uncertainties did you consider?",
                            url=purple_agent_url,
                            new_conversation=False,
                            timeout=60,
                        )
                        result["rebuttal_received"] = True
                        result["rebuttal_preview"] = rebuttal[:100] + "..." if len(rebuttal) > 100 else rebuttal
                    except Exception:
                        result["rebuttal_received"] = False
                
                # Add tool_calls if available (only if store_predicted is enabled to save space)
                if self.store_predicted and tool_calls_raw:
                    result["tool_calls"] = tool_calls_raw
                
                all_results.append(result)
                
            except Exception as e:
                all_results.append({
                    "example_id": example.example_id,
                    "dataset_type": example.dataset_type,
                    "error": str(e),
                    "score": 0.0,
                    "is_correct": False,
                })
        
        return all_results



class EvalRequest(BaseModel):
    """Evaluation request payload."""

    participants: dict[str, str]
    config: dict[str, Any] = {}


