"""AgentBeats Results Formatter

Formats evaluation results in AgentBeats-compliant JSON for leaderboard integration.
See: https://docs.agentbeats.dev/tutorial/
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


class AgentBeatsResultsFormatter:
    """Formats and saves evaluation results for AgentBeats leaderboard."""

    def __init__(
        self,
        scenario_id: str | None = None,
        green_agent_id: str | None = None,
        results_dir: str = "results",
    ):
        """
        Initialize the formatter.

        Args:
            scenario_id: AgentBeats scenario UUID (from scenario.toml)
            green_agent_id: Green agent's AgentBeats UUID
            results_dir: Directory to save results files
        """
        self.scenario_id = scenario_id or os.environ.get("AGENTBEATS_SCENARIO_ID", "")
        self.green_agent_id = green_agent_id or os.environ.get("AGENTBEATS_GREEN_AGENT_ID", "")
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def format_results(
        self,
        participant_id: str,
        participant_name: str,
        evaluation_results: dict[str, Any],
        by_dataset: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Format evaluation results in AgentBeats-compliant format (schema 1.0).

        Args:
            participant_id: Participant's AgentBeats UUID
            participant_name: Human-readable participant name
            evaluation_results: Raw evaluation results from Green Agent
            by_dataset: Per-dataset breakdown

        Returns:
            AgentBeats-compliant results dictionary
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        run_id = str(uuid4())

        # Check if this has unified scoring format
        overall_score = evaluation_results.get("overall_score", {})
        section_scores = evaluation_results.get("section_scores", {})
        eval_metadata = evaluation_results.get("evaluation_metadata", {})

        if overall_score:
            # Unified scoring - map to 1.0 format
            # Convert overall_score (0-100) to 0-1 scale for consistency
            score_value = overall_score.get("score", 0) / 100.0
            num_tasks = eval_metadata.get("num_tasks", 0)
            num_successful = eval_metadata.get("num_successful", 0)

            # Map section_scores to dataset_scores format
            dataset_scores = {}
            for section_name, ss in section_scores.items():
                dataset_scores[section_name] = {
                    "count": ss.get("task_count", 0),
                    "mean_score": ss.get("score", 0) / 100.0,  # Normalize to 0-1
                    "accuracy": ss.get("accuracy", 0),
                    "weight": ss.get("weight", 0),
                }
        else:
            # Legacy format
            score_value = evaluation_results.get("average_score", 0.0)
            num_tasks = evaluation_results.get("num_evaluated", 0)
            num_successful = evaluation_results.get("num_successful", 0)

            dataset_scores = {}
            if by_dataset:
                for ds_name, ds_data in by_dataset.items():
                    dataset_scores[ds_name] = {
                        "count": ds_data.get("count", 0),
                        "mean_score": ds_data.get("mean_score", 0.0),
                        "accuracy": ds_data.get("accuracy", 0.0),
                    }

        # Calculate accuracy from section scores or use provided
        if section_scores:
            total_correct = sum(
                ss.get("task_count", 0) * ss.get("accuracy", 0)
                for ss in section_scores.values()
            )
            total_tasks = sum(ss.get("task_count", 0) for ss in section_scores.values())
            accuracy = total_correct / total_tasks if total_tasks > 0 else 0
        else:
            accuracy = evaluation_results.get("accuracy", 0.0)

        # Structure for leaderboard queries:
        # - participants.purple_agent returns participant name
        # - results[1] returns the evaluation results (DuckDB is 1-indexed)
        result = {
            "schema_version": "1.0",
            "run_id": run_id,
            "scenario_id": self.scenario_id,
            "timestamp": timestamp,
            "green_agent": {
                "id": self.green_agent_id,
                "benchmark": evaluation_results.get("benchmark", "FAB++ Finance Agent Benchmark"),
                "version": evaluation_results.get("version", "1.0"),
            },
            "participants": {
                "purple_agent": participant_name,
                "participant_id": participant_id,
            },
            "results": [{
                "overall_score": overall_score if overall_score else {"score": score_value * 100},
                "section_scores": section_scores,
                "evaluation_metadata": {
                    "num_tasks": num_tasks,
                    "num_successful": num_successful,
                    "accuracy": round(accuracy, 4),
                },
                "dataset_scores": dataset_scores,
                "config": evaluation_results.get("config_summary", {}),
                "sampling_strategy": evaluation_results.get("sampling_strategy", "stratified"),
            }],
            "detailed_results": evaluation_results.get("results", evaluation_results.get("detailed_results", [])),
        }

        return result

    def save_results(
        self,
        results: dict[str, Any],
        filename: str | None = None,
    ) -> Path:
        """
        Save results to a JSON file.

        Args:
            results: AgentBeats-formatted results
            filename: Optional filename (defaults to run_id.json)

        Returns:
            Path to the saved results file
        """
        if filename is None:
            filename = f"{results['run_id']}.json"

        filepath = self.results_dir / filename

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)

        return filepath

    def save_leaderboard_entry(
        self,
        results: dict[str, Any],
    ) -> Path:
        """
        Save a compact leaderboard entry for DuckDB queries.

        Args:
            results: AgentBeats-formatted results (schema 1.0)

        Returns:
            Path to the saved leaderboard entry file
        """
        # Create leaderboard directory
        leaderboard_dir = self.results_dir / "leaderboard"
        leaderboard_dir.mkdir(parents=True, exist_ok=True)

        # Extract data from new structure
        participants = results.get("participants", {})
        eval_results = results.get("results", [{}])[0]  # First result
        section_scores = eval_results.get("section_scores", {})
        eval_metadata = eval_results.get("evaluation_metadata", {})
        overall_score = eval_results.get("overall_score", {})

        entry = {
            "run_id": results["run_id"],
            "scenario_id": results["scenario_id"],
            "timestamp": results["timestamp"],
            "schema_version": "1.0",
            "participant_id": participants.get("participant_id", ""),
            "participant_name": participants.get("purple_agent", ""),
            "overall_score": overall_score.get("score", 0),
            "accuracy": eval_metadata.get("accuracy", 0),
            "tasks_evaluated": eval_metadata.get("num_tasks", 0),
            "tasks_successful": eval_metadata.get("num_successful", 0),
            # Flatten section scores for easier querying
            **{
                f"{section}_score": ss.get("score", 0)
                for section, ss in section_scores.items()
            },
            **{
                f"{section}_weight": ss.get("weight", 0)
                for section, ss in section_scores.items()
            },
        }

        # Save as newline-delimited JSON for easy DuckDB loading
        filepath = leaderboard_dir / "entries.ndjson"
        with open(filepath, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

        return filepath


def format_and_save_results(
    participant_id: str,
    participant_name: str,
    evaluation_results: dict[str, Any],
    by_dataset: dict[str, Any] | None = None,
    scenario_id: str | None = None,
    green_agent_id: str | None = None,
    results_dir: str = "results",
) -> tuple[Path, Path]:
    """
    Convenience function to format and save results.

    Returns:
        Tuple of (full results path, leaderboard entry path)
    """
    formatter = AgentBeatsResultsFormatter(
        scenario_id=scenario_id,
        green_agent_id=green_agent_id,
        results_dir=results_dir,
    )

    results = formatter.format_results(
        participant_id=participant_id,
        participant_name=participant_name,
        evaluation_results=evaluation_results,
        by_dataset=by_dataset,
    )

    full_path = formatter.save_results(results)
    leaderboard_path = formatter.save_leaderboard_entry(results)

    return full_path, leaderboard_path
