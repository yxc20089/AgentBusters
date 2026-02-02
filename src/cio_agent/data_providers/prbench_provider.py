"""
PRBench dataset provider using HuggingFace API.

Dynamically fetches data from ScaleAI/PRBench on HuggingFace Hub.
PRBench is a professional reasoning benchmark for high-stakes tasks
in Finance and Legal domains.

Dataset structure:
- 4 splits: finance (600), legal (500), finance_hard (300), legal_hard (250)
- Multi-turn conversations (up to 10 turns per task)
- Rich rubric evaluation with weighted criteria (10-30 per task)

Usage:
    provider = PRBenchProvider(
        splits=["finance", "legal"],
        topics=None,  # All topics
        limit=100
    )
    examples = provider.load()
    templates = provider.to_templates()
"""

import logging
from typing import Any, Dict, List, Optional

from cio_agent.data_providers.base import DatasetExample, DatasetProvider
from cio_agent.models import (
    FABQuestionTemplate,
    GroundTruth,
    TaskCategory,
    TaskDifficulty,
    TaskRubric,
)

logger = logging.getLogger(__name__)

# Global cache for HuggingFace datasets (avoid re-downloading)
_dataset_cache: Dict[str, Any] = {}


class PRBenchProvider(DatasetProvider):
    """
    Provider for Scale AI PRBench dataset via HuggingFace API.

    Dynamically fetches data from:
    https://huggingface.co/datasets/ScaleAI/PRBench

    Supports 4 splits:
        - finance: 600 tasks in finance domain
        - legal: 500 tasks in legal domain
        - finance_hard: 300 challenging finance tasks
        - legal_hard: 250 challenging legal tasks
    """

    name = "prbench"

    # HuggingFace dataset identifier
    HF_DATASET_ID = "ScaleAI/PRBench"

    # Available splits
    AVAILABLE_SPLITS = ["finance", "legal", "finance_hard", "legal_hard"]

    # Map domain to TaskCategory
    DOMAIN_CATEGORY_MAP = {
        "Finance": TaskCategory.NUMERICAL_REASONING,
        "Legal": TaskCategory.QUALITATIVE_RETRIEVAL,
    }

    # Map split to difficulty
    SPLIT_DIFFICULTY_MAP = {
        "finance": TaskDifficulty.MEDIUM,
        "legal": TaskDifficulty.MEDIUM,
        "finance_hard": TaskDifficulty.EXPERT,
        "legal_hard": TaskDifficulty.EXPERT,
    }

    # Rubric weight class to numeric weight
    WEIGHT_CLASS_MAP = {
        "critically important": 3.0,
        "important": 2.0,
        "slightly important": 1.0,
        "detrimental": -1.0,
    }

    def __init__(
        self,
        splits: Optional[List[str]] = None,
        topics: Optional[List[str]] = None,
        limit: Optional[int] = None,
        include_reference_texts: bool = True,
    ):
        """
        Initialize PRBench provider with HuggingFace API.

        Args:
            splits: List of splits to include. Default: ["finance", "legal"]
            topics: Optional list of topics to filter by. None means all.
            limit: Optional limit on number of examples to load per split
            include_reference_texts: Include reference documents in metadata

        Raises:
            ValueError: If invalid split is specified
        """
        self.splits = splits or ["finance", "legal"]
        self.topics = topics
        self.limit = limit
        self.include_reference_texts = include_reference_texts

        # Validate splits
        invalid_splits = set(self.splits) - set(self.AVAILABLE_SPLITS)
        if invalid_splits:
            raise ValueError(
                f"Invalid splits: {invalid_splits}. "
                f"Valid splits: {self.AVAILABLE_SPLITS}"
            )

        # Update provider name to be unique
        splits_str = "_".join(sorted(self.splits))
        self.name = f"prbench_{splits_str}"

        logger.info(
            f"Initialized PRBenchProvider: splits={self.splits}, "
            f"topics={self.topics}, limit={self.limit}"
        )

    @classmethod
    def list_splits(cls) -> List[str]:
        """Return list of available splits."""
        return cls.AVAILABLE_SPLITS.copy()

    @classmethod
    def list_topics(cls) -> Dict[str, List[str]]:
        """
        Return available topics by domain.

        Note: Topics are discovered dynamically from the dataset.
        """
        return {
            "Finance": [
                "Risk Management & Stress Testing",
                "Financial Analysis",
                "Investment Strategy",
                "Corporate Finance",
                "Regulatory Compliance",
            ],
            "Legal": [
                "Contract Law",
                "Litigation",
                "Regulatory",
                "Corporate Governance",
                "Intellectual Property",
            ],
        }

    def _fetch_from_huggingface(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch dataset from HuggingFace Hub with caching.

        Returns:
            Dictionary mapping split names to list of records
        """
        cache_key = f"{self.HF_DATASET_ID}:{','.join(sorted(self.splits))}"

        if cache_key in _dataset_cache:
            logger.debug(f"Using cached dataset: {cache_key}")
            return _dataset_cache[cache_key]

        logger.info(f"Fetching from HuggingFace: {self.HF_DATASET_ID}")

        try:
            from datasets import load_dataset

            try:
                dataset = load_dataset(self.HF_DATASET_ID)
            except TypeError:
                # Fallback for older datasets versions that may require trust_remote_code
                dataset = load_dataset(self.HF_DATASET_ID, trust_remote_code=True)

            result = {}
            for split in self.splits:
                if split in dataset:
                    result[split] = list(dataset[split])
                    logger.info(f"Loaded {len(result[split])} records from split '{split}'")
                else:
                    logger.warning(f"Split '{split}' not found in dataset")

            # Cache for future use
            _dataset_cache[cache_key] = result
            return result

        except Exception as e:
            logger.error(f"Failed to fetch from HuggingFace: {e}")
            raise RuntimeError(
                f"Could not fetch PRBench data from HuggingFace. Error: {e}"
            )

    def _extract_conversation(self, item: Dict[str, Any]) -> str:
        """
        Extract full conversation from PRBench record.

        PRBench has multi-turn format with prompt_0 to prompt_9.
        We concatenate all turns into a single prompt.
        """
        turns = item.get("turns", 1)
        conversation_parts = []

        for i in range(turns):
            prompt_key = f"prompt_{i}"
            prompt = item.get(prompt_key, "")
            if prompt:
                if i > 0:
                    conversation_parts.append(f"\n--- Turn {i + 1} ---\n")
                conversation_parts.append(prompt)

        return "".join(conversation_parts)

    def _extract_reference_texts(self, item: Dict[str, Any]) -> List[str]:
        """Extract all reference texts from the record."""
        turns = item.get("turns", 1)
        all_refs = []

        for i in range(turns):
            ref_key = f"reference_texts_{i}"
            refs = item.get(ref_key, [])
            if refs:
                all_refs.extend(refs)

        return all_refs

    def _extract_rubric(self, item: Dict[str, Any]) -> TaskRubric:
        """
        Extract rubric criteria from PRBench record.

        PRBench rubric format:
        {
            "id": "...",
            "title": "criteria description",
            "annotations": {
                "criteria_category": "Risk & Regulatory Disclosure",
                "criteria_description": "...",
                "weight_class": "critically important",
                "critically_important_weight": 9
            }
        }
        """
        rubric_items = item.get("rubric", [])
        criteria = []
        penalty_conditions = []

        for rubric_item in rubric_items:
            if not isinstance(rubric_item, dict):
                continue

            title = rubric_item.get("title", "")
            annotations = rubric_item.get("annotations", {})
            weight_class = annotations.get("weight_class", "important")

            if not title:
                continue

            # Detrimental criteria are penalties
            if weight_class == "detrimental":
                penalty_conditions.append(title)
            else:
                criteria.append(title)

        return TaskRubric(criteria=criteria, penalty_conditions=penalty_conditions)

    def _extract_rubric_weights(self, item: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract rubric criteria with their weights.

        Returns:
            Dictionary mapping criteria to their weights
        """
        rubric_items = item.get("rubric", [])
        weights = {}

        for rubric_item in rubric_items:
            if not isinstance(rubric_item, dict):
                continue

            title = rubric_item.get("title", "")
            annotations = rubric_item.get("annotations", {})
            weight_class = annotations.get("weight_class", "important")

            if title:
                weights[title] = self.WEIGHT_CLASS_MAP.get(weight_class, 1.0)

        return weights

    def load(self) -> List[DatasetExample]:
        """Load examples from HuggingFace API."""
        import random

        data_by_split = self._fetch_from_huggingface()
        examples: List[DatasetExample] = []

        for split, records in data_by_split.items():
            for idx, item in enumerate(records):
                try:
                    # Filter by topic if specified
                    topic = item.get("topic", "")
                    if self.topics and topic not in self.topics:
                        continue

                    task_id = item.get("task", f"prbench_{split}_{idx}")
                    domain = item.get("field", "Finance")
                    turns = item.get("turns", 1)
                    scratchpad = item.get("scratchpad", "")

                    # Extract conversation
                    question = self._extract_conversation(item)
                    if not question:
                        continue

                    # Extract rubric
                    rubric = self._extract_rubric(item)
                    rubric_weights = self._extract_rubric_weights(item)

                    # Determine category and difficulty
                    category = self.DOMAIN_CATEGORY_MAP.get(
                        domain, TaskCategory.QUALITATIVE_RETRIEVAL
                    )
                    difficulty = self.SPLIT_DIFFICULTY_MAP.get(
                        split, TaskDifficulty.MEDIUM
                    )

                    # Build ground truth from scratchpad
                    ground_truth = GroundTruth(
                        macro_thesis=scratchpad,
                        key_themes=rubric.criteria[:5] if rubric.criteria else [],
                    )

                    # Build metadata
                    metadata = {
                        "split": split,
                        "domain": domain,
                        "topic": topic,
                        "turns": turns,
                        "expert_type": item.get("expert", ""),
                        "economic_pathway": item.get("economic_pathway", ""),
                        "decision_type": item.get("decision_type", ""),
                        "rubric_weights": rubric_weights,
                        "rubric_count": len(item.get("rubric", [])),
                    }

                    # Include reference texts if enabled
                    if self.include_reference_texts:
                        refs = self._extract_reference_texts(item)
                        if refs:
                            metadata["reference_texts"] = refs

                    example_id = f"prbench_{split}_{task_id}"

                    examples.append(
                        DatasetExample(
                            example_id=example_id,
                            question=question,
                            answer=scratchpad,  # Use scratchpad as reference answer
                            rubric=rubric,
                            category=category,
                            difficulty=difficulty,
                            ground_truth=ground_truth,
                            source="prbench_hf",
                            metadata=metadata,
                        )
                    )

                except Exception as e:
                    logger.warning(f"Failed to process item {idx} in split {split}: {e}")
                    continue

            # Apply per-split limit
            if self.limit:
                split_examples = [ex for ex in examples if ex.metadata.get("split") == split]
                if len(split_examples) > self.limit:
                    # Keep only up to limit per split
                    to_remove = set(
                        ex.example_id for ex in split_examples[self.limit:]
                    )
                    examples = [ex for ex in examples if ex.example_id not in to_remove]

        logger.info(
            f"Loaded {len(examples)} examples from PRBench "
            f"(splits={self.splits})"
        )
        return examples

    def to_templates(self) -> List[FABQuestionTemplate]:
        """Convert loaded examples to FAB question templates."""
        templates: List[FABQuestionTemplate] = []
        for ex in self.load():
            templates.append(
                FABQuestionTemplate(
                    template_id=ex.example_id,
                    category=ex.category,
                    template=ex.question,
                    difficulty=ex.difficulty,
                    metric="prbench",
                    rubric=ex.rubric or TaskRubric(criteria=[]),
                    requires_code_execution=False,
                )
            )
        return templates


def clear_cache():
    """Clear the dataset cache (useful for testing or memory management)."""
    global _dataset_cache
    _dataset_cache.clear()
    logger.info("PRBench dataset cache cleared")
