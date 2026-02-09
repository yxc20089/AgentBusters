"""
Unit tests for EvaluatorLLMConfig and per-evaluator LLM settings.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from evaluators.llm_utils import (
    EvaluatorLLMConfig,
    EvaluatorModelConfig,
    get_evaluator_llm_config,
    reset_evaluator_llm_config,
    get_model_for_evaluator,
    get_temperature_for_evaluator,
    get_max_tokens_for_evaluator,
    get_provider_for_evaluator,
    build_llm_client,
    build_llm_client_for_evaluator,
)


class TestEvaluatorModelConfig:
    """Test EvaluatorModelConfig dataclass."""

    def test_default_values(self):
        config = EvaluatorModelConfig()
        assert config.model == "gpt-4o-mini"
        assert config.temperature == 0.0
        assert config.max_tokens == 512
        assert config.provider is None

    def test_custom_values(self):
        config = EvaluatorModelConfig(
            model="gpt-4o",
            temperature=0.0,
            max_tokens=1000,
            provider="openai",
        )
        assert config.model == "gpt-4o"
        assert config.temperature == 0.0
        assert config.max_tokens == 1000
        assert config.provider == "openai"

    def test_get_provider_explicit(self):
        config = EvaluatorModelConfig(model="gpt-4o", provider="openai")
        assert config.get_provider() == "openai"

    def test_get_provider_auto_openai(self):
        config = EvaluatorModelConfig(model="gpt-4o-mini")
        assert config.get_provider() == "openai"

    def test_get_provider_auto_anthropic(self):
        config = EvaluatorModelConfig(model="claude-3-sonnet")
        assert config.get_provider() == "anthropic"


class TestEvaluatorLLMConfig:
    """Test EvaluatorLLMConfig dataclass."""

    def test_default_models(self):
        config = EvaluatorLLMConfig()
        assert config.macro.model == "gpt-4o-mini"
        assert config.execution.model == "gpt-4o-mini"
        assert config.gdpval.model == "gpt-4o"
        assert config.bizfinbench.model == "gpt-4o-mini"
        assert config.debate.model == "gpt-4o"
        assert config.prbench.model == "gpt-4o-mini"

    def test_all_temperatures_are_zero(self):
        """Ensure all temperatures are 0 for reproducibility."""
        config = EvaluatorLLMConfig()
        assert config.macro.temperature == 0.0
        assert config.execution.temperature == 0.0
        assert config.gdpval.temperature == 0.0
        assert config.bizfinbench.temperature == 0.0
        assert config.debate.temperature == 0.0
        assert config.prbench.temperature == 0.0
        assert config.default.temperature == 0.0

    def test_get_config_known_evaluator(self):
        config = EvaluatorLLMConfig()
        macro_config = config.get_config("macro")
        assert macro_config.model == "gpt-4o-mini"

    def test_get_config_unknown_evaluator(self):
        config = EvaluatorLLMConfig()
        unknown_config = config.get_config("unknown_evaluator")
        assert unknown_config.model == config.default.model

    def test_get_model(self):
        config = EvaluatorLLMConfig()
        assert config.get_model("gdpval") == "gpt-4o"
        assert config.get_model("macro") == "gpt-4o-mini"

    def test_get_temperature(self):
        config = EvaluatorLLMConfig()
        assert config.get_temperature("gdpval") == 0.0
        assert config.get_temperature("macro") == 0.0

    def test_get_max_tokens(self):
        config = EvaluatorLLMConfig()
        assert config.get_max_tokens("gdpval") == 2000
        assert config.get_max_tokens("macro") == 256


class TestEvaluatorLLMConfigFromEnv:
    """Test environment variable configuration."""

    def setup_method(self):
        """Reset config before each test."""
        reset_evaluator_llm_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_evaluator_llm_config()

    def test_from_env_default_model_override(self):
        with patch.dict(os.environ, {"EVAL_LLM_DEFAULT_MODEL": "gpt-4o"}):
            config = EvaluatorLLMConfig.from_env()
            assert config.default.model == "gpt-4o"
            # Per-evaluator should also use default if not explicitly set
            assert config.macro.model == "gpt-4o"

    def test_from_env_per_evaluator_model(self):
        with patch.dict(os.environ, {"EVAL_LLM_MACRO_MODEL": "claude-3-sonnet"}):
            config = EvaluatorLLMConfig.from_env()
            assert config.macro.model == "claude-3-sonnet"
            # Others should remain default
            assert config.execution.model == "gpt-4o-mini"

    def test_from_env_per_evaluator_temperature(self):
        # Note: temperature should stay 0 for reproducibility
        # but the system allows override for testing
        with patch.dict(os.environ, {"EVAL_LLM_GDPVAL_TEMPERATURE": "0.0"}):
            config = EvaluatorLLMConfig.from_env()
            assert config.gdpval.temperature == 0.0

    def test_from_env_per_evaluator_max_tokens(self):
        with patch.dict(os.environ, {"EVAL_LLM_MACRO_MAX_TOKENS": "500"}):
            config = EvaluatorLLMConfig.from_env()
            assert config.macro.max_tokens == 500

    def test_from_env_per_evaluator_provider(self):
        with patch.dict(os.environ, {"EVAL_LLM_MACRO_PROVIDER": "anthropic"}):
            config = EvaluatorLLMConfig.from_env()
            assert config.macro.provider == "anthropic"

    def test_from_env_invalid_temperature_ignored(self):
        with patch.dict(os.environ, {"EVAL_LLM_MACRO_TEMPERATURE": "invalid"}):
            config = EvaluatorLLMConfig.from_env()
            # Should keep default
            assert config.macro.temperature == 0.0

    def test_from_env_invalid_max_tokens_ignored(self):
        with patch.dict(os.environ, {"EVAL_LLM_MACRO_MAX_TOKENS": "invalid"}):
            config = EvaluatorLLMConfig.from_env()
            # Should keep default
            assert config.macro.max_tokens == 256


class TestGlobalConfigFunctions:
    """Test global config helper functions."""

    def setup_method(self):
        reset_evaluator_llm_config()

    def teardown_method(self):
        reset_evaluator_llm_config()

    def test_get_evaluator_llm_config_cached(self):
        config1 = get_evaluator_llm_config()
        config2 = get_evaluator_llm_config()
        assert config1 is config2  # Same instance

    def test_reset_evaluator_llm_config(self):
        config1 = get_evaluator_llm_config()
        reset_evaluator_llm_config()
        config2 = get_evaluator_llm_config()
        assert config1 is not config2  # Different instance

    def test_get_model_for_evaluator(self):
        assert get_model_for_evaluator("gdpval") == "gpt-4o"
        assert get_model_for_evaluator("macro") == "gpt-4o-mini"

    def test_get_temperature_for_evaluator(self):
        assert get_temperature_for_evaluator("gdpval") == 0.0
        assert get_temperature_for_evaluator("macro") == 0.0

    def test_get_max_tokens_for_evaluator(self):
        assert get_max_tokens_for_evaluator("gdpval") == 2000
        assert get_max_tokens_for_evaluator("macro") == 256

    def test_env_override_reflected_in_helpers(self):
        with patch.dict(os.environ, {"EVAL_LLM_MACRO_MODEL": "custom-model"}):
            reset_evaluator_llm_config()
            assert get_model_for_evaluator("macro") == "custom-model"

    def test_get_provider_for_evaluator_default(self):
        assert get_provider_for_evaluator("macro") == "openai"
        assert get_provider_for_evaluator("gdpval") == "openai"

    def test_get_provider_for_evaluator_anthropic_model(self):
        with patch.dict(os.environ, {"EVAL_LLM_MACRO_MODEL": "claude-3-sonnet"}):
            reset_evaluator_llm_config()
            assert get_provider_for_evaluator("macro") == "anthropic"

    def test_get_provider_for_evaluator_explicit_override(self):
        with patch.dict(os.environ, {"EVAL_LLM_MACRO_PROVIDER": "anthropic"}):
            reset_evaluator_llm_config()
            assert get_provider_for_evaluator("macro") == "anthropic"


class TestBuildLLMClient:
    """Test build_llm_client with provider parameter."""

    def test_build_llm_client_provider_parameter(self):
        # Without API keys, should return None but not raise
        # This test verifies the function handles the provider parameter without crashing
        result = build_llm_client(provider="openai")
        # Without OPENAI_API_KEY set, should return None
        if not os.getenv("OPENAI_API_KEY"):
            assert result is None

    def test_build_llm_client_existing_passthrough(self):
        mock_client = object()
        result = build_llm_client(existing=mock_client)
        assert result is mock_client

    def test_build_llm_client_for_evaluator_existing_passthrough(self):
        mock_client = object()
        result = build_llm_client_for_evaluator("macro", existing=mock_client)
        assert result is mock_client

    def test_build_llm_client_for_evaluator_uses_provider(self):
        # Without API keys, should return None but use correct provider logic
        reset_evaluator_llm_config()
        result = build_llm_client_for_evaluator("macro")
        # Without API keys set, should return None
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
            assert result is None


class TestEvaluatorIntegration:
    """Test evaluators use correct configuration."""

    def setup_method(self):
        reset_evaluator_llm_config()

    def teardown_method(self):
        reset_evaluator_llm_config()

    def test_macro_evaluator_uses_config(self):
        """Verify MacroEvaluator picks up per-evaluator config."""
        from unittest.mock import MagicMock
        from cio_agent.models import GroundTruth

        # Create a mock ground truth
        gt = GroundTruth(
            macro_thesis="Test thesis",
            key_themes=["theme1"],
            fundamental_data={},
            expected_recommendation="buy",
        )

        # Import after reset to get fresh config
        from evaluators.macro import MacroEvaluator

        evaluator = MacroEvaluator(ground_truth=gt, use_llm=False)

        # Verify it picked up the config values
        assert evaluator.llm_model == get_model_for_evaluator("macro")
        assert evaluator.llm_temperature == get_temperature_for_evaluator("macro")
        assert evaluator.llm_max_tokens == get_max_tokens_for_evaluator("macro")

    def test_gdpval_evaluator_uses_config(self):
        """Verify GDPValEvaluator picks up per-evaluator config."""
        from evaluators.gdpval_evaluator import GDPValEvaluator

        evaluator = GDPValEvaluator(use_llm=False)

        assert evaluator.llm_model == get_model_for_evaluator("gdpval")
        assert evaluator.llm_temperature == get_temperature_for_evaluator("gdpval")
        assert evaluator._llm_max_tokens == get_max_tokens_for_evaluator("gdpval")

    def test_bizfinbench_evaluator_uses_config(self):
        """Verify BizFinBenchEvaluator picks up per-evaluator config."""
        from evaluators.bizfinbench_evaluator import BizFinBenchEvaluator

        evaluator = BizFinBenchEvaluator(use_llm=False)

        assert evaluator.llm_model == get_model_for_evaluator("bizfinbench")
        assert evaluator.llm_temperature == get_temperature_for_evaluator("bizfinbench")
        assert evaluator._llm_max_tokens == get_max_tokens_for_evaluator("bizfinbench")

    def test_prbench_evaluator_uses_config(self):
        """Verify PRBenchEvaluator picks up per-evaluator config."""
        from evaluators.prbench_evaluator import PRBenchEvaluator

        evaluator = PRBenchEvaluator(use_llm=False)

        assert evaluator.llm_model == get_model_for_evaluator("prbench")
        assert evaluator.llm_temperature == get_temperature_for_evaluator("prbench")
        assert evaluator._llm_max_tokens == get_max_tokens_for_evaluator("prbench")

    def test_evaluator_respects_env_override(self):
        """Verify evaluators respect environment variable overrides."""
        with patch.dict(os.environ, {"EVAL_LLM_GDPVAL_MODEL": "custom-model-123"}):
            reset_evaluator_llm_config()

            from evaluators.gdpval_evaluator import GDPValEvaluator
            evaluator = GDPValEvaluator(use_llm=False)

            assert evaluator.llm_model == "custom-model-123"


class TestReproducibilityRequirements:
    """
    Tests to ensure LLM-as-judge reproducibility requirements are met.

    Per submission guidelines:
    - Use deterministic settings when relying on LLM judges
    - Fix relevant parameters (e.g., temperature)
    - Validate that repeated runs produce consistent scores
    """

    def test_all_evaluators_have_zero_temperature(self):
        """Critical: All evaluators must use temperature=0 for reproducibility."""
        config = EvaluatorLLMConfig()
        evaluators = ["macro", "execution", "gdpval", "bizfinbench", "debate", "public_csv"]

        for name in evaluators:
            temp = config.get_temperature(name)
            assert temp == 0.0, f"Evaluator '{name}' has temperature={temp}, must be 0.0"

    def test_default_temperature_is_zero(self):
        """Default config must also use temperature=0."""
        config = EvaluatorLLMConfig()
        assert config.default.temperature == 0.0


class TestExecutionEvaluatorConfig:
    """
    Test ExecutionEvaluator per-evaluator LLM configuration.

    Addresses Copilot review: Ensure ExecutionEvaluator correctly picks up
    model/temperature/max_tokens from EvaluatorLLMConfig.
    """

    def setup_method(self):
        reset_evaluator_llm_config()

    def teardown_method(self):
        reset_evaluator_llm_config()

    def _create_test_task(self):
        """Create a minimal test task with all required fields."""
        from datetime import datetime
        from cio_agent.models import Task, TaskCategory, TaskRubric, GroundTruth

        return Task(
            question_id="test_task",
            category=TaskCategory.NUMERICAL_REASONING,
            question="Test question",
            ticker="TEST",
            fiscal_year=2024,
            simulation_date=datetime(2024, 1, 1),
            rubric=TaskRubric(criteria=["test"], mandatory_elements=["test"]),
            ground_truth=GroundTruth(macro_thesis="test"),
        )

    def test_execution_evaluator_uses_default_config(self):
        """Verify ExecutionEvaluator picks up per-evaluator config by default."""
        from evaluators.execution import ExecutionEvaluator

        task = self._create_test_task()
        evaluator = ExecutionEvaluator(task=task)

        # Should use config values from EvaluatorLLMConfig
        assert evaluator.llm_model == get_model_for_evaluator("execution")
        assert evaluator.llm_temperature == get_temperature_for_evaluator("execution")
        assert evaluator.llm_max_tokens == get_max_tokens_for_evaluator("execution")

    def test_execution_evaluator_explicit_model_override(self):
        """Verify ExecutionEvaluator respects explicit model override."""
        from evaluators.execution import ExecutionEvaluator

        task = self._create_test_task()
        evaluator = ExecutionEvaluator(
            task=task,
            llm_model="custom-model-override",
        )

        assert evaluator.llm_model == "custom-model-override"
        # Temperature should still use config default
        assert evaluator.llm_temperature == get_temperature_for_evaluator("execution")

    def test_execution_evaluator_explicit_temperature_override(self):
        """Verify ExecutionEvaluator respects explicit temperature override."""
        from evaluators.execution import ExecutionEvaluator

        task = self._create_test_task()

        # Explicit temperature override (note: 0.0 for reproducibility is recommended)
        evaluator = ExecutionEvaluator(
            task=task,
            llm_temperature=0.5,
        )

        assert evaluator.llm_temperature == 0.5
        # Model should still use config default
        assert evaluator.llm_model == get_model_for_evaluator("execution")

    def test_execution_evaluator_respects_env_override(self):
        """Verify ExecutionEvaluator respects environment variable overrides."""
        with patch.dict(os.environ, {"EVAL_LLM_EXECUTION_MODEL": "env-override-model"}):
            reset_evaluator_llm_config()

            from evaluators.execution import ExecutionEvaluator

            task = self._create_test_task()
            evaluator = ExecutionEvaluator(task=task)

            assert evaluator.llm_model == "env-override-model"

    def test_execution_evaluator_uses_build_llm_client(self):
        """Verify ExecutionEvaluator uses build_llm_client_for_evaluator when no client provided."""
        from evaluators.execution import ExecutionEvaluator

        task = self._create_test_task()

        # Without API keys, llm_client will be None but the function should be called
        evaluator = ExecutionEvaluator(task=task)

        # Without API keys set, client should be None
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
            assert evaluator.llm_client is None

    def test_execution_evaluator_passthrough_client(self):
        """Verify ExecutionEvaluator uses provided client directly."""
        from evaluators.execution import ExecutionEvaluator

        task = self._create_test_task()
        mock_client = MagicMock()
        evaluator = ExecutionEvaluator(task=task, llm_client=mock_client)

        assert evaluator.llm_client is mock_client


class TestCallLLMModelSpecificKwargs:
    """
    Tests for call_llm model-specific parameter handling.
    
    Addresses Copilot review: Verify that the kwargs passed to 
    client.chat.completions.create are correct for different model types:
    - Regular models: max_tokens, temperature
    - Reasoning models (o1, o3, o4, gpt-5-nano): max_completion_tokens, no temperature
    """

    def test_regular_model_uses_max_tokens_and_temperature(self):
        """Regular models should use max_tokens and temperature."""
        from evaluators.llm_utils import call_llm
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test response"
        mock_client.chat.completions.create.return_value = mock_response
        
        call_llm(
            client=mock_client,
            prompt="test prompt",
            model="gpt-4o-mini",  # Regular model
            temperature=0.0,
            max_tokens=100,
            model_context_limit=128000,
        )
        
        # Verify the call was made with correct kwargs
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "max_tokens" in call_kwargs, "Regular model should use max_tokens"
        assert "max_completion_tokens" not in call_kwargs, "Regular model should NOT use max_completion_tokens"
        assert "temperature" in call_kwargs, "Regular model should use temperature"
        assert call_kwargs["temperature"] == 0.0

    def test_o1_model_uses_max_completion_tokens_no_temperature(self):
        """o1 models should use max_completion_tokens and no temperature."""
        from evaluators.llm_utils import call_llm
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test response"
        mock_client.chat.completions.create.return_value = mock_response
        
        call_llm(
            client=mock_client,
            prompt="test prompt",
            model="o1-mini",  # Reasoning model
            temperature=0.0,
            max_tokens=100,
            model_context_limit=128000,
        )
        
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "max_completion_tokens" in call_kwargs, "o1 model should use max_completion_tokens"
        assert "max_tokens" not in call_kwargs, "o1 model should NOT use max_tokens"
        assert "temperature" not in call_kwargs, "o1 model should NOT use temperature"

    def test_o3_model_uses_max_completion_tokens_no_temperature(self):
        """o3 models should use max_completion_tokens and no temperature."""
        from evaluators.llm_utils import call_llm
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test response"
        mock_client.chat.completions.create.return_value = mock_response
        
        call_llm(
            client=mock_client,
            prompt="test prompt",
            model="o3-mini",  # Reasoning model
            temperature=0.0,
            max_tokens=100,
            model_context_limit=128000,
        )
        
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "max_completion_tokens" in call_kwargs, "o3 model should use max_completion_tokens"
        assert "max_tokens" not in call_kwargs, "o3 model should NOT use max_tokens"
        assert "temperature" not in call_kwargs, "o3 model should NOT use temperature"

    def test_o4_model_uses_max_completion_tokens_no_temperature(self):
        """o4 models should use max_completion_tokens and no temperature."""
        from evaluators.llm_utils import call_llm
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test response"
        mock_client.chat.completions.create.return_value = mock_response
        
        call_llm(
            client=mock_client,
            prompt="test prompt",
            model="o4-mini",  # Reasoning model
            temperature=0.0,
            max_tokens=100,
            model_context_limit=128000,
        )
        
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "max_completion_tokens" in call_kwargs, "o4 model should use max_completion_tokens"
        assert "max_tokens" not in call_kwargs, "o4 model should NOT use max_tokens"
        assert "temperature" not in call_kwargs, "o4 model should NOT use temperature"

    def test_gpt5_nano_uses_max_completion_tokens_no_temperature(self):
        """gpt-5-nano should use max_completion_tokens and no temperature."""
        from evaluators.llm_utils import call_llm
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test response"
        mock_client.chat.completions.create.return_value = mock_response
        
        call_llm(
            client=mock_client,
            prompt="test prompt",
            model="gpt-5-nano-2025-08-07",  # Reasoning model
            temperature=0.0,
            max_tokens=100,
            model_context_limit=128000,
        )
        
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "max_completion_tokens" in call_kwargs, "gpt-5-nano should use max_completion_tokens"
        assert "max_tokens" not in call_kwargs, "gpt-5-nano should NOT use max_tokens"
        assert "temperature" not in call_kwargs, "gpt-5-nano should NOT use temperature"

    def test_gpt5_regular_uses_max_tokens_and_temperature(self):
        """gpt-5.2 (non-nano) should use max_tokens and temperature."""
        from evaluators.llm_utils import call_llm
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test response"
        mock_client.chat.completions.create.return_value = mock_response
        
        call_llm(
            client=mock_client,
            prompt="test prompt",
            model="gpt-5.2-2025-12-11",  # Regular GPT-5 model (not nano)
            temperature=0.0,
            max_tokens=100,
            model_context_limit=128000,
        )
        
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "max_tokens" in call_kwargs, "gpt-5.2 should use max_tokens"
        assert "max_completion_tokens" not in call_kwargs, "gpt-5.2 should NOT use max_completion_tokens"
        assert "temperature" in call_kwargs, "gpt-5.2 should use temperature"
        assert call_kwargs["temperature"] == 0.0

    def test_fallback_on_max_tokens_error(self):
        """Test fallback to max_completion_tokens when max_tokens is rejected."""
        from evaluators.llm_utils import call_llm
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "fallback response"
        
        # First call raises error mentioning both max_tokens and max_completion_tokens
        # Second call (retry) succeeds
        mock_client.chat.completions.create.side_effect = [
            Exception("Unsupported parameter: 'max_tokens' is not supported, use 'max_completion_tokens'"),
            mock_response,
        ]
        
        result = call_llm(
            client=mock_client,
            prompt="test prompt",
            model="gpt-4o-mini",  # Regular model, but API rejects max_tokens
            temperature=0.0,
            max_tokens=100,
            model_context_limit=128000,
        )
        
        assert result == "fallback response"
        # Verify retry was made
        assert mock_client.chat.completions.create.call_count == 2
        # Second call should use max_completion_tokens
        retry_kwargs = mock_client.chat.completions.create.call_args_list[1][1]
        assert "max_completion_tokens" in retry_kwargs

    def test_fallback_on_temperature_error(self):
        """Test fallback when temperature is rejected."""
        from evaluators.llm_utils import call_llm
        
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "fallback response"
        
        # First call raises error about temperature
        # Second call (retry) succeeds
        mock_client.chat.completions.create.side_effect = [
            Exception("This model does not support temperature parameter"),
            mock_response,
        ]
        
        result = call_llm(
            client=mock_client,
            prompt="test prompt",
            model="gpt-4o-mini",  # Regular model, but API rejects temperature
            temperature=0.0,
            max_tokens=100,
            model_context_limit=128000,
        )
        
        assert result == "fallback response"
        # Verify retry was made
        assert mock_client.chat.completions.create.call_count == 2
        # Second call should NOT have temperature
        retry_kwargs = mock_client.chat.completions.create.call_args_list[1][1]
        assert "temperature" not in retry_kwargs
