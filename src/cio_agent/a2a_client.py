"""
HTTP client to talk to the Purple Agent server.

Uses the Purple Agent's `/analyze` convenience endpoint to obtain an analysis
and wraps it into the Green-side AgentResponse / DebateRebuttal structures.
"""

from __future__ import annotations

import httpx
from datetime import datetime
from typing import Optional

from cio_agent.models import AgentResponse, DebateRebuttal, FinancialData, Task


class PurpleHTTPAgentClient:
    """Minimal HTTP client for the Purple Agent `/analyze` endpoint."""

    def __init__(
        self,
        base_url: str,
        agent_id: str = "purple-agent",
        model: str = "purple-http",
        timeout_seconds: int = 300,
    ):
        self.base_url = base_url.rstrip("/")
        self.agent_id = agent_id
        self.model = model
        self.timeout_seconds = timeout_seconds

    async def process_task(self, task: Task) -> AgentResponse:
        """Send a task to purple `/analyze` and wrap the response."""
        url = f"{self.base_url}/analyze"
        payload = {
            "question": task.question,
            "ticker": task.ticker,
            "simulation_date": task.simulation_date.isoformat(),
        }
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        analysis = data.get("analysis") or "No analysis returned."
        
        # Parse tool calls from response
        from cio_agent.models import ToolCall
        raw_tool_calls = data.get("tool_calls", [])
        tool_calls = []
        for tc in raw_tool_calls:
            try:
                tool_calls.append(ToolCall(
                    tool_name=tc.get("tool", "unknown"),
                    params=tc.get("params", {}),
                    timestamp=datetime.fromisoformat(tc.get("timestamp", datetime.now().isoformat())),
                    duration_ms=tc.get("elapsed_ms", 0),
                    success=not tc.get("is_error", False),
                    result=tc.get("result"),
                ))
            except Exception:
                pass  # Skip malformed tool calls

        return AgentResponse(
            agent_id=self.agent_id,
            task_id=task.question_id,
            analysis=analysis,
            recommendation=analysis,  # Same as analysis - recommendation field required by model
            extracted_financials=FinancialData(),
            tool_calls=tool_calls,
            code_executions=[],
            timestamp=datetime.now(),
            execution_time_seconds=0.0,
        )

    async def process_challenge(
        self,
        task_id: str,
        challenge: str,
        original_response: Optional[AgentResponse] = None,
        ticker: Optional[str] = None,
    ) -> DebateRebuttal:
        """
        Get a real rebuttal from the Purple Agent by calling /analyze with debate context.

        This reuses the /analyze endpoint by framing the debate as a contextual question
        that includes the original analysis and the counter-argument, prompting the LLM
        to generate a substantive defense.
        """
        original_analysis = original_response.analysis if original_response else "No original analysis."

        debate_prompt = f"""DEBATE REBUTTAL REQUEST

Your original analysis was:
---
{original_analysis}
---

A critical reviewer has challenged your analysis:
---
{challenge}
---

Respond to this challenge. Either:
1. Defend your original analysis with specific evidence and reasoning
2. Address each point raised by the reviewer
3. If valid criticisms, acknowledge them and refine your position

Be substantive and specific. Do not simply repeat your original analysis."""

        url = f"{self.base_url}/analyze"
        payload = {
            "question": debate_prompt,
            "ticker": ticker or "AAPL",  # Use provided ticker or default to a valid one
            "simulation_date": datetime.now().isoformat(),
        }

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        defense = data.get("analysis") or "No defense provided."

        # Detect if new evidence was cited (not in original analysis)
        new_evidence_cited = []
        evidence_indicators = [
            "according to", "data shows", "reported", "SEC filing",
            "10-K", "10-Q", "earnings call", "quarterly report"
        ]
        defense_lower = defense.lower()
        original_lower = original_analysis.lower()
        for indicator in evidence_indicators:
            if indicator in defense_lower and indicator not in original_lower:
                new_evidence_cited.append(f"New reference: {indicator}")
                break

        return DebateRebuttal(
            agent_id=self.agent_id,
            task_id=task_id,
            defense=defense,
            new_evidence_cited=new_evidence_cited,
            tool_calls=[],
        )
