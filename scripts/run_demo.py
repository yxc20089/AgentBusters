#!/usr/bin/env python3
"""
Demo script: Run Purple Agent analysis and Green Agent evaluation

This demonstrates the full CIO-Agent FAB++ system with:
1. Purple Agent performing financial analysis
2. Green Agent evaluating the Purple Agent's response
"""

import asyncio
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import sys
sys.path.insert(0, "src")

from purple_agent.agent import FinanceAnalysisAgent
from cio_agent.models import (
    Task,
    TaskCategory,
    TaskDifficulty,
    TaskRubric,
    GroundTruth,
    FinancialData,
    AgentResponse,
)
from cio_agent.evaluator import ComprehensiveEvaluator, EvaluationReporter
from cio_agent.orchestrator import MockAgentClient

console = Console()


async def run_demo():
    """Run the full demo pipeline."""

    console.print(Panel.fit(
        "[bold blue]AgentBusters Demo[/bold blue]\n\n"
        "Purple Agent (Finance Analyst) vs Green Agent (Evaluator)\n"
        "Task: NVIDIA Q3 FY2026 Earnings Analysis"
    ))

    # Create Purple Agent
    console.print("\n[cyan]1. Initializing Purple Agent...[/cyan]")
    purple_agent = FinanceAnalysisAgent(
        simulation_date=datetime(2025, 11, 20)
    )

    # Display Agent Card
    console.print(f"   Agent: {purple_agent.card.name}")
    console.print(f"   Skills: {len(purple_agent.card.skills)} available")
    console.print(f"   URL: {purple_agent.card.url}")

    # Define the evaluation task
    console.print("\n[cyan]2. Creating evaluation task...[/cyan]")

    task = Task(
        question_id="NVIDIA_Q3_FY2026_demo",
        category=TaskCategory.BEAT_OR_MISS,
        question=(
            "Did NVIDIA beat or miss analyst expectations in Q3 FY2026 "
            "(quarter ended October 26, 2025)? Provide the actual vs expected "
            "revenue and EPS figures."
        ),
        ticker="NVDA",
        fiscal_year=2026,
        simulation_date=datetime(2025, 11, 20),
        ground_truth=GroundTruth(
            macro_thesis=(
                "NVIDIA beat expectations with record $57B revenue (+62% YoY), "
                "EPS of $1.30 beat $1.25 consensus. Blackwell GPU demand drove growth."
            ),
            key_themes=["beat", "revenue", "EPS", "data center", "AI", "Blackwell"],
            financials=FinancialData(
                revenue=57_000_000_000,
                net_income=31_910_000_000,
                gross_margin=0.734,
                eps=1.30,
            ),
            expected_recommendation="Beat",
            numerical_answer=57_000_000_000,
        ),
        difficulty=TaskDifficulty.MEDIUM,
        rubric=TaskRubric(
            criteria=[
                "Correctly identify beat/miss status",
                "Provide actual vs expected figures",
                "Analyze key drivers",
            ],
            mandatory_elements=["beat determination", "revenue", "EPS"],
        ),
    )

    console.print(f"   Category: {task.category.value}")
    console.print(f"   Ticker: {task.ticker}")
    console.print(f"   Difficulty: {task.difficulty.value}")

    # Purple Agent generates analysis
    console.print("\n[cyan]3. Purple Agent analyzing...[/cyan]")

    analysis = await purple_agent.analyze(task.question, ticker=task.ticker)

    console.print(Panel(
        analysis[:1000] + "..." if len(analysis) > 1000 else analysis,
        title="Purple Agent Response",
        border_style="magenta"
    ))

    # Check beat/miss
    console.print("\n[cyan]4. Checking beat/miss determination...[/cyan]")

    beat_check = await purple_agent.check_earnings_beat(
        ticker="NVDA",
        actual_revenue=57_000_000_000,
        actual_eps=1.30,
        expected_revenue=54_920_000_000,
        expected_eps=1.25,
    )

    table = Table(title="Beat/Miss Analysis")
    table.add_column("Metric", style="cyan")
    table.add_column("Actual", style="green")
    table.add_column("Expected", style="yellow")
    table.add_column("Result", style="bold")

    if beat_check["revenue_beat"]:
        rev = beat_check["revenue_beat"]
        table.add_row(
            "Revenue",
            f"${rev['actual']/1e9:.1f}B",
            f"${rev['expected']/1e9:.1f}B",
            "BEAT" if rev['beat'] else "MISS"
        )

    if beat_check["eps_beat"]:
        eps = beat_check["eps_beat"]
        table.add_row(
            "EPS",
            f"${eps['actual']:.2f}",
            f"${eps['expected']:.2f}",
            "BEAT" if eps['beat'] else "MISS"
        )

    console.print(table)
    console.print(f"\n   [bold]Overall Assessment: {beat_check['overall_assessment']}[/bold]")

    # Create agent response for evaluation
    console.print("\n[cyan]5. Green Agent evaluating...[/cyan]")

    agent_response = AgentResponse(
        agent_id="purple-finance-agent",
        task_id=task.question_id,
        analysis=analysis,
        recommendation="Beat",
        confidence=0.9,
        financials_extracted=FinancialData(
            revenue=57_000_000_000,
            eps=1.30,
        ),
        code_traces=[],
        tool_calls=[],
    )

    # Run evaluation using mock agent
    mock_agent = MockAgentClient(
        agent_id="purple-finance-agent",
        model="gpt-4o",
    )
    # Inject the Purple Agent's actual analysis into the mock response
    mock_agent._override_response = agent_response

    evaluator = ComprehensiveEvaluator()
    result = await evaluator.run_full_evaluation(
        task=task,
        agent_client=mock_agent,
        conduct_debate=True,
    )

    # Override with actual response for scoring
    # (In a real scenario, this would come from the A2A protocol)

    # Display results
    console.print("\n" + "=" * 60)
    console.print("[bold green]EVALUATION COMPLETE[/bold green]")
    console.print("=" * 60)

    summary = EvaluationReporter.generate_summary(result)
    console.print(summary)

    # Final score
    console.print(Panel.fit(
        f"[bold green]Final Alpha Score: {result.alpha_score.score:.2f}[/bold green]\n\n"
        f"Macro: {result.role_score.macro.score:.1f}/100\n"
        f"Fundamental: {result.role_score.fundamental.score:.1f}/100\n"
        f"Execution: {result.role_score.execution.score:.1f}/100\n"
        f"Debate Multiplier: {result.debate_result.debate_multiplier}x",
        title="Results"
    ))

    return result


if __name__ == "__main__":
    asyncio.run(run_demo())
