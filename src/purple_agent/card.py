"""
Agent Card Definition for Purple Agent

Defines the capabilities and metadata for the Finance Analysis Agent
following the A2A protocol specification.
"""

from a2a.types import (
    AgentCard,
    AgentCapabilities,
    AgentSkill,
    AgentProvider,
)


def get_agent_card(host: str = "localhost", port: int = 8101) -> AgentCard:
    """
    Create the Agent Card for the Purple Finance Agent.

    The Agent Card advertises the agent's capabilities to other agents
    following the A2A protocol specification.

    Args:
        host: The hostname where the agent is running
        port: The port where the agent is listening

    Returns:
        AgentCard with full capability specification
    """
    return AgentCard(
        name="Purple Finance Agent",
        description=(
            "A specialized financial analysis agent for the AgentBeats competition. "
            "Capable of analyzing earnings reports, SEC filings, financial ratios, "
            "and providing investment recommendations based on fundamental analysis."
        ),
        url=f"http://{host}:{port}/",
        version="1.0.0",
        provider=AgentProvider(
            organization="AgentBusters Team",
            url="https://github.com/yxc20089/AgentBusters",
        ),
        capabilities=AgentCapabilities(
            streaming=True,
            push_notifications=False,
            state_transition_history=True,
        ),
        skills=[
            AgentSkill(
                id="earnings_analysis",
                name="Earnings Analysis",
                description=(
                    "Analyze quarterly and annual earnings reports to determine "
                    "beat/miss status, revenue trends, and profitability metrics."
                ),
                tags=["finance", "earnings", "analysis"],
                examples=[
                    "Did NVIDIA beat or miss Q3 FY2026 expectations?",
                    "Analyze Apple's Q4 2024 earnings report",
                    "What was Tesla's revenue growth in Q3 2024?",
                ],
            ),
            AgentSkill(
                id="sec_filing_analysis",
                name="SEC Filing Analysis",
                description=(
                    "Extract and analyze information from SEC filings including "
                    "10-K, 10-Q, 8-K, and proxy statements."
                ),
                tags=["finance", "sec", "filings", "10-k", "10-q"],
                examples=[
                    "What risks did Microsoft disclose in their latest 10-K?",
                    "Extract revenue breakdown from Amazon's 10-Q",
                    "Summarize the executive compensation from Tesla's proxy",
                ],
            ),
            AgentSkill(
                id="financial_ratio_calculation",
                name="Financial Ratio Calculation",
                description=(
                    "Calculate and analyze financial ratios including P/E, P/B, "
                    "ROE, ROA, debt ratios, and liquidity metrics."
                ),
                tags=["finance", "ratios", "valuation", "analysis"],
                examples=[
                    "Calculate NVIDIA's current P/E ratio",
                    "What is Apple's debt-to-equity ratio?",
                    "Compare ROE across FAANG companies",
                ],
            ),
            AgentSkill(
                id="market_analysis",
                name="Market Analysis",
                description=(
                    "Analyze market trends, sector performance, and macro-economic "
                    "factors affecting specific stocks or industries."
                ),
                tags=["finance", "market", "macro", "trends"],
                examples=[
                    "How is AI demand affecting semiconductor stocks?",
                    "What are the key drivers of cloud growth?",
                    "Analyze the impact of interest rates on tech valuations",
                ],
            ),
            AgentSkill(
                id="investment_recommendation",
                name="Investment Recommendation",
                description=(
                    "Provide investment recommendations with supporting thesis, "
                    "key risks, and price targets based on fundamental analysis."
                ),
                tags=["finance", "recommendation", "investment", "thesis"],
                examples=[
                    "Should I buy NVIDIA at current prices?",
                    "What's your rating on Microsoft stock?",
                    "Provide a bull/bear case for Tesla",
                ],
            ),
        ],
        default_input_modes=["text"],
        default_output_modes=["text"],
    )
