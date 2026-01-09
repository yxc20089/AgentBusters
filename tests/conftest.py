"""
Pytest configuration for AgentBusters tests.
"""

import pytest


def pytest_addoption(parser):
    """Add custom pytest options."""
    parser.addoption(
        "--agent-url",
        action="store",
        default="http://localhost:9109",
        help="URL of the agent to test"
    )


@pytest.fixture
def agent(request):
    """Get the agent URL from command line."""
    return request.config.getoption("--agent-url")
