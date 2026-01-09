# AgentBusters - CIO-Agent FAB++ System

A dynamic finance agent benchmark system for the [AgentBeats Competition](https://rdi.berkeley.edu/agentx-agentbeats). This project implements both **Green Agent** (Evaluator) and **Purple Agent** (Finance Analyst) using the A2A (Agent-to-Agent) protocol.

## 🚀 AgentBeats Platform Submission

This codebase is designed to work with the [AgentBeats platform](https://agentbeats.dev). The Green Agent follows the official [green-agent-template](https://github.com/RDI-Foundation/green-agent-template).

### Quick Start for AgentBeats

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
# source .venv/bin/activate   # Linux/Mac

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Start Green Agent A2A server
python src/cio_agent/a2a_server.py --host 0.0.0.0 --port 9109

# 4. Verify agent card (in another terminal)
curl http://localhost:9109/.well-known/agent.json

# 5. Run A2A conformance tests
python -m pytest tests/test_a2a_green.py -v --agent-url http://localhost:9109
```

### Docker Build & Publish

```bash
# Build Green Agent image
docker build -f Dockerfile.green -t ghcr.io/your-org/cio-agent-green:latest .

# Run locally
docker run -p 9109:9109 ghcr.io/your-org/cio-agent-green:latest --host 0.0.0.0

# Push to GitHub Container Registry
docker push ghcr.io/your-org/cio-agent-green:latest
```

The CI/CD workflow (`.github/workflows/test-and-publish-green.yml`) automatically builds and publishes on push to `main` or version tags.

---

## Overview

The CIO-Agent FAB++ system evaluates AI agents on financial analysis tasks using:

- **FAB++ (Finance Agent Benchmark)**: Dynamic variant with 537 questions across 9 categories
- **MCP Trinity**: SEC EDGAR, Yahoo Finance, and Python Sandbox servers
- **Adversarial Debate**: Counter-argument generation to test conviction
- **Alpha Score**: Comprehensive evaluation metric

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     AgentBusters System                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐         A2A Protocol        ┌───────────┐ │
│  │   Green Agent   │◄──────────────────────────►│  Purple   │ │
│  │   (Evaluator)   │                             │   Agent   │ │
│  │   CIO-Agent     │                             │ (Analyst) │ │
│  └────────┬────────┘                             └─────┬─────┘ │
│           │                                            │       │
│           ▼                                            ▼       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    MCP Trinity                          │   │
│  │  ┌──────────┐   ┌──────────────┐   ┌──────────────┐    │   │
│  │  │  SEC     │   │   Yahoo      │   │   Python     │    │   │
│  │  │  EDGAR   │   │   Finance    │   │   Sandbox    │    │   │
│  │  │  MCP     │   │   MCP        │   │   MCP        │    │   │
│  │  └──────────┘   └──────────────┘   └──────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+ (Python 3.13 recommended for AgentBeats)
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Docker (optional, for full stack deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/yxc20089/AgentBusters.git
cd AgentBusters

# Option 1: Using uv (recommended)
uv sync

# Option 2: Using pip
pip install -e ".[dev]"
```

### Running the Green Agent (A2A Server for AgentBeats)

```bash
# Start A2A server (AgentBeats compatible)
py -3.13 src/cio_agent/a2a_server.py --host 0.0.0.0 --port 9109

# With custom card URL
py -3.13 src/cio_agent/a2a_server.py --host 0.0.0.0 --port 9109 --card-url https://your-domain.com/
```

### Running the Green Agent (CLI for local testing)

```bash
# List available tasks
cio-agent list-tasks

# Run evaluation on a specific task
cio-agent evaluate --task-id FAB_001 --purple-endpoint http://localhost:9110

# Run the NVIDIA Q3 FY2026 test
python scripts/test_nvidia.py
```

### Running the Purple Agent (Finance Analyst)

```bash
# Start the A2A server
purple-agent serve --host 0.0.0.0 --port 8101

# Or use the simple test agent
py -3.13 src/simple_purple_agent.py --host 0.0.0.0 --port 9110

# Or run a direct analysis
purple-agent analyze "Did NVIDIA beat or miss Q3 FY2026 expectations?" --ticker NVDA
```

## MCP Server Configuration

The Purple Agent connects to MCP servers for real financial data:

| Server | Default URL | Purpose |
|--------|-------------|---------|
| SEC EDGAR MCP | `http://localhost:8101` | SEC filings, XBRL data |
| Yahoo Finance MCP | `http://localhost:8102` | Market data, statistics |
| Sandbox MCP | `http://localhost:8103` | Python code execution |

Configure via environment variables:

```bash
export MCP_EDGAR_URL=http://localhost:8101
export MCP_YFINANCE_URL=http://localhost:8102
export MCP_SANDBOX_URL=http://localhost:8103
```

## Docker Deployment

### Green Agent (AgentBeats Compatible)

```bash
# Build
docker build -f Dockerfile.green -t cio-agent-green .

# Run
docker run -p 9109:9109 cio-agent-green --host 0.0.0.0 --port 9109

# With API keys
docker run -p 9109:9109 -e OPENAI_API_KEY=sk-xxx cio-agent-green --host 0.0.0.0
```

### Full Stack (MCP + Purple + Green)

```bash
# Build all images
docker compose build

# Start services
docker compose up -d

# Check status
docker ps --filter "name=fab-plus"

# Stop services
docker compose down
```

External ports (default compose): Purple `9110->9110`, EDGAR `8101->8101`, YFinance `8102->8102`, Sandbox `8103->8103`.

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for LLM | - |
| `ANTHROPIC_API_KEY` | Anthropic API key for LLM | - |
| `LLM_MODEL` | Model to use | `gpt-4o` |
| `SIMULATION_DATE` | Date for temporal locking (YYYY-MM-DD) | Current date |
| `MCP_EDGAR_URL` | SEC EDGAR MCP server URL | `http://localhost:8101` |
| `MCP_YFINANCE_URL` | Yahoo Finance MCP server URL | `http://localhost:8102` |
| `MCP_SANDBOX_URL` | Sandbox MCP server URL | `http://localhost:8103` |

## Project Structure

```
AgentBusters/
├── src/
│   ├── cio_agent/           # Green Agent (Evaluator)
│   │   ├── a2a_server.py    # A2A server entry point (AgentBeats)
│   │   ├── green_executor.py # A2A protocol executor
│   │   ├── green_agent.py   # FAB++ evaluation logic
│   │   ├── messenger.py     # A2A messaging utilities
│   │   ├── models.py        # Core data models
│   │   ├── evaluator.py     # Comprehensive evaluator
│   │   ├── debate.py        # Adversarial debate manager
│   │   ├── task_generator.py # Dynamic task generation
│   │   └── cli.py           # CLI interface
│   │
│   ├── purple_agent/        # Purple Agent (Finance Analyst)
│   │   ├── server.py        # A2A FastAPI server
│   │   ├── executor.py      # A2A executor implementation
│   │   ├── agent.py         # Main agent class
│   │   └── cli.py           # CLI interface
│   │
│   ├── simple_purple_agent.py # Simple test Purple Agent
│   │
│   ├── mcp_servers/         # MCP servers (FastMCP)
│   │   ├── sec_edgar.py     # SEC EDGAR server
│   │   ├── yahoo_finance.py # Yahoo Finance server
│   │   └── sandbox.py       # Python execution sandbox
│   │
│   └── evaluators/          # Evaluation components
│       ├── macro.py         # Macro thesis evaluator
│       ├── fundamental.py   # Fundamental analysis evaluator
│       └── execution.py     # Execution quality evaluator
│
├── tests/
│   ├── test_a2a_green.py    # A2A conformance tests
│   ├── test_e2e.py          # E2E tests with real NVIDIA data
│   └── test_purple_agent.py # Purple Agent tests
│
├── .github/workflows/
│   └── test-and-publish-green.yml  # CI/CD for Green Agent
│
├── Dockerfile.green         # Green Agent container (AgentBeats)
├── Dockerfile               # Legacy Green Agent container
├── Dockerfile.purple        # Purple Agent container
└── pyproject.toml           # Project configuration
```

## Alpha Score Formula

The evaluation uses the Alpha Score metric:

```
Alpha Score = (RoleScore × DebateMultiplier) / (ln(1 + Cost) × (1 + LookaheadPenalty))
```

Where:
- **RoleScore**: Weighted combination of Macro (30%), Fundamental (40%), Execution (30%)
- **DebateMultiplier**: 0.5x - 1.2x based on conviction in adversarial debate
- **Cost**: Total USD cost of LLM and tool calls
- **LookaheadPenalty**: Penalty for temporal violations (accessing future data)

## Testing

```bash
# Run all tests
py -3.13 -m pytest tests/ -v

# Run A2A conformance tests
py -3.13 -m pytest tests/test_a2a_green.py -v --agent-url http://localhost:9109

# Run with coverage
py -3.13 -m pytest tests/ --cov=src --cov-report=html
```

## API Reference

### Green Agent A2A Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/.well-known/agent.json` | GET | Agent Card (A2A discovery) |
| `/` | POST | A2A JSON-RPC endpoint |

### Purple Agent A2A Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/.well-known/agent.json` | GET | Agent Card (A2A discovery) |
| `/health` | GET | Health check |
| `/analyze` | POST | Direct analysis (non-A2A) |
| `/` | POST | A2A JSON-RPC endpoint |

## Competition Info

This project is built for the [AgentBeats Finance Track](https://rdi.berkeley.edu/agentx-agentbeats):

- **Phase 1** (Jan 15, 2026): Green Agent submissions
- **Phase 2** (Feb 2026): Purple Agent submissions

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `py -3.13 -m pytest tests/ -v`
4. Submit a pull request

## Acknowledgments

- [AgentBeats Competition](https://rdi.berkeley.edu/agentx-agentbeats) by Berkeley RDI
- [A2A Protocol](https://a2a-protocol.org/) by Google
- [FAB Benchmark](https://github.com/financial-agent-benchmark/FAB) for task templates
- [green-agent-template](https://github.com/RDI-Foundation/green-agent-template) for A2A implementation reference

