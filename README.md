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

- Python 3.13 (recommended for AgentBeats)
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- vLLM, Ollama, or LM Studio (for local LLM deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/yxc20089/AgentBusters.git
cd AgentBusters

# Option 1: Using uv (recommended)
uv sync

# Option 2: Using pip
pip install -e ".[dev]"

# Option 3: Create .env file from template
cp .env.example .env
# Edit .env with your API keys and configuration
```

### Running the Green Agent (A2A Server for AgentBeats)

```bash
# Start A2A server (AgentBeats compatible)
python src/cio_agent/a2a_server.py --host 0.0.0.0 --port 9109

# With custom card URL
python src/cio_agent/a2a_server.py --host 0.0.0.0 --port 9109 --card-url https://your-domain.com/
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
python src/simple_purple_agent.py --host 0.0.0.0 --port 9110

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

Configure via environment variables or `.env` file:

```bash
export MCP_EDGAR_URL=http://localhost:8101
export MCP_YFINANCE_URL=http://localhost:8102
export MCP_SANDBOX_URL=http://localhost:8103
```

Or in `.env`:
```dotenv
MCP_EDGAR_URL=http://localhost:8101
MCP_YFINANCE_URL=http://localhost:8102
MCP_SANDBOX_URL=http://localhost:8103
```

## Local Deployment

### Quick Start (Minimal Setup)

```bash
# Terminal 1: Green Agent
python src/cio_agent/a2a_server.py --host 0.0.0.0 --port 9109

# Terminal 2: Purple Agent  
python src/simple_purple_agent.py --host 0.0.0.0 --port 9110

# Verify services
curl http://localhost:9109/.well-known/agent.json
curl http://localhost:9110/health

# Run tests
python -m pytest tests/test_e2e.py -v
```

### With Local vLLM

```bash
# Configure .env first (see Configuration section)
# Terminal 1: vLLM
export LIBRARY_PATH="/chronos_data/huixu/libcuda_stub:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/chronos_data/huixu/libcuda_stub:$LD_LIBRARY_PATH"
vllm serve openai/gpt-oss-20b --port 8000

# Terminal 2-3: Start agents (same as above)
```

### With MCP Servers (Optional)

```bash
# Terminal 4: SEC EDGAR MCP
python -m src.mcp_servers.sec_edgar --host 0.0.0.0 --port 8101

# Terminal 5: Yahoo Finance MCP
python -m src.mcp_servers.yahoo_finance --host 0.0.0.0 --port 8102

# Terminal 6: Sandbox MCP
python -m src.mcp_servers.sandbox --host 0.0.0.0 --port 8103
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

### Individual Service Build & Run

```bash
# Green Agent
docker build -f Dockerfile -t cio-agent-green .
docker run -p 9109:9109 cio-agent-green

# Purple Agent
docker build -f Dockerfile.purple -t purple-agent .
docker run -p 9110:9110 purple-agent

# MCP Servers
docker build -f Dockerfile.mcp-edgar -t mcp-edgar .
docker run -p 8101:8000 mcp-edgar

docker build -f Dockerfile.mcp-yahoo -t mcp-yahoo .
docker run -p 8102:8000 mcp-yahoo

docker build -f Dockerfile.mcp-sandbox -t mcp-sandbox .
docker run -p 8103:8000 mcp-sandbox
```

Port mapping: Green Agent `9109`, Purple Agent `9110`, EDGAR `8101`, YFinance `8102`, Sandbox `8103`.

## Configuration

### Environment Setup with `.env` File

```bash
# 1. Create .env from template
cp .env.example .env

# 2. Edit .env with your LLM configuration
```

**For local vLLM (gpt-oss-20b):**
```dotenv
LLM_PROVIDER=openai
OPENAI_API_BASE=http://localhost:8000/v1
OPENAI_API_KEY=dummy
LLM_MODEL=gpt-oss-20b
```

**For OpenAI API:**
```dotenv
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key
LLM_MODEL=gpt-4o
```

**For Anthropic API:**
```dotenv
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-your-key
LLM_MODEL=claude-3.5-sonnet
```

**MCP Servers (optional):**
```dotenv
MCP_EDGAR_URL=http://localhost:8101
MCP_YFINANCE_URL=http://localhost:8102
MCP_SANDBOX_URL=http://localhost:8103
```

The agents will automatically load `.env` on startup. Alternatively, you can use `export` commands instead of `.env` file.

### Environment Variables Reference

| Variable | Description | Example |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM provider | `openai`, `anthropic` |
| `LLM_MODEL` | Model name | `gpt-4o`, `claude-3.5-sonnet`, `gpt-oss-20b` |
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |
| `OPENAI_API_BASE` | Custom API endpoint (for local vLLM) | `http://localhost:8000/v1` |
| `ANTHROPIC_API_KEY` | Anthropic API key | `sk-ant-...` |
| `MCP_EDGAR_URL` | SEC EDGAR MCP server | `http://localhost:8101` |
| `MCP_YFINANCE_URL` | Yahoo Finance MCP server | `http://localhost:8102` |
| `MCP_SANDBOX_URL` | Sandbox MCP server | `http://localhost:8103` |

## Running with Local LLM (vLLM + gpt-oss-20b)

### 1. Install and Start vLLM

```bash
# Install vLLM
pip install vllm

# Start vLLM server (Terminal 1)
export LIBRARY_PATH="/chronos_data/huixu/libcuda_stub:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/chronos_data/huixu/libcuda_stub:$LD_LIBRARY_PATH"
vllm serve openai/gpt-oss-20b --port 8000

# For multi-GPU: add --tensor-parallel-size=2
```

### 2. Configure .env for Local LLM

```bash
cp .env.example .env
```

Edit `.env`:
```dotenv
LLM_PROVIDER=openai
OPENAI_API_BASE=http://localhost:8000/v1
OPENAI_API_KEY=dummy
LLM_MODEL=gpt-oss-20b
```

### 3. Start Agents

```bash
# Terminal 2: Green Agent
python src/cio_agent/a2a_server.py --host 0.0.0.0 --port 9109

# Terminal 3: Purple Agent
python src/simple_purple_agent.py --host 0.0.0.0 --port 9110

# Terminal 4: Run tests
python -m pytest tests/test_e2e.py -v
```

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
python -m pytest tests/ -v

# Run A2A conformance tests
python -m pytest tests/test_a2a_green.py -v --agent-url http://localhost:9109

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
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
3. Run tests: `python -m pytest tests/ -v`
4. Submit a pull request

## Acknowledgments

- [AgentBeats Competition](https://rdi.berkeley.edu/agentx-agentbeats) by Berkeley RDI
- [A2A Protocol](https://a2a-protocol.org/) by Google
- [FAB Benchmark](https://github.com/financial-agent-benchmark/FAB) for task templates
- [green-agent-template](https://github.com/RDI-Foundation/green-agent-template) for A2A implementation reference

