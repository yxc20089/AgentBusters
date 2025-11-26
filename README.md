# AgentBusters FAB++ Demo (Green ↔ Purple)

This guide shows how to run the green evaluator against the baseline purple agent, record the A2A dialogue (task → response → challenge → rebuttal), and capture outputs for a demo.

## 1) What’s included
- Green evaluator (CIO-Agent) CLI.
- Baseline purple A2A HTTP agent (heuristic responses).
- A2A HTTP client so green can talk to purple via `--purple-endpoint`.

Note: MCP server code/contexts are not included. When using Docker, run green with `--no-deps` unless you supply your own MCP images/contexts.

## 2) Prerequisites
- Python 3.11+ (for local runs).
- Docker Desktop (for containerized runs).
- Optional: `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` in `.env` for real LLM calls (not required; heuristics work without).

Example `.env` (repo root):
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

## 3) Option A: Local run (no Docker)
```powershell
# Create and activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .

# Start purple (keep this window open)
python -m cio_agent.cli run-purple --port 8090
```
New terminal (point green at purple):
```powershell
python -m cio_agent.cli evaluate --task-id FAB_050 --date 2024-01-01 --output summary --purple-endpoint http://localhost:8090/a2a
```
You should see `Agent: purple-baseline` in the green output and A2A logs in the purple window.

## 4) Option B: Docker
Start purple container:
```powershell
docker compose up -d purple-agent
```
Run green container (skip MCP deps, because MCP folders are not in this repo):
```powershell
docker compose run --rm --no-deps cio-agent cio-agent evaluate --task-id FAB_050 --date 2024-01-01 --output summary --purple-endpoint http://purple-agent:8090/a2a
```
- If purple runs on host, use `http://host.docker.internal:8090/a2a`.
- Rebuild purple after code changes: `docker compose build purple-agent`.

## 5) Verify A2A dialogue
- Green output should show `Agent: purple-baseline`.
- Purple logs should show `a2a_received` / `a2a_replied` for `task_assignment`, `task_response`, `challenge`, `rebuttal`.

## 6) Manual A2A calls (for recording)
POST to purple at `/a2a`:
- `task_assignment` JSON → `task_response`
- `challenge` JSON → `rebuttal`

PowerShell example (here-string starts at column 1):
```powershell
$body = @'
{
  "protocol_version": "1.0",
  "message_type": "task_assignment",
  "sender_id": "cio-agent-green",
  "receiver_id": "purple-baseline",
  "timestamp": "2025-01-20T00:00:00Z",
  "payload": {
    "task_id": "demo_task_001",
    "question": "Did NVIDIA beat analyst EPS estimates in Q3 FY2026?",
    "category": "Beat or Miss",
    "simulation_date": "2025-11-20T00:00:00",
    "available_tools": ["sec-edgar-mcp", "yahoo-finance-mcp", "mcp-sandbox"],
    "deadline_seconds": 1800,
    "difficulty": "medium",
    "fiscal_year": 2026,
    "ticker": "NVDA",
    "requires_code_execution": false
  }
}
'@
Invoke-RestMethod -Method Post -Uri http://localhost:8090/a2a -Body $body -ContentType "application/json"
```
Then send a `challenge` to get a `rebuttal`:
```powershell
$challenge = @'
{
  "protocol_version": "1.0",
  "message_type": "challenge",
  "sender_id": "cio-agent-green",
  "receiver_id": "purple-baseline",
  "timestamp": "2025-01-20T00:05:00Z",
  "payload": {
    "task_id": "demo_task_001",
    "challenge": "Valuation looks stretched; justify your BUY."
  }
}
'@
Invoke-RestMethod -Method Post -Uri http://localhost:8090/a2a -Body $challenge -ContentType "application/json"
```

## 7) Demo capture tips
- Record two terminals: green output (scores/Alpha) and purple logs (A2A send/receive).
- Use `--output markdown` for richer green output.
- Keep purple running in its own window so logs are visible during recording.

## 8) Tests
Run locally (no Docker needed):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]

# all tests
pytest
# or just A2A basics
pytest tests/test_a2a_basics.py
```
