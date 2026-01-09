"""
Simple Purple Agent for Testing

A minimal finance agent implementation for testing the Green Agent evaluation.
This agent provides basic responses for FAB++ evaluation tasks.
"""

import argparse
import json
import uvicorn
from dotenv import load_dotenv
from starlette.responses import JSONResponse
from starlette.routing import Route

# Load environment variables from .env file
load_dotenv()

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Task,
    TaskState,
    UnsupportedOperationError,
    InvalidRequestError,
    Part,
    TextPart,
    DataPart,
)
from a2a.utils.errors import ServerError
from a2a.utils import (
    new_agent_text_message,
    new_task,
    get_message_text,
)


TERMINAL_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected
}


class SimpleFinanceAgent:
    """
    Simple finance agent for testing.
    
    Provides basic financial analysis responses.
    """
    
    def __init__(self):
        pass

    async def run(self, message_text: str, updater: TaskUpdater) -> None:
        """Generate a simple financial analysis response."""
        
        # Parse incoming task
        try:
            task_data = json.loads(message_text)
            ticker = task_data.get("ticker", "UNKNOWN")
            question = task_data.get("question", message_text)
            category = task_data.get("category", "general")
        except json.JSONDecodeError:
            ticker = "UNKNOWN"
            question = message_text
            category = "general"

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Analyzing {ticker}...")
        )

        # Generate response based on category
        if category == "beat_or_miss":
            analysis = f"""
## {ticker} Earnings Analysis

Based on available financial data, {ticker} has demonstrated strong quarterly performance.

### Key Metrics:
- Revenue: Exceeded analyst expectations
- EPS: Beat consensus estimates
- Gross Margin: Maintained strong profitability

### Recommendation: **BEAT**

The company showed solid execution across all business segments, with particular strength in core operations. Forward guidance remains positive.

### Key Drivers:
1. Strong demand in primary markets
2. Operational efficiency improvements
3. Favorable market conditions

*This analysis is based on available public information.*
"""
        else:
            analysis = f"""
## {ticker} Financial Analysis

Responding to: {question}

### Summary:
The company demonstrates solid fundamentals with stable revenue growth and margin performance.

### Key Observations:
1. Revenue trends remain positive
2. Cost management is effective
3. Market position is strong

*This is a simplified analysis for testing purposes.*
"""

        # Add artifact with the analysis
        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=analysis)),
                Part(root=DataPart(data={
                    "ticker": ticker,
                    "recommendation": "Beat" if category == "beat_or_miss" else "Hold",
                    "confidence": 0.75,
                })),
            ],
            name="financial_analysis",
        )


class SimplePurpleExecutor(AgentExecutor):
    """Executor for the simple purple agent."""
    
    def __init__(self):
        self.agents: dict[str, SimpleFinanceAgent] = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        msg = context.message
        if not msg:
            raise ServerError(error=InvalidRequestError(message="Missing message"))

        task = context.current_task
        if task and task.status.state in TERMINAL_STATES:
            raise ServerError(error=InvalidRequestError(
                message=f"Task {task.id} already processed"
            ))

        if not task:
            task = new_task(msg)
            await event_queue.enqueue_event(task)

        context_id = task.context_id
        agent = self.agents.get(context_id)
        if not agent:
            agent = SimpleFinanceAgent()
            self.agents[context_id] = agent

        updater = TaskUpdater(event_queue, task.id, context_id)

        await updater.start_work()
        try:
            message_text = get_message_text(msg)
            await agent.run(message_text, updater)
            if not updater._terminal_state_reached:
                await updater.complete()
        except Exception as e:
            await updater.failed(
                new_agent_text_message(f"Error: {e}", context_id=context_id, task_id=task.id)
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())


def main():
    """Start the simple purple agent server."""
    parser = argparse.ArgumentParser(description="Simple Purple Agent for testing")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9010)
    parser.add_argument("--card-url", type=str, help="URL for agent card")
    args = parser.parse_args()

    skill = AgentSkill(
        id="finance-analysis",
        name="Financial Analysis",
        description="Provides basic financial analysis for testing purposes",
        tags=["finance", "analysis", "testing"],
        examples=["Analyze NVDA earnings"]
    )

    agent_card = AgentCard(
        name="Simple Finance Agent (Test)",
        description="A simple finance agent for testing the FAB++ benchmark",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill]
    )

    request_handler = DefaultRequestHandler(
        agent_executor=SimplePurpleExecutor(),
        task_store=InMemoryTaskStore(),
    )
    
    server_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    app = server_app.build()
    
    # Add /health endpoint for health checks
    async def health_endpoint(request):
        """Health check endpoint."""
        return JSONResponse({
            "status": "healthy",
            "service": "simple-purple-agent",
            "port": args.port
        })
    
    # Add /analyze endpoint for non-A2A compatibility
    async def analyze_endpoint(request):
        """Simple analyze endpoint for testing."""
        try:
            data = await request.json()
            question = data.get("question", "")
            
            # Return a simple analysis
            return JSONResponse({
                "status": "success",
                "analysis": f"Analysis for: {question}",
                "recommendation": "HOLD",
                "confidence": 0.5
            })
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)}, status_code=400)
    
    app.routes.append(Route("/health", endpoint=health_endpoint, methods=["GET"]))
    app.routes.append(Route("/analyze", endpoint=analyze_endpoint, methods=["POST"]))
    
    print(f"Starting Simple Purple Agent...")
    print(f"  URL: http://{args.host}:{args.port}/")
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
