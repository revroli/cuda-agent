from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    TextPart,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils.artifact import new_text_artifact
from a2a.utils.message import new_agent_text_message
from a2a.utils.task import new_task

import uvicorn
from pydantic_ai import Agent

give_advice=AgentSkill(
    id="give_advice",
    name ="Profile & Report",
    description=(
        "Profile CUDA code and report the most useful performance findings to the facilitator. "
        "Focus on measurements, hotspots, bottlenecks, and evidence from profiling output rather than on proposing code improvements."
    ),
    tags=["cuda", "profiling", "performance", "report"],
    examples=[
        "Profile kernel `compute()` and report runtime breakdowns, memory throughput, and occupancy trends.",
        "Identify potential bank conflicts, divergence, and cache inefficiencies in `kernel.cu` and summarize the evidence.",
        "Provide the facilitator with the profiling commands used, the key metrics observed, and any notable regressions or plateaus."],
)
class AdvisorExecutor(AgentExecutor):
    """CUDA Advisor Agent Executor Implementation."""

    def __init__(self, profiler_agent: Agent) -> None:
        self.agent = profiler_agent

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute the agent process and enqueue the final response."""
        task = context.current_task or new_task(context.message)
        await event_queue.enqueue_event(task)

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(
                    state=TaskState.working,
                    message=new_agent_text_message('Profiling and analyzing CUDA sources...'),
                ),
                final=False
            )
        )

        message = context.message
        extracted_text = " ".join([part.root.text for part in message.parts if type(part.root) is TextPart])
        response = await self.agent.run(extracted_text)

        await event_queue.enqueue_event(
            TaskArtifactUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                artifact=new_text_artifact(name='result', text=response.output),
            )
        )
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(state=TaskState.completed),
                final=True
            )
        )

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Raise exception as cancel is not supported."""
        raise Exception('cancel not supported')

def init_advisor_agent(agent: Agent, host: str, port: int) -> A2AStarletteApplication:
    advisor_avatar_url = f"http://{host}:{port}/"
    advisor_interface_url = f"http://{host}:{port}"

    advisor_agent_card = AgentCard(
        name="CUDA Profiler Agent",
        url=advisor_avatar_url,
        version="1.0.0",
        description=(
            "An agent that profiles CUDA code, extracts performance-critical information, and reports concise, evidence-based findings to the facilitator."
        ),
        skills=[give_advice],
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(
            streaming=True
        ),
        supported_interfaces=[
            AgentInterface(
                transport='JSONRPC',
                url=advisor_interface_url,
            )
        ],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=AdvisorExecutor(profiler_agent = agent),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=advisor_agent_card,
        http_handler=request_handler,
    )

    return server

def run_advisor_agent(host: str, port: int, server: A2AStarletteApplication):
    uvicorn.run(server.build(), host=host, port=port, log_level="info")