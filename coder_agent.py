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

from dotenv import load_dotenv
import uvicorn
from pydantic_ai import Agent
import os


load_dotenv()

MODEL = os.getenv("OPENAI_MODEL")

# Agent Card configuration from environment (with sensible defaults)
CODER_HOST = os.getenv("CODER_HOST")
CODER_PORT = int(os.getenv("CODER_PORT", "10021"))
CODER_AVATAR_URL = f"http://{CODER_HOST}:{CODER_PORT}/"
CODER_INTERFACE_URL = f"http://{CODER_HOST}:{CODER_PORT}"


write_code = AgentSkill(
    id="write_code",
    name ="Write & Optimize",
    description=(
        "Turn advisor recommendations into concrete code edits, patches, and test/benchmark steps."
    ),
    tags=["cuda", "implementation", "optimization", "patch"],
    examples=[
        "Apply a patch to reduce global memory accesses in `kernel.cu` and adjust threadblock sizes.",
        "Replace coalesced loads with vectorized loads and add `__restrict__` annotations to pointers.",
        "Introduce `__launch_bounds__` and adjust unrolling pragmas to improve occupancy."],
)

send_message = AgentSkill(
    id="send_message",
    name="Send Status Update",
    description=(
        "Send a short progress update, blocker, or clarification request back to the facilitator."
    ),
    tags=["communication", "facilitator", "status"],
    examples=[
        "Send the facilitator a concise note that the kernel patch is ready for review.",
        "Ask the facilitator for a decision when there are two viable optimization paths.",
        "Report that the requested code change was applied and needs profiling feedback.",
    ],
)

coder_agent_card = AgentCard(
    name="CUDA Coding Agent",
    url=CODER_AVATAR_URL,
    version="1.0.0",
    description=(
        "An agent that receives profiling-driven recommendations and implements optimized CUDA code changes as patches, with tests and benchmarks."
    ),
    skills=[write_code, send_message],
    default_input_modes=['text'],
    default_output_modes=['text'],
    capabilities=AgentCapabilities(
        streaming=True
    ),
    supported_interfaces=[
        AgentInterface(
            transport='JSONRPC',
            url=CODER_INTERFACE_URL,
        )
    ],
)


class CoderExecutor(AgentExecutor):
    """CUDA Coding Agent Executor Implementation."""

    def __init__(self, coder_agent: Agent) -> None:
        self.agent = coder_agent

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
                    message=new_agent_text_message('Implementing suggested CUDA optimizations and producing patches...'),
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

def init_coder_agent(agent: Agent) -> A2AStarletteApplication:
    request_handler = DefaultRequestHandler(
        agent_executor=CoderExecutor(coder_agent=agent),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=coder_agent_card,
        http_handler=request_handler,
    )

    return server

def run_coder_agent(host: str, port: int, server: A2AStarletteApplication):
    uvicorn.run(server.build(), host=host, port=port, log_level="info")