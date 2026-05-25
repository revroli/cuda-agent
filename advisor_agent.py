from uuid import uuid4

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Message,
    Message,
    MessageSendParams,
    Part,
    Part,
    Role,
    Role,
    SendMessageRequest,
    SendMessageRequest,
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

from a2a.client import A2AClient, A2ACardResolver
from typing import List, Dict
import httpx

from dotenv import load_dotenv
import uvicorn
from pydantic_ai import Agent
import os



load_dotenv()

MODEL = os.getenv("OPENAI_MODEL")

# Agent Card configuration from environment (with sensible defaults)
ADVISOR_HOST = os.getenv("ADVISOR_HOST")
ADVISOR_PORT = int(os.getenv("ADVISOR_PORT", "10020"))
ADVISOR_AVATAR_URL = f"http://{ADVISOR_HOST}:{ADVISOR_PORT}/"
ADVISOR_INTERFACE_URL = f"http://{ADVISOR_HOST}:{ADVISOR_PORT}"

class AgentRegistry:
    agent_client_map: Dict[str, A2AClient] = {}
    agent_card_map: Dict[str, AgentCard] = {}

    def __init__(self, agent_urls: List[str]):
        self.agent_urls = agent_urls

    async def _create_client(self, url: str):
        async with httpx.AsyncClient() as httpx_client:
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url=url,
            )

            public_card = (
                await resolver.get_agent_card()
            )


        self.agent_card_map[public_card.name] = public_card

        async_httpx_client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
        client = A2AClient(
            httpx_client=async_httpx_client,
            agent_card=public_card,
        )
        self.agent_client_map[public_card.name] = client

    async def create_clients(self):
        for url in self.agent_urls:
            await self._create_client(url)

give_advice=AgentSkill(
    id="give_advice",
    name ="Profile & Recommend",
    description=(
        "Analyze CUDA sources and profiling outputs, identify performance hotspots and root causes, "
        "and provide prioritized, actionable code changes the coding agent can implement."
    ),
    tags=["cuda", "profiling", "performance", "advice"],
    examples=[
        "Profile kernel `compute()` and recommend memory and launch-configuration changes to reduce runtime.",
        "Identify potential bank conflicts and divergent branches in `kernel.cu` and propose code-level fixes.",
        "Suggest concrete changes (e.g., loop unrolling, shared memory use, threadblock sizing) to improve occupancy and throughput."],
)

"""advisor_agent_card = AgentCard(
    name="CUDA Profiling Advisor",
    url=ADVISOR_AVATAR_URL,
    version="1.0.0",
    description=(
        "An agent that profiles CUDA code, extracts performance-critical information, and provides concise, prioritized recommendations "
        "for a coding agent to apply changes to the source code."
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
            url=ADVISOR_INTERFACE_URL,
        )
    ],
)"""


class AdvisorAgent():
    """CUDA Advisor Agent Executor Implementation."""

    def __init__(self, advisor_agent: Agent, registry: AgentRegistry) -> None:
        self.registry = registry
        self.agent = advisor_agent
        self.agent.tool_plain(self.send_message)
        
        

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
    
    async def send_message(self, agent_name: str, message: str):
      """Send a message to another agent."""
      agent_client = self.registry.agent_client_map[agent_name]
      if not agent_client:
        return "The given agent does not exist. Choose an existing one."


      parts = [Part(text=message)]
      message = Message(
          role=Role.user,
          parts=parts,
          message_id=uuid4().hex,
      )

      try:
        response = await agent_client.send_message(SendMessageRequest(id=uuid4().hex, params=MessageSendParams(message=message)))
        return response.root.result.artifacts

      except Exception:
        return "The given agent is not available currently."

      return ""

    async def run(self, message: str):
        response = await self.agent.run(message)
        return response.output
    
    
    


"""request_handler = DefaultRequestHandler(
    agent_executor=AdvisorExecutor(advisor_agent_card, registry=None),
    task_store=InMemoryTaskStore(),
)

server = A2AStarletteApplication(
    agent_card=advisor_agent_card,
    http_handler=request_handler,
)

def run_advisor_agent(host: str, port: int):
    uvicorn.run(server.build(), host=host, port=port, log_level="info")"""