from uuid import uuid4

from a2a.types import (
    AgentCard,
    Message,
    Message,
    MessageSendParams,
    Part,
    Part,
    Role,
    Role,
    SendMessageRequest,
    SendMessageRequest,
)
from a2a.client import A2AClient, A2ACardResolver
from typing import List, Dict
import httpx

from pydantic_ai import Agent

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
            
class SeniorProgrammerAgent:
    def __init__(self, registry: AgentRegistry, model: str):
        self.registry = registry
        self.agent = self._create_agent(model)

    def _agent_description(self, agent_name: str):
        agent_card = self.registry.agent_card_map[agent_name]
        return (
            f"Agent name: {agent_card.name}\n"
            f"Agent description: {agent_card.description}\n"
            f"Agent skills: {', '.join([f'{skill.name} (examples: {', '.join(skill.examples)})' for skill in agent_card.skills])}\n"
        )

    def system_instruction(self):
        return (
            "You are the CUDA facilitator. You do not write CUDA code yourself unless explicitly asked for a final summary; you coordinate the specialist agents. "
            "Your job is to turn the user's request into precise, executable instructions for the coding agent, then use the profiler's measurements to decide the next step. "
            "Always prefer delegation over explanation: if the coding agent should act, send it a message instead of narrating the change in your own answer. "
            "If the profiler has findings, use them to refine the next coding instructions. If a specialist needs clarification or a decision, message them directly. "
            "Continue iterating until several consecutive rounds show a plateau or no meaningful improvement, then report the best result and why further changes were not justified. "
            "Never stop after a single pass if there is still a plausible improvement path. "
            "\n===Agents===\n"
            f"{'\n'.join([self._agent_description(agent) for agent in self.registry.agent_card_map.keys()])}"
        )

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

    def _create_agent(self, model: str):
        return Agent(
            model,
            system_prompt=self.system_instruction(),
            tools=[self.send_message]
        )

    async def run(self, message: str):
        response = await self.agent.run(message)
        return response.output