import asyncio
import os
import threading
import logfire
import paramiko
from sshtunnel import SSHTunnelForwarder
from mcp import ClientSession
from mcp.client.sse import sse_client
from pydantic_ai import Agent


from dotenv import load_dotenv

from coder_agent import init_coder_agent, run_coder_agent
from advisor_agent import init_advisor_agent, run_advisor_agent
from facilitator_agent import AgentRegistry, SeniorProgrammerAgent
import nest_asyncio
from IPython.display import display, Markdown

load_dotenv()

remote_host = os.getenv("REMOTE_HOST")
remote_user = os.getenv("REMOTE_USER")
remote_cmd = os.getenv("REMOTE_CMD")
ssh_port = os.getenv("SSH_PORT")
mcp_server_remote_port = os.getenv("MCP_SERVER_REMOTE_PORT")
mcp_server_local_port = os.getenv("MCP_SERVER_LOCAL_PORT")
model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

remote_key_path = os.getenv("SSH_KEY_PATH")
   
def init_ssh_client():
    if not remote_host or not remote_user or not ssh_port:
        raise ValueError("REMOTE_HOST, REMOTE_USER, and SSH_PORT must be set")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(remote_host, port=int(ssh_port), username=remote_user)
    except Exception as e:
        raise RuntimeError(
            f"Failed to open SSH connection to {remote_user}@{remote_host}:{ssh_port}"
        ) from e
        
    print(f"SSH connection established to {remote_user}@{remote_host}:{ssh_port}")
    return ssh    
    
def start_mcp_server_over_ssh(ssh: paramiko.SSHClient):
    if not remote_cmd:
        raise ValueError("REMOTE_CMD must be set")

    try:
        ssh.exec_command(remote_cmd, get_pty=True)
    except Exception as e:
        raise RuntimeError(f"Failed to start MCP server over SSH using command: {remote_cmd}") from e

    print(f"Started MCP server on remote host using command: {remote_cmd}")

async def tunnel_port_over_ssh():
    """Start an SSH tunnel as a subprocess.Popen so we can terminate it later.
    Returns the Popen object.
    """
    if not ssh_port or not mcp_server_remote_port or not mcp_server_local_port:
        raise ValueError("SSH_PORT, MCP_SERVER_REMOTE_PORT, and MCP_SERVER_LOCAL_PORT must be set to create a tunnel")

    with SSHTunnelForwarder(
        (remote_host, int(ssh_port)),
        ssh_username=remote_user,
        ssh_pkey=remote_key_path,
        remote_bind_address=('127.0.0.1', int(mcp_server_remote_port)),  # A szerver belső portja, ahol az MCP fut
        local_bind_address=('127.0.0.1', int(mcp_server_local_port))   # Amilyen porton az otthoni laptopodon akarod elérni
    ) as tunnel:
        
        print(f"SSH tunnel established: localhost:{mcp_server_local_port} -> {remote_host}:{mcp_server_remote_port}")
        # Csatlakozunk a távoli MCP-hez az alagúton át
        await asyncio.sleep(1)  # Várunk egy kicsit, hogy biztosan létrejöjjön az alagút
        print(f"Connecting to MCP server through SSH tunnel on localhost:{mcp_server_local_port}...")
        
        
        async with sse_client(f"http://127.0.0.1:{mcp_server_local_port}/sse") as (read_stream, write_stream):
            print("Connected to MCP server through SSH tunnel. Starting agent...")
            async with ClientSession(read_stream, write_stream) as mcp_session:
                await mcp_session.initialize()
                
                print("[MCP] Eszközök beolvasása a távoli Linuxról...")
                mcp_tools_response = await mcp_session.list_tools()

                # Létrehozzuk a Pydantic AI ágenst
                coder_agent = Agent(
                    model,
                    system_prompt=(
                        "You are a CUDA coding agent that implements high-performance, production-ready CUDA code. "
                        "Your job is to receive concise, prioritized recommendations from a CUDA Profiling Advisor, then produce concrete, minimal, and correct code changes to realize those optimizations. "
                        "You must not use profiling, benchmarking, or measurement tools. Those are reserved for the other agent. "
                        "If you need a decision, have a blocker, or want to report completion, use the send_message tool to notify the facilitator. "
                        "You may use any other tool available on the MCP server to inspect files, edit code, reason about the codebase, or validate non-profiling behavior. "
                        "When given an advisor's recommendation, respond with: 1) a short plan of code edits, 2) patch-style diffs or precise file/line edits the coding agent can apply, and 3) a short rationale and estimated performance impact. "
                        "Prefer safe, incremental changes that preserve correctness. When multiple options exist, prioritize changes by expected impact and implementation risk."
                    ),
                )
                                
                profiler_agent = Agent(
                    model,
                    system_prompt=(
                        "You are the profiler agent in a multi-agent CUDA workflow. "
                        "Your primary job is to analyze CUDA source files and profiling output, extract the most important performance information (hotspots, kernel runtimes, memory bandwidth, occupancy, divergent branches, shared memory usage, bank conflicts), and report those findings to the facilitator. "
                        "You may run shell commands on the remote server using the tool `execute_command(command: str) -> str` to run profilers (nvprof, nsight, nvbench, nvtx, nsys, perf), collect outputs, and reproduce measurements. "
                        "Do not focus on proposing code improvements or optimization advice; instead, provide clear evidence, metrics, comparisons, regressions, and plateaus so the facilitator can decide what the coding agent should do next. "
                        "If you need a decision, a narrower target, or want to report that the search has plateaued, use the send_message tool to notify the facilitator. "
                        "When you answer, include: 1) a short summary of findings, 2) the key profiling metrics and observations, 3) the commands and minimal steps used to reproduce the numbers, and 4) any notable stability, regression, or plateau information. "
                        "Keep answers concise, technical, and centered on profiling evidence."
                    ),
                )

                # 2. AZ MCP ESZKÖZÖK REGISZTRÁLÁSA (A Pydantic AI varázslata)
                for tool in mcp_tools_response.tools:
                    
                    # Készítünk egy dinamikus wrapper függvényt, amit az ágens meg tud hívni
                    # A lambda vagy belső függvény segít átadni a hívást az MCP session-nek
                    async def dynamic_tool_wrapper(tool_name=tool.name, **kwargs):
                        # Ez fut le, amikor az OpenAI úgy dönt, hogy megnyomja a gombot
                        print(f"[Pydantic AI -> MCP] Eszköz futtatása: {tool_name} -> {kwargs}")
                        result = await mcp_session.call_tool(tool_name, arguments=kwargs)
                        return result.content

                    # Beállítjuk a metaadatokat, hogy a GPT-4o tudja, mire jó a függvény
                    dynamic_tool_wrapper.__name__ = tool.name
                    dynamic_tool_wrapper.__doc__ = tool.description

                    # Regisztráljuk az eszközt az ágensbe
                    coder_agent.tool_plain(dynamic_tool_wrapper)
                    profiler_agent.tool_plain(dynamic_tool_wrapper)
                    

                print(f"[MCP] {len(mcp_tools_response.tools)} eszköz sikeresen csatolva az ágenshez.")
                nest_asyncio.apply()
                
                logfire.configure(send_to_logfire='if-token-present')
                logfire.instrument_pydantic_ai()

                coder_server = init_coder_agent(coder_agent)
                advisor_server = init_advisor_agent(profiler_agent)
                
                host_ip = "127.0.0.1"
                host_port_coder = 8000
                host_port_advisor = 8001


                coder_daemon_thread = threading.Thread(target=run_coder_agent, args = (host_ip, host_port_coder, coder_server), daemon=True)
                coder_daemon_thread.start()

                advisor_daemon_thread = threading.Thread(target=run_advisor_agent, args = (host_ip, host_port_advisor, advisor_server), daemon=True)
                advisor_daemon_thread.start()



                agent_urls = [f"http://{host_ip}:{host_port_coder}", f"http://{host_ip}:{host_port_advisor}"]
                registry = AgentRegistry(agent_urls)
                await registry.create_clients()
                
                senior_agent = SeniorProgrammerAgent(registry=registry, model=model)

                    
                """coder_client = A2AClient(
                    f"http://{host_ip}:{host_port}",
                    httpx.AsyncClient(timeout=httpx.Timeout(100.0, connect=20.0)),
                )"""
                
                result = await senior_agent.run("Create a code that adds two vectors together in CUDA.")
                



                """response = await client.send_message(message=Message(
                    message_id="1",
                    role="user",
                    kind="message",
                    parts=[TextPart(kind="text", text=f"Write a CUDA kernel that adds two vectors together and returns the result and create a .cu file from it.")]
                ))"""
                
                #final_response = response["result"]
                #print(f"\n[Final Result]: {response}")
                
                display(Markdown(result))
                


if __name__ == "__main__":
    ssh = init_ssh_client()
    start_mcp_server_over_ssh(ssh)
    asyncio.run(tunnel_port_over_ssh())
