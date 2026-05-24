import asyncio
import os
import paramiko
from sshtunnel import SSHTunnelForwarder
from mcp import ClientSession
from mcp.client.sse import sse_client
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel


from dotenv import load_dotenv

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
        
        """print(f"SSH tunnel established: localhost:{mcp_server_local_port} -> {remote_host}:{mcp_server_remote_port}")
        # 1. Létrehozunk egy aszinkron HTTPX klienst, ami bírja a folyamatos adatfolyamot (Stream)
        async with httpx.AsyncClient() as http_client:
        
            print(f"SSH tunnel established: localhost:{mcp_server_local_port} -> {remote_host}:{mcp_server_remote_port}")
        # Bekopogunk az /mcp végpontra az elvárt session ID-ért
        # A timeout=None azért kell, mert az MCP egy nyitott, hosszú kapcsolat
            async with http_client.stream("GET", f"http://127.0.0.1:{mcp_server_local_port}/mcp", timeout=None) as response:
                
                if response.status_code != 200:
                    print(f"[HIBA] A szerver {response.status_code} kóddal elutasította a kapcsolatot!")
                    return
                    
                print("[SIKER] A szerver fogadta a kapcsolatot, inicializálás...")

                # Készítünk két aszinkron sort (Queue), amik a Linux írás/olvasás stream-eket szimulálják
                read_stream = asyncio.Queue()
                write_stream = asyncio.Queue()
                """
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
                agent = Agent(
                    model=OpenAIModel(model),
                    system_prompt="Te egy segítőkész asszisztens vagy, aki eléri a távoli GPU szervert az eszközein keresztül."
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
                    agent.tool_plain(dynamic_tool_wrapper)

                print(f"[MCP] {len(mcp_tools_response.tools)} eszköz sikeresen csatolva az ágenshez.")

                # 3. Futtatás - felhasználói prompttal (párbeszéd)
                chat_history = []
                max_turns = 10
                while True:
                    user_prompt = input("\n[User]: ").strip()
                    if not user_prompt:
                        continue
                    if user_prompt.lower() in {"exit", "quit"}:
                        break

                    print("[Agent]: Gondolkodom es futtatom a szukseges lepeseket...")

                    chat_history.append(("User", user_prompt))
                    chat_history = chat_history[-max_turns * 2 :]
                    history_text = "\n".join(f"{role}: {text}" for role, text in chat_history)
                    combined_prompt = f"Conversation so far:\n{history_text}\nAssistant:"

                    # A Pydantic AI teljesen atveszi az iranyitast:
                    # Ha kell, meghivja a tavoli fuggvenyt, megvarja az eredmenyt, es magatol ujraprobalja
                    result = await agent.run(combined_prompt)
                    ai_text = result.output
                    chat_history.append(("Assistant", ai_text))

                    print(f"\n[AI Vegso Valasz]: {ai_text}")


if __name__ == "__main__":
    ssh = init_ssh_client()
    start_mcp_server_over_ssh(ssh)
    asyncio.run(tunnel_port_over_ssh())
