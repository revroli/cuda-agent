# MI-agents

## Setup

1) Create and activate a Python virtual environment.
2) Install dependencies:

```
pip install -r requirements.txt
```

## Folder structure

- Agents/ : Local agent code and supporting modules.
- remote_files/ : Files mirrored from or used by the remote environment.
- requirements.txt : Python dependencies.
- run_mcp_ssh.py : Entry point that starts the SSH tunnel and runs the agent workflow.
- .env : Local environment configuration (not committed).

## .env configuration

Create a .env file in the project root with the following variables:

```
REMOTE_HOST=
REMOTE_USER=
REMOTE_CMD=
SSH_PORT=
MCP_SERVER_REMOTE_PORT=
MCP_SERVER_LOCAL_PORT=
SSH_KEY_PATH=
OPENAI_MODEL=
```

### Notes

- REMOTE_HOST: Remote SSH host (IP or hostname).
- REMOTE_USER: SSH username.
- REMOTE_CMD: Command to start the MCP server on the remote host.
- SSH_PORT: SSH port on the remote host (usually 22).
- MCP_SERVER_REMOTE_PORT: Port where the MCP server listens on the remote host.
- MCP_SERVER_LOCAL_PORT: Local port to bind the tunnel to.
- SSH_KEY_PATH: Path to the SSH private key used to connect.
- OPENAI_MODEL: Model name to use (default is gpt-4o-mini if not set).
