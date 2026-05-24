import asyncio
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List
import signal
import atexit
import shlex
import paramiko
from sshtunnel import SSHTunnelForwarder

from dotenv import load_dotenv

load_dotenv()

remote_host = os.getenv("REMOTE_HOST")
remote_user = os.getenv("REMOTE_USER")
remote_cmd = os.getenv("REMOTE_CMD")
ssh_port = os.getenv("SSH_PORT")
mcp_server_remote_port = os.getenv("MCP_SERVER_REMOTE_PORT")
mcp_server_local_port = os.getenv("MCP_SERVER_LOCAL_PORT")

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

def tunnel_port_over_ssh():
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

        print(f"Connection successful! Tunnel is live at http://localhost:{tunnel.local_bind_port}.")
        print("You can now start the AI agent and point it at the local port.")  
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nAlagút lezárása...")  


if __name__ == "__main__":
    ssh = init_ssh_client()
    start_mcp_server_over_ssh(ssh)
    tunnel_port_over_ssh()

    #asyncio.run(main())