import asyncio
import json
import os
import subprocess
import sys
from typing import Any, Dict, List
import signal
import atexit
import shlex

from dotenv import load_dotenv

load_dotenv()

remote_host = os.getenv("REMOTE_HOST")
remote_user = os.getenv("REMOTE_USER")
remote_cmd = os.getenv("REMOTE_CMD")
remote_port = os.getenv("REMOTE_PORT")
local_port = os.getenv("LOCAL_PORT")
# runtime handles
_remote_pid: str | None = None
_tunnel_proc: subprocess.Popen | None = None
    
def start_mcp_server_over_ssh():
    """Start the remote MCP server in background and record its PID."""
    global _remote_pid

    if not (remote_host and remote_user and remote_cmd):
        raise RuntimeError("REMOTE_HOST, REMOTE_USER and REMOTE_CMD must be set to start remote server")

    # Use nohup + & and echo $! to print the PID of the backgrounded process.
    # Quote the remote command so it's safe to pass through ssh.
    quoted_cmd = shlex.quote(remote_cmd)
    remote_launch = f"nohup {quoted_cmd} > /tmp/mcp_server.log 2>&1 & echo $!"

    ssh_cmd = [
        "ssh",
        "-p",
        f"{remote_port}",
        f"{remote_user}@{remote_host}",
        remote_launch,
    ]

    try:
        out = subprocess.check_output(ssh_cmd, text=True, stderr=subprocess.STDOUT)
        pid = out.strip().splitlines()[-1]
        _remote_pid = pid
        print(f"Remote server started with PID {_remote_pid}")
    except subprocess.CalledProcessError as exc:
        print("Failed to start remote MCP server:", exc.output, file=sys.stderr)
        raise

def tunnel_port_over_ssh():
    """Start an SSH tunnel as a subprocess.Popen so we can terminate it later.

    Returns the Popen object.
    """
    global _tunnel_proc

    if not (remote_host and remote_user and remote_port and local_port):
        raise RuntimeError("REMOTE_HOST, REMOTE_USER, REMOTE_PORT and LOCAL_PORT must be set to create tunnel")

    ssh_cmd = [
        "ssh",
        "-N",
        "-L",
        f"{local_port}:localhost:{remote_port}",
        f"{remote_user}@{remote_host}",
    ]

    # Start the ssh tunnel and keep the process handle so we can terminate it on exit.
    _tunnel_proc = subprocess.Popen(ssh_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Tunnel started, local:{local_port} -> remote:{remote_port} (ssh pid={_tunnel_proc.pid})")
    return _tunnel_proc


def stop_mcp_server_over_ssh():
    """Stop the remote MCP server using the PID we recorded (best-effort)."""
    global _remote_pid
    if not _remote_pid:
        return

    ssh_cmd = [
        "ssh",
        "-p",
        f"{remote_port}",
        f"{remote_user}@{remote_host}",
        f"kill -TERM {_remote_pid} || kill -9 {_remote_pid} || true; rm -f /tmp/mcp_server.pid",
    ]
    try:
        subprocess.run(ssh_cmd, check=False)
        print(f"Requested stop for remote PID {_remote_pid}")
    except Exception as exc:
        print("Error while stopping remote server:", exc, file=sys.stderr)
    finally:
        _remote_pid = None


def stop_tunnel():
    """Terminate the local ssh tunnel process if we started one."""
    global _tunnel_proc
    if _tunnel_proc:
        try:
            _tunnel_proc.terminate()
            _tunnel_proc.wait(timeout=5)
            print(f"Tunnel process {_tunnel_proc.pid} terminated")
        except Exception:
            try:
                _tunnel_proc.kill()
            except Exception:
                pass
        finally:
            _tunnel_proc = None
    def _cleanup_and_exit(code: int = 0):
        stop_tunnel()
        stop_mcp_server_over_ssh()
        sys.exit(code)


    def _signal_handler(signum, frame):
        print(f"Received signal {signum}; cleaning up...", file=sys.stderr)
        _cleanup_and_exit(0)


    # Register handlers
    try:
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)
    except Exception:
        # On Windows some signals/handlers may not be available; KeyboardInterrupt will still work.
        pass

    atexit.register(_cleanup_and_exit)


# Note: OpenAI/MCP agent code removed for simpler testing. The script
# now only starts the remote server and tunnel, then waits until terminated.


async def main() -> None:
    print("Running. Press Ctrl+C to stop and clean up the remote server and tunnel.")
    # Wait forever until signal/KeyboardInterrupt triggers cleanup
    await asyncio.Event().wait()


if __name__ == "__main__":
    start_mcp_server_over_ssh()
    tunnel_port_over_ssh()
    print(
        f"Started MCP server on {remote_host}:{remote_port} and tunneled to local port "
        f"{local_port}"
    )
    asyncio.run(main())