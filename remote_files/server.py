from fastmcp import FastMCP
import os
import re
import subprocess
import threading
import time
import socket
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv
from starlette.middleware import Middleware

load_dotenv()

HTTP_HOST = os.environ.get("CUDA_MCP_HTTP_HOST", "127.0.0.1")
HTTP_PORT = int(os.environ.get("CUDA_MCP_HTTP_PORT", "4111"))
TRANSPORT = os.environ.get("CUDA_MCP_TRANSPORT", "sse")

cuda_mcp_server = FastMCP("CUDA_writer")

workspace_root = Path(os.environ.get("CUDA_MCP_WORKSPACE_ROOT", ".")).resolve()
makefile_path = workspace_root / "makefile"

print("workspace_root:", workspace_root)
print("cwd:", os.getcwd())
cmd = ["make", "-B", f"SRC=vector_add.cu"]

print("cmd:", cmd)


@cuda_mcp_server.tool()
def create_program(program_name: str, code: str) -> str:
    """Create a CUDA source file and compile via Make.

    - `program_name`: Base name for the program. If it ends with `.cu`, the suffix is removed.
        Only letters, digits, underscore, dash, and dot are allowed.
    - `code`: Full CUDA source content written to `<program_name>.cu` under `workspace_root`.

    Behavior:
    - Writes `<program_name>.cu` to disk.
    - Invokes: `make SRC=<program_name>.cu`.
    - Raises `RuntimeError` with captured stdout/stderr if `make` fails.
    """
    
    if not program_name or not program_name.strip():
        raise ValueError("program_name is required.")
    program_name = program_name.strip()
    if program_name.endswith(".cu"):
        program_name = program_name[:-3]
    if not re.match(r"^[A-Za-z0-9_.-]+$", program_name):
        raise ValueError("program_name contains invalid characters.")

    src_path = workspace_root / f"{program_name}.cu"
    exe_path = workspace_root / f"{program_name}.exe"

    src_path.write_text(code, encoding="utf-8")

    cmd = ["make", "-B", f"SRC={program_name}.cu"]
    result = subprocess.run(
        cmd,
        cwd=workspace_root,
        capture_output=True,
        text=True,
        check=False,
    )            
    
    return {
        "write_code_successful": True
        "make_exit_code": result.returncode,
        "make_stdout": result.stdout,
        "make_stderr": result.stderr
    }

@cuda_mcp_server.tool()
def read_makefile() -> str:
    """Read and return the Makefile under `workspace_root` as a string."""
    return makefile_path.read_text(encoding="utf-8")


@cuda_mcp_server.tool()
def list_build_artifacts() -> List[str]:
    """List CUDA sources, binaries, and Makefile under `workspace_root`.

    Returns relative paths for files ending in `.cu`, `.exe`, or named `makefile`/`Makefile`.
    """
    matches: List[str] = []
    for path in workspace_root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if name.lower() == "makefile" or name.endswith(".cu") or name.endswith(".exe"):
            matches.append(str(path.relative_to(workspace_root)))
    return sorted(matches)


@cuda_mcp_server.tool()
def read_cuda_source(file_name: str) -> str:
    """Read and return the contents of a .cu file under `workspace_root`.

    - `file_name`: Relative path or base name of a `.cu` file.
    """
    if not file_name or not file_name.strip():
        raise ValueError("file_name is required.")
    file_name = file_name.strip()

    if not file_name.endswith(".cu"):
        file_name = f"{file_name}.cu"

    candidate = (workspace_root / file_name).resolve()
    if workspace_root not in candidate.parents and candidate != workspace_root:
        raise ValueError("file_name must be within the workspace.")
    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError(f"CUDA source not found: {candidate}")

    return candidate.read_text(encoding="utf-8")

@cuda_mcp_server.tool()
def profile_binary(binary_name: str) -> str:
    """Profile a binary with Nsight Compute (ncu).

        - `binary_name`: The binary to run. May be absolute, relative, or a name in `workspace_root`.
            If it does not end with `.exe`, the suffix is added.

    Behavior:
        - Runs: `ncu --log-file <log> <binary_name>` in `workspace_root`.
        - Returns the profiling log contents on success (no program stdout/stderr).
        - Raises `RuntimeError` with captured stderr if `ncu` fails.
    """
    # Expect at least the binary name/path as the first argument. Any following
    # items are passed to the profiled binary as its command-line args.

    target = binary_name.strip()
    if not target:
        raise ValueError("binary_name is required.")

    if not target.endswith(".exe"):
        target = f"{target}.exe"
    
    # Build ncu command. Use `--` to separate ncu options from the profiled binary.
    log_path = workspace_root / f"{Path(target).stem}.ncu.log"
    ncu_cmd = ["ncu", "--log-file", str(log_path), str(target)]

    result = subprocess.run(
        ncu_cmd,
        cwd=workspace_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"ncu failed (exit {result.returncode})\n"
            f"stderr:\n{result.stderr}"
        )
    # Return profiling output only.
    if log_path.exists():
        return log_path.read_text(encoding="utf-8")
    return ""

if __name__ == "__main__":
    cuda_mcp_server.run(transport=TRANSPORT, host=HTTP_HOST, port=HTTP_PORT)
