"""
Small helpers shared by the MCP notebooks.

- setup_notebook(): fixes Windows + Jupyter issues when launching stdio MCP servers
- start_server() / stop_server(): run an HTTP MCP server in the background from a notebook
- parse_sse(): extract JSON-RPC messages from a "text/event-stream" response body
"""
import asyncio
import atexit
import json
import os
import socket
import subprocess
import sys
import time
import warnings

_RUNNING = []


def setup_notebook():
    """Call once at the top of every notebook."""
    if sys.platform == "win32":
        # 1. Subprocesses (stdio servers) need the Proactor event loop on Windows
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            if not isinstance(asyncio.get_event_loop_policy(), asyncio.WindowsProactorEventLoopPolicy):
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    # 2. Jupyter replaces sys.stderr with an object that (on Windows) has no real file
    #    descriptor, and stdio_client() hands sys.stderr to the server process -> "fileno" error.
    #    We change only stdio_client's default `errlog` to the real stderr. We do NOT replace
    #    sys.stderr globally: that would hide error tracebacks (e.g. ExceptionGroup) from the notebook.
    if "ipykernel" in sys.modules and sys.__stderr__ is not None:
        import mcp.client.stdio as _stdio

        func = getattr(_stdio.stdio_client, "__wrapped__", None)
        if func is not None and func.__defaults__:
            func.__defaults__ = (sys.__stderr__,)
    print("Notebook ready | Python", sys.version.split()[0], "|", sys.platform)


def _port_in_use(port, host="127.0.0.1"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def start_server(script, *args, port=8000, timeout=30):
    """Start `python script *args` in the background and wait until `port` accepts connections."""
    if _port_in_use(port):
        raise RuntimeError(
            f"Port {port} is already in use (maybe a server from a previous run).\n"
            f"  Windows: netstat -ano | findstr :{port}   then   taskkill /PID <PID> /F\n"
            f"  Linux:   lsof -i :{port}                  then   kill <PID>\n"
            "Or simply restart the Jupyter kernel."
        )
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", os.path.basename(script).replace(".py", "") + f"_{port}.log")
    log = open(log_path, "w", encoding="utf-8")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen([sys.executable, script, *args], stdout=log, stderr=subprocess.STDOUT, env=env)
    proc.log_path = log_path
    _RUNNING.append(proc)
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            log.close()
            raise RuntimeError(f"Server exited early. Log ({log_path}):\n" + open(log_path, encoding="utf-8").read())
        if _port_in_use(port):
            print(f"Server started: {script} {' '.join(args)}  (port {port}, pid {proc.pid}, log: {log_path})")
            return proc
        time.sleep(0.3)
    stop_server(proc)
    raise TimeoutError(f"Server did not open port {port} within {timeout}s. See {log_path}")


def stop_server(proc):
    """Stop a server started with start_server()."""
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    print(f"Server stopped (pid {proc.pid})")


@atexit.register
def _cleanup():
    for p in _RUNNING:
        try:
            stop_server(p)
        except Exception:
            pass


def parse_sse(text):
    """Return the JSON objects found in the `data:` lines of an SSE body."""
    messages = []
    for line in text.splitlines():
        if line.startswith("data:"):
            messages.append(json.loads(line[len("data:"):].strip()))
    return messages


def show_json(obj):
    """Pretty-print a dict / JSON-serialisable object (keeps non-English text readable)."""
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump(exclude_none=True, by_alias=True)
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
