"""PORT-aware process launcher: python -m deployment.start api|frontend [--check]."""
import argparse
import ipaddress
import os
import shlex
import sys
from urllib.parse import urlsplit


def api_url(environment):
    url = environment.get("MOVIEMATCH_API_URL", "").strip()
    if not url:
        hostport = environment.get("MOVIEMATCH_API_HOSTPORT", "").strip()
        if hostport:
            url = f"http://{hostport}"
    if not url:
        raise ValueError("Set MOVIEMATCH_API_URL or MOVIEMATCH_API_HOSTPORT for deployed Streamlit; localhost fallback is disabled in this launcher")
    parsed = urlsplit(url)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("Invalid MOVIEMATCH_API_URL: expected an HTTP(S) API base URL")
    host = parsed.hostname.lower().rstrip(".")
    if host == "localhost" or host.endswith(".localhost"):
        raise ValueError("Deployed Streamlit must not use localhost for its API")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address and (address.is_loopback or address.is_unspecified):
        raise ValueError("Deployed Streamlit must not use a loopback API address")
    if parsed.port is not None and not 1 <= parsed.port <= 65535:
        raise ValueError("Invalid API port")
    return url.rstrip("/")


def launch_command(role, environment):
    port = environment.get("PORT", "10000")
    if not port.isascii() or not port.isdigit() or not 1 <= int(port) <= 65535:
        raise ValueError("PORT must be an integer between 1 and 65535")
    if role == "api":
        return [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0",
                "--port", port, "--workers", "1"]
    if role == "frontend":
        environment["MOVIEMATCH_API_URL"] = api_url(environment)
        return [sys.executable, "-m", "streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0",
                f"--server.port={port}", "--server.headless=true", "--browser.gatherUsageStats=false"]
    raise ValueError("Unknown service role")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=["api", "frontend"])
    parser.add_argument("--check", action="store_true", help="Validate and print command without starting a server")
    args = parser.parse_args()
    environment = dict(os.environ)
    try:
        command = launch_command(args.role, environment)
    except ValueError as exc:
        parser.exit(1, f"Deployment configuration error: {exc}\n")
    if args.check:
        print(shlex.join(command))
    else:
        os.execvpe(command[0], command, environment)


if __name__ == "__main__":
    main()
