"""Bind the actual listener before publishing its address; never steal a port."""

import errno
import json
from http.server import ThreadingHTTPServer
import os
import socket
from socketserver import TCPServer
from urllib.parse import urlsplit
from .platforms import process_matches


def proxy_record(archive):
    try:
        record = json.loads((archive / "proxy.json").read_text())
        return record if process_matches(record) else None
    except (OSError, ValueError, KeyError):
        return None


def managed_port(archive):
    try:
        record = json.loads((archive / "runtime.json").read_text())
        if process_matches(record):
            argv = record["argv"]
            return int(argv[argv.index("--port") + 1])
    except (OSError, ValueError, KeyError, IndexError):
        pass
    return None


class LocalServer(ThreadingHTTPServer):
    allow_reuse_address = os.name != "nt"
    daemon_threads = True

    def server_bind(self):
        if os.name == "nt":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        # HTTPServer's default bind performs a reverse-DNS lookup, which can
        # stall localhost startup on macOS with an unavailable resolver.
        TCPServer.server_bind(self)
        self.server_name = "localhost"
        self.server_port = self.server_address[1]


def bind_server(handler, preferred, upstream_url):
    target = urlsplit(upstream_url)
    if target.hostname in ("127.0.0.1", "localhost", "::1") and target.port == preferred:
        preferred = 0
    try:
        return LocalServer(("127.0.0.1", preferred), handler)
    except OSError as exc:
        if exc.errno not in (errno.EADDRINUSE, errno.EACCES):
            raise
        return LocalServer(("127.0.0.1", 0), handler)
