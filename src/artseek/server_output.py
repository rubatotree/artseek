from __future__ import annotations

import os
import stat
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path


@contextmanager
def server_output(data_dir: Path) -> Iterator[None]:
    """Give background servers a log sink that outlives their launcher's pipes.

    Keep terminals and explicit file redirections intact. Inherited pipes can
    disappear when a desktop launcher exits; model progress writes must not
    depend on that launcher's lifetime.
    """
    pipe_fds = set()
    redirects = []
    for stream, redirect in ((sys.stdout, redirect_stdout), (sys.stderr, redirect_stderr)):
        if stream is None:
            redirects.append(redirect)
            continue
        try:
            is_pipe = stat.S_ISFIFO(os.fstat(stream.fileno()).st_mode)
        except (AttributeError, OSError, ValueError):
            # In-memory streams used by embedding applications have no OS handle.
            continue
        if is_pipe:
            redirects.append(redirect)
            pipe_fds.add(stream.fileno())

    with ExitStack() as stack:
        if redirects:
            log_dir = data_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log = stack.enter_context(
                (log_dir / "server.log").open("a", encoding="utf-8", buffering=1)
            )
            # Libraries imported before serve() may already hold the original
            # stream object. Repair its descriptor too, not just sys.stderr.
            for fd in pipe_fds:
                previous = os.dup(fd)
                stack.callback(os.close, previous)
                stack.callback(os.dup2, previous, fd)
                os.dup2(log.fileno(), fd)
            for redirect in redirects:
                stack.enter_context(redirect(log))
        yield
