"""Make every job dump its own stacks on request, and on a crash.

Python imports `sitecustomize` automatically at interpreter startup if it is
anywhere on sys.path, so putting this directory on PYTHONPATH installs the hook in
every job process without touching any repo's code.

Why it is needed: when a job stopped making progress there was no way to see where
it was. `py-spy` and `gdb` both need to attach to a non-child process, which
`kernel.yama.ptrace_scope = 1` on this host forbids, and the jobs are not our
children from the shell we debug from. `PYTHONFAULTHANDLER=1` alone does not help
either -- it covers fatal signals like SIGSEGV, not a live hang.

So the job registers SIGUSR1 for itself. Sending it costs the job nothing until it
arrives, and then every thread's stack lands in the job's own directory, next to
its log. That is how a hang inside generated code, inside an HTTP wait, or inside a
pool shutdown becomes distinguishable from the outside.
"""

import os

try:
    import faulthandler
    import signal

    # Fatal-signal traceback: a segfault in a native extension (torch, pyg) would
    # otherwise leave nothing but a non-zero exit code.
    _path = os.environ.get("SHIM_STACKDUMP")
    if _path:
        # Line-buffered append, kept open for the process's lifetime: faulthandler
        # writes from a signal handler, where opening a file is not safe.
        _handle = open(_path, "a", buffering=1)
        faulthandler.enable(file=_handle, all_threads=True)
        if hasattr(signal, "SIGUSR1"):
            faulthandler.register(signal.SIGUSR1, file=_handle,
                                  all_threads=True, chain=False)
    else:
        faulthandler.enable(all_threads=True)
        if hasattr(signal, "SIGUSR1"):
            faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
except Exception:  # noqa: BLE001
    # A debugging aid must never be the reason a job fails to start.
    pass
