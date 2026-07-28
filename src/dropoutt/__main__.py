"""Entry point for `python -m dropoutt`.

The `dropoutt` console script lands in the environment's `bin/`, which is not
always on `PATH` — module-load systems and batch schedulers frequently leave it
off, and on a shared cluster the user may not be able to change that. Running
the package as a module works from any interpreter that can import it, so this
is the invocation that always works. See docs/portability.md.
"""

from __future__ import annotations

from .cli import run

if __name__ == "__main__":
    run()
