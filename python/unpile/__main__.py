# SPDX-License-Identifier: MIT OR Apache-2.0
"""`python -m unpile ...` runs the binary with the same arguments."""

from __future__ import annotations

import os
import subprocess
import sys

from ._find import find_unpile_bin


def _run() -> None:
    exe = find_unpile_bin()
    if sys.platform == "win32":
        try:
            sys.exit(subprocess.run([exe, *sys.argv[1:]], check=False).returncode)
        except KeyboardInterrupt:
            sys.exit(2)
    os.execvp(exe, [exe, *sys.argv[1:]])


if __name__ == "__main__":
    _run()
