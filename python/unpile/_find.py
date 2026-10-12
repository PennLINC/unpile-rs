# SPDX-License-Identifier: MIT OR Apache-2.0
"""Locate the `unpile` binary that the wheel installed as a console script."""

from __future__ import annotations

import os
import shutil
import sys
import sysconfig


class UnpileNotFound(FileNotFoundError):
    pass


def find_unpile_bin() -> str:
    """Return the path of the unpile binary.

    `UNPILE_BIN` overrides the search; otherwise the interpreter's script directories are tried,
    then the user scheme, then `PATH`.
    """
    env = os.environ.get("UNPILE_BIN")
    if env:
        return env
    exe = "unpile" + (sysconfig.get_config_var("EXE") or "")
    targets = [
        sysconfig.get_path("scripts"),
        sysconfig.get_path("scripts", vars={"base": sys.base_prefix}),
        _user_scripts(),
    ]
    seen: list[str] = []
    for target in targets:
        if not target or target in seen:
            continue
        seen.append(target)
        path = os.path.join(target, exe)
        if os.path.isfile(path):
            return path
    on_path = shutil.which("unpile")
    if on_path:
        return on_path
    locations = "\n".join(f" - {t}" for t in seen)
    raise UnpileNotFound(f"the unpile binary was not found in:\n{locations}\nnor on PATH; set UNPILE_BIN to its path")


def _user_scripts() -> str | None:
    try:
        scheme = sysconfig.get_preferred_scheme("user") if sys.version_info >= (3, 10) else ("nt_user" if os.name == "nt" else "posix_user")
        return sysconfig.get_path("scripts", scheme=scheme)
    except (KeyError, LookupError):
        return None
