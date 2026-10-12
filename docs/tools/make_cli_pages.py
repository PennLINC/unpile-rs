#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Regenerate docs/cli/*.md from `unpile --help` and its subcommands.

Usage: python docs/tools/make_cli_pages.py [path/to/unpile]
CI runs this after building and fails if the committed pages differ.
"""
import subprocess
import sys
from pathlib import Path

INTRO = {
    "fit": "Estimate the field from one EPI image and a reference, from an EPI image and a GRE fieldmap, or from two "
           "images of opposite phase-encoding polarity. See {doc}`../user/gre-refinement` and {doc}`../user/fieldmapless`.",
    "export": "Write an existing field (and a pose) as ITK transforms, with the internally resampled images for "
              "comparison against `antsApplyTransforms`. See {doc}`../user/inputs-outputs`.",
    "check": "Compare the operators with the stored PyTorch reference dumps (`testdata/rpe`). See {doc}`../user/validation`.",
}


def help_text(binary, *args):
    return subprocess.run([binary, *args, "--help"], check=True, capture_output=True, text=True).stdout.rstrip()


def main():
    root = Path(__file__).resolve().parents[2]
    binary = sys.argv[1] if len(sys.argv) > 1 else str(root / "target" / "release" / "unpile")
    out = root / "docs" / "cli"
    out.mkdir(exist_ok=True)
    pages = ["index"] + list(INTRO)
    (out / "index.md").write_text(
        "# Command line\n\nOne binary, three subcommands. The global options `--device`, `--threads` and "
        "`--membership` go before the subcommand.\n\n```text\n" + help_text(binary) + "\n```\n\n"
        "```{toctree}\n:maxdepth: 1\n\n" + "\n".join(INTRO) + "\n```\n"
    )
    for sub, intro in INTRO.items():
        (out / f"{sub}.md").write_text(f"# unpile {sub}\n\n{intro}\n\n```text\n" + help_text(binary, sub) + "\n```\n")
    print("wrote", ", ".join(f"docs/cli/{p}.md" for p in pages))


if __name__ == "__main__":
    main()
