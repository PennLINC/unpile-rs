"""Sphinx configuration for the unpile documentation."""

import os
import re

project = "unpile"
author = "the PennLINC developers"
copyright = "2026, the PennLINC developers"
with open(os.path.join(os.path.dirname(__file__), "..", "Cargo.toml")) as f:
    release = re.search(r'^version\s*=\s*"([^"]+)"', f.read(), re.M).group(1)
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
]
myst_enable_extensions = ["dollarmath", "colon_fence", "deflist"]
myst_heading_anchors = 3
# The diffusion book explains the physics; these pages say what unpile does with it and link to
# https://pennlinc.github.io/diffusion-book/book/preprocessing/susceptibility-distortion by anchor.
intersphinx_mapping = {
    "qsiprep": ("https://qsiprep.readthedocs.io/en/latest", None),
}
exclude_patterns = ["_build", "tools"]
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = f"unpile {release}"
