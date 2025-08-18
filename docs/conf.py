"""Sphinx configuration for votingai documentation."""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Project information
project = "AutoGen Voting Extension"
copyright = "2025, Tejas Dharani"
author = "Tejas Dharani"
release = "0.1.0"
version = "0.1.0"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "myst_parser",
]

# Autodoc configuration
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
}

# Napoleon configuration
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# MyST configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "tasklist",
]

# Templates and static files
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# HTML output configuration
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "canonical_url": "",
    "analytics_id": "",
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "vcs_pageview_mode": "",
    "style_nav_header_background": "#2980B9",
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]

# HTML context
html_context = {
    "display_github": True,
    "github_user": "your-username",
    "github_repo": "votingai",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# LaTeX configuration
latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
    "preamble": "",
    "fncychap": "",
    "printindex": "",
}

latex_documents = [
    (
        "index",
        "votingai.tex",
        "AutoGen Voting Extension Documentation",
        "Tejas Dharani",
        "manual",
    ),
]

# Manual page configuration
man_pages = [
    (
        "index",
        "votingai",
        "AutoGen Voting Extension Documentation",
        [author],
        1,
    )
]

# Texinfo configuration
texinfo_documents = [
    (
        "index",
        "votingai",
        "AutoGen Voting Extension Documentation",
        author,
        "votingai",
        "Democratic consensus for Microsoft AutoGen multi-agent systems.",
        "Miscellaneous",
    ),
]

# Epub configuration
epub_title = project
epub_exclude_files = ["search.html"]
