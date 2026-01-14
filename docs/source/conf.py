# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------
project = 'torch-sla'
copyright = '2024, walker chi'
author = 'walker chi'
version = '0.1.0'
release = '0.1.0'

# -- SEO & Metadata ----------------------------------------------------------
# Project description for search engines
html_short_title = 'torch-sla'
html_baseurl = 'https://walkerchi.github.io/torch-sla/'

# Open Graph metadata for social sharing
ogp_site_url = 'https://walkerchi.github.io/torch-sla/'
ogp_site_name = 'torch-sla: Torch Sparse Linear Algebra'
ogp_image = '_static/logo.jpg'
ogp_description_length = 200
ogp_type = 'website'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',  # Link to external docs
    'sphinx.ext.graphviz',     # For vector diagrams
]

# Graphviz settings
graphviz_output_format = 'svg'

# Intersphinx mapping for cross-references
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
}

templates_path = ['_templates']
exclude_patterns = ['setup.py', '__init__.py']

# Language settings
language = 'en'

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'
html_static_path = ['_static']
html_css_files = ['custom.css']
autodoc_member_order = 'bysource'

# Furo theme options with SEO enhancements
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/walkerchi/torch-sla",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}

# Logo and title
html_logo = "_static/logo.jpg"
html_title = f"torch-sla v{version}"
html_favicon = "_static/logo.jpg"

# Additional SEO meta tags
html_context = {
    'description': 'torch-sla: Differentiable sparse linear algebra library for PyTorch with CUDA support. Solve sparse linear systems with automatic differentiation.',
    'keywords': 'PyTorch, sparse matrix, linear algebra, CUDA, cuSOLVER, cuDSS, sparse solver, differentiable, autograd, scientific computing, FEM, CFD',
    'author': 'walker chi',
    'og_title': 'torch-sla: Torch Sparse Linear Algebra',
    'og_description': 'Differentiable sparse linear equation solver for PyTorch with multiple backends (SciPy, Eigen, cuSOLVER, cuDSS). Full gradient support via autograd.',
    'og_image': '_static/logo.jpg',
    'twitter_card': 'summary_large_image',
}

# Sitemap for search engines
html_extra_path = []

# Search engine optimization
html_use_index = True
html_split_index = False
html_copy_source = True
html_show_sourcelink = True
html_show_sphinx = False
html_show_copyright = True

# JSON-LD structured data will be added via template