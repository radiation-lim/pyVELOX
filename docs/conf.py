# docs/conf.py
import os
import sys

sys.path.insert(0, os.path.abspath('..'))

project = 'pyVELOX'
author = 'Joshua Müller'
copyright = '2026, Leipzig Institute for Meteorology'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'myst_nb',
]

root_doc = 'index'
exclude_patterns = [
    '_build', 'Thumbs.db', '.DS_Store', '.jupyter_cache',
    # only the numbered example notebooks belong in the docs
    'notebooks/[!0-9]*.ipynb', 'notebooks/*-checkpoint.ipynb',
]

# API reference: numpy-style docstrings, members in source order
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autodoc_default_options = {'members': True, 'undoc-members': True}
napoleon_google_docstring = False
napoleon_use_rtype = False
napoleon_use_ivar = True  # 'Attributes' as fields: dataclass members are not documented twice
autosummary_generate = False

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'xarray': ('https://docs.xarray.dev/en/stable', None),
    'dask': ('https://docs.dask.org/en/stable', None),
    'distributed': ('https://distributed.dask.org/en/stable', None),
}

# notebooks are rendered with the outputs stored in them -- executing them
# needs the campaign archive
nb_execution_mode = 'off'
myst_enable_extensions = ['colon_fence', 'deflist']
myst_heading_anchors = 3  # README links to its own sections, e.g. #configuration

html_theme = 'sphinx_book_theme'
html_title = 'pyVELOX'
html_logo = '../logo.png'
html_theme_options = {
    'repository_url': 'https://github.com/radiation-lim/pyVELOX',
    'use_repository_button': True,
    'show_toc_level': 2,
}
