# docs/conf.py
import os
import sys

sys.path.insert(0, os.path.abspath('..'))

project = 'pyVELOX'
author = 'Joshua Müller'
copyright = '2026, Joshua Müller'

extensions = [
    'autodoc2',
    'myst_nb',
]

autodoc2_packages = [
    {'path': '../velox_tools', 'exclude_dirs': ['notebooks', '__pycache__', 'data']},
]
autodoc2_render_plugin = 'myst'
autodoc2_index_template = None

myst_enable_extensions = ['colon_fence', 'deflist']

# myst-nb: execute notebooks at build time so examples stay honest, but
# don't fail the whole build if one needs data/paths only available on
# the cluster this package is developed on
nb_execution_mode = 'off'

root_doc = 'index'

html_theme = 'sphinx_book_theme'
html_title = 'pyVELOX'
exclude_patterns = [
    '_build', 'Thumbs.db', '.DS_Store',
    # only the two example notebooks are referenced from the toctree;
    # exclude the rest of velox_tools/notebooks/ (dev notebooks, not docs)
    'notebooks/pushbroom.ipynb', 'notebooks/georeff.ipynb',
    'notebooks/correct_fixed_pattern.ipynb', 'notebooks/correct_fixed_pattern2.ipynb',
    'notebooks/*-checkpoint.ipynb',
]
