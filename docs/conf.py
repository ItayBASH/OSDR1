# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "osdr"
copyright = "2024, Jonathan Somer"
author = "Jonathan Somer"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_copybutton",
    "autoclasstoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc.typehints",
    "sphinx.ext.autosummary",
    "nbsphinx",  # jupyter notebooks
    "sphinx_gallery.load_style",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/JonathanSomer/osdr",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "Twitter",
            "url": "https://twitter.com/jonathan_somer_",
            "icon": "fa-brands fa-twitter",
        },
        {
            "name": "Website",
            "url": "https://jonathansomer.github.io/",
            "icon": "fa-solid fa-globe",
            "type": "fontawesome",
        },
    ]
}

# nbsphinx parameters:
nbsphinx_execute = "never"


# suppress some warnings:
# nitpick_ignore_regex = [("py:class", ".*")]

# autodoc parameters:
autodoc_default_options = {
    "members": True,
    "special-members": "__init__, __add__",
    "private-members": False,
    "inherited-members": False,
    "undoc-members": False,
    "exclude-members": "__weakref__",
}

# autosummary:
autosummary_generate = True
