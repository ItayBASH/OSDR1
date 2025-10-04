![](https://img.shields.io/badge/python-3.10+-blue.svg)
[![codecov](https://codecov.io/gh/JonathanSomer/osdr/branch/v5/graph/badge.svg?token=ybWCucx2Ha)](https://codecov.io/gh/JonathanSomer/osdr)
![Test](https://github.com/JonathanSomer/osdr/actions/workflows/test.yml/badge.svg)
![Format](https://github.com/JonathanSomer/osdr/actions/workflows/format.yml/badge.svg)
![Lint](https://github.com/JonathanSomer/osdr/actions/workflows/lint.yml/badge.svg)
![Typecheck](https://github.com/JonathanSomer/osdr/actions/workflows/typecheck.yml/badge.svg)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square)](https://opensource.org/licenses/Apache-2.0)


# Getting Started

## Requirements

Make sure you have Python 3.10 or higher installed.

```bash
python --version
```

[Optional] setup and activate a new virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate
```

## Installation

### Option 1: Install the package from Github (recommended)

```bash
pip install git+https://github.com/JonathanSomer/osdr.git
```

### Option 2: Install the package locally for development

The following commands will clone the repo and install the `tdm` package locally:

```bash
git clone git@github.com:JonathanSomer/osdr.git
pip install -e ./osdr
```

> **Note:** The `-e` flag is used to install the package in editable mode, which allows you to make changes to the code and have them reflected in the installed package without needing to reinstall.

> **Tip:** You can clone the repo into any directory, just make sure to modify the path `./osdr` provided to the `pip install` command.

## Testing the installation

After installation the following command should display the value `1e-06`:

```bash
python -c "from tdm.utils import microns; print(microns(1))"
```


## Next steps

Check out some [examples](https://jonathansomer.github.io/osdr/examples.html), including figures from our paper "Temporal Tissue Dynamics from a Spatial Snapshot" (Somer, Mannor, Alon, 2025).



