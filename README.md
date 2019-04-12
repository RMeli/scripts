[![Build Status](https://travis-ci.org/RMeli/scripts.svg?branch=master)](https://travis-ci.org/RMeli/scripts)
[![DOI](https://zenodo.org/badge/176330398.svg)](https://zenodo.org/badge/latestdoi/176330398)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Scripts

## Description

```
python -m scripts.MODULE
```

| Module            | Description                                   |
| :---              | ---                                           |
| `md.rmsd`         | Compute RMSD on an MD trajectory.             |
| `plot.hist`       | Plot histogram.                               |
| `plot.roc`        | Plot ROC curve for binary classifier.         |
| `plot.xy`         | Plot Y versus X.                              |


## Installation

```bash
pip install .
```

### Development

```bash
pip install -e .
```

## Documentation

This Python package is documented using [Sphinx](http://www.sphinx-doc.org/en/master/index.html). Python docstrings follow the [Google Python Style Guidelines](http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)

### Automatic Generation

```bash
cd docs
sphinx-apidoc -f -o source/ ../scripts/ 
```

### Build

```bash
cd docs
make html
```

### GitHub Pages