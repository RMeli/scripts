# Scripts

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