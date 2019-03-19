# Scripts

## Documentation

This Python package is documented using [Sphinx](http://www.sphinx-doc.org/en/master/index.html). Python docstrings follow the [Google Python Style Guidelines](http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)

### Automatic Generation

```
cd docs
sphinx-apidoc -f -o source/ ../scripts/ 
```

### Build

```
cd docs
make html
```

### GitHub Pages