## Contributing to Documentation

### Main website

The main documentation website is built using [mkdocs-material](https://squidfunk.github.io/mkdocs-material/).
To build the docs, first install requirements:

```bash
pip install -r docs/requirements.txt
```

Then build and start the docs server:

```bash
cd docs
mkdocs serve
```

### Python Generated Doc

Python code documentation is built using Sphinx in [lance-python-doc](https://github.com/lancedb/lance-python-doc),
and published through [Github Pages](https://lancedb.github.io/lance-python-doc/) in ReadTheDocs style.

### Rust Generated Doc

Rust code documentation is built and published to the [Rust official docs website](https://docs.rs/lance/latest/lance/)
as a part of the release process.
