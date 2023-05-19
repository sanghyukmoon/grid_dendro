# Building Documentation

We use `jupyter-book`.

An example can be found at https://github.com/changgoo/PRFM

To build the document without a source code package sitting in the same folder, this trick helps.

```sh
PYTHONPATH=./ jupyter-book build docs/
```

To publish the document on `gh-pages` for the first time,
please follow the [instruction](https://jupyterbook.org/en/stable/start/publish.html#publish-your-book-online-with-github-pages)

In short, do the followings.

```sh
pip install ghp-import
ghp-import -n -p -f docs/_build/html
```

We then use GitHub Workflow and GitHub Action for continuous deployment.