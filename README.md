# Solver Sandbox

Linear solver tests using scipy.sparse

# Installation

Install project and dependencies (preferably on a virtual environment).

The `--editable` flag is optional but recommended for development.
```
python -m venv env
source env/bin/activate
pip install --editable .
```

Uncompress example problems with
```
cd examples
chmod u+x extract
./extract
```

Provided tests must be run from the test folder for now
```
cd tests && python -m unittest
```