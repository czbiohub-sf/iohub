# Contributing guide

Thanks for your interest in contributing to `iohub`!

Please see the following steps for our workflow.

## Getting started

Please read the [README](./README.md) for an overview of the project,
and how you can install and use the package.

## Issues

We use [issues](https://github.com/czbiohub-sf/iohub/issues) to track
bug reports, feature requests, and provide user support.

Before opening a new issue, please first search existing issues (including closed ones),
to see if there is an existing discussion about it.

### Bug report

We welcome bug reports!

When reporting a bug (something should work but does not),
please include the following in the order mentioned:

- A concise summary of the symptom in the title
- A description of the problem
- Minimal code/command examples to reproduce
- Error message, log output, or screenshot
- `iohub` version, Python version and build, and platform/OS information
- (Optional) potential cause

### Feature request

Feature requests ask for new functionalities or performance improvements to be added to the package.
Please address following questions in the issue:

- What is the new feature?
- iohub's main mission is to provide an efficient Python library and command line interface to access ND bioimaging data.
Is the feature within the scope of this project or another project?
- Who will be the users?
- (Optional) what is the best way to implement it?
- (Optional) what are the alternatives?

### Prioritization

As is the case with any software project, possible improvements vastly out number the community's capacity.
We use releases as a reference to focus the effort on near-term priorities.
PRs within the scope of the next release will receive more attention.
A list of planned release milestones can be found [here](https://github.com/czbiohub-sf/iohub/milestones).

### Documentation change

If you find that any documentation in this project is incomplete, inaccurate, or ambiguous,
please open an issue.
We welcome contributions to the documentation from users,
particularly user guides that we can collaboratively edit.

## Making changes

Any change made to the `main` branch or release maintenance branches
need to be proposed in a [pull request](https://github.com/czbiohub-sf/iohub/pulls) (PR).

If there is an issue that can be addressed by the PR, please reference it.

If there is not a relevant issue, please either open an issue first,
or describe the bug fixed or feature implemented in the PR.

### Setting up development environment

For local development, first install [Git](https://git-scm.com/)
and Python with an environment management tool
(e.g. [miniforge](https://github.com/conda-forge/miniforge), a minimal community distribution of Conda).

If you use Conda, set up an environment with:

```sh
conda create -n iohub-dev python=3.11
conda activate iohub-dev
```

If you have push permission to the repository,
clone the repository (the code blocks below are shell commands):

```sh
cd # to the directory you want to work in
git clone https://github.com/czbiohub-sf/iohub.git
```

Otherwise, you can follow [these instructions](https://docs.github.com/en/get-started/quickstart/fork-a-repo)
to [fork](https://github.com/czbiohub-sf/iohub/fork) the repository.

Then install the package in editable mode with the development dependencies:

```sh
cd iohub/ # or the renamed project root directory
pip install -e ".[dev]"
```

Then make the changes and [track them with Git](https://docs.github.com/en/get-started/using-git/about-git#example-contribute-to-an-existing-repository).

### Developing documentation

#### Building the HTML version locally

Inside `/docs` folder

```shell
pip install "/PATH/TO/iohub[doc]"
make clean && make build
```

Generated HTML documentation can be found in
the ``build/html`` directory. Open ``build/html/index.html`` to view the home
page for the documentation.

#### Writing examples

Example scripts in the `docs/examples` directory
are automatically compiled to RST with `sphinx-gallery`
in the `docs/source/auto_examples` directory.
Files that start with `run_` in the file name
are executed during the build,
and output (stdout, matplotlib plot) are rendered in HTML.

They can also be executed as plain Python scripts
or interactive code blocks in some IDEs (VS Code, PyCharm, Spyder etc.).

See the syntax documentation
[here](https://sphinx-gallery.github.io/stable/syntax.html).

### Testing

If you made code changes, make sure that there are also tests for them!
Local test runs and coverage check can be invoked by:

```sh
# in the project root directory
pytest --cov=iohub tests/
```

`iohub` uses [Hypothesis](https://hypothesis.readthedocs.io/en/latest/index.html)
together with [pytest](https://docs.pytest.org/).
See [this paper](https://conference.scipy.org/proceedings/scipy2020/zac_hatfield-dodds.html)
for how this can reveal more bugs.

### Code style

We use [pre-commit](https://pre-commit.com/) to sort imports with [isort](https://github.com/PyCQA/isort), format code with [black](https://black.readthedocs.io/en/stable/), and lint with [flake8](https://github.com/PyCQA/flake8) automatically prior to each commit. To minimize test errors when submitting pull requests, please install pre-commit in your environment as follows:

```bash
pre-commit install
```

When these packages are executed within the project root directory, they should automatically use the [project settings](./pyproject.toml).

[NumPy style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html) are used for API documentation.
