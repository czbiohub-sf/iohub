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

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

#### Install uv

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or see [uv installation docs](https://docs.astral.sh/uv/getting-started/installation/) for other methods.

#### Clone the repository

If you have push permission to the repository:

```sh
cd # to the directory you want to work in
git clone https://github.com/czbiohub-sf/iohub.git
```

Otherwise, you can follow [these instructions](https://docs.github.com/en/get-started/quickstart/fork-a-repo)
to [fork](https://github.com/czbiohub-sf/iohub/fork) the repository.

#### Install dependencies

```sh
cd iohub/
uv sync --all-extras --all-groups
```

#### Dependency groups

The project uses [dependency groups](https://docs.astral.sh/uv/concepts/projects/dependencies/#dependency-groups) for development tools and [optional dependencies](https://docs.astral.sh/uv/concepts/projects/dependencies/#optional-dependencies) for user-facing extras.

**Full installation (recommended):**

```sh
uv sync --all-extras --all-groups
```

**Selective installation:**

| Group | Purpose |
|-------|---------|
| `test` | Testing tools (pytest, hypothesis, etc.) |
| `acquire-zarr` | Acquire-zarr reader (requires glibc 2.35+) |
| `doc` | Documentation (sphinx, etc.) |
| `pre-commit` | Pre-commit hooks |
| `dev` | Includes test, acquire-zarr, doc, pre-commit |

| Extra | Purpose |
|-------|---------|
| `tensorstore` | TensorStore array access |

**Examples:**

```sh
# Install specific groups
uv sync --group test --group doc

# Install specific extras
uv sync --extra tensorstore

# Combine groups and extras
uv sync --group test --extra tensorstore

# On older glibc systems (e.g., Rocky Linux 8), omit acquire-zarr:
uv sync --group test --group doc --group pre-commit --all-extras
```

Then make the changes and [track them with Git](https://docs.github.com/en/get-started/using-git/about-git#example-contribute-to-an-existing-repository).

### Developing documentation

#### Prerequisites

Install documentation dependencies and a [forked version of `sphinx-polyversion`](https://github.com/ziw-liu/sphinx-polyversion/tree/iohub-staging) (temporary fix for compatibility):

```shell
uv sync --group doc
uv pip install --force-reinstall git+https://github.com/ziw-liu/sphinx-polyversion.git@iohub-staging
```

#### Building the HTML version locally

Inside `/docs` folder:

```shell
make clean
uv run sphinx-polyversion poly.py -vvv --local
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

See the [syntax documentation](https://sphinx-gallery.github.io/stable/syntax.html).

### Testing

If you made code changes, make sure that there are also tests for them!
Local test runs and coverage check can be invoked by:

```sh
# in the project root directory
uv run pytest --cov=iohub tests/
```

`iohub` uses [Hypothesis](https://hypothesis.readthedocs.io/en/latest/index.html)
together with [pytest](https://docs.pytest.org/).
See [this paper](https://conference.scipy.org/proceedings/scipy2020/zac_hatfield-dodds.html)
for how this can reveal more bugs.

### Code style

We use [pre-commit](https://pre-commit.com/) to automatically format and lint code prior to each commit. To minimize test errors when submitting pull requests, install the pre-commit hooks:

```bash
# Using uvx (recommended - no need to install pre-commit as a dependency)
uvx pre-commit install

# Or using prek (https://github.com/tweag/prek - a faster pre-commit runner)
uvx prek install

# Or if you prefer to sync the pre-commit group
uv sync --group pre-commit
uv run pre-commit install
```

To run pre-commit manually on all files:

```bash
uvx pre-commit run --all-files
# or
uvx prek run --all-files
```

When these tools are executed within the project root directory, they should automatically use the [project settings](./pyproject.toml).

[NumPy style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html) are used for API documentation.
