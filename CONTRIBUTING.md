# Contributing guide

Thanks for your interest in contributing to `iohub`!

Please see the following steps for our workflow.

## Getting started

Please read the [README](./README.md) for an overview of the project,
and how you can install and use the package.

## Issues

We use [issues](https://github.com/czbiohub/iohub/issues) to track
bug reports, feature requests, and provide user support.

Before opening a new issue, please first search existing issues (including closed ones),
to see if there is an existing discussion about it.

### Bug report

When reporting a bug (something should work but does not), please include:

- A concise summary of the symptom in the title
- A description of the problem
- Error message, log output, or sreenshot
- Minimal code/command examples to reproduce
- `iohub` version, Python version and build, and platform/OS information
- (Optional) potential cause

### Feature request

Feature requests ask for new functionalities or performance improvements to be added to the package.
Please consider the questions points and address them in the issue:

- What is the new feature?
- Is it within the scope of this project?
- Who will be the users?
- (Optional) what is the best way to implement it?
- (Optional) what are the alternatives?

### Documentation change

If you find that any documentation in this project is incomplete, inaccurate, or ambiguous,
please open an issue.

### Project management

Issues can also be used to track our project management and higher level decision making process.

## Making changes

Any change made to the `main` branch or release maintenance branches
need to be proposed in a [pull request](https://github.com/czbiohub/iohub/pulls) (PR).

If there is an issue that can be addressed by the PR, please reference it.

If there is not a relevant issue, please either open an issue first,
or describe the bug fixed or feature implemented in the PR.

### Setting up developing environment

For local development, first install [Git](https://git-scm.com/)
and Python with an environment management tool
(e.g. [miniforge](https://github.com/conda-forge/miniforge), a minimal community distribution of Conda).

If you use Conda, set up an environment with:

```sh
conda create -n iohub-dev python=3.9
conda activate iohub-dev
```

If you have push permission to the repository,
clone the repository (the code blocks below are shell commands):

```sh
cd # to the directory you want to work in
git clone https://github.com/czbiohub/iohub.git
```

Otherwise, you can follow [these instructions](https://docs.github.com/en/get-started/quickstart/fork-a-repo)
to [fork](https://github.com/czbiohub/iohub/fork) the repository.

Then install the package in editable mode with the development dependencies:

```sh
cd iohub/ # or the renamed project root directory
pip install -e ".[dev]"
```

Then make the changes and [track them with Git](https://docs.github.com/en/get-started/using-git/about-git#example-contribute-to-an-existing-repository).

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

We use [black](https://black.readthedocs.io/en/stable/) to format our code.
Black installed with `iohub` should automatically use the [settings](./pyproject.toml) in the repository.

[NumPy style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html) are used for API documentation.

### Focusing effort

As is the case with any software project, possible improvements vastly out number the community's capacity.
We use releases as a reference to focus the effort on near-term priorities.
PRs within the scope of the next release should expect to receive more attention.
A list of planned release milestones can be found [here](https://github.com/czbiohub/iohub/milestones).
