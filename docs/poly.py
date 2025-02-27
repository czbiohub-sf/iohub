from datetime import datetime
from pathlib import Path

from sphinx_polyversion import apply_overrides
from sphinx_polyversion.driver import DefaultDriver
from sphinx_polyversion.git import Git, GitRef, GitRefType, file_predicate
from sphinx_polyversion.pyvenv import Pip, VenvWrapper
from sphinx_polyversion.sphinx import SphinxBuilder

#: Regex matching the branches to build docs for
BRANCH_REGEX = r"^main$"

#: Regex matching the tags to build docs for
TAG_REGEX = r"^v\d+\.\d+\.\d+$"

#: Output dir relative to project root
#: !!! This name has to be choosen !!!
OUTPUT_DIR = "docs/build"

#: Source directory
SOURCE_DIR = "docs/"

#: Arguments to pass to `pip install`
PIP_ARGS = ["."]

#: Arguments to pass to `sphinx-build`
SPHINX_ARGS = []

#: Mock data used for building local version
MOCK_DATA = {
    "revisions": [
        GitRef(
            name="v0.1.0",
            obj="",
            ref="",
            type_=GitRefType.TAG,
            date=datetime(year=2024, month=2, day=13),
        )
    ],
    "current": GitRef(
        name="local",
        obj="",
        ref="",
        type_=GitRefType.BRANCH,
        date=datetime.now(),
    ),
}

#: Whether to build using only local files and mock data
MOCK = False

# Load overrides read from commandline to global scope
apply_overrides(globals())
# Determine repository root directory
root = Git.root(Path(__file__).parent)

# Setup driver and run it
src = Path(SOURCE_DIR)
DefaultDriver(
    root,
    OUTPUT_DIR,
    vcs=Git(
        branch_regex=r"polyversion",
        tag_regex=r"",
        buffer_size=1 * 10**9,  # 1 GB
        predicate=file_predicate([src]),  # exclude refs without source dir
    ),
    builder=SphinxBuilder(src / "source", args=SPHINX_ARGS),
    env=Pip.factory(
        venv=Path(".venv"),
        args=PIP_ARGS,
        creator=VenvWrapper(),
    ),
    # template_dir=root / src / "templates",
    static_dir=root / src / "source" / "_static",
    mock=MOCK_DATA,
).run()
