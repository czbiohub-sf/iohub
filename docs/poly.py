from datetime import datetime
from functools import partial
from pathlib import Path

from sphinx_polyversion import apply_overrides
from sphinx_polyversion.driver import DefaultDriver
from sphinx_polyversion.git import (
    Git,
    GitRef,
    GitRefType,
    closest_tag,
    file_predicate,
)
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
PIP_ARGS = [".[doc]"]

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
ENVIRONMENT = (
    {
        None: Pip.factory(
            venv=Path(".venv") / "default",
            args=PIP_ARGS,
            creator=VenvWrapper(),
        ),
        "v0.1.0": Pip.factory(
            venv=Path(".venv") / "v0.1.0",
            args=PIP_ARGS
            + [
                "sphinxcontrib-applehelp<=1.0.4",
                "sphinxcontrib-devhelp<=1.0.2",
                "sphinxcontrib-htmlhelp<=2.0.1",
                "sphinxcontrib-qthelp<=1.0.3",
                "sphinxcontrib-serializinghtml<=1.1.5",
            ],
            creator=VenvWrapper(),
        ),
        "main": Pip.factory(
            venv=Path(".venv") / "main",
            args=PIP_ARGS + ["importlib_metadata"],
            creator=VenvWrapper(),
        ),
    }
    if not MOCK
    else Pip.factory(
        venv=Path(".venv") / "local",
        args=PIP_ARGS + ["importlib_metadata"],
        creator=VenvWrapper(),
    )
)

BUILDER = (
    {
        None: SphinxBuilder(src / "source", args=[]),
        "v0.1.0": SphinxBuilder(src / "source", args=["-D", "plot_gallery=0"]),
        "main": SphinxBuilder(src / "source", args=[]),
    }
    if not MOCK
    else SphinxBuilder(src / "source", args=[])
)

DefaultDriver(
    root,
    OUTPUT_DIR,
    vcs=Git(
        branch_regex=BRANCH_REGEX,
        tag_regex=TAG_REGEX,
        buffer_size=1 * 10**9,  # 1 GB
        predicate=file_predicate([src]),  # exclude refs without source dir
    ),
    builder=BUILDER,
    env=ENVIRONMENT,
    # template_dir=root / src / "templates",
    static_dir=root / src / "source" / "_static",
    mock=MOCK_DATA,
    selector=partial(closest_tag, root),
).run(MOCK)
