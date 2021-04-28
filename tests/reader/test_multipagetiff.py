import numpy as np
import pytest

from waveorder.io.multipagetiff import MicromanagerOmeTiffReader


def test_constructor(setup_mm2):
    mmr = MicromanagerOmeTiffReader()
