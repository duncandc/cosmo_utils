"""
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from ..interpolated_functions import interpolated_distmod


def test_dist_mod():

    z = np.linspace(0,1,100)
    d = interpolated_distmod(z)

    assert np.all(d >= 0.0)
