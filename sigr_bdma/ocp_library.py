"""Literature electrode OCPs.

graphite_ai2020 : measured graphite OCP of the Enertech cell, Ai et al.,
                  J. Electrochem. Soc. 167 (2020) 013512. Data file as distributed
                  with PyBaMM (BSD-3-Clause, see data/LICENSE-PyBaMM.txt).
si_verbrugge2016: silicon lithiation / delithiation OCP polynomial fits from
                  Verbrugge, Baker, Xiao, J. Electrochem. Soc. 163 (2016) A262
                  (PyBaMM functions silicon_ocp_*_Mark2016).
"""
from __future__ import annotations

from importlib import resources

import numpy as np

from .ocp import SiOCP

_EPS = 1e-6


def _load_graphite_ai2020():
    with resources.files(__package__).joinpath("data/graphite_ocp_Enertech_Ai2020.csv").open() as f:
        d = np.genfromtxt(f, delimiter=",", comments="#")
    return d[:, 0], d[:, 1]


_GR_X, _GR_U = _load_graphite_ai2020()


def graphite_ai2020(x):
    return np.interp(x, _GR_X, _GR_U)


_SI_LITH = [-96.63, 372.6, -587.6, 489.9, -232.8, 62.99, -9.286, 0.8633]
_SI_DELITH = [-51.02, 161.3, -205.7, 140.2, -58.76, 16.87, -3.792, 0.9937]


def _poly(c, x):
    x = np.clip(np.asarray(x, float), _EPS, 1 - _EPS)
    return np.polyval(c, x) + 1e-4 * (1 / x + 1 / (x - 1))


def si_verbrugge2016() -> SiOCP:
    return SiOCP("Si-Verbrugge2016",
                 lambda x: _poly(_SI_LITH, x),
                 lambda x: _poly(_SI_DELITH, x))
