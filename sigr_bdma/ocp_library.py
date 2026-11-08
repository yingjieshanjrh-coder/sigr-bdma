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


# ---------------------------------------------------------------------------
# OCP collection of the TUM-EES DegradationModeAnalysis framework (MIT licence,
# Rehm et al., J. Power Sources 670 (2026) 239418). Not redistributed here: clone
# https://github.com/TUM-EES/DegradationModeAnalysis and pass its path. Cite the framework
# paper AND the original source of each curve (see input_data/*/README in that repo).
# ---------------------------------------------------------------------------
def _read_tum_mat(path):
    import scipy.io as sio
    m = sio.loadmat(path)
    s = m[[k for k in m if not k.startswith("__")][0]][0, 0]
    c = np.ravel(s["normalizedCapacity"][()]).astype(float)
    v = np.ravel(s["voltage"][()]).astype(float)
    ok = np.isfinite(c) & np.isfinite(v)
    c, v = c[ok], v[ok]
    x = c if np.corrcoef(c, v)[0, 1] < 0 else 1.0 - c      # x = lithiation fraction
    o = np.argsort(x)
    x, v = x[o], np.minimum.accumulate(v[o])                # enforce U decreasing in x
    x, idx = np.unique(x, return_index=True)
    return x, v[idx]


def load_tum_ocps(repo_path):
    """Returns {'graphite': {name: {'lith': f, 'delith': f}}, 'silicon': {...}} with
    callables U(x), x = lithiation fraction 0..1. Missing branches are absent."""
    import glob
    import os
    import re
    out = {"graphite": {}, "silicon": {}}
    for kind, sub, pat in (("graphite", "graphite", r"Gr_(Lithiation|Delithiation)_(\w+)\.mat"),
                           ("silicon", "silicon", r"Si(?:Reconstr)?_(Lithiation|Delithiation)_(\w+?)(?:_P45B.*)?\.mat")):
        for f in glob.glob(os.path.join(repo_path, "input_data", sub, "*.mat")):
            m = re.match(pat, os.path.basename(f))
            if not m:
                continue
            branch = "lith" if m.group(1) == "Lithiation" else "delith"
            x, v = _read_tum_mat(f)
            out[kind].setdefault(m.group(2), {})[branch] = (
                lambda xx, x=x, v=v: np.interp(xx, x, v))
    return out
