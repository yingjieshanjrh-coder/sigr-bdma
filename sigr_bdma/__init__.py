"""sigr-bdma: Bayesian degradation-mode analysis for Si–graphite blend anodes."""
from .ocp import (SiOCP, U_graphite, U_nmc811, extract_si_from_blend, mix_si,  # noqa: F401
                  placeholder_si)
from .cell import Branch, Cell  # noqa: F401
from .inference import Param, Posterior, Problem, default_params, ensemble_mcmc  # noqa: F401
from .modes import MODE_NAMES, paired_modes, summarize_modes, trajectory  # noqa: F401
from .design import evaluate_design, fisher_covariance  # noqa: F401
from .io import branch_from_arrays, pulse_resistance, split_cycler_log  # noqa: F401
from . import io  # noqa: F401
from .workflow import fit_series, fit_state  # noqa: F401

__version__ = "0.3.1"
from .ocp_library import graphite_ai2020, si_verbrugge2016  # noqa: F401,E402
from . import datasets  # noqa: F401,E402
from . import sensitivity  # noqa: F401,E402
