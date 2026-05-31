"""
General purpose utilities for PyBERT.

Original author: David Banas <capn.freako@gmail.com>

Original date:   June 16, 2024

Copyright (c) 2024 David Banas; all rights reserved World wide.

Historical lineage:
    pybert/pybert_cntrl.py => pybert/utility.py => pybert/utility/__init__.py

A refactoring of the `pybert.utility` module, as per Issue #147.
"""

from .channel    import *   # noqa: F401,F403
from .functional import *   # noqa: F401,F403
try:
    from .ibisami import *  # noqa: F401,F403
except ImportError:
    # `ibisami` requires the optional `pyibisami` dependency. Allow the rest of
    # `pybert.utility` (and consumers like `pybert.models.viterbi`) to import
    # without it; only IBIS-AMI functionality is unavailable when it's missing.
    pass
from .jitter     import *   # noqa: F401,F403
from .math       import *   # noqa: F401,F403
from .python     import *   # noqa: F401,F403
from .sigproc    import *   # noqa: F401,F403
from .sparam     import *   # noqa: F401,F403
