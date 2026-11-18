"""Report the Python and library versions the analysis runs on."""

import sys
from importlib.metadata import version

MAIN_LIBRARIES = ["pandas", "numpy", "scikit-learn", "matplotlib", "seaborn"]


def describe_environment():
    """Return the Python version and the main library versions, one per line."""
    lines = [f"Python {sys.version.split()[0]}"]
    lines += [f"{library} {version(library)}" for library in MAIN_LIBRARIES]
    return "\n".join(lines)
