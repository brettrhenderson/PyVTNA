"""A python package for performing variable time normalization analysis of chemical reaction rate laws."""

# Add imports here
from .vtna import VTNA
# from pyvtna import align
# from pyvtna import metrics
# from pyvtna import notebook
# from pyvtna import readers
# from pyvtna import signal
# from pyvtna import testing
# from pyvtna import visualizers
# from pyvtna import vtna


# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
