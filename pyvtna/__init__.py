"""A python package for performing variable time normalization analysis of chemical reaction rate laws."""

# Add imports here
from .vtna import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
