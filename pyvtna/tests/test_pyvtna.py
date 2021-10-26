"""
Unit and regression test for the pyvtna package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import pyvtna


def test_pyvtna_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "pyvtna" in sys.modules
