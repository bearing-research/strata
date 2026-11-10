"""Tests for ``imported_names``; cellgraph's own suite covers the cell analysis."""

from strata.notebook.analyzer import imported_names


class TestImportedNames:
    """imported_names: re-importable bindings, used to pick the log level when an upstream
    variable's artifact is unexpectedly absent.
    """

    def test_plain_and_aliased_imports(self):
        names = imported_names("import numpy as np\nimport math\nimport matplotlib.pyplot as plt\n")
        assert names == {"np", "math", "plt"}

    def test_from_imports(self):
        assert imported_names("from os import path, getcwd as cwd\n") == {"path", "cwd"}

    def test_star_import_contributes_nothing(self):
        assert imported_names("from os import *\n") == set()

    def test_non_imports_excluded(self):
        assert imported_names("x = 1\ndef f():\n    import json\n    return json\n") == set()

    def test_syntax_error_is_empty(self):
        assert imported_names("import (((") == set()

    def test_dotted_import_binds_the_root(self):
        assert imported_names("import os.path\n") == {"os"}
