import os
import subprocess
import sys

import pytest

from tests.shim_helpers import write_interpreter_shim

# In a process of its own: descriptor 255 here may be anything's.
_PASS_THE_HIGHEST = """
import os, resource, subprocess, sys

# Where bash moves the script it reads: the highest descriptor below 256 the limit allows.
high = min(resource.getrlimit(resource.RLIMIT_NOFILE)[0], 256) - 1
read_end, write_end = os.pipe()
os.dup2(write_end, high, inheritable=False)
held = "import os, sys; os.fstat(int(sys.argv[1]))"
sys.exit(subprocess.run([sys.argv[1], "-c", held, str(high)], pass_fds=(high,)).returncode)
"""


@pytest.mark.skipif(os.name == "nt", reason="a shell script is no Windows interpreter")
def test_the_shim_hands_on_the_descriptors_it_was_passed(tmp_path):
    """A harness whose event pipe was descriptor 255 started without it under macOS's /bin/sh."""
    shim = tmp_path / "python"
    write_interpreter_shim(shim)

    passed = subprocess.run(
        [sys.executable, "-c", _PASS_THE_HIGHEST, str(shim)], capture_output=True, text=True
    )

    assert passed.returncode == 0, passed.stderr


@pytest.mark.skipif(os.name == "nt", reason="a shell script is no Windows interpreter")
def test_the_shim_runs_this_interpreter_with_the_environment_it_was_given(tmp_path):
    shim = tmp_path / "python"
    write_interpreter_shim(shim, {"SHIM_SAYS": "two words"})

    ran = subprocess.run(
        [shim, "-c", "import os, sys; print(sys.executable, os.environ['SHIM_SAYS'], sep='|')"],
        capture_output=True,
        text=True,
        check=True,
    )

    assert ran.stdout.strip() == f"{sys.executable}|two words"
