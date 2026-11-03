"""Every name importable from ``strata.notebook.executor`` stays importable as paths move out.

Listed from the code and tests before the first extraction: what the module defines, then what
the repo takes through it. A patch through the module reaches a moved path only if the test
follows the move; this list keeps the name, not the patch.
"""

from __future__ import annotations

import pytest

import strata.notebook.executor as executor_module

DEFINED = """
    BatchCellResult BatchExecutionResult CellExecutionResult CellExecutor
    DEFAULT_CELL_TIMEOUT_SECONDS RemoteExecutionError _ARTIFACT_EXT_BY_CONTENT_TYPE
    _CellProvenance _DISPLAY_TRANSIENT_KEYS _JOB_POLL_SECONDS _MODULE_TO_PACKAGE
    _OUTPUT_VARIABLE _QUEUE_DETAIL_MAX_CHARS _add_fetch_inputs _add_harness_params
    _artifact_content_type _detect_missing_module _drain_stream _each_files _each_output_read
    _fenced_result _loop_folders _loop_run_token _merge_variant_writes _merged_test_runs
    _monotonic _mount_sync_failure_result _no_interpreter_message _pack_folder_inputs
    _queue_report _refused_batch _refused_result _resolve_worker_token _store_failure_result
    _team_cache_publish_policy _variant_test_run _worker_mount_bytes _worker_mount_writes
    _writes_entry cell_timeout_message display_metadata_params effective_worker_name
    is_cell_batchable logger partition_batchable_runs
""".split()
TAKEN_THROUGH_IT = """
    TableWriteError TeamStore blocking_diagnostic hand_over let_through pull_cell_outputs
    resolve_harness_user tempfile time
""".split()


@pytest.mark.parametrize("name", DEFINED + TAKEN_THROUGH_IT)
def test_a_name_stays_importable(name):
    assert hasattr(executor_module, name)


def test_a_moved_result_is_the_same_class():
    from strata.notebook.execution_result import CellExecutionResult

    assert executor_module.CellExecutionResult is CellExecutionResult
