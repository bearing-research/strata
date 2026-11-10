"""Build reliability: the runner's id, its lease settings, and its heartbeat.

The ledger's own lease and finalize rules are tested in vestigo.
"""

import asyncio
import uuid
from unittest.mock import patch

import pytest
from vestigo.build_store import reset_build_store
from vestigo.provenance import TransformSpec
from vestigo.store import get_artifact_store, reset_artifact_store

from strata.transforms.registry import (
    TransformDefinition,
    TransformRegistry,
    reset_transform_registry,
    set_transform_registry,
)
from strata.transforms.runner import BuildRunner, RunnerConfig
from tests.conftest import seed_build_targets


@pytest.fixture
def artifact_dir(tmp_path):
    artifact_path = tmp_path / "artifacts"
    artifact_path.mkdir(parents=True)
    return artifact_path


@pytest.fixture
def artifact_store(artifact_dir):
    reset_artifact_store()
    store = get_artifact_store(artifact_dir)
    yield store
    reset_artifact_store()


@pytest.fixture
def build_store(artifact_dir):
    reset_build_store()
    from vestigo.build_store import get_build_store

    seed_build_targets(artifact_dir)
    db_path = artifact_dir / "artifacts.sqlite"
    store = get_build_store(db_path)
    yield store
    reset_build_store()


@pytest.fixture
def transform_registry():
    reset_transform_registry()
    registry = TransformRegistry(
        enabled=True,
        definitions=[
            TransformDefinition(
                ref="test_executor@v1",
                executor_url="http://localhost:9999",
                timeout_seconds=30.0,
                max_output_bytes=1024 * 1024,
            ),
        ],
    )
    set_transform_registry(registry)
    yield registry
    reset_transform_registry()


class TestBuildRunnerLeaseIntegration:
    """The build runner with leases."""

    @pytest.fixture
    def runner_config(self):
        return RunnerConfig(
            poll_interval_ms=50,
            max_concurrent_builds=5,
            max_builds_per_tenant=2,
            default_timeout_seconds=30.0,
            default_max_output_bytes=1024 * 1024,
            lease_duration_seconds=2.0,
            heartbeat_interval_seconds=0.5,
            runner_id="test-runner-1",
        )

    @pytest.fixture
    def build_runner(
        self, runner_config, artifact_store, build_store, transform_registry, artifact_dir
    ):
        runner = BuildRunner(
            config=runner_config,
            artifact_store=artifact_store,
            build_store=build_store,
            transform_registry=transform_registry,
            artifact_dir=artifact_dir,
        )
        return runner

    def test_runner_has_unique_id(self, build_runner):
        assert build_runner._runner_id == "test-runner-1"

    def test_runner_config_has_lease_settings(self, runner_config):
        assert runner_config.lease_duration_seconds == 2.0
        assert runner_config.heartbeat_interval_seconds == 0.5
        assert runner_config.runner_id == "test-runner-1"

    def test_runner_generates_id_if_not_provided(
        self, artifact_store, build_store, transform_registry, artifact_dir
    ):
        config = RunnerConfig()
        runner = BuildRunner(
            config=config,
            artifact_store=artifact_store,
            build_store=build_store,
            transform_registry=transform_registry,
            artifact_dir=artifact_dir,
        )
        assert runner._runner_id.startswith("runner-")
        assert len(runner._runner_id) > 7  # "runner-" + UUID fragment

    @pytest.mark.asyncio
    async def test_heartbeat_renews_leases(self, build_runner, artifact_store, build_store):
        """The heartbeat loop renews leases for running builds."""
        artifact_id = str(uuid.uuid4())
        version = artifact_store.create_artifact(
            artifact_id=artifact_id,
            provenance_hash=f"hash-{uuid.uuid4()}",
            transform_spec=TransformSpec(
                executor="test_executor@v1",
                params={},
                inputs=[],
            ),
        )
        build_id = str(uuid.uuid4())
        build_store.create_build(
            build_id=build_id,
            artifact_id=artifact_id,
            version=version,
            executor_ref="test_executor@v1",
            executor_url="http://localhost:9999",
        )

        original_renew = build_store.renew_lease
        renew_calls = []

        def track_renew(bid, owner, duration):
            result = original_renew(bid, owner, duration)
            renew_calls.append((bid, owner, result))
            return result

        with patch.object(build_store, "renew_lease", track_renew):
            # Run slowly so the heartbeat fires.
            async def slow_execute(build, already_claimed=False):
                if not already_claimed:
                    build_store.claim_build(
                        build.build_id,
                        build_runner._runner_id,
                        build_runner.config.lease_duration_seconds,
                    )
                await asyncio.sleep(0.8)  # Let the heartbeat run.
                build_store.complete_build(build.build_id)

            with patch.object(build_runner, "_execute_build", slow_execute):
                await build_runner.start()
                await asyncio.sleep(1.2)
                await build_runner.stop()

        successful_renewals = [c for c in renew_calls if c[2]]
        assert len(successful_renewals) >= 1
