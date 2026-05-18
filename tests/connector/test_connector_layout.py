# SPDX-License-Identifier: Apache-2.0

# Standard
import inspect

# Third Party
import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("cupy")
pytest.importorskip("kvikio")

# First Party
from daser.connector.daser_connector import DaserConnector
from daser.connector.metadata import DaserConnectorMeta, ReqLoadSpec, ReqStoreSpec


def test_connector_entrypoint_delegates_scheduler_and_worker_methods() -> None:
    """DaserConnector keeps vLLM's entrypoint while role logic lives elsewhere."""
    assert (
        inspect.getmodule(DaserConnector).__name__ == "daser.connector.daser_connector"
    )
    assert (
        inspect.getmodule(DaserConnector.get_num_new_matched_tokens).__name__
        == "daser.connector.scheduler"
    )
    assert (
        inspect.getmodule(DaserConnector.start_load_kv).__name__
        == "daser.connector.worker"
    )


def test_connector_metadata_lives_in_dedicated_module() -> None:
    """Scheduler/worker metadata types are shared from connector.metadata."""
    assert ReqLoadSpec.__module__ == "daser.connector.metadata"
    assert ReqStoreSpec.__module__ == "daser.connector.metadata"
    assert DaserConnectorMeta.__module__ == "daser.connector.metadata"
