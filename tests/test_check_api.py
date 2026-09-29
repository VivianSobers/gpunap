from gpunap import results
from gpunap.check import api


def test_all_functions_and_init_is_ok():
    assert api.verdict([], 0) == results.OK


def test_missing_function_is_a_hazard():
    assert api.verdict(["cuCheckpointProcessLock"], 0) == results.HAZARD


def test_failed_init_is_a_hazard():
    assert api.verdict([], 100) == results.HAZARD


def test_registry_lists_api_first():
    from gpunap.check import registry
    assert registry.CHECKS[0].name == "api" and registry.by_name()["api"].mode == "default"
