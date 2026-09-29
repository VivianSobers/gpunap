import json
import os

from gpunap import results
from gpunap.check import registry

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu", "expectations.json")


def test_expectations_cover_every_check_on_both_drivers():
    checks = json.load(open(PATH))["checks"]
    assert sorted(checks) == sorted(c.name for c in registry.CHECKS)
    for name, e in checks.items():
        assert e["595"] in results.VERDICTS and e["580"] in results.VERDICTS, name
        assert e["evidence"], name
