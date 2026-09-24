from pathlib import Path

from trace_platform.api import TraceAPI
from trace_platform.classifier import BaseClassifier
from trace_platform.ledger import HashChainLedger


if __name__ == "__main__":
    api = TraceAPI(classifier=BaseClassifier(), ledger=HashChainLedger())

    scan_result = api.scan(image=None, batch_id="batch-001", location="A1", device_id="device-01")
    print(scan_result)

    trace_result = api.trace("batch-001")
    print(trace_result)

    verify_result = api.verify(scan_result["record_hash"])
    print(verify_result)
