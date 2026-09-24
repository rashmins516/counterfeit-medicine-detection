from trace_platform.services import SightingConsumer
from trace_platform.trace_query_api import TraceQueryAPI


if __name__ == "__main__":
    consumer = SightingConsumer()
    payload = {
        "batch_id": "batch-001",
        "location": "A1",
        "device_id": "device-01",
        "verdict": "suspicious",
        "confidence": 0.91,
        "image_hash": "abc123",
        "record_hash": "record-001",
    }
    print(consumer.ingest(payload))

    query_api = TraceQueryAPI()
    print(query_api.get_trace("batch-001"))
