from prometheus_client import start_http_server, Counter, Histogram
import time
import random

REQUEST_COUNT = Counter("request_count_total", "Total request ke model")
PREDICTION_COUNT = Counter("prediction_count_total", "Total prediksi model")
INFERENCE_TIME = Histogram("inference_duration_seconds", "Waktu inference model")

start_http_server(8001)

print("Exporter running on port 8001")

def mock_inference():
    time.sleep(random.uniform(0.1, 0.5))

while True:
    REQUEST_COUNT.inc()

    start = time.time()
    mock_inference()
    INFERENCE_TIME.observe(time.time() - start)

    PREDICTION_COUNT.inc()
