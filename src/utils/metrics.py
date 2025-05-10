from prometheus_client import Histogram, Counter, start_http_server

start_http_server(8001)

REQUEST_LATENCY = Histogram('request_latency_seconds', 'Latency of generation requests')
REQUEST_COUNT   = Counter('request_count', 'Total generation requests')

class Metrics:
    def observe(self, latency: float):
        REQUEST_LATENCY.observe(latency)
        REQUEST_COUNT.inc()