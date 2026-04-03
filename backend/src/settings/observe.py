import os

os.environ.setdefault("LITELLM_LOG", "ERROR")


def setup_telemetry():
    os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    os.environ.setdefault("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
