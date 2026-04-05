import os

_ = os.environ.setdefault("LITELLM_LOG", "ERROR")


def setup_telemetry():
    _ = os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    _ = os.environ.setdefault("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
