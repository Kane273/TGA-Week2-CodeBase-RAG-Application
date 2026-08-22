from contextlib import contextmanager
import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from app.config import Settings


def configure_langsmith(settings: Settings) -> None:
    """Configure optional hosted tracing while keeping local mode network-free."""
    if not settings.langsmith_tracing:
        os.environ["LANGSMITH_TRACING"] = "false"
        return
    if not settings.langsmith_api_key:
        raise ValueError("LANGSMITH_API_KEY is required when LANGSMITH_TRACING=true")
    os.environ.update({
        "LANGSMITH_TRACING": "true",
        "LANGSMITH_API_KEY": settings.langsmith_api_key,
        "LANGSMITH_ENDPOINT": settings.langsmith_endpoint,
        "LANGSMITH_PROJECT": settings.langsmith_project,
    })
    if settings.langsmith_workspace_id:
        os.environ["LANGSMITH_WORKSPACE_ID"] = settings.langsmith_workspace_id


def configure_telemetry(app, settings: Settings) -> None:
    if not settings.otel_enabled:
        return
    provider = TracerProvider(resource=Resource.create({"service.name": "codebase-rag"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(
        endpoint=settings.otel_exporter_otlp_endpoint, insecure=True
    )))
    trace.set_tracer_provider(provider)
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)


@contextmanager
def span(name: str, **attributes):
    tracer = trace.get_tracer("codebase-rag")
    with tracer.start_as_current_span(name, attributes=attributes) as current:
        yield current
