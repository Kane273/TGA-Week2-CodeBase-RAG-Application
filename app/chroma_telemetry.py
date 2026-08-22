from chromadb.telemetry.product import ProductTelemetryClient, ProductTelemetryEvent
from overrides import override


class NoopProductTelemetry(ProductTelemetryClient):
    """Disable Chroma product analytics for private enterprise deployments."""

    @override
    def capture(self, event: ProductTelemetryEvent) -> None:
        return None
