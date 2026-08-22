# Deployment

1. Replace every `REPLACE_ME` secret and configure an OIDC issuer before shared use.
2. Build and sign the image, run tests/Trivy, and provision PostgreSQL/Chroma backups.
3. Install an ingress controller, metrics server, RWX storage class, and create `rag-tls` in `codebase-rag`.
4. Apply `deploy/k8s/base.yaml`, preload Ollama models, and verify readiness plus Prometheus targets.
5. Run golden retrieval and k6 smoke tests before shifting traffic.
6. Roll back the deployment image and active index version when quality gates fail.
