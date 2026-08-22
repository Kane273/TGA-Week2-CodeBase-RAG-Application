# Incident response

1. Confirm impact with `/api/health/live`, `/api/health/ready`, Prometheus alerts, and correlated `X-Request-ID` logs.
2. Contain the issue by scaling workers to zero for ingestion incidents or disabling ingress for suspected compromise.
3. Preserve PostgreSQL, Redis, Chroma, audit logs, and affected upload artifacts before remediation.
4. Rotate database/OIDC/TLS secrets when exposure is possible; no third-party API keys are required by default.
5. Restore from a verified backup, run retrieval evaluations, and reopen traffic gradually.
6. Record timeline, customer impact, root cause, corrective actions, owners, and due dates.
