# Access review

1. Run `uv run python scripts/access-review.py` monthly and attach the output to the access-review ticket.
2. Repository owners confirm each subject and least-privilege role (`reader`, `writer`, or `owner`).
3. Administrators remove stale grants and review `repository.grant.updated` audit events.
4. Review Keycloak administrators, service accounts, dormant users, and OIDC client redirect URIs.
5. Retain approval evidence according to organizational policy.
