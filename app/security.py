from dataclasses import dataclass

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select

from app.config import get_settings
from app.database import RepositoryGrant, session_scope


@dataclass(frozen=True)
class Principal:
    subject: str
    roles: frozenset[str]


bearer = HTTPBearer(auto_error=False)


def current_principal(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> Principal:
    settings = get_settings()
    if not settings.auth_enabled:
        return Principal("local-developer", frozenset({"admin", "developer"}))
    if not credentials:
        raise HTTPException(status_code=401, detail="Bearer token required")
    try:
        jwks = jwt.PyJWKClient(settings.oidc_jwks_url or
                              f"{settings.oidc_issuer}/protocol/openid-connect/certs")
        key = jwks.get_signing_key_from_jwt(credentials.credentials)
        claims = jwt.decode(credentials.credentials, key.key, algorithms=["RS256"],
                            audience=settings.oidc_audience, issuer=settings.oidc_issuer)
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid identity token") from exc
    roles = claims.get("realm_access", {}).get("roles", [])
    return Principal(str(claims["sub"]), frozenset(roles))


def require_admin(principal: Principal = Depends(current_principal)) -> Principal:
    if "admin" not in principal.roles:
        raise HTTPException(status_code=403, detail="Administrator role required")
    return principal


def authorize_repository(principal: Principal, repository_id: str, write: bool = False) -> None:
    if "admin" in principal.roles:
        return
    with session_scope() as session:
        grant = session.scalar(select(RepositoryGrant).where(
            RepositoryGrant.repository_id == repository_id,
            RepositoryGrant.subject == principal.subject,
        ))
    permitted = {"owner", "writer"} if write else {"owner", "writer", "reader"}
    if not grant or grant.role not in permitted:
        raise HTTPException(status_code=403, detail="Repository access denied")
