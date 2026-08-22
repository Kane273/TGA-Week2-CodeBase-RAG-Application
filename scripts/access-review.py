import json

from app.database import RepositoryGrant, RepositoryRecord, session_scope


with session_scope() as session:
    rows = session.query(RepositoryGrant, RepositoryRecord).join(
        RepositoryRecord, RepositoryGrant.repository_id == RepositoryRecord.id
    ).all()
    print(json.dumps([
        {"repository": repository.name, "subject": grant.subject, "role": grant.role}
        for grant, repository in rows
    ], indent=2))
