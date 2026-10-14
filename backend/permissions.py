import logging
from dataclasses import dataclass
from typing import Callable, FrozenSet, Iterable, Optional, Set

from fastapi import Depends, HTTPException

from src.plugin_api import AccountType, User, get_current_active_user, get_user, list_user_ids

from . import core_bridge, permissions_store
from .permissions_store import (  # noqa: F401
    CAPABILITIES,
    DEFAULT_CAPABILITIES,
    DEFAULT_SOURCES,
    Row,
)

logger = logging.getLogger("plugins.model-browser")

PURPOSES = ("download", "request")


@dataclass(frozen=True)
class Params:
    sources: FrozenSet[str] = frozenset()
    model_types: FrozenSet[str] = frozenset()
    max_file_gb: Optional[float] = None
    assign_to: str = "user"
    allow_remote: bool = False

    def __post_init__(self) -> None:
        for name in ("sources", "model_types"):
            object.__setattr__(self, name, frozenset(getattr(self, name)))


NO_PARAMS = Params()


@dataclass(frozen=True)
class Grant:
    capabilities: FrozenSet[str]
    sources: FrozenSet[str]
    model_types: FrozenSet[str]
    max_file_gb: Optional[float]
    assign_to: str
    allow_remote: bool
    unrestricted: bool
    request_params: Params = NO_PARAMS

    @property
    def download_params(self) -> Params:
        return Params(
            sources=self.sources,
            model_types=self.model_types,
            max_file_gb=self.max_file_gb,
            assign_to=self.assign_to,
            allow_remote=self.allow_remote,
        )

    def params(self, purpose: str = "download") -> Params:
        if purpose == "download":
            return self.download_params
        if purpose == "request":
            return self.request_params
        raise ValueError(f"unknown purpose {purpose!r}")

    def has(self, capability: str) -> bool:
        return self.unrestricted or capability in self.capabilities

    def allows_source(self, source: str, purpose: str = "download") -> bool:
        return self.unrestricted or source in self.params(purpose).sources

    def allows_type(self, model_type: str, purpose: str = "download") -> bool:
        if self.unrestricted:
            return True
        model_types = self.params(purpose).model_types
        return not model_types or model_type in model_types


@dataclass(frozen=True)
class Principal:
    user: User
    grant: Grant

    @property
    def is_admin(self) -> bool:
        return self.grant.unrestricted


ADMIN_GRANT = Grant(
    capabilities=frozenset(CAPABILITIES),
    sources=frozenset(),
    model_types=frozenset(),
    max_file_gb=None,
    assign_to="user",
    allow_remote=True,
    unrestricted=True,
)


def is_admin(user: User) -> bool:
    return user.account_type == AccountType.ADMIN


def merge_params(rows: Iterable[Row], *, remote: bool = True) -> Params:
    rows = list(rows)
    sizes = [row.max_file_gb for row in rows]
    every_type = any(not row.model_types for row in rows)
    return Params(
        sources=frozenset().union(*(row.sources for row in rows)),
        model_types=frozenset() if every_type else frozenset().union(*(row.model_types for row in rows)),
        max_file_gb=None if not sizes or None in sizes else max(sizes),
        assign_to="group" if any(row.assign_to == "group" for row in rows) else "user",
        allow_remote=remote and any(row.allow_remote for row in rows),
    )


def merge_rows(rows: Iterable[Row]) -> Grant:
    rows = list(rows)
    downloads = merge_params(row for row in rows if "download" in row.capabilities)
    requests = merge_params((row for row in rows if "request" in row.capabilities), remote=False)
    return Grant(
        capabilities=frozenset().union(*(row.capabilities for row in rows)),
        sources=downloads.sources,
        model_types=downloads.model_types,
        max_file_gb=downloads.max_file_gb,
        assign_to=downloads.assign_to,
        allow_remote=downloads.allow_remote,
        unrestricted=False,
        request_params=requests,
    )


def resolve_grant(user: User) -> Grant:
    if is_admin(user):
        return ADMIN_GRANT
    groups = core_bridge.user_groups(user.id)
    rows = permissions_store.load_rows()
    return merge_rows(permissions_store.row_or_default(rows, group["id"]) for group in groups)


def forbidden(capability: str) -> HTTPException:
    return HTTPException(
        status_code=403,
        detail={"success": False, "error": "forbidden", "message": f"This action needs the '{capability}' permission"},
    )


def require(capability: str) -> Callable[..., Principal]:
    if capability not in CAPABILITIES:
        raise ValueError(f"unknown capability {capability!r}")

    def dependency(user: User = Depends(get_current_active_user)) -> Principal:
        grant = resolve_grant(user)
        if not grant.has(capability):
            raise forbidden(capability)
        return Principal(user=user, grant=grant)

    dependency.__name__ = f"require_{capability}"
    return dependency


def require_any(*capabilities: str) -> Callable[..., Principal]:
    if not capabilities:
        raise ValueError("require_any needs at least one capability")
    unknown = [capability for capability in capabilities if capability not in CAPABILITIES]
    if unknown:
        raise ValueError(f"unknown capabilities {unknown!r}")

    def dependency(user: User = Depends(get_current_active_user)) -> Principal:
        grant = resolve_grant(user)
        if not any(grant.has(capability) for capability in capabilities):
            names = ", ".join(f"'{capability}'" for capability in capabilities)
            raise HTTPException(
                status_code=403,
                detail={
                    "success": False,
                    "error": "forbidden",
                    "message": f"This action needs one of these permissions: {names}",
                },
            )
        return Principal(user=user, grant=grant)

    dependency.__name__ = "require_any_" + "_".join(capabilities)
    return dependency


require_browse = require("browse")
require_request = require("request")
require_download = require("download")
require_approve = require("approve")


def current_principal(user: User = Depends(get_current_active_user)) -> Principal:
    return Principal(user=user, grant=resolve_grant(user))


def principal_for(user: User) -> Principal:
    return Principal(user=user, grant=resolve_grant(user))


def can_approve(principal: Principal) -> bool:
    return principal.grant.has("approve")


def user_by_id(user_id: Optional[str]) -> Optional[User]:
    if not user_id:
        return None
    try:
        return get_user(user_id)
    except Exception:
        logger.warning("[MODEL-BROWSER] Could not look user %s up", user_id, exc_info=True)
        return None


def admin_by_id(user_id: Optional[str]) -> Optional[User]:
    user = user_by_id(user_id)
    return user if user is not None and is_admin(user) else None


def acting_admin(requested_by: Optional[str], decided_by: Optional[str], group_id: str) -> Optional[User]:
    for user_id in (requested_by, decided_by):
        admin = admin_by_id(user_id)
        if admin is not None:
            return admin
    try:
        row = permissions_store.load_rows().get(group_id)
    except Exception:
        logger.warning("[MODEL-BROWSER] Could not read the permissions of user group %s", group_id, exc_info=True)
        return None
    return admin_by_id(row.updated_by) if row is not None else None


def approver_ids() -> Set[str]:
    rows = permissions_store.load_rows()
    ids: Set[str] = set()
    for group in core_bridge.list_groups():
        if "approve" in permissions_store.row_or_default(rows, group["id"]).capabilities:
            ids.update(core_bridge.group_member_ids(group["id"]))
    for user_id in list_user_ids():
        if user_id in ids:
            continue
        user = user_by_id(user_id)
        if user is not None and is_admin(user):
            ids.add(user_id)
    return ids
