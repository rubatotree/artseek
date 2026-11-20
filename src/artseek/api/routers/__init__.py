"""Domain-split FastAPI routers for ArtSeek v2."""

from . import (
    artists,
    collections,
    gallery,
    jobs,
    media,
    notes,
    pages,
    provenance,
    search,
    system,
    tags,
)

ALL_ROUTERS = (
    pages.router,
    gallery.router,
    search.router,
    media.router,
    collections.router,
    provenance.router,
    artists.router,
    tags.router,
    jobs.router,
    system.router,
    notes.router,
)

__all__ = ["ALL_ROUTERS"]
