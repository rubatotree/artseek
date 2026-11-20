"""Compatibility re-export of tag dictionary lookup."""

from .core.tag_dictionary import TagEntry, expand_tag_aliases, search_tags, tag_info

__all__ = ["TagEntry", "expand_tag_aliases", "search_tags", "tag_info"]
