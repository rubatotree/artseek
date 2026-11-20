"""Compatibility re-export of pure boolean tag query DSL."""

from .core.tag_query import (
    And,
    Atom,
    Node,
    Not,
    Or,
    TagQueryError,
    Token,
    evaluate_tag_expression,
    evaluate_tag_expression_with_universe,
    parse_tag_expression,
)

__all__ = [
    "And",
    "Atom",
    "Node",
    "Not",
    "Or",
    "TagQueryError",
    "Token",
    "evaluate_tag_expression",
    "evaluate_tag_expression_with_universe",
    "parse_tag_expression",
]
