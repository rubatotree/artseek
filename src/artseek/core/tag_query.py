"""Pure boolean `#tag` expression parser and evaluator."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


class TagQueryError(ValueError):
    def __init__(self, message: str, position: int = 0):
        super().__init__(message)
        self.position = position


@dataclass(frozen=True)
class Atom:
    value: str


@dataclass(frozen=True)
class Not:
    child: Node


@dataclass(frozen=True)
class And:
    left: Node
    right: Node


@dataclass(frozen=True)
class Or:
    left: Node
    right: Node


Node = Atom | Not | And | Or


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    position: int


def _tokens(text: str) -> list[Token]:
    result: list[Token] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char.isspace():
            index += 1
            continue
        if char in "()+-*&|!":
            result.append(Token(char, char, index))
            index += 1
            continue
        if char == "#":
            start = index
            index += 1
            if index < len(text) and text[index] == '"':
                index += 1
                value_start = index
                while index < len(text) and text[index] != '"':
                    index += 1
                if index >= len(text):
                    raise TagQueryError("tag 引号未闭合", index)
                value = text[value_start:index]
                index += 1
            else:
                value_start = index
                while index < len(text) and (text[index].isalnum() or text[index] in "_-.:/"):
                    if text[index] == "-" and (
                        index + 1 >= len(text)
                        or text[index + 1].isspace()
                        or text[index + 1] == "#"
                    ):
                        break
                    index += 1
                value = text[value_start:index]
            if not value:
                raise TagQueryError("缺少 tag 名称", start)
            result.append(Token("atom", value, start))
            continue
        if char.isalpha():
            start = index
            while index < len(text) and text[index].isalpha():
                index += 1
            word = text[start:index].casefold()
            if word in {"and", "or", "not"}:
                result.append(Token(word, word, start))
            else:
                raise TagQueryError(f"未知词：{text[start:index]}；tag 请使用 # 前缀", start)
            continue
        raise TagQueryError(f"无法识别字符：{char}", index)
    result.append(Token("eof", "", len(text)))
    return result


class _Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.index = 0
        self.nodes = 0

    def peek(self) -> Token:
        return self.tokens[self.index]

    def take(self) -> Token:
        token = self.tokens[self.index]
        self.index += 1
        return token

    def node(self, value: Node) -> Node:
        self.nodes += 1
        if self.nodes > 200:
            raise TagQueryError("表达式过于复杂", self.peek().position)
        return value

    def parse(self) -> Node:
        node = self.parse_or()
        if self.peek().kind != "eof":
            raise TagQueryError(f"多余内容：{self.peek().value}", self.peek().position)
        return node

    def parse_or(self) -> Node:
        node = self.parse_and()
        while self.peek().kind in {"+", "or", "|"}:
            self.take()
            node = self.node(Or(node, self.parse_and()))
        return node

    def parse_and(self) -> Node:
        node = self.parse_unary()
        while True:
            token = self.peek()
            if token.kind in {"*", "and", "&"}:
                self.take()
                node = self.node(And(node, self.parse_unary()))
                continue
            if token.kind in {"atom", "not", "-", "!"}:
                node = self.node(And(node, self.parse_unary()))
                continue
            return node

    def parse_unary(self) -> Node:
        token = self.peek()
        if token.kind in {"-", "not", "!"}:
            self.take()
            return self.node(Not(self.parse_unary()))
        if token.kind == "(":
            self.take()
            node = self.parse_or()
            if self.peek().kind != ")":
                raise TagQueryError("缺少右括号", self.peek().position)
            self.take()
            return node
        if token.kind == "atom":
            self.take()
            return self.node(Atom(token.value))
        raise TagQueryError(f"这里需要 tag 或括号，得到：{token.value or '结尾'}", token.position)


def parse_tag_expression(text: str) -> Node:
    if not str(text or "").strip():
        raise TagQueryError("tag 表达式为空")
    return _Parser(_tokens(str(text))).parse()


def evaluate_tag_expression(
    node: Node,
    resolve_atom: Callable[[str], set[str]],
) -> set[str]:
    if isinstance(node, Atom):
        return set(resolve_atom(node.value))
    if isinstance(node, Not):
        raise TagQueryError("NOT 必须与范围一起求值")
    if isinstance(node, And):
        return evaluate_tag_expression(node.left, resolve_atom) & evaluate_tag_expression(
            node.right, resolve_atom
        )
    if isinstance(node, Or):
        return evaluate_tag_expression(node.left, resolve_atom) | evaluate_tag_expression(
            node.right, resolve_atom
        )
    raise TagQueryError("未知表达式节点")


def evaluate_tag_expression_with_universe(
    node: Node,
    resolve_atom: Callable[[str], set[str]],
    universe: set[str],
) -> set[str]:
    if isinstance(node, Atom):
        return set(resolve_atom(node.value))
    if isinstance(node, Not):
        return universe - evaluate_tag_expression_with_universe(
            node.child, resolve_atom, universe
        )
    if isinstance(node, And):
        return evaluate_tag_expression_with_universe(
            node.left, resolve_atom, universe
        ) & evaluate_tag_expression_with_universe(node.right, resolve_atom, universe)
    if isinstance(node, Or):
        return evaluate_tag_expression_with_universe(
            node.left, resolve_atom, universe
        ) | evaluate_tag_expression_with_universe(node.right, resolve_atom, universe)
    raise TagQueryError("未知表达式节点")
