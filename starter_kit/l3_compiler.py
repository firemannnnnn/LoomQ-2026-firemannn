#!/usr/bin/env python3
"""LoomQ L3 hybrid compiler: Hybrid-QASM -> (quantum ops, RISC-V assembly).

Hybrid-QASM extends OpenQASM 2.0 with one `classical { ... }` block whose mini
grammar is: integer literals, register variables r1..r9 (mapped to RISC-V
x1..x9), operators `+ - == !=`, `if/else` and sequential assignments. Measured
bits c[k] are injected into x10, x11, ... by the evaluator.

This module implements a real parser/compiler (not pattern matching): the
classical block is tokenised and compiled to RISC-V assembly for the official
TinyRISCVEmulator instruction subset (li/add/sub/addi/beq/bne/j).
"""

from __future__ import annotations

import re
from typing import List, Tuple

# ---- tokeniser ------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<comment>//[^\n]*)
  | (?P<num>\d+)
  | (?P<reg>r[1-9]|x[1-9]\d*)
  | (?P<cbit>c\[\d+\])
  | (?P<eq>==)
  | (?P<ne>!=)
  | (?P<assign>=)
  | (?P<lbrace>\{)
  | (?P<rbrace>\})
  | (?P<lparen>\()
  | (?P<rparen>\))
  | (?P<semi>;)
  | (?P<comma>,)
  | (?P<plus>\+)
  | (?P<minus>-)
  | (?P<kw>[A-Za-z_][A-Za-z0-9_]*)
""",
    re.VERBOSE,
)

TOKEN_TYPES = {
    "num": "num",
    "reg": "reg",
    "cbit": "cbit",
    "eq": "eq",
    "ne": "ne",
    "assign": "assign",
    "lbrace": "lbrace",
    "rbrace": "rbrace",
    "lparen": "lparen",
    "rparen": "rparen",
    "semi": "semi",
    "plus": "plus",
    "minus": "minus",
    "kw": "kw",
}


def _tokenize(text: str) -> List[Tuple[str, str]]:
    tokens: List[Tuple[str, str]] = []
    pos = 0
    while pos < len(text):
        match = _TOKEN_RE.match(text, pos)
        if not match:
            raise ValueError(f"unexpected token near: {text[pos:pos+20]!r}")
        kind = match.lastgroup
        value = match.group()
        pos = match.end()
        if kind in ("ws", "comment"):
            continue
        tokens.append((TOKEN_TYPES[kind], value))
    return tokens


# ---- AST nodes -------------------------------------------------------------

class Assign:
    def __init__(self, target: str, expr):
        self.target = target
        self.expr = expr


class IfStmt:
    def __init__(self, left, op: str, right, then_block, else_block):
        self.left = left
        self.op = op
        self.right = right
        self.then_block = then_block
        self.else_block = else_block


class Num:
    def __init__(self, value: int):
        self.value = value


class RegRef:
    def __init__(self, name: str):
        self.name = name  # r1..r9 or x10.. (cbit mapped)


class BinaryOp:
    def __init__(self, op: str, left, right):
        self.op = op
        self.left = left
        self.right = right


class CbitRef:
    def __init__(self, index: int):
        self.index = index


# ---- classical-block parser ------------------------------------------------

class ClassicalParser:
    def __init__(self, tokens: List[Tuple[str, str]]):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else (None, None)

    def next(self):
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, kind, value=None):
        k, v = self.next()
        if k != kind or (value is not None and v != value):
            raise ValueError(f"expected {value or kind!r}, got {v!r}")
        return v

    def parse_block(self):
        self.expect("lbrace")
        stmts = []
        while True:
            k, v = self.peek()
            if k == "rbrace":
                self.next()
                break
            if k in (None, "eof"):
                raise ValueError("classical block not closed")
            stmts.append(self.parse_stmt())
        return stmts

    def parse_stmt(self):
        k, v = self.peek()
        if k == "kw" and v == "if":
            return self.parse_if()
        if k == "reg":
            return self.parse_assign()
        if k == "cbit":  # allow direct cbit read like `c[0]` in expressions
            return self.parse_assign()
        raise ValueError(f"unexpected statement start: {v!r}")

    def parse_assign(self):
        k, v = self.next()
        target = v
        self.expect("assign")
        expr = self.parse_expr()
        self.expect("semi")
        return Assign(target, expr)

    def parse_if(self):
        self.next()  # if
        self.expect("lparen")
        left = self.parse_expr()
        k, op = self.next()
        if k not in ("eq", "ne"):
            raise ValueError(f"expected == or != in if condition, got {op!r}")
        right = self.parse_expr()
        self.expect("rparen")
        then_block = self.parse_block()
        else_block = []
        k, v = self.peek()
        if k == "kw" and v == "else":
            self.next()
            else_block = self.parse_block()
        return IfStmt(left, op, right, then_block, else_block)

    # expression: additive (+/-) over atoms
    def parse_expr(self):
        left = self.parse_atom()
        while True:
            k, v = self.peek()
            if k in ("plus", "minus"):
                self.next()
                right = self.parse_atom()
                left = BinaryOp("+" if k == "plus" else "-", left, right)
            else:
                break
        return left

    def parse_atom(self):
        k, v = self.next()
        if k == "num":
            return Num(int(v))
        if k == "reg":
            return RegRef(v)
        if k == "cbit":
            return CbitRef(int(v[2:-1]))  # 'c[0]' -> '0'
        if k == "minus":  # unary minus
            atom = self.parse_atom()
            return BinaryOp("-", Num(0), atom)
        raise ValueError(f"unexpected atom: {v!r}")


# ---- RISC-V code generation -------------------------------------------------

class AsmBuilder:
    def __init__(self):
        self.lines: List[str] = []
        self.label_count = 0

    def emit(self, line: str):
        self.lines.append(line)

    def new_label(self, prefix="L"):
        self.label_count += 1
        return f"{prefix}_{self.label_count}"

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


_TEMP = "x20"  # temp register for constants in comparisons
_CBIT_BASE = 10  # c[k] -> x10+k


def _resolve_reg(name: str, cbit_base: int) -> str:
    if name.startswith("r"):
        idx = int(name[1:])
        return f"x{idx}"
    if name.startswith("x"):
        return name
    if name.startswith("c["):
        return f"x{cbit_base + int(name[2:-1])}"
    raise ValueError(f"cannot resolve register {name!r}")


def _compile_expr_to_reg(expr, builder: AsmBuilder, target: str, cbit_base: int) -> None:
    """Compile expr so its value ends up in `target`."""
    if isinstance(expr, Num):
        builder.emit(f"li {target}, {expr.value}")
    elif isinstance(expr, RegRef) or isinstance(expr, CbitRef):
        src = _resolve_reg(expr.name if isinstance(expr, RegRef) else f"c[{expr.index}]", cbit_base)
        if src != target:
            builder.emit(f"add {target}, {src}, x0")
    elif isinstance(expr, BinaryOp) and expr.op == "+":
        left, right = expr.left, expr.right
        # li/addi fast path when one side is a literal
        if isinstance(left, Num) and isinstance(right, (RegRef, CbitRef)):
            builder.emit(f"addi {target}, {_resolve_reg(right.name if isinstance(right, RegRef) else f'c[{right.index}]', cbit_base)}, {left.value}")
        elif isinstance(right, Num) and isinstance(left, (RegRef, CbitRef)):
            builder.emit(f"addi {target}, {_resolve_reg(left.name if isinstance(left, RegRef) else f'c[{left.index}]', cbit_base)}, {right.value}")
        elif isinstance(left, Num) and isinstance(right, Num):
            builder.emit(f"li {target}, {left.value + right.value}")
        else:
            lsrc = _resolve_reg(left.name, cbit_base) if isinstance(left, RegRef) else (f"x{cbit_base + left.index}" if isinstance(left, CbitRef) else None)
            rsrc = _resolve_reg(right.name, cbit_base) if isinstance(right, RegRef) else (f"x{cbit_base + right.index}" if isinstance(right, CbitRef) else None)
            if lsrc is None:
                _compile_expr_to_reg(left, builder, _TEMP, cbit_base)
                lsrc = _TEMP
            if rsrc is None:
                _compile_expr_to_reg(right, builder, target, cbit_base)
                rsrc = target
                builder.emit(f"add {target}, {lsrc}, {rsrc}")
            else:
                builder.emit(f"add {target}, {lsrc}, {rsrc}")
    elif isinstance(expr, BinaryOp) and expr.op == "-":
        left, right = expr.left, expr.right
        if isinstance(right, Num) and isinstance(left, (RegRef, CbitRef)):
            builder.emit(f"addi {target}, {_resolve_reg(left.name if isinstance(left, RegRef) else f'c[{left.index}]', cbit_base)}, {-right.value}")
        elif isinstance(left, Num) and isinstance(right, Num):
            builder.emit(f"li {target}, {left.value - right.value}")
        else:
            lsrc = _resolve_reg(left.name, cbit_base) if isinstance(left, RegRef) else (f"x{cbit_base + left.index}" if isinstance(left, CbitRef) else None)
            rsrc = _resolve_reg(right.name, cbit_base) if isinstance(right, RegRef) else (f"x{cbit_base + right.index}" if isinstance(right, CbitRef) else None)
            if lsrc is None:
                _compile_expr_to_reg(left, builder, _TEMP, cbit_base)
                lsrc = _TEMP
            if rsrc is None:
                _compile_expr_to_reg(right, builder, target, cbit_base)
                rsrc = target
                builder.emit(f"sub {target}, {lsrc}, {rsrc}")
            else:
                builder.emit(f"sub {target}, {lsrc}, {rsrc}")
    else:
        raise ValueError(f"unsupported expression: {expr!r}")


def _compile_stmt(stmt, builder: AsmBuilder, cbit_base: int) -> None:
    if isinstance(stmt, Assign):
        _compile_expr_to_reg(stmt.expr, builder, _resolve_reg(stmt.target, cbit_base), cbit_base)
        return
    if isinstance(stmt, IfStmt):
        else_label = builder.new_label("else")
        end_label = builder.new_label("end")
        # Evaluate both sides into scratch registers x20 / x21. User variables
        # only map to x1..x9 and cbits to x10+, so x20/x21 are safe.
        _compile_expr_to_reg(stmt.left, builder, _TEMP, cbit_base)
        _compile_expr_to_reg(stmt.right, builder, "x21", cbit_base)
        # Branch to else when the condition is FALSE.
        if stmt.op == "==":
            builder.emit(f"bne {_TEMP}, x21, {else_label}")
        else:  # "!="
            builder.emit(f"beq {_TEMP}, x21, {else_label}")
        for s in stmt.then_block:
            _compile_stmt(s, builder, cbit_base)
        builder.emit(f"j {end_label}")
        builder.emit(f"{else_label}:")
        for s in stmt.else_block:
            _compile_stmt(s, builder, cbit_base)
        builder.emit(f"{end_label}:")
        return
    raise ValueError(f"unsupported statement: {stmt!r}")


# ---- Hybrid-QASM splitting ---------------------------------------------------

def _split_hybrid(source: str) -> Tuple[str, str]:
    """Split Hybrid-QASM into (quantum part, classical block text)."""
    classical_match = re.search(
        r"classical\s*\{(.*)\}", source, re.DOTALL
    )
    quantum_part = source
    classical_text = ""
    if classical_match:
        classical_text = classical_match.group(1)
        quantum_part = source[: classical_match.start()] + source[classical_match.end():]
    return quantum_part, classical_text


def compile_hybrid(hybrid_qasm_str: str) -> Tuple[List[str], str]:
    """Compile a Hybrid-QASM program.

    Returns (quantum_ops, assembly):
      quantum_ops: list of quantum gate / measure instruction strings
      assembly:    RISC-V assembly text for the classical control logic
    """
    quantum_part, classical_text = _split_hybrid(hybrid_qasm_str)

    # Quantum operations: keep every non-classical, non-declaration line.
    quantum_ops: List[str] = []
    for line in quantum_part.splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith(("OPENQASM", "include", "qreg", "creg")):
            continue
        if line.endswith(";"):
            line = line[:-1]
        quantum_ops.append(line)

    builder = AsmBuilder()
    if classical_text.strip():
        tokens = _tokenize("{" + classical_text + "}")
        parser = ClassicalParser(tokens)
        stmts = parser.parse_block()
        for stmt in stmts:
            _compile_stmt(stmt, builder, _CBIT_BASE)
    return quantum_ops, builder.text()
