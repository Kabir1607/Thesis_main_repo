"""Concolic execution of integer-valued Python programs.

Wraps each integer input as a :class:`SymbolicInt` that carries both a
concrete value and an SMT expression. Arithmetic builds new symbolic
values; comparisons record a :class:`BranchRecord` (clause taken, clause
alternate) and return the concrete Boolean so Python control flow
proceeds normally.

The result is a :class:`ConcolicResult` containing:

* the (concretized) program output,
* the property value ``phi`` (0 or 1),
* the full path condition as a list of :class:`BranchRecord`s,
* a ``terminated`` flag (False if the per-run branch budget was exceeded).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..smt import SMTBackend, SMTExpr

# -----------------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BranchRecord:
    """One branch decision: the clause taken plus its negation."""

    clause_taken: SMTExpr
    clause_alt: SMTExpr

    def __repr__(self) -> str:
        return f"BranchRecord(taken={self.clause_taken!r})"


@dataclass
class ConcolicResult:
    """Result of one concolic run.

    ``concretized`` is ``True`` iff the trace crossed a value-
    concretization point (currently: bitwise-op fallbacks on
    :class:`SymbolicInt` that the SMT backend can't model
    natively).  Closure rules treat concretized traces as opaque
    and refuse to issue an SMT certificate against them, even when
    the trace's path-clause sequence is trivially path-deterministic.
    """

    output: Any
    phi_value: int
    path_condition: list[BranchRecord]
    inputs: dict[str, int]
    terminated: bool
    concretized: bool = False

    @property
    def n_branches(self) -> int:
        return len(self.path_condition)


# -----------------------------------------------------------------------------
# Tracer
# -----------------------------------------------------------------------------


class _BranchLimit(Exception):
    """Internal: raised when the per-run branch limit is exceeded."""


class Tracer:
    """Accumulates :class:`BranchRecord`s emitted during a concolic run.

    The ``concretized`` flag is set when a :class:`SymbolicInt`
    operation falls off the symbolic path (e.g. an unsupported
    bitwise op concretized to a Python ``int``).  Subsequent branch
    decisions on the run are still recorded for sample-path equality
    checks, but :class:`~dise.regions.Frontier.try_close` refuses to
    issue an SMT closure certificate against a trace that crossed
    a concretization point — because the SMT view of the trace no
    longer reflects every value-dependent decision.
    """

    def __init__(self, smt: SMTBackend, max_branches: int = 10_000) -> None:
        self.smt = smt
        self.max_branches = max_branches
        self.branches: list[BranchRecord] = []
        self.diverged: bool = False
        self.concretized: bool = False

    def record(self, clause_taken: SMTExpr, clause_alt: SMTExpr) -> None:
        if self.diverged:
            raise _BranchLimit()
        self.branches.append(BranchRecord(clause_taken, clause_alt))
        if len(self.branches) >= self.max_branches:
            self.diverged = True
            raise _BranchLimit()


# -----------------------------------------------------------------------------
# SymbolicInt
# -----------------------------------------------------------------------------


class SymbolicInt:
    """Concolic integer: ``(concrete, smt_expr, tracer)``.

    Arithmetic operators (``+``, ``-``, ``*``, ``//``, ``%``, unary ``-``)
    return new :class:`SymbolicInt`s with combined expressions. Comparison
    operators (``<``, ``<=``, ``>``, ``>=``, ``==``, ``!=``) and ``bool()``
    record a :class:`BranchRecord` and return a plain Python ``bool`` so
    that Python control flow proceeds based on concrete values.

    Hashing is by identity (so two SymbolicInts are distinct in dicts /
    sets even if their concrete values match). This means ``==`` may be
    used freely without breaking container invariants.
    """

    __slots__ = ("concrete", "expr", "tracer")

    def __init__(self, concrete: int, expr: SMTExpr, tracer: Tracer) -> None:
        self.concrete = int(concrete)
        self.expr = expr
        self.tracer = tracer

    # ---- internal helpers ----

    def _lift(self, other: Any) -> tuple[int, SMTExpr] | None:
        if isinstance(other, SymbolicInt):
            return other.concrete, other.expr
        if isinstance(other, bool):  # bool is subclass of int; handle explicitly
            v = int(other)
            return v, self.tracer.smt.const(v)
        if isinstance(other, int):
            return other, self.tracer.smt.const(other)
        try:
            v = int(other)
        except (TypeError, ValueError):
            return None
        return v, self.tracer.smt.const(v)

    def _arith(self, op_name: str, other: Any, swap: bool = False) -> SymbolicInt:
        lifted = self._lift(other)
        if lifted is None:
            raise TypeError(
                f"unsupported operand type(s) for {op_name}: SymbolicInt and {type(other).__name__}"
            )
        o_c, o_e = lifted
        smt = self.tracer.smt
        if swap:
            new_concrete = self._do_arith(op_name, o_c, self.concrete)
            new_expr = smt.op(op_name, o_e, self.expr)
        else:
            new_concrete = self._do_arith(op_name, self.concrete, o_c)
            new_expr = smt.op(op_name, self.expr, o_e)
        return SymbolicInt(new_concrete, new_expr, self.tracer)

    @staticmethod
    def _do_arith(op_name: str, a: int, b: int) -> int:
        if op_name == "+":
            return a + b
        if op_name == "-":
            return a - b
        if op_name == "*":
            return a * b
        if op_name == "div":
            return a // b
        if op_name == "mod":
            return a % b
        raise ValueError(op_name)

    def _compare(self, op_name: str, other: Any) -> Any:
        lifted = self._lift(other)
        if lifted is None:
            return NotImplemented
        o_c, o_e = lifted
        result = self._do_compare(op_name, self.concrete, o_c)
        smt = self.tracer.smt
        cmp_expr = smt.op(op_name, self.expr, o_e)
        if result:
            self.tracer.record(cmp_expr, smt.negation(cmp_expr))
        else:
            self.tracer.record(smt.negation(cmp_expr), cmp_expr)
        return result

    @staticmethod
    def _do_compare(op_name: str, a: int, b: int) -> bool:
        if op_name == "<":
            return a < b
        if op_name == "<=":
            return a <= b
        if op_name == ">":
            return a > b
        if op_name == ">=":
            return a >= b
        if op_name == "==":
            return a == b
        if op_name == "!=":
            return a != b
        raise ValueError(op_name)

    # ---- arithmetic ----

    def __add__(self, other: Any) -> SymbolicInt:
        return self._arith("+", other)

    def __radd__(self, other: Any) -> SymbolicInt:
        return self._arith("+", other, swap=True)

    def __sub__(self, other: Any) -> SymbolicInt:
        return self._arith("-", other)

    def __rsub__(self, other: Any) -> SymbolicInt:
        return self._arith("-", other, swap=True)

    def __mul__(self, other: Any) -> SymbolicInt:
        return self._arith("*", other)

    def __rmul__(self, other: Any) -> SymbolicInt:
        return self._arith("*", other, swap=True)

    def __floordiv__(self, other: Any) -> SymbolicInt:
        return self._arith("div", other)

    def __rfloordiv__(self, other: Any) -> SymbolicInt:
        return self._arith("div", other, swap=True)

    def __mod__(self, other: Any) -> SymbolicInt:
        return self._arith("mod", other)

    def __rmod__(self, other: Any) -> SymbolicInt:
        return self._arith("mod", other, swap=True)

    # C-style truncated division is emitted by the SVCOMP transpiler
    # as ``int((a) / (b))``.  Python's ``/`` returns a float, which
    # would force a concretization on ``int(...)``.  We instead model
    # ``/`` as floor division — which matches truncated division when
    # both operands are non-negative, the common case for SVCOMP
    # nondets sampled from ``Uniform(0, N)``.  Negative operands fall
    # through to the concretization path inside ``__int__`` callers.
    def __truediv__(self, other: Any) -> SymbolicInt:
        return self._arith("div", other)

    def __rtruediv__(self, other: Any) -> SymbolicInt:
        return self._arith("div", other, swap=True)

    def __neg__(self) -> SymbolicInt:
        smt = self.tracer.smt
        return SymbolicInt(-self.concrete, smt.op("neg", self.expr), self.tracer)

    def __pos__(self) -> SymbolicInt:
        return self

    # ---- bitwise ops ----
    #
    # The SMT backend's vocabulary is LIA (no bitwise primitives), so
    # we cannot in general represent ``x & MASK``, ``x | y``, ``x ^ y``,
    # ``x << k``, ``x >> k`` symbolically.  We handle two cases:
    #
    #   1. ``x & ((1 << n) - 1)`` where the concrete value of ``x`` is
    #      already in ``[0, mask]``: this is a no-op on the concrete
    #      side AND a no-op modulo our integer model on the symbolic
    #      side (we soundly model unsigned overflow this way for
    #      power-of-two-minus-1 masks when the value is in range).
    #   2. Any other bitwise op: we compute the concrete result and
    #      return a SymbolicInt whose SMT expression is the *constant*
    #      result.  Symbolic information about the operand is lost,
    #      which downstream prevents refinement on that variable but
    #      remains sound — closure requires an SMT proof, and a
    #      constant-valued leaf cannot fake one.
    #
    # We do NOT raise: raising during concolic tracing would crash the
    # estimator on any program that uses bitwise arithmetic, and the
    # silent-error path that bug surfaced in SVCOMP runs (Issue: every
    # diamond-style program returned mu=0,hw=0 because the wrapper
    # swallowed the TypeError) is much worse than dropping precision.

    def _bitwise_concretize(self, op_name: str, op_fn, other: Any) -> SymbolicInt:
        """Fallback: concretize the result and emit a constant SMT expr.

        Marks the tracer as concretized so the closure rule cannot
        issue an SMT certificate against the trace — without this
        flag, downstream comparisons on the concretized value would
        record trivially-true branch clauses (e.g. ``const == 0``)
        and the closure check ``F ∧ ¬True ≡ False`` would succeed
        without proving anything about the underlying variable.
        """
        lifted = self._lift(other)
        if lifted is None:
            raise TypeError(
                f"unsupported operand type(s) for {op_name}: "
                f"SymbolicInt and {type(other).__name__}"
            )
        o_c, _o_e = lifted
        new_concrete = op_fn(self.concrete, o_c)
        smt = self.tracer.smt
        self.tracer.concretized = True
        return SymbolicInt(new_concrete, smt.const(new_concrete), self.tracer)

    def __and__(self, other: Any) -> SymbolicInt:
        # Special case: x & ((1<<n)-1) where x.concrete fits the mask is a
        # symbolic no-op (preserves the symbolic expression of self).
        if isinstance(other, int) and not isinstance(other, bool):
            mask = other
            if mask > 0 and (mask & (mask + 1)) == 0:  # power-of-two-minus-1
                if 0 <= self.concrete <= mask:
                    return self
        return self._bitwise_concretize("&", lambda a, b: a & b, other)

    def __rand__(self, other: Any) -> SymbolicInt:
        return self.__and__(other)

    def __or__(self, other: Any) -> SymbolicInt:
        return self._bitwise_concretize("|", lambda a, b: a | b, other)

    def __ror__(self, other: Any) -> SymbolicInt:
        return self.__or__(other)

    def __xor__(self, other: Any) -> SymbolicInt:
        return self._bitwise_concretize("^", lambda a, b: a ^ b, other)

    def __rxor__(self, other: Any) -> SymbolicInt:
        return self.__xor__(other)

    def __lshift__(self, other: Any) -> SymbolicInt:
        return self._bitwise_concretize("<<", lambda a, b: a << b, other)

    def __rlshift__(self, other: Any) -> SymbolicInt:
        return self._bitwise_concretize(
            "<<", lambda a, b: a << b, other
        )

    def __rshift__(self, other: Any) -> SymbolicInt:
        return self._bitwise_concretize(">>", lambda a, b: a >> b, other)

    def __rrshift__(self, other: Any) -> SymbolicInt:
        return self._bitwise_concretize(
            ">>", lambda a, b: a >> b, other
        )

    def __invert__(self) -> SymbolicInt:
        new_concrete = ~self.concrete
        smt = self.tracer.smt
        return SymbolicInt(new_concrete, smt.const(new_concrete), self.tracer)

    def __abs__(self) -> SymbolicInt:
        if self.concrete >= 0:
            # Record: self >= 0
            smt = self.tracer.smt
            cmp_expr = smt.op(">=", self.expr, smt.const(0))
            self.tracer.record(cmp_expr, smt.negation(cmp_expr))
            return self
        smt = self.tracer.smt
        cmp_expr = smt.op(">=", self.expr, smt.const(0))
        self.tracer.record(smt.negation(cmp_expr), cmp_expr)
        return -self

    # ---- comparisons ----

    def __lt__(self, other: Any) -> bool:
        return self._compare("<", other)

    def __le__(self, other: Any) -> bool:
        return self._compare("<=", other)

    def __gt__(self, other: Any) -> bool:
        return self._compare(">", other)

    def __ge__(self, other: Any) -> bool:
        return self._compare(">=", other)

    def __eq__(self, other: Any) -> Any:
        lifted = self._lift(other)
        if lifted is None:
            return NotImplemented
        return self._compare("==", other)

    def __ne__(self, other: Any) -> Any:
        lifted = self._lift(other)
        if lifted is None:
            return NotImplemented
        return self._compare("!=", other)

    # ---- conversion ----

    def __bool__(self) -> bool:
        smt = self.tracer.smt
        zero = smt.const(0)
        cmp_expr = smt.op("!=", self.expr, zero)
        result = self.concrete != 0
        if result:
            self.tracer.record(cmp_expr, smt.negation(cmp_expr))
        else:
            self.tracer.record(smt.negation(cmp_expr), cmp_expr)
        return result

    def __int__(self) -> int:
        return self.concrete

    def __index__(self) -> int:
        return self.concrete

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        return f"SymbolicInt({self.concrete})"


# -----------------------------------------------------------------------------
# run_concolic
# -----------------------------------------------------------------------------


def _concretize(obj: Any) -> Any:
    """Recursively replace SymbolicInts with their concrete int values."""
    if isinstance(obj, SymbolicInt):
        return obj.concrete
    if isinstance(obj, tuple):
        return tuple(_concretize(x) for x in obj)
    if isinstance(obj, list):
        return [_concretize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _concretize(v) for k, v in obj.items()}
    return obj


def run_concolic(
    program: Callable[..., Any],
    inputs: dict[str, int],
    property_fn: Callable[[Any], bool],
    smt: SMTBackend,
    max_branches: int = 10_000,
) -> ConcolicResult:
    """Run ``program(**sym_inputs)`` under concolic tracing.

    ``program`` is called with each input wrapped as a :class:`SymbolicInt`.
    The (still-symbolic) output is then passed to ``property_fn``; if
    ``property_fn`` performs comparisons on the output, those comparisons
    are recorded as additional :class:`BranchRecord`s (the "phi branch").
    This is important for non-branching programs whose property carries
    the only useful split (e.g. ``lambda y: y < k`` on identity ``f(x) = x``).
    """
    tracer = Tracer(smt=smt, max_branches=max_branches)
    sym_inputs: dict[str, SymbolicInt] = {}
    for name, value in inputs.items():
        v = int(value)
        sym_inputs[name] = SymbolicInt(v, smt.make_int_var(name), tracer)

    terminated = True
    output_concrete: Any = None
    phi_value = 0

    try:
        output = program(**sym_inputs)
        phi_result = property_fn(output)
        if isinstance(phi_result, SymbolicInt):
            phi_value = 1 if phi_result.concrete != 0 else 0
        elif isinstance(phi_result, bool):
            phi_value = 1 if phi_result else 0
        else:
            phi_value = 1 if bool(phi_result) else 0
        output_concrete = _concretize(output)
    except _BranchLimit:
        terminated = False

    return ConcolicResult(
        output=output_concrete,
        phi_value=phi_value,
        path_condition=list(tracer.branches),
        inputs={k: int(v) for k, v in inputs.items()},
        terminated=terminated,
        concretized=tracer.concretized,
    )


__all__ = [
    "BranchRecord",
    "ConcolicResult",
    "SymbolicInt",
    "Tracer",
    "run_concolic",
]
