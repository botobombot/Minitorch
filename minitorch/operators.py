"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    return x * y
    # TODO: Implement for Task 0.1.


def id(x: float) -> float:
    return x
    # TODO: Implement for Task 0.1.


def add(x: float, y: float) -> float:
    return x + y
    # TODO: Implement for Task 0.1.


def neg(x: float) -> float:
    return -x
    # TODO: Implement for Task 0.1.

def lt(x: float, y: float) -> float:
    return 1.0 if x < y else 0.0
    # TODO: Implement for Task 0.1.
    

def eq(x: float, y: float) -> float:
    return 1.0 if x == y else 0.0
    # TODO: Implement for Task 0.1.
    

def max(x: float, y: float) -> float:
    return x if x > y else y
    # TODO: Implement for Task 0.1.
    

def is_close(x: float, y: float) -> float:
    return 1.0 if abs(x - y) < 1e-2 else 0.0
    # TODO: Implement for Task 0.1.


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)
    # TODO: Implement for Task 0.1.


def relu(x: float) -> float:
    return x if x > 0 else 0.0
    # TODO: Implement for Task 0.1.


EPS = 1e-6


def log(x: float) -> float:
    "$f(x) = log(x)$"
    return math.log(x + EPS)


def exp(x: float) -> float:
    "$f(x) = e^{x}$"
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    return d / x
    # TODO: Implement for Task 0.1.


def inv(x: float) -> float:
    return 1.0 / x
    # TODO: Implement for Task 0.1.


def inv_back(x: float, d: float) -> float:
    return -d / (x * x)
    # TODO: Implement for Task 0.1.
    

def relu_back(x: float, d: float) -> float:
    return d if x > 0 else 0.0
    # TODO: Implement for Task 0.1.
    

# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    def apply(ls: Iterable[float]) -> Iterable[float]:
        return [fn(x) for x in ls]
    return apply
    # TODO: Implement for Task 0.3

def negList(ls: Iterable[float]) -> Iterable[float]:
    return map(neg)(ls)
    # TODO: Implement for Task 0.3

def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    def apply(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        return [fn(x, y) for x, y in zip(ls1, ls2)]
    return apply
    # TODO: Implement for Task 0.3.


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    return zipWith(add)(ls1, ls2)
    # TODO: Implement for Task 0.3.


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    def apply(ls: Iterable[float]) -> float:
        result = start
        for x in ls:
            result = fn(result, x)
        return result
    return apply
    # TODO: Implement for Task 0.3.


def sum(ls: Iterable[float]) -> float:
    return reduce(add, 0.0)(ls)
    # TODO: Implement for Task 0.3.


def prod(ls: Iterable[float]) -> float:
    return reduce(mul, 1.0)(ls)
    # TODO: Implement for Task 0.3.
