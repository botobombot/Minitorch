from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals1 = list(vals)
    vals2 = list(vals)

    vals1[arg] = vals1[arg] + epsilon
    vals2[arg] = vals2[arg] - epsilon

    return (f(*vals1) - f(*vals2)) / (2 * epsilon)
    raise NotImplementedError('Need to implement for Task 1.1')


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    visited = set()
    order = []

    def visit(var: Variable) -> None:
        if var.unique_id in visited or var.is_constant():
            return

        visited.add(var.unique_id)

        for parent in var.parents:
            visit(parent)

        order.append(var)

    visit(variable)
    order.reverse()
    return order


def backpropagate(variable: Variable, deriv: Any) -> None:
    ordered = topological_sort(variable)
    derivatives = {variable.unique_id: deriv}

    for var in ordered:
        d = derivatives.get(var.unique_id, 0.0)

        if var.is_leaf():
            var.accumulate_derivative(d)
        else:
            for parent, parent_deriv in var.chain_rule(d):
                if parent.unique_id in derivatives:
                    derivatives[parent.unique_id] += parent_deriv
                else:
                    derivatives[parent.unique_id] = parent_deriv


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
