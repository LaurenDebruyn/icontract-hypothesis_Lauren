import ast
import importlib
import inspect
import textwrap
import typing
from dataclasses import dataclass
from typing import Callable, Union, Dict, Optional, Tuple, List, TypeVar, Type
import icontract
import hypothesis
import astunparse
import black
import networkx as nx
from hypothesis.strategies import SearchStrategy
from mypy.build import Any
import regex as re
import sys

import hypothesis.strategies._internal.types

from icontract_hypothesis_Lauren.generate_property_table import Table, generate_property_table, print_pretty_table, \
    generate_dag_from_table
from icontract_hypothesis_Lauren.property_table_to_strategies import generate_strategies, SymbolicStrategy

T = TypeVar("T")  # pylint: disable=invalid-name


def type_hint_to_str(type_hint: Type) -> str:
    type_hints: List[str] = []
    is_optional = False
    if typing.get_origin(type_hint):
        origin_type_hint = typing.get_origin(type_hint)
        if isinstance(origin_type_hint, typing._SpecialForm):
            type_hints.append(origin_type_hint._name)
        elif sys.version_info < (3, 9) and origin_type_hint in [list, set, tuple, dict]:
            type_hints.append(origin_type_hint.__name__.capitalize())
        else:
            type_hints.append(origin_type_hint.__name__)
        type_hint_args: List[str] = []
        for type_hint_arg in typing.get_args(type_hint):
            if type_hint_arg == type(None):
                is_optional = True
            else:
                type_hint_args.append(type_hint_to_str(type_hint_arg))
        type_hints.append("[" + ", ".join(type_hint_args) + "]")
    else:
        type_hints.append(type_hint.__name__)

    result = "".join(type_hints)
    if is_optional:
        return f"Optional[{result}]"
    return "".join(type_hints)


@dataclass
class StrategyFactory:
    func: Callable
    localns: Optional[Dict[str, Any]] = None
    globalns: Optional[Dict[str, Any]] = None


    def _dependency_ordered_arguments(self) -> List[str]:
        dependency_graph = generate_dag_from_table(self.generate_property_table_without_failed_contracts())
        return list(reversed(list(nx.topological_sort(dependency_graph))))

    def _arguments_original_order(self) -> List[str]:
        args = list(inspect.signature(self.func).parameters.keys())
        if 'self' in args:
            args.remove('self')
        elif 'cls' in args:
            args.remove('cls')
        type_hints = typing.get_type_hints(self.func)
        return [arg for arg in args if arg in type_hints]

    def _get_type_hints(self) -> List[str]:
        type_hints = []
        table = self.generate_property_table_without_failed_contracts()
        for arg in self._arguments_original_order():
            row_type = table.get_row_by_var_id(arg).type
            type_hints.append(type_hint_to_str(row_type))
        return type_hints

    def generate_property_table_without_failed_contracts(self) -> Table:
        _, table = generate_property_table(self.func)
        return table

    def generate_property_table(self) -> Tuple[List[Tuple[ast.AST, Optional[str]]], Table]:
        failed_contracts, table = generate_property_table(self.func)
        return failed_contracts, table

    def generate_strategies(self) -> Dict[str, Union[SymbolicStrategy]]:
        try:
            table = self.generate_property_table_without_failed_contracts()
            strategies = generate_strategies(table)
        except NotImplementedError as e:
            raise e
        return strategies

    def debug_table(self):
        print_pretty_table(self.generate_property_table_without_failed_contracts())

    def generate_composite_strategy(self) -> str:
        tab = ' ' * 4
        strategies = self.generate_strategies()
        failed_contracts, _ = self.generate_property_table()
        ordered_var_ids = self._dependency_ordered_arguments()

        func_args = []

        type_hints = ", ".join(self._get_type_hints())

        if re.match(r'__(.*)__', self.func.__name__):
            strategy_name = f"strategy_{self.func.__qualname__.lower().replace('.', '')}"
        else:
            strategy_name = f"strategy_{self.func.__name__}"

        if len(self._get_type_hints()) == 0:
            return ''
        composite_strategy = f"@hypothesis.strategies.composite\ndef {strategy_name}(draw) -> Tuple[{type_hints}]:\n"
        for var_id in ordered_var_ids:
            strategy = strategies[var_id]
            composite_strategy += f"{tab}{var_id} = draw({strategy.represent()})\n"
            func_args.append(var_id)
        for failed_contract in failed_contracts:
            failed_contract_str = astunparse.unparse(failed_contract[0]).rstrip('\n')
            composite_strategy += f"{tab}hypothesis.assume({failed_contract_str})\n"
        composite_strategy += f"{tab}return "
        composite_strategy += ", ".join(self._arguments_original_order())
        return black.format_str(composite_strategy, mode=black.FileMode())

    def generate_hypothesis_strategy(self) -> SearchStrategy:
        strategy_str = self.generate_composite_strategy()
        if self.localns:
            for k in self.localns.keys():
                if k not in locals().keys():
                    locals()[k] = self.localns[k]
        if self.globalns:
            for k in self.globalns.keys():
                if k not in globals().keys():
                    globals()[k] = self.globalns[k]
        try:
            exec(strategy_str)
        except NameError as e:
            raise NameError(textwrap.dedent(
                f"""{e.args[0]}
                    Have you included the global namespace in the constructor of StrategyFactory?"""
            ))

        if re.match(r'__(.*)__', self.func.__name__):
            strategy_name = f"strategy_{self.func.__qualname__.lower().replace('.', '')}"
        else:
            strategy_name = f"strategy_{self.func.__name__}"

        strategy = locals()[strategy_name]
        assert isinstance(strategy(), SearchStrategy)
        return strategy()


def _register_with_hypothesis(cls: Type[T]) -> None:
    """
    Register ``cls`` with Hypothesis based on the inferred strategy.
    The registration is necessary so that the preconditions on the __init__ are propagated
    in ``hypothesis.strategies.builds``.
    """
    # We should not register abstract classes as this will mislead Hypothesis to instantiate
    # them. However, if a class provides a ``__new__``, we can instantiate it even if it is
    # abstract!
    # pylint: disable=comparison-with-callable
    if inspect.isabstract(cls) and getattr(cls, "__new__") == object.__new__:
        return

    if cls in hypothesis.strategies._internal.types._global_type_lookup:
        # Do not re-register
        return

    init = getattr(cls, "__init__")

    if inspect.isfunction(init):
        # If there is is no checker nor pre-conditions on ``__init__``, we should simply delegate
        # the case to Hypothesis and not register the strategy for this type.

        checker = icontract._checkers.find_checker(init)
        if checker is None:
            return

        maybe_preconditions = getattr(checker, "__preconditions__", None)
        if maybe_preconditions is None or len(maybe_preconditions) == 0:
            return

        # We have to add the ``a_type`` itself to a local namespace for forward references.
        #
        # This is needed if we register a class through the ``icontract.DBCMeta``
        # meta-class where it references itself. For example, a node in a linked list.

        strategy_factory = StrategyFactory(init, localns={cls.__name__: cls})
        strategy = strategy_factory.generate_hypothesis_strategy()

    elif isinstance(init, icontract._checkers._SLOT_WRAPPER_TYPE):
        # We have to distinguish this special case which is used by named tuples and
        # possibly other optimized data structures.
        # In those cases, we have to infer the strategy based on __new__ instead of __init__.

        # If there is is no checker nor pre-conditions on ``__init__``, we should simply delegate
        # the case to Hypothesis and not register the strategy for this type.

        new = getattr(cls, "__new__")

        assert (
            new is not None
        ), "Expected __new__ in {} if __init__ is a slot wrapper.".format(cls)

        # We have to add the ``a_type`` itself to a local namespace for forward references.
        # This is usually the case for the return value of ``__new__``.
        #
        # In particular, the class is not available in the module while we are
        # registering it through the ``icontract.DBCMeta`` meta-class as its loading is still
        # in progress.

        strategy_factory = StrategyFactory(new, localns={cls.__name__: cls})
        if len(strategy_factory._get_type_hints()) == 0:
            return
        strategy = strategy_factory.generate_hypothesis_strategy()
    else:
        raise AssertionError(
            "Expected __init__ to be either a function or a slot wrapper, but got: {}".format(
                type(init)
            )
        )
    # strategy = hypothesis.strategies.builds(cls, strategy)
    if len(strategy_factory._get_type_hints()) > 1:
        pack_repr = f"lambda d: {cls.__name__}(*d)"

        pack = lambda d: cls(*d)  # type: ignore
    else:
        pack_repr = f"lambda d: {cls.__name__}(d)"

        pack = lambda d: cls(d)  # type: ignore
    pack.__icontract_hypothesis_source_code__ = pack_repr  # type: ignore

    strategy = strategy.map(pack=pack)
    hypothesis.strategies.register_type_strategy(custom_type=cls, strategy=strategy)


def hook_into_icontract_and_hypothesis(localns: Optional[Dict[str, Any]] = None, globalns: Optional[Dict[str, Any]] = None) -> None:
    """
    Redirect ``icontract._metaclass._register_for_hypothesis``.
    All the classes previously registered by icontract are now re-registered
    by ``_register_with_hypothesis``.
    """
    if not hasattr(icontract._metaclass, "_CONTRACT_CLASSES"):
        return  # already hooked in

    icontract._metaclass._register_for_hypothesis = _register_with_hypothesis

    if localns:
        for k in localns.keys():
            if k not in locals().keys():
                locals()[k] = localns[k]
    if globalns:
        for k in globalns.keys():
            if k not in globals().keys():
                globals()[k] = globalns[k]

    for cls in icontract._metaclass._CONTRACT_CLASSES:
        _register_with_hypothesis(cls)

    # Delete in order to fail fast
    del icontract._metaclass._CONTRACT_CLASSES

    # Monkey-patch lambda source so that we do not have to introduce
    # strategy classes just for this functionality.
    # See https://github.com/HypothesisWorks/hypothesis/issues/2713
    upstream_extract_lambda_source = (
        hypothesis.internal.reflection.extract_lambda_source
    )
    hypothesis.internal.reflection.extract_lambda_source = lambda f: (
        getattr(f, "__icontract_hypothesis_source_code__", None)
        or upstream_extract_lambda_source(f)  # type: ignore
    )
