import ast
import inspect
import typing
from dataclasses import dataclass
from typing import Callable, Union, Dict, Optional, Tuple, List

import astunparse  # TODO replace by astor
import black
import networkx as nx

from icontract_hypothesis_Lauren.generate_symbol_table import Table, generate_symbol_table, print_pretty_table, \
    generate_dag_from_table
from icontract_hypothesis_Lauren.property_table_to_strategies import SymbolicIntegerStrategy, SymbolicTextStrategy, \
    SymbolicFromRegexStrategy, generate_strategies


@dataclass
class StrategyFactory:
    func: Callable

    def _dependency_ordered_arguments(self) -> List[str]:
        dependency_graph = generate_dag_from_table(self.generate_property_table_without_failed_contracts())
        return list(reversed(list(nx.topological_sort(dependency_graph))))

    def _arguments_original_order(self) -> List[str]:
        return list(inspect.signature(self.func).parameters.keys())

    def _get_type_hints(self) -> List[str]:
        type_hints = []
        table = self.generate_property_table_without_failed_contracts()
        for arg in self._arguments_original_order():
            row_type = table.get_row_by_var_id(arg).type
            if typing.get_origin(row_type) == list:
                type_str = ['List']
                sub_type = typing.get_args(row_type)[0]
                while typing.get_origin(sub_type) == list:
                    type_str.append('[List')
                    sub_type = typing.get_args(sub_type)[0]
                type_str.append('[' + sub_type.__name__ + ']'*len(type_str))
                type_hints.append("".join(type_str))
            else:
                type_hints.append(table.get_row_by_var_id(arg).type.__name__)
        return type_hints

    def generate_property_table_without_failed_contracts(self) -> Table:
        _, table = generate_symbol_table(self.func)
        return table

    def generate_property_table(self) -> Tuple[List[Tuple[ast.AST, Optional[str]]], Table]:
        failed_contracts, table = generate_symbol_table(self.func)
        return failed_contracts, table

    def generate_strategies(self) -> Dict[str, Union[SymbolicIntegerStrategy, SymbolicTextStrategy, SymbolicFromRegexStrategy]]:
        try:
            table = self.generate_property_table_without_failed_contracts()
            strategies = generate_strategies(table)
        except NotImplementedError as e:
            raise e
        return strategies

    def debug_table(self):
        print_pretty_table(self.generate_property_table_without_failed_contracts())

    def generate_composite_strategy(self) -> str:
        tab = ' '*4
        strategies = self.generate_strategies()
        failed_contracts, _ = self.generate_property_table()
        ordered_var_ids = self._dependency_ordered_arguments()

        func_args = []

        type_hints = ", ".join(self._get_type_hints())

        composite_strategy = f"@hypothesis.strategies.composite\ndef strategy_{self.func.__name__}(draw) -> Tuple[{type_hints}]:\n"
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
