import ast
from dataclasses import dataclass
from typing import Callable, Union, Dict, Optional, Tuple, List

import astunparse
from icontract import require

from icontract_hypothesis_Lauren.generate_symbol_table import Table, generate_symbol_table
from icontract_hypothesis_Lauren.property_table_to_strategies import SymbolicIntegerStrategy, SymbolicTextStrategy, \
    SymbolicFromRegexStrategy, generate_strategies


@dataclass
class StrategyFactory:
    func: Callable

    def generate_property_table_without_failed_contracts(self) -> Table:
        _, table = generate_symbol_table(self.func)
        return table

    def generate_property_table(self) -> Tuple[List[Tuple[ast.AST, Optional[str]]], Table]:
        failed_contracts, table = generate_symbol_table(self.func)
        return failed_contracts, table

    def generate_strategies(self) -> Dict[str, Union[SymbolicIntegerStrategy, SymbolicTextStrategy, SymbolicFromRegexStrategy]]:
        table = self.generate_property_table_without_failed_contracts()
        return generate_strategies(table)

    def generate_composite_strategy(self) -> str:
        # TODO take into account dependency graph
        tab = ' '*4
        strategies = self.generate_strategies()
        failed_contracts, _ = self.generate_property_table()

        func_args = []

        composite_strategy = f"@st.composite\ndef test_strategy(draw) -> ...:\n"
        for var_id, strategy in strategies.items():
            composite_strategy += f"{tab}{var_id} = draw({strategy.get_strategy()})\n"
            func_args.append(var_id)
        for failed_contract in failed_contracts:
            failed_contract_str = astunparse.unparse(failed_contract[0]).rstrip('\n')
            composite_strategy += f"{tab}assume({failed_contract_str})\n"
        composite_strategy += f"{tab}return "
        composite_strategy += ", ".join(func_args)
        composite_strategy += "\n"
        return composite_strategy


@require(lambda n1, n2: n1 < n2 and n2 >= 100)
@require(lambda n1: n1 % 2 == 0)
def int_func_with_assume(n1: int, n2: int):
    pass


@require(lambda s: s.isidentifier())
def str_func_with_filter(s: str):
    pass


@require(lambda n1, n2: n1 < n2 and n1 >= 0 and n1 <= 1000 and n2 <= 750)
@require(lambda s: s.startswith('abc') and s.contains('hey'))
def int_str_func(n1: int, n2: int, s: str):
    pass


if __name__ == '__main__':
    # strategy_factory = StrategyFactory(int_func_with_assume)
    strategy_factory = StrategyFactory(str_func_with_filter)
    # strategy_factory = StrategyFactory(int_str_func)
    print(strategy_factory.generate_composite_strategy())
