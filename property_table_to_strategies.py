import abc
import ast
import typing
from dataclasses import dataclass
from typing import List, Union, Dict, Tuple, Set, Sequence, Callable, Hashable, Optional

import networkx as nx
from hypothesis.strategies._internal.strategies import Ex

import icontract_hypothesis_Lauren.generate_symbol_table
from icontract_hypothesis_Lauren import generate_symbol_table
from icontract_hypothesis_Lauren.generate_symbol_table import Lambda, property_as_lambdas, \
    represent_property_arguments, decrement_property, increment_property, add_property_arguments_to_property


##
# Classes
##


class SymbolicStrategy:

    @abc.abstractmethod
    def represent(self) -> str:
        ...


@dataclass
class SymbolicIntegerStrategy(SymbolicStrategy):
    var_id: str
    min_value: Sequence[str]
    max_value: Sequence[str]
    filters: Sequence[Lambda]

    def represent(self):
        result: List[str] = [f'hypothesis.strategies.integers(']
        if self.min_value:
            if len(self.min_value) == 1:
                result.append(f'min_value={self.min_value[0]}')
            elif len(self.min_value) > 1:
                result.append(f'min_value=max({", ".join(str(e) for e in self.min_value)})')
        if self.min_value and self.max_value:
            result.append(', ')
        if self.max_value:
            if len(self.max_value) == 1:
                result.append(f'max_value={self.max_value[0]}')
            elif len(self.max_value) > 1:
                result.append(f'max_value=min({", ".join(str(e) for e in self.max_value)})')
        result.append(')')
        for lambda_filter in self.filters:
            result.append(f'.filter({lambda_filter})')
        return "".join(result)


@dataclass
class SymbolicTextStrategy(SymbolicStrategy):
    var_id: str
    blacklist_categories: Sequence[Set[str]]
    whitelist_categories: Sequence[Set[str]]
    min_size: Sequence[str]
    max_size: Sequence[str]
    filters: Sequence[Lambda]

    def represent(self):
        result = [f'hypothesis.strategies.text(']
        if len(self.blacklist_categories) > 0 or len(self.whitelist_categories) > 0:
            result.append('alphabet=hypothesis.strategies.characters(')
        if len(self.blacklist_categories) >= 1:
            result.append(f'blacklist_categories={tuple(sorted(set.intersection(*self.blacklist_categories)))}')
        if len(self.whitelist_categories) >= 1:
            if len(self.blacklist_categories) > 0:
                result.append(', ')
            result.append(f'whitelist_categories={tuple(sorted(set.intersection(*self.whitelist_categories)))}')
        if len(self.blacklist_categories) > 0 or len(self.whitelist_categories) > 0:
            result.append(')')

        if (len(self.blacklist_categories) > 0 or len(self.whitelist_categories) > 0) and len(self.min_size) > 0:
            result.append(', ')
        if len(self.min_size) == 1:
            result.append(f'min_size={self.min_size[0]}')
        elif len(self.min_size) > 1:
            result.append(f'min_size=max({self.min_size})')

        if (len(self.blacklist_categories) > 0 or len(self.whitelist_categories) > 0 or len(self.min_size) > 0) \
                and len(self.max_size) > 0:
            result.append(', ')
        if len(self.max_size) == 1:
            result.append(f'max_size={self.max_size[0]}')
        elif len(self.max_size) > 1:
            result.append(f'max_size=min({self.max_size})')

        result.append(')')

        for f in self.filters:
            result.append(f'.filter({f})')

        return "".join(result)


@dataclass
class SymbolicFromRegexStrategy(SymbolicStrategy):
    var_id: str
    regexps: Sequence[str]
    full_match: bool
    filters: Sequence[Lambda]

    def represent(self):
        result = [f'hypothesis.strategies.from_regex(regex=r"']
        if len(self.regexps) == 0:
            result.append('.*')
        elif len(self.regexps) == 1:
            result.append(self.regexps[0])
        else:
            for regexp in self.regexps:
                result.append(f'(?={regexp})')
        result.append('"')

        if self.full_match:
            result.append(f', fullmatch={self.full_match}')  # noqa

        result.append(')')

        for f in self.filters:
            result.append(f'.filter({f})')

        return "".join(result)


UniqueBy = Union[Callable[[Ex], Hashable], Tuple[Callable[[Ex], Hashable], ...]]


@dataclass
class SymbolicListStrategy(SymbolicStrategy):
    var_id: str
    elements: SymbolicStrategy
    min_size: Sequence[str]
    max_size: Sequence[str]
    unique_by: Sequence[UniqueBy]
    unique: bool
    filters: Sequence[Lambda]

    def represent(self) -> str:
        result = [f'hypothesis.strategies.lists(elements={self.elements.represent()}']
        if self.min_size:
            min_size_str = ", ".join(self.min_size)
            if len(self.min_size) == 1:
                result.append(f', min_size={min_size_str}')
            else:
                result.append(f', min_size=max({min_size_str})')
        if self.max_size:
            max_size_str = ", ".join(self.max_size)
            if len(self.max_size) == 1:
                result.append(f', min_size={max_size_str}')
            else:
                result.append(f', min_size=max({max_size_str})')
        if self.unique_by:
            raise NotImplementedError
        if self.unique:
            result.append(f', unique=True')

        result.append(')')

        for f in self.filters:
            result.append(f'.filter({f})')

        return "".join(result)


##
# Implementation
##


def generate_strategies(table: generate_symbol_table.Table):
    # variable identifier ->  strategy how to generate the identifier
    strategies: Dict[str, SymbolicStrategy] = dict()

    for row in table.get_rows():
        if row.kind == generate_symbol_table.Kind.BASE:
            strategies[row.var_id] = _infer_strategy(row, table)
    # TODO fix dependencies
    dependency_graph = nx.topological_sort(
        icontract_hypothesis_Lauren.generate_symbol_table.generate_dag_from_table(table)
    )
    for row_id in dependency_graph:
        row = table.get_row_by_var_id(row_id)
        if row.type == int:
            strategy = strategies[row_id]
            assert isinstance(strategy, SymbolicIntegerStrategy)
            for row_property in row.properties.values():
                if isinstance(row_property.identifier, ast.LtE):
                    for dependency in row_property.free_variables():
                        affected_strategy = strategies[dependency]
                        assert isinstance(affected_strategy, SymbolicIntegerStrategy)
                        new_min_value = list(affected_strategy.min_value)
                        new_min_value.extend(strategy.min_value)
                        strategies[dependency] = SymbolicIntegerStrategy(
                            var_id=affected_strategy.var_id,
                            min_value=new_min_value,
                            max_value=affected_strategy.max_value,
                            filters=affected_strategy.filters
                        )
                elif isinstance(row_property.identifier, ast.Lt):
                    for dependency in row_property.free_variables():
                        affected_strategy = strategies[dependency]
                        assert isinstance(affected_strategy, SymbolicIntegerStrategy)
                        new_min_value = list(affected_strategy.min_value)
                        new_min_value.extend([
                            f'{int(arg) + 1}' if arg.isnumeric() else f'{arg} + 1'
                            for arg in strategy.min_value
                        ])
                        strategies[dependency] = SymbolicIntegerStrategy(
                            var_id=affected_strategy.var_id,
                            min_value=new_min_value,
                            max_value=affected_strategy.max_value,
                            filters=affected_strategy.filters
                        )
                elif isinstance(row_property.identifier, ast.GtE):
                    for dependency in row_property.free_variables():
                        affected_strategy = strategies[dependency]
                        assert isinstance(affected_strategy, SymbolicIntegerStrategy)
                        new_max_value = list(affected_strategy.max_value)
                        new_max_value.extend(strategy.max_value)
                        strategies[dependency] = SymbolicIntegerStrategy(
                            var_id=affected_strategy.var_id,
                            min_value=affected_strategy.min_value,
                            max_value=new_max_value,
                            filters=affected_strategy.filters
                        )
                elif isinstance(row_property.identifier, ast.Gt):
                    for dependency in row_property.free_variables():
                        affected_strategy = strategies[dependency]
                        assert isinstance(affected_strategy, SymbolicIntegerStrategy)
                        new_max_value = list(affected_strategy.max_value)
                        new_max_value.extend([
                            f'{int(arg) - 1}' if arg.isnumeric() else f'{arg} - 1'
                            for arg in strategy.max_value
                        ])
                        strategies[dependency] = SymbolicIntegerStrategy(
                            var_id=affected_strategy.var_id,
                            min_value=affected_strategy.min_value,
                            max_value=new_max_value,
                            filters=affected_strategy.filters
                        )

    return strategies


def _infer_strategy(row: generate_symbol_table.Row, table: generate_symbol_table.Table) -> SymbolicStrategy:
    if isinstance(row.type, int) or row.type == int:
        return _infer_int_strategy(row, table)
    elif isinstance(row.type, str) or row.type == str:
        if any(row_property in ['contains', 'in', 're.match', 'startswith', 'endswith']
               for row_property in row.properties.keys()):
            return _infer_from_regex_strategy(row, table)
        else:
            return _infer_text_strategy(row, table)
    elif isinstance(typing.get_origin(row.type), list) or typing.get_origin(row.type) == list:
        return _infer_list_strategy(row, table)
    else:
        raise NotImplementedError


##
# Integers
##


def _infer_int_strategy(row: generate_symbol_table.Row,
                        table: generate_symbol_table.Table) -> SymbolicIntegerStrategy:
    max_value_constraints: List[str] = []  # TODO rename
    min_value_constraints: List[str] = []
    filters: List[Lambda] = []

    for property_identifier, row_property in row.properties.items():

        if property_identifier == '<':
            row_property_decremented = decrement_property(row_property)
            max_value_constraints.extend(represent_property_arguments(row_property_decremented))
        elif property_identifier == '<=':
            max_value_constraints.extend(represent_property_arguments(row_property))
        elif property_identifier == '>':
            row_property_incremented = increment_property(row_property)
            min_value_constraints.extend(represent_property_arguments(row_property_incremented))
        elif property_identifier == '>=':
            min_value_constraints.extend(represent_property_arguments(row_property))
        else:
            filters.extend(property_as_lambdas(row_property))

    for link_row in table.get_rows():
        if link_row.parent == row.var_id and link_row.kind == generate_symbol_table.Kind.LINK:
            link_strategy = _infer_int_strategy(link_row, table)
            min_value_constraints.extend(link_strategy.min_value)
            max_value_constraints.extend(link_strategy.max_value)
            filters.extend(link_strategy.filters)

    return SymbolicIntegerStrategy(var_id=row.var_id,
                                   min_value=min_value_constraints,
                                   max_value=max_value_constraints,
                                   filters=filters)


##
# Strings
##

def _infer_text_strategy(row: generate_symbol_table.Row,
                         table: generate_symbol_table.Table) -> SymbolicTextStrategy:
    blacklist_categories: List[Set[str]] = []
    whitelist_categories: List[Set[str]] = []
    min_size: List[Union[int, str]] = []
    max_size: List[Union[int, str]] = []
    filters: List[Tuple[str, Set[str]]](row.var_id) = []

    for property_identifier, row_property in row.properties.items():
        if property_identifier == 'isalnum':
            whitelist_categories.append({'Ll', 'Lu', 'Nd'})
        elif property_identifier == 'isalpha':
            whitelist_categories.append({'Ll', 'Lu'})
        elif property_identifier == 'isdigit':
            whitelist_categories.append({'Nd'})
        elif property_identifier == 'islower':
            whitelist_categories.append({'Ll'})
        elif property_identifier == 'isnumeric':
            whitelist_categories.append({'Nd', 'Nl', 'No'})
        elif property_identifier == 'isspace':
            whitelist_categories.append({'Zs'})
        elif property_identifier == 'isupper':
            whitelist_categories.append({'Lu'})
        elif property_identifier == 'isdecimal':
            whitelist_categories.append({'Nd'})
        elif row.kind == generate_symbol_table.Kind.LINK:
            # link_property = row.var_id[len(row.parent) + 1:]  # TODO better way?
            link_property = row.var_id[:-(len(row.parent) + 2)]
            if link_property == 'len':
                if property_identifier == '<':
                    row_property_decremented = decrement_property(row_property)
                    max_size.extend(represent_property_arguments(row_property_decremented))
                elif property_identifier == '<=':
                    max_size.extend(represent_property_arguments(row_property))
                elif property_identifier == '>':
                    row_property_increment = increment_property(row_property)
                    min_size.extend(represent_property_arguments(row_property_increment))
                elif property_identifier == '>=':
                    min_size.extend(represent_property_arguments(row_property))
                else:
                    raise NotImplementedError  # TODO better exception
            else:
                raise NotImplementedError  # TODO better exception
        else:
            filters.extend(property_as_lambdas(row_property))

    for link_row in table.get_rows():
        if link_row.parent == row.var_id and link_row.kind == generate_symbol_table.Kind.LINK:
            link_strategy = _infer_text_strategy(link_row, table)
            blacklist_categories.extend(link_strategy.blacklist_categories)
            whitelist_categories.extend(link_strategy.whitelist_categories)
            min_size.extend(link_strategy.min_size)
            max_size.extend(link_strategy.max_size)
            filters.extend(link_strategy.filters)

    return SymbolicTextStrategy(var_id=row.var_id,
                                blacklist_categories=blacklist_categories,
                                whitelist_categories=whitelist_categories,
                                min_size=min_size,
                                max_size=max_size,
                                filters=filters)


def _infer_from_regex_strategy(row: generate_symbol_table.Row,
                               table: generate_symbol_table.Table) -> SymbolicFromRegexStrategy:
    regexps: List[str] = []
    full_match: bool = False
    filters: List[Lambda] = []

    for property_identifier, row_property in row.properties.items():
        if property_identifier == 'isalnum':
            regexps.append(r'^[0-9a-zA-Z]+$')
            full_match = True
        elif property_identifier == 'isalpha':
            regexps.append(r'^[a-zA-Z]+$')
            full_match = True
        elif property_identifier == 'isdigit':
            regexps.append(r'^[0-9]*$')
            full_match = True
        elif property_identifier == 'islower':
            regexps.append(r'^[a-z]$')
            full_match = True
        elif property_identifier == 'isnumeric':
            regexps.append(r'^(-[0-9]*|[0-9]*)$')
            full_match = True
        elif property_identifier == 'isspace':
            regexps.append(r'^\s+$')
            full_match = True
        elif property_identifier == 'isupper':
            regexps.append(r'^[A-Z]+$')
            full_match = True
        elif property_identifier == 'isdecimal':
            regexps.append(r'^\d*\.?\d+$')
            full_match = True
        elif property_identifier == 're.match':
            # re.match(r'..', s), we only want the first argument and we don't want any leading/ending \'
            regexps.extend(
                [arg.split(',')[0][1:] for arg in represent_property_arguments(row_property)])  # TODO better way?
            full_match = True
        elif property_identifier == 'contains' or property_identifier == 'in':
            regexps.extend([arg.strip("\'") for arg in represent_property_arguments(row_property)])
        elif property_identifier == 'startswith':
            stripped_args = [arg.strip("\'") for arg in represent_property_arguments(row_property)]
            regexps.extend([f'^{arg}' for arg in stripped_args])
        elif property_identifier == 'endswith':
            stripped_args = [arg.strip("\'") for arg in represent_property_arguments(row_property)]
            regexps.extend([f'.*{arg}$' for arg in stripped_args])
            full_match = True
        elif row.kind == generate_symbol_table.Kind.LINK:
            # link_property = row.var_id[len(row.parent) + 1:]  # TODO better way?
            link_property = row.var_id[:-(len(row.parent) + 2)]
            # TODO better way: row_property.left_function_call
            if link_property == 'len':
                if property_identifier == '<':
                    filters.extend(property_as_lambdas(row_property))
                elif property_identifier == '<=':
                    filters.extend(property_as_lambdas(row_property))
                elif property_identifier == '>':
                    filters.extend(property_as_lambdas(row_property))
                elif property_identifier == '>=':
                    filters.extend(property_as_lambdas(row_property))
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            filters.extend(property_as_lambdas(row_property))

    for link_row in table.get_rows():
        if link_row.parent == row.var_id and link_row.kind == generate_symbol_table.Kind.LINK:
            link_strategy = _infer_from_regex_strategy(link_row, table)
            regexps.extend(link_strategy.regexps)
            full_match = full_match or link_strategy.full_match
            filters.extend(link_strategy.filters)

    return SymbolicFromRegexStrategy(var_id=row.var_id,
                                     regexps=regexps,
                                     full_match=full_match,
                                     filters=filters)


##
# Lists
##

# TODO: "nesting" also needs to apply for all the iterable types (sets, dictionaries, ordered dictionaries...)

def _infer_list_strategy(row: generate_symbol_table.Row, table: generate_symbol_table.Table) -> SymbolicListStrategy:
    elements: Optional[SymbolicStrategy] = None
    min_size: List[str] = []
    max_size: List[str] = []
    unique_by: List[UniqueBy] = []
    unique: bool = False
    filters: List[Lambda] = []

    for other_row in table.get_rows():
        if other_row.parent == row.var_id:
            if other_row.kind == generate_symbol_table.Kind.LINK:
                link_property = other_row.var_id[:-(len(other_row.parent) + 2)]

                for property_identifier, row_property in other_row.properties.items():
                    if link_property == 'len':
                        if property_identifier == '<':
                            row_property_decremented = decrement_property(row_property)
                            max_size.extend(represent_property_arguments(row_property_decremented))
                        elif property_identifier == '<=':
                            max_size.extend(represent_property_arguments(row_property))
                        elif property_identifier == '>':
                            row_property_increment = increment_property(row_property)
                            min_size.extend(represent_property_arguments(row_property_increment))
                        elif property_identifier == '>=':
                            min_size.extend(represent_property_arguments(row_property))
                        else:
                            filters.extend(property_as_lambdas(row_property))
                    else:
                        filters.extend(property_as_lambdas(row_property))
            elif other_row.kind == generate_symbol_table.Kind.UNIVERSAL_QUANTIFIER:
                elements = _infer_strategy(other_row, table)
            else:
                raise NotImplementedError

    if not elements:
        elements_type = typing.get_args(row.type)[0]
        elements = _strategy_from_type(elements_type)

    return SymbolicListStrategy(var_id=row.var_id,
                                elements=elements,
                                min_size=min_size,
                                max_size=max_size,
                                unique_by=unique_by,
                                unique=False or unique,
                                filters=filters)


def _strategy_from_type(strategy_type: typing.Type) -> SymbolicStrategy:
    if strategy_type == int:
        return SymbolicIntegerStrategy(var_id='_',
                                       min_value=[],
                                       max_value=[],
                                       filters=[])
    elif strategy_type == str:
        return SymbolicTextStrategy(var_id='_',
                                    blacklist_categories=[],
                                    whitelist_categories=[],
                                    min_size=[],
                                    max_size=[],
                                    filters=[])
    elif typing.get_origin(strategy_type) == list:
        return SymbolicListStrategy(var_id='_',
                                    elements=_strategy_from_type(typing.get_args(strategy_type)[0]),
                                    min_size=[],
                                    max_size=[],
                                    unique_by=[],
                                    unique=False,
                                    filters=[])
    else:
        raise NotImplementedError(strategy_type)
