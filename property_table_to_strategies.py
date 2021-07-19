import abc
import ast
import typing
from dataclasses import dataclass
from itertools import chain
import regex as re
from typing import List, Union, Dict, Tuple, Set, Sequence, Callable, Hashable, Optional

from hypothesis.strategies._internal.strategies import Ex

from icontract_hypothesis_Lauren import generate_symbol_table
from icontract_hypothesis_Lauren.generate_symbol_table import Lambda, property_to_lambdas, \
    represent_property_arguments, Property, PropertyArgument


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
    # TODO: remove nx, implement Tarjan's algorithm ourselves --> see mristin,
    #   https://github.com/aas-core-works/abnf-to-regexp
    # TODO: check for cycles?
    return strategies


def _increment_property(prop: Property) -> Property:
    new_property_arguments = []
    for property_argument in prop.property_arguments:
        old_argument = property_argument.argument[0]
        if isinstance(old_argument, ast.Constant):
            old_argument.value = int(old_argument.value) + 1
            new_property_arguments.append(PropertyArgument((old_argument,),
                                                           property_argument.free_variables))
        else:
            new_property_arguments.append(PropertyArgument((ast.BinOp(property_argument.argument[0], ast.Add(),  # noqa
                                                                      ast.Constant(1, lineno=0, col_offset=0,
                                                                                   kind=None),
                                                                      lineno=0, col_offset=0, kind=None),),
                                                           property_argument.free_variables))

    return Property(identifier=prop.identifier,
                    property_arguments=new_property_arguments,
                    left_function_call=prop.left_function_call,
                    var_id=prop.var_id,
                    is_routine=prop.is_routine,
                    var_is_caller=prop.var_is_caller)


def _decrement_property(prop: Property) -> Property:
    new_property_arguments = []
    for property_argument in prop.property_arguments:
        old_argument = property_argument.argument[0]
        if isinstance(old_argument, ast.Constant):
            old_argument.value = int(old_argument.value) - 1
            new_property_arguments.append(PropertyArgument((old_argument,),
                                                           property_argument.free_variables))
        else:
            new_property_arguments.append(PropertyArgument((ast.BinOp(property_argument.argument[0], ast.Sub(),  # noqa
                                                                      ast.Constant(1, lineno=0, col_offset=0,
                                                                                   kind=None),
                                                                      lineno=0, col_offset=0, kind=None),),
                                                           property_argument.free_variables))
    return Property(identifier=prop.identifier,
                    property_arguments=new_property_arguments,
                    left_function_call=prop.left_function_call,
                    var_id=prop.var_id,
                    is_routine=prop.is_routine,
                    var_is_caller=prop.var_is_caller)


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
        # TODO infer from_type strategy
        raise NotImplementedError("Only integers, strings and lists are currently supported")


##
# Integers
##


def _infer_int_strategy(row: generate_symbol_table.Row,
                        table: generate_symbol_table.Table) -> SymbolicIntegerStrategy:
    # TODO
    # max_value: List[str] = []
    # min_value: List[str] = []
    max_value: List[Property] = []
    min_value: List[Property] = []
    filters: List[Lambda] = []

    contains_max_value_free_variables = False
    contains_min_value_free_variables = False

    for property_identifier, row_property in row.properties.items():
        if property_identifier == '<':
            # TODO only need ast, not whole property
            row_property_decremented = _decrement_property(row_property)
            row_property_decremented.identifier = '<='
            max_value.append(row_property_decremented)
            if row_property.free_variables():
                contains_max_value_free_variables = True
        elif property_identifier == '<=':
            max_value.append(row_property)
            if row_property.free_variables():
                contains_max_value_free_variables = True
        elif property_identifier == '>':
            row_property_incremented = _increment_property(row_property)
            row_property_incremented.identifier = '>='
            min_value.append(row_property_incremented)
            if row_property.free_variables():
                contains_min_value_free_variables = True
        elif property_identifier == '>=':
            min_value.append(row_property)
            if row_property.free_variables():
                contains_min_value_free_variables = True
        else:
            filters.extend(property_to_lambdas(row_property))

    # TODO can I fix dependencies here?
    # TODO make this one variable
    if contains_max_value_free_variables or contains_min_value_free_variables:
        # turn min values into filters
        for prop in min_value:
            filters.extend(property_to_lambdas(prop))
        min_value = []
    # TODO remove?
    # elif contains_min_value_free_variables:
    #     # turn max values into filters
    #     for prop in max_value:
    #         filters.extend(property_to_lambdas(prop))
    #     max_value = []

    min_value_deserialized: List[str] = list(chain(*[represent_property_arguments(prop) for prop in min_value]))
    max_value_deserialized: List[str] = list(chain(*[represent_property_arguments(prop) for prop in max_value]))

    # TODO correct to do this afterwards?
    for link_row in table.get_rows():
        if link_row.parent == row.var_id and link_row.kind == generate_symbol_table.Kind.LINK:
            link_strategy = _infer_int_strategy(link_row, table)
            min_value_deserialized.extend(link_strategy.min_value)
            max_value_deserialized.extend(link_strategy.max_value)
            filters.extend(link_strategy.filters)

    return SymbolicIntegerStrategy(var_id=row.var_id,
                                   min_value=min_value_deserialized,
                                   max_value=max_value_deserialized,
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
                    row_property_decremented = _decrement_property(row_property)
                    max_size.extend(represent_property_arguments(row_property_decremented))
                elif property_identifier == '<=':
                    max_size.extend(represent_property_arguments(row_property))
                elif property_identifier == '>':
                    row_property_increment = _increment_property(row_property)
                    min_size.extend(represent_property_arguments(row_property_increment))
                elif property_identifier == '>=':
                    min_size.extend(represent_property_arguments(row_property))
                else:
                    raise NotImplementedError  # TODO better exception
            else:
                raise NotImplementedError  # TODO better exception
        else:
            filters.extend(property_to_lambdas(row_property))

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
        # TODO make distinction between match & fullmatch?
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
                    filters.extend(property_to_lambdas(row_property))
                elif property_identifier == '<=':
                    filters.extend(property_to_lambdas(row_property))
                elif property_identifier == '>':
                    filters.extend(property_to_lambdas(row_property))
                elif property_identifier == '>=':
                    filters.extend(property_to_lambdas(row_property))
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            filters.extend(property_to_lambdas(row_property))

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

    for property_identifier, row_property in row.properties.items():
        if property_identifier == 'SpecialProperties.IS_UNIQUE':
            unique = True
        else:
            raise NotImplementedError("'is_unique' is currently the only property applying directly to a list.")

    for other_row in table.get_rows():
        if other_row.parent == row.var_id:
            if other_row.kind == generate_symbol_table.Kind.LINK:
                link_property = other_row.var_id[:-(len(other_row.parent) + 2)]

                for property_identifier, row_property in other_row.properties.items():
                    if link_property == 'len':
                        if property_identifier == '<':
                            row_property_decremented = _decrement_property(row_property)
                            max_size.extend(represent_property_arguments(row_property_decremented))
                        elif property_identifier == '<=':
                            max_size.extend(represent_property_arguments(row_property))
                        elif property_identifier == '>':
                            row_property_increment = _increment_property(row_property)
                            min_size.extend(represent_property_arguments(row_property_increment))
                        elif property_identifier == '>=':
                            min_size.extend(represent_property_arguments(row_property))
                        else:
                            filters.extend(property_to_lambdas(row_property))
                    else:
                        filters.extend(property_to_lambdas(row_property))
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
