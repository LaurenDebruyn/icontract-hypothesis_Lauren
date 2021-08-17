import abc
import ast
import collections.abc
import typing
from dataclasses import dataclass
from icontract import require
from itertools import chain
from typing import List, Union, Dict, Tuple, Set, Sequence, Callable, Hashable, Optional, Type
import sys
from hypothesis.strategies._internal.strategies import Ex

from icontract_hypothesis_Lauren import generate_symbol_table
from icontract_hypothesis_Lauren.generate_symbol_table import Lambda, property_to_lambdas, \
    represent_property_arguments, Property, PropertyArgument


##
# Classes
##


@dataclass
class SymbolicStrategy:
    var_id: str
    filters: Sequence[Lambda]

    @abc.abstractmethod
    def represent(self) -> str:
        ...


@dataclass
class SymbolicIntegerStrategy(SymbolicStrategy):
    min_value: Sequence[str]
    max_value: Sequence[str]

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
class SymbolicFloatStrategy(SymbolicStrategy):
    min_value: Sequence[str]
    max_value: Sequence[str]
    allow_nan: bool = None
    allow_infinity: bool = None
    width: int = 64
    exclude_min = False
    exclude_max = False

    def represent(self):
        result: List[str] = [f'hypothesis.strategies.floats(']
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
    blacklist_categories: Sequence[Set[str]]
    whitelist_categories: Sequence[Set[str]]
    min_size: Sequence[str]
    max_size: Sequence[str]

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
    regexps: Sequence[str]
    full_match: bool

    def represent(self):
        result = [f'hypothesis.strategies.from_regex(regex=r']
        if len(self.regexps) == 0:
            result.append('.*')
        elif len(self.regexps) == 1:
            result.append(self.regexps[0])
        else:
            result.append('\"')
            for regexp in self.regexps:
                stripped_regexp = regexp.strip('\"')
                result.append(f'(?={stripped_regexp})')
            result.append('\"')

        if self.full_match:
            result.append(f', fullmatch={self.full_match}')  # noqa

        result.append(')')

        for f in self.filters:
            result.append(f'.filter({f})')

        return "".join(result)


UniqueBy = Union[Callable[[Ex], Hashable], Tuple[Callable[[Ex], Hashable], ...]]


@dataclass
class SymbolicListStrategy(SymbolicStrategy):
    elements: SymbolicStrategy
    min_size: Sequence[str]
    max_size: Sequence[str]
    unique_by: Sequence[UniqueBy]
    unique: bool

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


@dataclass
class SymbolicDictionaryStrategy(SymbolicStrategy):
    keys: SymbolicStrategy
    values: SymbolicStrategy
    min_size: Sequence[str]
    max_size: Sequence[str]

    def represent(self) -> str:
        result = [f'hypothesis.strategies.dictionaries(keys={self.keys.represent()}, values={self.values.represent()}']
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

        result.append(')')

        for f in self.filters:
            result.append(f'.filter({f})')

        return "".join(result)


@dataclass
class SymbolicSetStrategy(SymbolicStrategy):
    elements: SymbolicStrategy
    min_size: Sequence[str]
    max_size: Sequence[str]

    def represent(self) -> str:
        result = [f'hypothesis.strategies.sets(elements={self.elements.represent()}']
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

        result.append(')')

        for f in self.filters:
            result.append(f'.filter({f})')

        return "".join(result)


@dataclass
class SymbolicFromTypeStrategy(SymbolicStrategy):
    type: type

    def represent(self) -> str:
        type_str = type_hint_to_str(self.type)
        result = [f'hypothesis.strategies.from_type(thing={type_str})']
        for f in self.filters:
            result.append(f'.filter({f})')
        return "".join(result)


@dataclass
class SymbolicOneOfStrategy(SymbolicStrategy):
    strategies: List[SymbolicStrategy]

    def represent(self) -> str:
        strategies_str = ", ".join([strategy.represent() for strategy in self.strategies])
        result = [f'hypothesis.strategies.one_of({strategies_str})']
        for f in self.filters:
            result.append(f'.filter({f})')
        return "".join(result)


@dataclass
class SymbolicNoneStrategy(SymbolicStrategy):

    def represent(self) -> str:
        return 'hypothesis.strategies.none()'


##
# Implementation
##


def generate_strategies(table: generate_symbol_table.Table) -> Dict[str, SymbolicStrategy]:
    # variable identifier ->  strategy how to generate the identifier
    strategies: Dict[str, SymbolicStrategy] = dict()

    for row in table.get_rows():
        if row.kind == generate_symbol_table.Kind.BASE:
            strategies[row.var_id] = _infer_strategy(row, table)
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
    elif isinstance(row.type, float) or row.type == float:
        return _infer_float_strategy(row, table)
    elif isinstance(row.type, str) or row.type == str:
        if any(row_property in ['contains', 'in', 're.match', 'regex.match', 're.fullmatch', 'regex.fullmatch',
                                'startswith', 'endswith']
               for row_property in row.properties.keys()):
            return _infer_from_regex_strategy(row, table)
        else:
            return _infer_text_strategy(row, table)
    elif isinstance(typing.get_origin(row.type), list) or typing.get_origin(row.type) == list or \
            typing.get_origin(row.type) == collections.abc.Collection:
        return _infer_list_strategy(row, table)
    elif isinstance(typing.get_origin(row.type), dict) or typing.get_origin(row.type) == dict or \
         typing.get_origin(row.type) == collections.abc.Mapping:
        return _infer_dictionary_strategy(row, table)
    elif isinstance(typing.get_origin(row.type), set) or typing.get_origin(row.type) == set:
        return _infer_set_strategy(row, table)
    elif isinstance(typing.get_origin(row.type), typing._SpecialForm):
        return _infer_one_of_strategy(row, table)
    else:
        return _infer_from_type_strategy(row, table)


##
# Integers
##


def _infer_int_strategy(row: generate_symbol_table.Row,
                        table: generate_symbol_table.Table) -> SymbolicIntegerStrategy:
    max_value: List[Property] = []
    min_value: List[Property] = []
    filters: List[Lambda] = []

    min_value_contains_free_variables = False
    max_value_contains_free_variables = False

    for property_identifier, row_property in row.properties.items():
        if property_identifier == '<':
            row_property_decremented = _decrement_property(row_property)
            row_property_decremented.identifier = '<='
            max_value.append(row_property_decremented)
            if row_property.free_variables():
                max_value_contains_free_variables = True
        elif property_identifier == '<=':
            max_value.append(row_property)
            if row_property.free_variables():
                max_value_contains_free_variables = True
        elif property_identifier == '>':
            row_property_incremented = _increment_property(row_property)
            row_property_incremented.identifier = '>='
            min_value.append(row_property_incremented)
            if row_property.free_variables():
                min_value_contains_free_variables = True
        elif property_identifier == '>=':
            min_value.append(row_property)
            if row_property.free_variables():
                min_value_contains_free_variables = True
        else:
            filters.extend(property_to_lambdas(row_property))

    if min_value_contains_free_variables or max_value_contains_free_variables:
        # turn min values into filters
        for prop in min_value:
            filters.extend(property_to_lambdas(prop))
        min_value = []


    min_value_deserialized: List[str] = list(chain(*[represent_property_arguments(prop) for prop in min_value]))
    max_value_deserialized: List[str] = list(chain(*[represent_property_arguments(prop) for prop in max_value]))

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


def _infer_float_strategy(row: generate_symbol_table.Row,
                        table: generate_symbol_table.Table) -> SymbolicFloatStrategy:
    max_value: List[Property] = []
    min_value: List[Property] = []
    filters: List[Lambda] = []

    min_value_contains_free_variables = False
    max_value_contains_free_variables = False

    for property_identifier, row_property in row.properties.items():
        if property_identifier == '<':
            row_property_decremented = _decrement_property(row_property)
            row_property_decremented.identifier = '<='
            max_value.append(row_property_decremented)
            if row_property.free_variables():
                max_value_contains_free_variables = True
        elif property_identifier == '<=':
            max_value.append(row_property)
            if row_property.free_variables():
                max_value_contains_free_variables = True
        elif property_identifier == '>':
            row_property_incremented = _increment_property(row_property)
            row_property_incremented.identifier = '>='
            min_value.append(row_property_incremented)
            if row_property.free_variables():
                min_value_contains_free_variables = True
        elif property_identifier == '>=':
            min_value.append(row_property)
            if row_property.free_variables():
                min_value_contains_free_variables = True
        else:
            filters.extend(property_to_lambdas(row_property))

    if min_value_contains_free_variables or max_value_contains_free_variables:
        for prop in min_value:
            filters.extend(property_to_lambdas(prop))
        min_value = []

    min_value_deserialized: List[str] = list(chain(*[represent_property_arguments(prop) for prop in min_value]))
    max_value_deserialized: List[str] = list(chain(*[represent_property_arguments(prop) for prop in max_value]))

    for link_row in table.get_rows():
        if link_row.parent == row.var_id and link_row.kind == generate_symbol_table.Kind.LINK:
            link_strategy = _infer_int_strategy(link_row, table)
            min_value_deserialized.extend(link_strategy.min_value)
            max_value_deserialized.extend(link_strategy.max_value)
            filters.extend(link_strategy.filters)

    return SymbolicFloatStrategy(var_id=row.var_id,
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
                elif property_identifier == '==':
                    min_size.extend(represent_property_arguments(row_property))
                    max_size.extend(represent_property_arguments(row_property))
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
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
        elif property_identifier in ['re.match', 'regex.match']:
            # re.match(r'..', s), we only want the first argument and we don't want any leading/ending \'
            regexps.extend(
                [arg.rsplit(',', 1)[0][1:] for arg in represent_property_arguments(row_property)])
        elif property_identifier in ['re.fullmatch', 'regex.fullmatch']:
            # re.match(r'..', s), we only want the first argument and we don't want any leading/ending \'
            regexps.extend(
                [arg.rsplit(',', 1)[0][1:] for arg in represent_property_arguments(row_property)])
            full_match = True
        elif property_identifier == 'contains' or property_identifier == 'in':
            regexps.extend([arg.strip("\'") for arg in represent_property_arguments(row_property)])
        elif property_identifier == 'startswith':
            stripped_args = [arg.strip('\"') for arg in represent_property_arguments(row_property)]
            regexps.extend([f'\"^{arg}\"' for arg in stripped_args])
        elif property_identifier == 'endswith':
            stripped_args = [arg.strip('\"') for arg in represent_property_arguments(row_property)]
            regexps.extend([f'\".*{arg}$\"' for arg in stripped_args])
            full_match = True
        elif row.kind == generate_symbol_table.Kind.LINK:
            link_property = row.var_id[:-(len(row.parent) + 2)]
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


def _infer_list_strategy(row: generate_symbol_table.Row, table: generate_symbol_table.Table) -> SymbolicListStrategy:
    elements: Optional[SymbolicStrategy] = None
    min_size: List[str] = []
    max_size: List[str] = []
    unique_by: List[UniqueBy] = []
    unique: bool = False
    filters: List[Lambda] = []

    for property_identifier, row_property in row.properties.items():
        if property_identifier == 'IS_UNIQUE':
            unique = True
        else:
            raise NotImplementedError("'is_unique' is currently the only property applying directly to a list.")

    for other_row in table.get_rows():
        if other_row.parent == row.var_id:
            if other_row.kind == generate_symbol_table.Kind.LINK:
                link_property = other_row.var_id[:-(len(other_row.parent) + 2)]

                for property_identifier, row_property in other_row.properties.items():
                    parents = []
                    parent = other_row.parent
                    while parent:
                        parents.append(parent)
                        parent = table.get_row_by_var_id(parent).parent
                    if any(parent in row_property.free_variables() for parent in parents):
                        filters.extend(property_to_lambdas(row_property))
                    elif link_property == 'len':
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
                if row.var_id in {e for s in other_row.get_dependencies().values() for e in s}:
                    for row_property in other_row.properties.values():
                        lambdas = property_to_lambdas(row_property)
                        for l in lambdas:
                            if row.var_id in l.free_variables and other_row.var_id in l.free_variables:
                                l.free_variables.remove(other_row.var_id)
                                l.condition = f"all({l.condition} for {other_row.var_id} in {row.var_id})"
                        filters.extend(lambdas)
                else:
                    elements = _infer_strategy(other_row, table)
                    filters_to_delete = []
                    for f in elements.filters:
                        if row.var_id in f.free_variables:
                            new_filter = Lambda(condition=f"all({f.condition} for {other_row.var_id} in {row.var_id})",
                                                free_variables=[row.var_id])
                            filters.append(new_filter)
                            filters_to_delete.append(f)
                    elements.filters = [f for f in elements.filters if f not in filters_to_delete]
            else:
                raise NotImplementedError

    if not elements:
        elements_type = typing.get_args(row.type)[0]
        elements = _strategy_from_type_hint(elements_type)

    return SymbolicListStrategy(var_id=row.var_id,
                                elements=elements,
                                min_size=min_size,
                                max_size=max_size,
                                unique_by=unique_by,
                                unique=False or unique,
                                filters=filters)


def _infer_dictionary_strategy(row: generate_symbol_table.Row, table: generate_symbol_table.Table) -> SymbolicDictionaryStrategy:
    keys: Optional[SymbolicStrategy] = None
    values: Optional[SymbolicStrategy] = None
    min_size: List[str] = []
    max_size: List[str] = []
    filters: List[Lambda] = []

    for other_row in table.get_rows():
        if row.var_id == other_row.var_id:
            pass
        elif other_row.parent == row.var_id:
            if other_row.kind == generate_symbol_table.Kind.LINK:
                link_property = other_row.var_id[:-(len(other_row.parent) + 2)]

                for property_identifier, row_property in other_row.properties.items():
                    parents = []
                    parent = other_row.parent
                    while parent:
                        parents.append(parent)
                        parent = table.get_row_by_var_id(parent).parent
                    if any(parent in row_property.free_variables() for parent in parents):
                        filters.extend(property_to_lambdas(row_property))
                    elif link_property == 'len':
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
                if row.var_id in {e for s in other_row.get_dependencies().values() for e in s}:
                    for row_property in other_row.properties.values():
                        lambdas = property_to_lambdas(row_property)
                        for l in lambdas:
                            if row.var_id in l.free_variables and other_row.var_id in l.free_variables:
                                l.free_variables.remove(other_row.var_id)
                                l.condition = f"all({l.condition} for {other_row.var_id} in {row.var_id})"
                        filters.extend(lambdas)
                else:
                    elements = _infer_strategy(other_row, table)
                    filters_to_delete = []
                    for f in elements.filters:
                        if row.var_id in f.free_variables:
                            new_filter = Lambda(condition=f"all({f.condition} for {other_row.var_id} in {row.var_id})",
                                                free_variables=[row.var_id])
                            filters.append(new_filter)
                            filters_to_delete.append(f)
                    elements.filters = [f for f in elements.filters if f not in filters_to_delete]
            else:
                raise NotImplementedError
        elif other_row.parent == f'{row.var_id}.keys()':
            if other_row.kind == generate_symbol_table.Kind.UNIVERSAL_QUANTIFIER:
                keys = _infer_strategy(other_row, table)
            else:
                raise NotImplementedError
        elif other_row.parent == f'{row.var_id}.values()':
            if other_row.kind == generate_symbol_table.Kind.UNIVERSAL_QUANTIFIER:
                values = _infer_strategy(other_row, table)
            else:
                raise NotImplementedError

    if not keys:
        keys_type = typing.get_args(row.type)[0]
        keys = _strategy_from_type_hint(keys_type)

    if not values:
        values_type = typing.get_args(row.type)[1]
        values = _strategy_from_type_hint(values_type)

    return SymbolicDictionaryStrategy(var_id=row.var_id,
                                      keys=keys,
                                      values=values,
                                      min_size=min_size,
                                      max_size=max_size,
                                      filters=filters)


def _infer_set_strategy(row: generate_symbol_table.Row, table: generate_symbol_table.Table) -> SymbolicSetStrategy:
    elements: Optional[SymbolicStrategy] = None
    min_size: List[str] = []
    max_size: List[str] = []
    filters: List[Lambda] = []

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
        elements = _strategy_from_type_hint(elements_type)

    return SymbolicSetStrategy(var_id=row.var_id,
                               elements=elements,
                               min_size=min_size,
                               max_size=max_size,
                               filters=filters)


def _infer_from_type_strategy(row: generate_symbol_table.Row, table: generate_symbol_table.Table) \
        -> SymbolicFromTypeStrategy:
    filters: List[Lambda] = []

    filters.extend([
        lambda_property
        for row_property in row.properties.values()
        for lambda_property in property_to_lambdas(row_property)
    ])

    for other_row in table.get_rows():
        if typing.get_origin(row.type) in [dict, collections.abc.Mapping] and \
                other_row.kind == generate_symbol_table.Kind.UNIVERSAL_QUANTIFIER:
            if row.var_id in {e for s in other_row.get_dependencies().values() for e in s}:
                for row_property in other_row.properties.values():
                    lambdas = property_to_lambdas(row_property)
                    for l in lambdas:
                        if row.var_id in l.free_variables and other_row.var_id in l.free_variables:
                            l.free_variables.remove(other_row.var_id)
                            l.condition = f"all({l.condition} for {other_row.var_id} in {other_row.parent})"
                    filters.extend(lambdas)
            else:
                elements = _infer_strategy(other_row, table)
                filters_to_delete = []
                for f in elements.filters:
                    if row.var_id in f.free_variables:
                        new_filter = Lambda(condition=f"all({f.condition} for {other_row.var_id} in {other_row.parent})",
                                            free_variables=[row.var_id])
                        filters.append(new_filter)
                        filters_to_delete.append(f)
                elements.filters = [f for f in elements.filters if f not in filters_to_delete]
        elif other_row.parent == row.var_id:
            if other_row.kind == generate_symbol_table.Kind.LINK:
                filters.extend([
                    lambda_property
                    for row_property in other_row.properties.values()
                    for lambda_property in property_to_lambdas(row_property)
                ])
            elif other_row.kind == generate_symbol_table.Kind.UNIVERSAL_QUANTIFIER:
                if row.var_id in {e for s in other_row.get_dependencies().values() for e in s}:
                    for row_property in other_row.properties.values():
                        lambdas = property_to_lambdas(row_property)
                        for l in lambdas:
                            if row.var_id in l.free_variables and other_row.var_id in l.free_variables:
                                l.free_variables.remove(other_row.var_id)
                                l.condition = f"all({l.condition} for {other_row.var_id} in {row.var_id})"
                        filters.extend(lambdas)
                else:
                    elements = _infer_strategy(other_row, table)
                    filters_to_delete = []
                    for f in elements.filters:
                        if row.var_id in f.free_variables:
                            new_filter = Lambda(condition=f"all({f.condition} for {other_row.var_id} in {row.var_id})",
                                                free_variables=[row.var_id])
                            filters.append(new_filter)
                            filters_to_delete.append(f)
                    elements.filters = [f for f in elements.filters if f not in filters_to_delete]
            else:
                raise NotImplementedError

    # Assume that the other variables are local variables in the strategy
    for f in filters:
        f.free_variables = [row.var_id]

    return SymbolicFromTypeStrategy(
        var_id=row.var_id,
        type=row.type,
        filters=filters
    )


def _infer_one_of_strategy(row: generate_symbol_table.Row, table: generate_symbol_table.Table) \
        -> SymbolicOneOfStrategy:
    filters: List[Lambda] = []

    filters.extend([
        lambda_property
        for row_property in row.properties.values()
        for lambda_property in property_to_lambdas(row_property)
    ])

    strategies = []

    union_types = typing.get_args(row.type)
    for type_hint in union_types:
        strategies.append(_strategy_from_type_hint(type_hint))

    return SymbolicOneOfStrategy(
        var_id=row.var_id,
        strategies=strategies,
        filters=filters
    )


def _strategy_from_type_hint(strategy_type: typing.Type) -> SymbolicStrategy:
    if strategy_type == int:
        return SymbolicIntegerStrategy(var_id='_',
                                       min_value=[],
                                       max_value=[],
                                       filters=[])
    elif strategy_type == float:
        return SymbolicFloatStrategy(var_id='_',
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
                                    elements=_strategy_from_type_hint(typing.get_args(strategy_type)[0]),
                                    min_size=[],
                                    max_size=[],
                                    unique_by=[],
                                    unique=False,
                                    filters=[])
    elif typing.get_origin(strategy_type) == dict or typing.get_origin(strategy_type) == collections.abc.Mapping:
        return SymbolicDictionaryStrategy(var_id='_',
                                          keys=_strategy_from_type_hint(typing.get_args(strategy_type)[0]),
                                          values=_strategy_from_type_hint(typing.get_args(strategy_type)[1]),
                                          min_size=[],
                                          max_size=[],
                                          filters=[])
    elif typing.get_origin(strategy_type) == set:
        return SymbolicSetStrategy(var_id='_',
                                   elements=_strategy_from_type_hint(typing.get_args(strategy_type)[0]),
                                   min_size=[],
                                   max_size=[],
                                   filters=[])
    elif strategy_type == type(None):
        return SymbolicNoneStrategy(var_id='_',
                                    filters=[])
    else:
        return SymbolicFromTypeStrategy(var_id='_',
                                        type=strategy_type,
                                        filters=[])



def type_hint_to_str(type_hint: Type) -> str:
    type_hints: List[str] = []
    if typing.get_origin(type_hint):
        origin_type_hint = typing.get_origin(type_hint)
        if isinstance(origin_type_hint, typing._SpecialForm):
            type_hints.append(origin_type_hint._name)
        elif sys.version_info < (3, 9) and origin_type_hint in [list, set, tuple, dict]:
            type_hints.append(typing.get_origin(type_hint).__name__.capitalize())
        else:
            type_hints.append(typing.get_origin(type_hint).__name__)
        type_hint_args: List[str] = []
        for type_hint_arg in typing.get_args(type_hint):
            type_hint_args.append(type_hint_to_str(type_hint_arg))
        type_hints.append("[" + ", ".join(type_hint_args) + "]")
    else:
        type_hints.append(type_hint.__name__)
    return "".join(type_hints)
