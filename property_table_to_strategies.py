import abc

from hypothesis.strategies._internal.strategies import Ex
from icontract import require, ensure
import typing
from typing import List, Union, Dict, Tuple, Set, Sequence, Callable, Hashable, Optional
from icontract_hypothesis_Lauren import generate_symbol_table
from dataclasses import dataclass
import itertools


##
# CLASSES
##


class SymbolicStrategy:
    @abc.abstractmethod
    def represent(self) -> str:
        ...


@dataclass
class SymbolicIntegerStrategy(SymbolicStrategy):
    var_id: str
    min_value: Sequence[Union[int, str]]
    max_value: Sequence[Union[int, str]]
    filters: Sequence[Tuple[str, Set[str]]]

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
        for f in self.filters:
            free_variables = [self.var_id]
            free_variables.extend(f[1])
            result.append(f'.filter(lambda {", ".join(free_variables)}: {f[0]})')
        return "".join(result)


@dataclass
class SymbolicTextStrategy(SymbolicStrategy):
    var_id: str
    blacklist_categories: Sequence[Set[str]]
    whitelist_categories: Sequence[Set[str]]
    min_size: Sequence[Union[int, str]]
    max_size: Sequence[Union[int, str]]
    filters: Sequence[Tuple[str, Set[str]]]

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
            free_variables = [self.var_id]
            free_variables.extend(f[1])
            result.append(f'.filter(lambda {", ".join(free_variables)}: {f[0]})')

        return "".join(result)


@dataclass
class SymbolicFromRegexStrategy(SymbolicStrategy):
    var_id: str
    regexps: Sequence[str]
    full_match: bool
    # TODO remove
    min_size: Sequence[Union[int, str]]
    max_size: Sequence[Union[int, str]]
    filters: Sequence[Tuple[str, Set[str]]]

    def represent(self):
        result = [f'hypothesis.strategies..from_regex(regex=r"']
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
            free_variables = [self.var_id]
            free_variables.extend(f[1])
            result.append(f'.filter(lambda {", ".join(free_variables)}: {f[0]})')

        return "".join(result)


UniqueBy = Union[Callable[[Ex], Hashable], Tuple[Callable[[Ex], Hashable], ...]]


@dataclass
class SymbolicListStrategy(SymbolicStrategy):
    var_id: str
    elements: SymbolicStrategy
    min_size: Sequence[Union[int, str]]
    max_size: Sequence[Union[int, str]]
    unique_by: Sequence[UniqueBy]
    unique: bool
    filters: Sequence[Tuple[str, Set[str]]]

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
        # TODO filters
        return "".join(result)


##
# IMPLEMENTATION
##


def generate_strategies(table: generate_symbol_table.Table):
    strategies: Dict[str, Union[SymbolicStrategy]] = dict()
    for row in table.get_rows():
        if row.kind == generate_symbol_table.Kind.BASE:
            strategies[row.var_id] = _infer_strategy(row, table)
    # for other_row in table.get_rows():
    #     if other_row.kind == generate_symbol_table.Kind.LINK:
    #         parent_var_id = other_row.parent
    #         parent_strategy = strategies[parent_var_id]
    #         # TODO what with lists??
    #         strategies[parent_var_id] = _infer_link_strategy(parent_strategy, other_row, table)
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
# INTEGERS
##


def _infer_int_strategy(row: generate_symbol_table.Row,
                        table: generate_symbol_table.Table) -> SymbolicIntegerStrategy:
    row_properties = row.properties
    max_value_constraints: List[Union[int, str]] = []
    min_value_constraints: List[Union[int, str]] = []
    filters: List[Tuple[str, Set[str]]] = []
    # TODO better name for args
    for row_property, (args, free_variables) in row_properties.items():
        # TODO this should already have happened in the property table
        args = [
            int(arg) if arg.isnumeric() else arg
            for arg in args
        ]

        if row_property == '<':
            max_value_constraints.extend([
                str(int(arg) - 1) if isinstance(arg, int) or arg.isnumeric() else f'{arg}-1'
                for arg in args
            ])
        elif row_property == '<=':
            max_value_constraints.extend(args)
        elif row_property == '>':
            min_value_constraints.extend([
                str(int(arg) + 1) if isinstance(arg, int) or arg.isnumeric() else f'{arg}+1'
                for arg in args
            ])
        elif row_property == '>=':
            min_value_constraints.extend(args)
        else:
            filters.append((f"{row.var_id}{row_property}{args}", free_variables))

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
# STRINGS
##


def _infer_text_strategy(row: generate_symbol_table.Row,
                         table: generate_symbol_table.Table) -> SymbolicTextStrategy:
    row_properties = row.properties
    blacklist_categories: List[Set[str]] = []
    whitelist_categories: List[Set[str]] = []
    min_size: List[Union[int, str]] = []
    max_size: List[Union[int, str]] = []
    filters: List[Tuple[str, Set[str]]](row.var_id) = []

    for row_property, (args, free_vars) in row_properties.items():
        if row_property == 'isalnum':
            whitelist_categories.append({'Ll', 'Lu', 'Nd'})
        elif row_property == 'isalpha':
            whitelist_categories.append({'Ll', 'Lu'})
        elif row_property == 'isdigit':
            whitelist_categories.append({'Nd'})
        elif row_property == 'islower':
            whitelist_categories.append({'Ll'})
        elif row_property == 'isnumeric':
            whitelist_categories.append({'Nd', 'Nl', 'No'})
        elif row_property == 'isspace':
            whitelist_categories.append({'Zs'})
        elif row_property == 'isupper':
            whitelist_categories.append({'Lu'})
        elif row_property == 'isdecimal':
            whitelist_categories.append({'Nd'})
        elif row.kind == generate_symbol_table.Kind.LINK:
            link_property = row.var_id[len(row.parent) + 1:]
            if link_property == 'len':
                if row_property == '<':
                    args = [int(arg) - 1 if arg.isnumeric() else f'{arg}-1' for arg in args]
                    max_size.extend(args)
                elif row_property == '<=':
                    max_size.extend(args)
                elif row_property == '>':
                    args = [int(arg) + 1 if arg.isnumeric() else f'{arg}+1' for arg in args]
                    min_size.extend(args)
                elif row_property == '>=':
                    min_size.extend(args)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            filter_args = ", ".join(args)
            if row.var_id in free_vars:  # s.func(..)
                filters.append((f"{row.var_id}.{row_property}({filter_args})", free_vars))
            else:  # func(..s..)  TODO does this actually occur?
                filters.append((f"{row_property}({filter_args})", free_vars))
            # TODO add row property as filter

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
    row_properties = row.properties
    regexps: List[str] = []
    full_match: bool = False
    min_size: List[Union[int, str]] = []
    max_size: List[Union[int, str]] = []
    filters: List[Tuple[str, Set[str]]] = []
    for row_property, (args, free_vars) in row_properties.items():
        if row_property == 'isalnum':
            regexps.append(r'^[0-9a-zA-Z]+$')
            full_match = True
        elif row_property == 'isalpha':
            regexps.append(r'^[a-zA-Z]+$')
            full_match = True
        elif row_property == 'isdigit':
            regexps.append(r'^[0-9]*$')
            full_match = True
        elif row_property == 'islower':
            regexps.append(r'^[a-z]$')
            full_match = True
        elif row_property == 'isnumeric':
            regexps.append(r'^(-[0-9]*|[0-9]*)$')
            full_match = True
        elif row_property == 'isspace':
            regexps.append(r'^\s+$')
            full_match = True
        elif row_property == 'isupper':
            regexps.append(r'^[A-Z]+$')
            full_match = True
        elif row_property == 'isdecimal':
            regexps.append(r'^\d*\.?\d+$')
            full_match = True
        elif row_property == 're.match':
            # re.match(r'..', s), we only want the first argument and we don't want any leading/ending \'
            regexps.extend([arg[0].strip("\'") for arg in args])
            full_match = True
        elif row_property == 'contains' or row_property == 'in':
            regexps.extend([arg.strip("\'") for arg in args])
        elif row_property == 'startswith':
            stripped_args = [arg.strip("\'") for arg in args]
            regexps.extend([f'^{arg}' for arg in stripped_args])
        elif row_property == 'endswith':
            stripped_args = [arg.strip("\'") for arg in args]
            regexps.extend([f'.*{arg}$' for arg in stripped_args])
            full_match = True
        elif row.kind == generate_symbol_table.Kind.LINK:
            link_property = row.var_id[len(row.parent) + 1:]
            if link_property == 'len':
                if row_property == '<':
                    args_str = ", ".join(args)
                    if len(args) > 1:
                        filters.append((f'len({row.parent}) < min({args_str})', free_vars))
                    else:
                        filters.append((f'len({row.parent}) < {args_str}', free_vars))
                elif row_property == '<=':
                    args_str = ", ".join(args)
                    if len(args) > 1:
                        filters.append((f'len({row.parent}) <= min({args_str})', free_vars))
                    else:
                        filters.append((f'len({row.parent}) <= {args_str}', free_vars))
                elif row_property == '>':
                    args_str = ", ".join(args)
                    if len(args) > 1:
                        filters.append((f'len({row.parent}) > max({args_str})', free_vars))
                    else:
                        filters.append((f'len({row.parent}) > {args_str}', free_vars))
                elif row_property == '>=':
                    args_str = ", ".join(args)
                    if len(args) > 1:
                        filters.append((f'len({row.parent}) >= max({args_str})', free_vars))
                    else:
                        filters.append((f'len({row.parent}) >= {args_str}', free_vars))
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            filter_args = ", ".join(args)
            if row.var_id in free_vars:  # s.func(..)
                filters.append((f"{row.var_id}.{row_property}({filter_args})", free_vars))
            else:  # func(..s..)  TODO does this actually occur?
                filters.append((f"{row_property}({filter_args})", free_vars))
    for link_row in table.get_rows():
        if link_row.parent == row.var_id and link_row.kind == generate_symbol_table.Kind.LINK:
            link_strategy = _infer_from_regex_strategy(link_row, table)
            regexps.extend(link_strategy.regexps)
            full_match = full_match or link_strategy.full_match
            min_size.extend(link_strategy.min_size)
            max_size.extend(link_strategy.max_size)
            filters.extend(link_strategy.filters)
    return SymbolicFromRegexStrategy(var_id=row.var_id,
                                     regexps=regexps,
                                     full_match=full_match,
                                     min_size=min_size,
                                     max_size=max_size,
                                     filters=filters)


##
# LISTS
##

def _infer_list_strategy(row: generate_symbol_table.Row, table: generate_symbol_table.Table) -> SymbolicListStrategy:
    elements: Optional[SymbolicStrategy] = None
    min_size: List[Union[int, str]] = []
    max_size: List[Union[int, str]] = []
    unique_by: List[UniqueBy] = []
    unique: bool = False
    filters: List[Tuple[str, Set[str]]] = []

    # TODO
    for other_row in table.get_rows():
        if other_row.parent == row.var_id:
            if other_row.kind == generate_symbol_table.Kind.LINK:
                link_property = other_row.var_id[len(other_row.parent) + 1:]
                row_properties = other_row.properties
                for row_property, (args, free_vars) in row_properties.items():
                    if link_property == 'len':
                        if row_property == '<':
                            max_size.extend([
                                str(int(arg) - 1) if isinstance(arg, int) or arg.isnumeric() else f'{arg}-1'
                                for arg in args
                            ])
                        elif row_property == '<=':
                            max_size.extend(args)
                        elif row_property == '>':
                            min_size.extend([
                                # str(int(arg) + 1) if arg.isnumeric() else f'{arg}+1'
                                str(int(arg) + 1) if isinstance(arg, int) or arg.isnumeric() else f'{arg}+1'
                                for arg in args
                            ])
                        elif row_property == '>=':
                            min_size.extend(args)
                        else:
                            filters.append((f"{row.var_id}{row_property}{args}", free_vars))
                    else:
                        filter_args = ", ".join(args)
                        if other_row.var_id in free_vars:  # s.func(..)
                            filters.append((f"{other_row.var_id}.{row_property}({filter_args})", free_vars))
                        else:  # func(..s..)  TODO does this actually occur?
                            filters.append((f"{row_property}({filter_args})", free_vars))
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


##
# LINK
##

# TODO remove
# @ensure(lambda parent_strategy, result: type(parent_strategy) == type(result))
# def _infer_link_strategy(parent_strategy: [SymbolicStrategy],
#                          link_row: generate_symbol_table.Row,
#                          table: generate_symbol_table.Table) -> SymbolicStrategy:
#     if isinstance(parent_strategy, SymbolicIntegerStrategy):
#         link_strategy = _infer_int_strategy(link_row)
#         # TODO better way to concatenate two sequences?
#         new_min_value = list(itertools.chain.from_iterable([parent_strategy.min_value, link_strategy.min_value]))
#         new_max_value = list(itertools.chain.from_iterable([parent_strategy.max_value, link_strategy.max_value]))
#         new_filters = list(itertools.chain.from_iterable([parent_strategy.filters, link_strategy.filters]))
#         return SymbolicIntegerStrategy(var_id=parent_strategy.var_id,
#                                        min_value=new_min_value,
#                                        max_value=new_max_value,
#                                        filters=new_filters)
#     elif isinstance(parent_strategy, SymbolicTextStrategy):
#         link_strategy = _infer_text_strategy(link_row)
#         new_blacklist_categories = list(itertools.chain.from_iterable([parent_strategy.blacklist_categories,
#                                                                        link_strategy.blacklist_categories]))
#         new_whitelist_categories = list(itertools.chain.from_iterable([parent_strategy.whitelist_categories,
#                                                                        link_strategy.whitelist_categories]))
#         new_min_size = list(itertools.chain.from_iterable([parent_strategy.min_size,
#                                                            link_strategy.min_size]))
#         new_max_size = list(itertools.chain.from_iterable([parent_strategy.max_size,
#                                                            link_strategy.max_size]))
#         new_filters = list(itertools.chain.from_iterable([parent_strategy.filters,
#                                                           link_strategy.filters]))
#         return SymbolicTextStrategy(var_id=parent_strategy.var_id,
#                                     blacklist_categories=new_blacklist_categories,
#                                     whitelist_categories=new_whitelist_categories,
#                                     min_size=new_min_size,
#                                     max_size=new_max_size,
#                                     filters=new_filters)
    # elif isinstance(parent_strategy, SymbolicFromRegexStrategy):
    #     link_strategy = _infer_from_regex_strategy(link_row)
    #     new_regexps = list(itertools.chain.from_iterable([parent_strategy.regexps, link_strategy.regexps]))
    #     new_full_match = parent_strategy.full_match or link_strategy.full_match
    #     new_min_size = list(itertools.chain.from_iterable([parent_strategy.min_size, link_strategy.min_size]))
    #     new_max_size = list(itertools.chain.from_iterable([parent_strategy.max_size, link_strategy.max_size]))
    #     new_filters = list(itertools.chain.from_iterable([parent_strategy.filters, link_strategy.filters]))
    #     return SymbolicFromRegexStrategy(var_id=parent_strategy.var_id,
    #                                      regexps=new_regexps,
    #                                      full_match=new_full_match,
    #                                      min_size=new_min_size,
    #                                      max_size=new_max_size,
    #                                      filters=new_filters)
    # elif isinstance(parent_strategy, SymbolicListStrategy):
    #     link_strategy = _infer_list_strategy(link_row, table)
    #     new_elements = link_strategy.elements
    #     new_min_size = list(itertools.chain.from_iterable([parent_strategy.min_size, link_strategy.min_size]))
    #     new_max_size = list(itertools.chain.from_iterable([parent_strategy.max_size, link_strategy.max_size]))
    #     new_unique_by = list(itertools.chain.from_iterable([parent_strategy.unique_by, link_strategy.unique_by]))
    #     new_unique = parent_strategy.unique or link_strategy.unique
    #     new_filters = list(itertools.chain.from_iterable([parent_strategy.filters, link_strategy.filters]))
    #     return SymbolicListStrategy(var_id=parent_strategy.var_id,
    #                                 elements=new_elements,
    #                                 min_size=new_min_size,
    #                                 max_size=new_max_size,
    #                                 unique_by=new_unique_by,
    #                                 unique=new_unique,
    #                                 filters=new_filters)
    # else:
    #     raise NotImplementedError


# region Description
# TESTS
# endregion


@require(lambda n1: n1 > 0)
def example_function_1(n1: int) -> None:
    pass


@require(lambda n1: n1 < 10)
@require(lambda n2: n2 > 100 and n2 < 1000)
def example_function_2(n1: int, n2: int) -> None:
    pass


@require(lambda n1, n2: n1 < 10 and n1 < n2)
def example_function_3(n1: int, n2: int) -> None:
    pass


@require(lambda s: s.isidentifier())
def example_function_4(s: str):
    pass


if __name__ == '__main__':
    _, t = generate_symbol_table.generate_symbol_table(example_function_4)
    generate_symbol_table.print_pretty_table(t)
    print(generate_strategies(t))
    # print(typing_extensions.get_type_hints(example_function_1))
