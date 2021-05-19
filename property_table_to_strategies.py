from icontract import require
from generate_symbol_table import Table, Row, Kind
from typing import List, Union, Dict, Any, Tuple, Set, Optional, Sequence
from icontract_hypothesis_Lauren.generate_symbol_table import generate_symbol_table, print_pretty_table
from dataclasses import dataclass, field
import ast


###########
# CLASSES #
###########


# @dataclass
# class SymbolicIntegerStrategy:
#     var_id: str
#     min_value: List[Union[int, str]] = field(default_factory=list)
#     max_value: List[Union[int, str]] = field(default_factory=list)
#     filters: List[Tuple[str, Set[str]]] = field(default_factory=list)
#
#     def add_min_value_constraints(self, min_constraints: List[Union[int, str]]) -> None:
#         for min_constraint in min_constraints:
#             self.min_value.append(min_constraint)
#
#     def add_max_value_constraints(self, max_constraints: List[Union[int, str]]) -> None:
#         for max_constraint in max_constraints:
#             self.max_value.append(max_constraint)
#
#     def add_filter(self, new_filter: str, free_variables: Set[str]) -> None:
#         self.filters.append((new_filter, free_variables))
#
#     def get_strategy(self):
#         # def __repr__(self):
#         result = f'st.integers('
#         if len(self.min_value) == 1:
#             result += f'min_value={self.min_value[0]}'
#         elif len(self.min_value) > 1:
#             result += f'min_value=max({", ".join(str(e) for e in self.min_value)})'
#         if len(self.min_value) > 0 and len(self.max_value) > 0:
#             result += ', '
#         if len(self.max_value) == 1:
#             result += f'max_value={self.max_value[0]}'
#         elif len(self.max_value) > 1:
#             result += f'max_value=min({", ".join(str(e) for e in self.max_value)}))'
#         result += ')'
#         for f in self.filters:
#             free_variables = [self.var_id]
#             free_variables.extend(f[1])
#             result += f'.filter(lambda {", ".join(free_variables)}: {f[0]})'
#         return result


@dataclass
class SymbolicIntegerStrategy:
    var_id: str
    min_value: Sequence[Union[int, str]]
    max_value: Sequence[Union[int, str]]
    filters: Sequence[Tuple[str, Set[str]]]

    def represent(self):
        # def __repr__(self):
        result = f'st.integers('
        if self.min_value:
            if len(self.min_value) == 1:
                result += f'min_value={self.min_value[0]}'
            elif len(self.min_value) > 1:
                result += f'min_value=max({", ".join(str(e) for e in self.min_value)})'
        if self.min_value and self.max_value:
            result += ', '
        if self.max_value:
            if len(self.max_value) == 1:
                result += f'max_value={self.max_value[0]}'
            elif len(self.max_value) > 1:
                result += f'max_value=min({", ".join(str(e) for e in self.max_value)}))'
        result += ')'
        for f in self.filters:
            free_variables = [self.var_id]
            free_variables.extend(f[1])
            result += f'.filter(lambda {", ".join(free_variables)}: {f[0]})'
        return result


@dataclass
class SymbolicTextStrategy:
    var_id: str
    blacklist_categories: Sequence[Set[str]]
    whitelist_categories: Sequence[Set[str]]
    min_size: Sequence[Union[int, str]]
    max_size: Sequence[Union[int, str]]
    filters: Sequence[Tuple[str, Set[str]]]

    def represent(self):
        # def __repr__(self):
        result = f'st.text('
        if len(self.blacklist_categories) > 0 or len(self.whitelist_categories) > 0:
            result += 'alphabet=st.characters('
        if len(self.blacklist_categories) >= 1:
            result += f'blacklist_categories={tuple(sorted(set.intersection(*self.blacklist_categories)))}'
        if len(self.whitelist_categories) >= 1:
            if len(self.blacklist_categories) > 0:
                result += ', '
            result += f'whitelist_categories={tuple(sorted(set.intersection(*self.whitelist_categories)))}'
        if len(self.blacklist_categories) > 0 or len(self.whitelist_categories) > 0:
            result += ')'

        if (len(self.blacklist_categories) > 0 or len(self.whitelist_categories) > 0) and len(self.min_size) > 0:
            result += ', '
        if len(self.min_size) == 1:
            result += f'min_size={self.min_size[0]}'
        elif len(self.min_size) > 1:
            result += f'min_size=max({self.min_size})'

        if (len(self.blacklist_categories) > 0 or len(self.whitelist_categories) > 0 or len(self.min_size) > 0) \
                and len(self.max_size) > 0:
            result += ', '
        if len(self.max_size) == 1:
            result += f'max_size={self.max_size[0]}'
        elif len(self.max_size) > 1:
            result += f'max_size=min({self.max_size})'

        result += ')'

        for f in self.filters:
            free_variables = [self.var_id]
            free_variables.extend(f[1])
            result += f'.filter(lambda {", ".join(free_variables)}: {f[0]})'

        return result


# @dataclass
# class SymbolicTextStrategy:
#     var_id: str
#     blacklist_categories: List[Set[str]] = field(default_factory=list)
#     whitelist_categories: List[Set[str]] = field(default_factory=list)
#     min_size: List[Union[int, str]] = field(default_factory=list)
#     max_size: List[Union[int, str]] = field(default_factory=list)
#     filters: List[Tuple[str, Set[str]]] = field(default_factory=list)
#
#     def add_blacklist_categories(self, new_blacklist_categories: List[Set[str]]) -> None:
#         for new_blacklist_category in new_blacklist_categories:
#             self.blacklist_categories.append(new_blacklist_category)
#
#     def add_whitelist_categories(self, new_whitelist_categories: List[Set[str]]) -> None:
#         for new_whitelist_category in new_whitelist_categories:
#             self.whitelist_categories.append(new_whitelist_category)
#
#     def add_max_size_constraints(self, max_constraints: List[Union[int, str]]) -> None:
#         for max_constraint in max_constraints:
#             self.max_size.append(max_constraint)
#
#     def add_min_size_constraints(self, min_constraints: List[Union[int, str]]) -> None:
#         for min_constraint in min_constraints:
#             self.min_size.append(min_constraint)
#
#     def add_filter(self, new_filter: str, free_variables: Set[str]) -> None:
#         self.filters.append((new_filter, free_variables))
#
#     def get_strategy(self):
#         # def __repr__(self):
#         result = f'st.text('
#         if len(self.blacklist_categories) > 0 or len(self.whitelist_categories) > 0:
#             result += 'alphabet=st.characters('
#         if len(self.blacklist_categories) >= 1:
#             result += f'blacklist_categories={tuple(sorted(set.intersection(*self.blacklist_categories)))}'
#         if len(self.whitelist_categories) >= 1:
#             if len(self.blacklist_categories) > 0:
#                 result += ', '
#             result += f'whitelist_categories={tuple(sorted(set.intersection(*self.whitelist_categories)))}'
#         if len(self.blacklist_categories) > 0 or len(self.whitelist_categories) > 0:
#             result += ')'
#
#         if (len(self.blacklist_categories) > 0 or len(self.whitelist_categories) > 0) and len(self.min_size) > 0:
#             result += ', '
#         if len(self.min_size) == 1:
#             result += f'min_size={self.min_size[0]}'
#         elif len(self.min_size) > 1:
#             result += f'min_size=max({self.min_size})'
#
#         if (len(self.blacklist_categories) > 0 or len(self.whitelist_categories) > 0 or len(self.min_size) > 0) \
#                 and len(self.max_size) > 0:
#             result += ', '
#         if len(self.max_size) == 1:
#             result += f'max_size={self.max_size[0]}'
#         elif len(self.max_size) > 1:
#             result += f'max_size=min({self.max_size})'
#
#         result += ')'
#
#         for f in self.filters:
#             free_variables = [self.var_id]
#             free_variables.extend(f[1])
#             result += f'.filter(lambda {", ".join(free_variables)}: {f[0]})'
#
#         return result


@dataclass
class SymbolicFromRegexStrategy:
    var_id: str
    regexps: Sequence[str]
    full_match: bool
    min_size: Sequence[Union[int, str]]
    max_size: Sequence[Union[int, str]]
    filters: Sequence[Tuple[str, Set[str]]]

    def represent(self):
        # def __repr__(self):
        result = f'st.from_regex(regex=r"'
        if len(self.regexps) == 0:
            result += '.*'
        elif len(self.regexps) == 1:
            result += self.regexps[0]
        else:
            for regexp in self.regexps:
                result += f'(?={regexp})'
        result += '"'

        if self.full_match:
            result += f', fullmatch={self.full_match}'  # noqa

        result += ')'

        for f in self.filters:
            free_variables = [self.var_id]
            free_variables.extend(f[1])
            result += f'.filter(lambda {", ".join(free_variables)}: {f[0]})'

        return result


##################
# IMPLEMENTATION #
##################


def generate_strategies(table: Table):
    strategies: Dict[str, Union[SymbolicIntegerStrategy, SymbolicTextStrategy, SymbolicFromRegexStrategy]] = dict()
    for row in table.get_rows():
        if row.kind == Kind.BASE:
            strategies[row.var_id] = _infer_strategy(row)
    return strategies


def _infer_strategy(row: Row):
    if isinstance(row.type, int) or row.type == int:
        return _infer_int_strategy(row)
    elif isinstance(row.type, str) or row.type == str:
        return _infer_str_strategy(row)
    else:
        raise NotImplementedError
    # Call infer_strategy on all children of row and process them accordingly.
    # Children will be a 'link', 'universal quantifier' or 'existential quantifier' row.
    # We have to merge all these strategies into one.


############
# INTEGERS #
############


def _infer_int_strategy(row) -> SymbolicIntegerStrategy:
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
                arg - 1 if isinstance(arg, int) else f'{arg}-1'
                for arg in args
            ])
        elif row_property == '<=':
            max_value_constraints.extend(args)
        elif row_property == '>':
            min_value_constraints.extend([
                arg + 1 if isinstance(arg, int) else f'{arg}+1'
                for arg in args
            ])
        elif row_property == '>=':
            min_value_constraints.extend(args)
        else:
            filters.append((f"{row.var_id}{row_property}{args}", free_variables))
    return SymbolicIntegerStrategy(var_id=row.var_id,
                                   min_value=min_value_constraints,
                                   max_value=max_value_constraints,
                                   filters=filters)


###########
# STRINGS #
###########


def _infer_str_strategy(row: Row) -> Union[SymbolicFromRegexStrategy, SymbolicTextStrategy]:
    # TODO if text properties present, it becomes a from_text strategy, else it will be a text strategy.
    row_properties = row.properties
    if any(row_property in ['contains', 'in', 're.match', 'startswith', 'endswith']
           for row_property in row.properties.keys()):
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
            elif row_property == '<':
                # TODO
                args = [arg - 1 if isinstance(arg, int) else f'{arg}-1' for arg in args]
                raise NotImplementedError
            elif row_property == '<=':
                # TODO
                raise NotImplementedError
            elif row_property == '>':
                # TODO
                args = [arg + 1 if isinstance(arg, int) else f'{arg}+1' for arg in args]
                raise NotImplementedError
            elif row_property == '>=':
                # TODO
                raise NotImplementedError
            else:
                filter_args = ", ".join(args)
                if row.var_id in free_vars:  # s.func(..)
                    filters.append((f"{row.var_id}.{row_property}({filter_args})", free_vars))
                else:  # func(..s..)  TODO does this actually occur?
                    filters.append((f"{row_property}({filter_args})", free_vars))
                # TODO add row property as filter
        return SymbolicFromRegexStrategy(var_id=row.var_id,
                                         regexps=regexps,
                                         full_match=full_match,
                                         min_size=min_size,
                                         max_size=max_size,
                                         filters=filters)
    else:
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
            elif row_property == '<':
                # TODO
                args = [arg - 1 if isinstance(arg, int) else f'{arg}-1' for arg in args]
                raise NotImplementedError
            elif row_property == '<=':
                # TODO
                raise NotImplementedError
            elif row_property == '>':
                # TODO
                args = [arg + 1 if isinstance(arg, int) else f'{arg}+1' for arg in args]
                raise NotImplementedError
            elif row_property == '>=':
                # TODO
                raise NotImplementedError
            else:
                filter_args = ", ".join(args)
                if row.var_id in free_vars:  # s.func(..)
                    filters.append((f"{row.var_id}.{row_property}({filter_args})", free_vars))
                else:  # func(..s..)  TODO does this actually occur?
                    filters.append((f"{row_property}({filter_args})", free_vars))
                # TODO add row property as filter
        strategy = SymbolicTextStrategy(var_id=row.var_id,
                                        blacklist_categories=blacklist_categories,
                                        whitelist_categories=whitelist_categories,
                                        min_size=min_size,
                                        max_size=max_size,
                                        filters=filters)

    return strategy


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
    _, t = generate_symbol_table(example_function_4)
    print_pretty_table(t)
    print(generate_strategies(t))
    # print(typing_extensions.get_type_hints(example_function_1))
