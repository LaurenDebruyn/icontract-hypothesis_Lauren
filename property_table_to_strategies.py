from icontract import require
from generate_symbol_table import Table, Row, Kind
from typing import List, Union, Dict, Any, Tuple, Set, Optional
from icontract_hypothesis_Lauren.generate_symbol_table import generate_symbol_table, print_pretty_table
from dataclasses import dataclass, field
import ast


###########
# CLASSES #
###########


@dataclass
class SymbolicIntegerStrategy:
    var_id: str
    min_value: List[Union[int, str]] = field(default_factory=list)
    max_value: List[Union[int, str]] = field(default_factory=list)
    filters: List[Tuple[str, Set[str]]] = field(default_factory=list)

    def add_min_value_constraints(self, min_constraints: List[Union[int, str]]) -> None:
        for min_constraint in min_constraints:
            self.min_value.append(min_constraint)

    def add_max_value_constraints(self, max_constraints: List[Union[int, str]]) -> None:
        for max_constraint in max_constraints:
            self.max_value.append(max_constraint)

    def add_filter(self, new_filter: str, free_variables: Set[str]) -> None:
        self.filters.append((new_filter, free_variables))

    def get_strategy(self):
    # def __repr__(self):
        result = f'st.integers('
        if len(self.min_value) == 1:
            result += f'min_value={self.min_value[0]}'
        elif len(self.min_value) > 1:
            result += f'min_value=max({", ".join(str(e) for e in self.min_value)})'
        if len(self.min_value) > 0 and len(self.max_value) > 0:
            result += ', '
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
    blacklist_categories: List[Set[str]] = field(default_factory=list)
    whitelist_categories: List[Set[str]] = field(default_factory=list)
    min_size: List[Union[int, str]] = field(default_factory=list)
    max_size: List[Union[int, str]] = field(default_factory=list)
    filters: List[Tuple[str, Set[str]]] = field(default_factory=list)

    def add_blacklist_categories(self, new_blacklist_categories: List[Set[str]]) -> None:
        for new_blacklist_category in new_blacklist_categories:
            self.blacklist_categories.append(new_blacklist_category)

    def add_whitelist_categories(self, new_whitelist_categories: List[Set[str]]) -> None:
        for new_whitelist_category in new_whitelist_categories:
            self.whitelist_categories.append(new_whitelist_category)

    def add_max_size_constraints(self, max_constraints: List[Union[int, str]]) -> None:
        for max_constraint in max_constraints:
            self.max_size.append(max_constraint)

    def add_min_size_constraints(self, min_constraints: List[Union[int, str]]) -> None:
        for min_constraint in min_constraints:
            self.min_size.append(min_constraint)

    def add_filter(self, new_filter: str, free_variables: Set[str]) -> None:
        self.filters.append((new_filter, free_variables))

    def get_strategy(self):
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


@dataclass
class SymbolicFromRegexStrategy:
    var_id: str
    regexps: List[str] = field(default_factory=list)
    full_match: bool = False
    # TODO introduce extra field for min and max size?
    filters: List[Tuple[str, Set[str]]] = field(default_factory=list)

    def add_regexps(self, regexps: List[str], full_match: bool) -> None:
        self.regexps.extend(regexps)
        self.full_match = self.full_match or full_match

    def add_filter(self, new_filter: str, free_variables: Set[str]) -> None:
        self.filters.append((new_filter, free_variables))

    def get_strategy(self):
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
            strategies[row.var_id] = infer_strategy(row)
    return strategies


def infer_strategy(row: Row):
    if isinstance(row.type, int) or row.type == int:
        return infer_int_strategy(row)
    elif isinstance(row.type, str) or row.type == str:
        return infer_str_strategy(row)
    else:
        raise NotImplementedError
    # Call infer_strategy on all children of row and process them accordingly.
    # Children will be a 'link', 'universal quantifier' or 'existential quantifier' row.
    # We have to merge all these strategies into one.


############
# INTEGERS #
############


def infer_int_strategy(row) -> SymbolicIntegerStrategy:
    row_properties = row.properties
    strategy = SymbolicIntegerStrategy(row.var_id)
    for row_property, (args, _) in row_properties.items():
        # TODO this should already have happened in the property table
        args = list(map(lambda arg: int(arg) if arg.isnumeric() else arg, args))
        if row_property in int_properties:
            strategy_attribute, attribute_args = int_properties[row_property](args)
            if strategy_attribute == 'min_value':
                strategy.add_min_value_constraints(attribute_args)
            elif strategy_attribute == 'max_value':
                strategy.add_max_value_constraints(attribute_args)
            else:
                raise NotImplementedError  # TODO define better error
        else:
            # TODO add filter
            raise NotImplementedError

    return strategy


def int_lt(args: List[Union[int, str]]) -> Tuple[str, List[Any]]:
    args = list(map(lambda arg: arg - 1 if isinstance(arg, int) else f'{arg}-1', args))
    return 'max_value', args


def int_lte(args: List[Union[int, str]]) -> Tuple[str, List[Any]]:
    return 'max_value', args


def int_gt(args: List[Union[int, str]]) -> Tuple[str, List[Any]]:
    args = list(map(lambda arg: arg + 1 if isinstance(arg, int) else f'{arg}+1', args))
    return 'min_value', args


def int_gte(args: List[Union[int, str]]) -> Tuple[str, List[Any]]:
    return 'min_value', args


int_properties = {'<': int_lt,
                  '<=': int_lte,
                  '>': int_gt,
                  '>=': int_gte}


###########
# STRINGS #
###########


def infer_str_strategy(row) -> Union[SymbolicFromRegexStrategy, SymbolicTextStrategy]:
    # TODO if text properties present, it becomes a from_text strategy, else it will be a text strategy.
    row_properties = row.properties
    if is_regex_strategy_needed(row):
        strategy = SymbolicFromRegexStrategy(row.var_id)

        for row_property, (args, _) in row_properties.items():
            # TODO check if row property in regex_properties, else add filter
            strategy_attribute, (regexps, full_match) = regex_properties[row_property](args)  # TODO can I simply pass args?
            if strategy_attribute == 'regex':
                strategy.add_regexps([regexps], full_match)
            elif strategy_attribute == 'regexps':
                strategy.add_regexps(regexps, full_match)
            else:
                raise NotImplementedError  # TODO better fault handling
    else:
        strategy = SymbolicTextStrategy(row.var_id)

        for row_property, (args, free_vars) in row_properties.items():
            if row_property in text_properties:
                strategy_attribute, attribute_args = text_properties[row_property]()
                if strategy_attribute == 'blacklist_categories':
                    strategy.add_blacklist_categories([attribute_args])
                elif strategy_attribute == 'whitelist_categories':
                    strategy.add_whitelist_categories([attribute_args])
                elif strategy_attribute == 'min_size':
                    assert isinstance(attribute_args, str) or isinstance(attribute_args, int)
                    strategy.add_min_size_constraints([attribute_args])
                elif strategy_attribute == 'max_size':
                    assert isinstance(attribute_args, str) or isinstance(attribute_args, int)
                    strategy.add_max_size_constraints([attribute_args])
                else:
                    raise NotImplementedError  # TODO define better exception
            else:
                filter_args = ", ".join(args)
                if row.var_id in free_vars:  # s.func(..)
                    strategy.add_filter(f"{row.var_id}.{row_property}({filter_args})", free_vars)
                else:  # func(..s..)  TODO does this actually occur?
                    strategy.add_filter(f"{row_property}({filter_args})", free_vars)
                # TODO add row property as filter
                # raise NotImplementedError(f'row property: {row_property}, text properties: {text_properties.keys()}')

    return strategy


def is_regex_strategy_needed(row: Row) -> bool:
    regex_only_properties = ['contains', 'in', 're.match', 'startswith', 'endswith']
    if any(row_property in regex_only_properties for row_property in row.properties.keys()):
        return True
    else:
        return False


def text_isalnum() -> Tuple[str, Set[str]]:
    return 'whitelist_categories', {'Ll', 'Lu', 'Nd'}


def text_isalpha() -> Tuple[str, Set[str]]:
    return 'whitelist_categories', {'Ll', 'Lu'}


def text_isdigit() -> Tuple[str, Set[str]]:
    return 'whitelist_categories', {'Nd'}


def text_islower() -> Tuple[str, Set[str]]:
    return 'whitelist_categories', {'Ll'}


def text_isnumeric() -> Tuple[str, Set[str]]:
    return 'whitelist_categories', {'Nd', 'Nl', 'No'}


def text_isspace() -> Tuple[str, Set[str]]:
    return 'whitelist_categories', {'Zs'}


def text_isupper() -> Tuple[str, Set[str]]:
    return 'whitelist_categories', {'Lu'}


def text_isdecimal() -> Tuple[str, Set[str]]:
    return 'whitelist_categories', {'Nd'}


def text_len_lt(args: List[Union[int, str]]) -> Tuple[str, List[Any]]:
    return 'max_size', args


def text_len_lte(args: List[Union[int, str]]) -> Tuple[str, List[Any]]:
    args = list(map(lambda arg: arg - 1 if isinstance(arg, int) else f'{arg}-1', args))
    return 'max_size', args


def text_len_gt(args: List[Union[int, str]]) -> Tuple[str, List[Any]]:
    return 'min_size', args


def text_len_gte(args: List[Union[int, str]]) -> Tuple[str, List[Any]]:
    args = list(map(lambda arg: arg + 1 if isinstance(arg, int) else f'{arg}+1', args))
    return 'min_size', args


def regex_isalnum() -> Tuple[str, Tuple[str, bool]]:
    return 'regex', (r'^[0-9a-zA-Z]+$', True)


def regex_isalpha() -> Tuple[str, Tuple[str, bool]]:
    return 'regex', (r'^[a-zA-Z]+$', True)


def regex_isdigit() -> Tuple[str, Tuple[str, bool]]:
    return 'regex', (r'^[0-9]*$', True)


def regex_islower() -> Tuple[str, Tuple[str, bool]]:
    return 'regex', (r'^[a-z]$', True)


def regex_isnumeric() -> Tuple[str, Tuple[str, bool]]:
    return 'regex', (r'^(-[0-9]*|[0-9]*)$', True)


def regex_isspace() -> Tuple[str, Tuple[str, bool]]:
    return 'regex', (r'^\s+$', True)


def regex_isupper() -> Tuple[str, Tuple[str, bool]]:
    return 'regex', (r'^[A-Z]+$', True)


def regex_isdecimal() -> Tuple[str, Tuple[str, bool]]:
    return 'regex', (r'^\d*\.?\d+$', True)


def regex_re_match(args: List[str]) -> Tuple[str, Tuple[List[str], bool]]:
    # re.match(r'..', s), we only want the first argument and we don't want any leading/ending \'
    return 'regexps', (list(map(lambda arg: arg[0].strip("\'"), args)), True)


def regex_contains(args: List[str]) -> Tuple[str, Tuple[List[str], bool]]:
    args = list(map(lambda arg: arg.strip("\'"), args))
    return 'regexps', (args, False)


def regex_in(args: List[str]) -> Tuple[str, Tuple[List[str], bool]]:
    return regex_contains(args)


def regex_startswith(args: List[str]) -> Tuple[str, Tuple[List[str], bool]]:
    args = list(map(lambda arg: arg.strip("\'"), args))
    return 'regexps', (list(map(lambda arg: f'^{arg}', args)), False)


def regex_endswith(args: List[str]) -> Tuple[str, Tuple[List[str], bool]]:
    args = list(map(lambda arg: arg.strip("\'"), args))
    return 'regexps', (list(map(lambda arg: f'.*{arg}$', args)), True)


def regex_len_lt(args: List[Union[int, str]]) -> Tuple[str, List[Any]]:
    return 'max_size', args


def regex_len_lte(args: List[Union[int, str]]) -> Tuple[str, List[Any]]:
    args = list(map(lambda arg: arg - 1 if isinstance(arg, int) else f'{arg}-1', args))
    return 'max_size', args


def regex_len_gt(args: List[Union[int, str]]) -> Tuple[str, List[Any]]:
    return 'min_size', args


def regex_len_gte(args: List[Union[int, str]]) -> Tuple[str, List[Any]]:
    args = list(map(lambda arg: arg + 1 if isinstance(arg, int) else f'{arg}+1', args))
    return 'min_size', args


text_properties = {
    'isalnum': text_isalnum,
    'isalpha': text_isalpha,
    'isdigit': text_isdigit,
    'islower': text_islower,
    'isnumeric': text_isnumeric,
    'isspace': text_isspace,
    'isupper': text_isupper,
    'isdecimal': text_isdecimal,
    'len_lt': text_len_lt,
    'len_lte': text_len_lte,
    'len_gt': text_len_gt,
    'len_gte': text_len_gte,
}

regex_properties = {
    'isalnum': regex_isalnum,
    'isalpha': regex_isalpha,
    'isdigit': regex_isdigit,
    'islower': regex_islower,
    'isnumeric': regex_isnumeric,
    'isspace': regex_isspace,
    'isupper': regex_isupper,
    'isdecimal': regex_isdecimal,
    're.match': regex_re_match,
    'contains': regex_contains,
    'in': regex_in,
    'startswith': regex_startswith,
    'endswith': regex_endswith,
    'len_lt': regex_len_lt,
    'len_lte': regex_len_lte,
    'len_gt': regex_len_gt,
    'len_gte': regex_len_gte,
}


#########
# TESTS #
#########


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
