import inspect
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, Tuple, Iterable, TypeVar, Callable, Any, cast
import typing
from enum import Enum
import ast
from icontract import require
import icontract
from tabulate import tabulate
import astunparse
from sympy import S
import regex as re

# Shamelessly stolen from icontract
CallableT = TypeVar('CallableT', bound=Callable[..., Any])


class Kind(Enum):
    BASE = 0
    ATTRIBUTE = 1
    UNIVERSAL_QUANTIFIER = 2
    DISJUNCTION = 3
    LINK = 4


@dataclass
class Row:
    var_id: str
    kind: Kind
    type: typing.Type
    function: str
    parent: Optional['Row']
    properties: Dict[str, Tuple[Set[str], Set[str]]] = field(default_factory=dict)  # operator : (arguments, vars)

    def __repr__(self) -> str:
        return f'{self.var_id}\t{self.kind}\t{self.type}\t{self.function}' \
               f'\t{self.parent}\t{self.properties}'

    def add_property(self, op: str, comp: str, variables: Set[str]) -> None:
        if op in self.properties.keys():
            new_args = self.properties[op][0]
            new_args.add(comp)
            new_vars = self.properties[op][1]
            new_vars = new_vars.union(variables)
            self.properties[op] = (new_args, new_vars)
        else:
            self.properties[op] = ({comp}, variables)

    def add_properties(self, properties: Dict[str, Tuple[Set[str], Set[str]]]) -> None:
        for op, (property_set, variables) in properties.items():
            for p in property_set:
                self.add_property(op, p, variables)


@dataclass
class Table:
    _rows: List[Row] = field(default_factory=list)

    def get_rows(self) -> List[Row]:
        return self._rows

    def get_row_by_var_id(self, var_id: str) -> Optional[Row]:
        for row in self._rows:
            if row.var_id == var_id:
                return row

    @require(lambda self, index: 0 <= index < len(self._rows))
    def get_row_by_index(self, index: int) -> Row:
        return self._rows[index]

    def add_row(self, row: Row) -> None:
        existing_row = self.get_row_by_var_id(row.var_id)
        # If row already exists (same id and corresponding function), the properties are added to the existing row.
        if existing_row and existing_row.function == row.function:
            existing_row.add_properties(row.properties)
        else:
            self._rows.append(row)

    def __repr__(self) -> str:
        result = 'IDX\tVAR_ID\tENTRY_TYPE\tTYPE_HINT\tFUNCTION\tPARENT\tPROPERTIES\n'
        for idx, row in enumerate(self._rows):
            result += f'{idx}\t{row}\n'
        return result


def extract_variables_from_expression(root: ast.AST) -> Set[str]:
    variables_with_functions = set(sorted({node.id for node in ast.walk(root) if isinstance(node, ast.Name)}))
    if isinstance(root, ast.Constant):
        return set()
    free_symbols = set(map(str, S(astunparse.unparse(root)).free_symbols))
    return free_symbols.intersection(variables_with_functions)


def visualize_operation(op: ast.Expression) -> str:
    if isinstance(op, ast.Gt):
        return '>'
    elif isinstance(op, ast.GtE):
        return '>='
    elif isinstance(op, ast.Lt):
        return '<'
    elif isinstance(op, ast.LtE):
        return '<='
    elif isinstance(op, ast.Eq):
        return '=='
    elif isinstance(op, ast.Add):
        return '+'
    elif isinstance(op, ast.Sub):
        return '-'
    elif isinstance(op, ast.Mult):
        return '*'
    elif isinstance(op, ast.Div):
        return '/'
    else:
        # If it is not a operator that is handled above, we simply return the expression as a string.
        return str(op)


# TODO needs a better name
def visualize_comparators(comp: ast.Expression) -> str:
    if isinstance(comp, ast.Name):
        return comp.id
    elif isinstance(comp, ast.Constant):
        return comp.value
    elif isinstance(comp, ast.BinOp):
        return f'{visualize_comparators(comp.left)} {visualize_operation(comp.op)} {visualize_comparators(comp.right)}'
    else:
        # If it is not a expression that is handled above, we simply return the expression as a string.
        return str(comp)


def parse_comparison(expr: ast.Compare) -> List[Row]:
    rows: List[Row] = []
    left = expr.left
    ops = expr.ops
    comparators = expr.comparators
    for op in ops:
        if isinstance(left, ast.Name):  # Only add a new row if the left side is a variable
            var_id = left.id
            row_kind = Kind.BASE
            add_to_rows = True
        elif isinstance(left, ast.Call):
            var_id = f'{left.args[0].id}.{left.func.id}'
            row_kind = Kind.LINK
            add_to_rows = True
        else:
            var_id = None
            row_kind = None
            add_to_rows = False

        if add_to_rows:
            row = Row(var_id,
                      row_kind,
                      # TODO not correct, this should be passed on, or I should get this information from elsewhere
                      int,
                      'dummy_function',
                      None,
                      {visualize_operation(op): ({visualize_comparators(comparators[0])},
                                                 extract_variables_from_expression(comparators[0]))})
            rows.append(row)

        left = comparators[0]
        comparators = comparators[1:]
    return rows


# TODO this will probably has to become more precise to handle different arguments, ...
@require(lambda expr: isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute))
def parse_attribute(expr: ast.Call) -> List[Row]:
    assert len(expr.args) == 1
    return [Row(f'{visualize_comparators(expr.func.value)}',
                Kind.ATTRIBUTE,
                str,
                'dummy_function',
                None,
                {expr.func.attr: ({visualize_comparators(expr.args[0])},
                                  extract_variables_from_expression(expr.args[0]))})]


@require(lambda expr: isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == 'all'
                      and isinstance(expr.args[0], ast.GeneratorExp))
def parse_universal_quantifier(expr: ast.expr) -> List[Row]:
    target = expr.args[0].generators[0].target  # base
    it = expr.args[0].generators[0].iter  # parent
    compare = expr.args[0].elt
    rows = parse_expression(compare)
    for row in rows:
        # TODO more checks needed
        row.parent = visualize_comparators(it)
        row.kind = Kind.UNIVERSAL_QUANTIFIER
    return rows


@require(lambda expr: isinstance(expr, ast.BoolOp))
def parse_boolean_operator(expr: ast.expr) -> List[Row]:
    """Only AND-operations are currently supported."""
    if not isinstance(expr.op, ast.And):
        assert NotImplementedError(f'{expr.op} is not supported, only AND-operations are currently supported.')
    else:
        result = []
        for clause in expr.values:
            result.extend(parse_expression(clause))
        return result


def parse_expression(expr: ast.expr) -> List[Row]:
    rows: List[Row] = []
    if isinstance(expr, ast.Compare):
        rows.extend(parse_comparison(expr))
    elif isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
        rows.extend(parse_attribute(expr))
    elif isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == 'all' \
            and isinstance(expr.args[0], ast.GeneratorExp):
        rows.extend(parse_universal_quantifier(expr))
    elif isinstance(expr, ast.BoolOp):
        rows.extend(parse_boolean_operator(expr))
    else:
        raise NotImplementedError('Only comparisons and var with attribute calls are currently supported')

    return rows


def generate_symbol_table(func: CallableT, func_name: Optional[str] = None) -> Table:
    table = Table()

    preconditions = get_contracts(func)

    for conjunction in preconditions:
        for contract in conjunction:
            body_node = _body_node_from_condition(condition=contract.condition)
            if body_node:
                rows = parse_expression(body_node)
                for row in rows:
                    if func_name:
                        row.function = func_name
                    table.add_row(row)
    return table


def print_pretty_table(table: Table) -> None:
    rows = list(map(lambda row: [table.get_rows().index(row), row.var_id, row.kind, row.type, row.function,
                                 row.parent, row.properties],
                    table.get_rows()))
    pretty_table = tabulate(rows,
                            headers=['IDX', 'VAR_ID', 'KIND', 'TYPE', 'FUNCTION', 'PARENT', 'PROPERTIES'])
    print(pretty_table)


#############################
# Connection with contracts #
#############################


# Shamelessly stolen from icontract
def find_checker(func: CallableT) -> Optional[CallableT]:
    """Iterate through the decorator stack till we find the contract checker."""
    contract_checker = None  # type: Optional[CallableT]
    for a_wrapper in _walk_decorator_stack(func):
        if hasattr(a_wrapper, "__preconditions__") or hasattr(a_wrapper, "__postconditions__"):
            contract_checker = a_wrapper

    return contract_checker


# Shamelessly stolen from icontract
def _walk_decorator_stack(func: CallableT) -> Iterable['CallableT']:
    """
    Iterate through the stack of decorated functions until the original function.

    Assume that all decorators used functools.update_wrapper.
    """
    while hasattr(func, "__wrapped__"):
        yield func

        func = getattr(func, "__wrapped__")

    yield func


# Shamelessly stolen from icontract
def get_contracts(func: CallableT) -> Optional[List[List[icontract._types.Contract]]]:
    checker = find_checker(func)

    preconditions = None  # type: Optional[List[List[icontract._types.Contract]]]

    if checker:
        maybe_preconditions = getattr(checker, "__preconditions__", None)

        if maybe_preconditions is not None:
            assert isinstance(maybe_preconditions, list)
            assert all(isinstance(conjunction, list) for conjunction in maybe_preconditions)
            assert all(
                isinstance(contract, icontract._types.Contract)
                for conjunction in maybe_preconditions
                for contract in conjunction
            )

            preconditions = cast(List[List[icontract._types.Contract]], maybe_preconditions)

    return preconditions


# Shamelessly stolen from icontract
def _body_node_from_condition(condition: Callable[..., Any]) -> Optional[ast.expr]:
    """Try to extract the body node of the contract's lambda condition."""
    if not icontract._represent.is_lambda(a_function=condition):
        return None

    lines, condition_lineno = inspect.findsource(condition)
    filename = inspect.getsourcefile(condition)
    assert filename is not None

    decorator_inspection = icontract._represent.inspect_decorator(
        lines=lines, lineno=condition_lineno, filename=filename
    )
    lambda_inspection = icontract._represent.find_lambda_condition(
        decorator_inspection=decorator_inspection
    )

    assert (
            lambda_inspection is not None
    ), "Expected lambda_inspection to be non-None if _is_lambda is True on: {}".format(
        condition
    )

    body_node = lambda_inspection.node.body

    return body_node


############
# EXAMPLES #
############


def example_table_1() -> Table:
    t = Table()
    expressions = [
        'n1 > n2 > 4',
        'n1 < 100',
        'n1 < n4',
        'n2 < 300 + n3',
        'n3 < n4 > n1',
        's.startswith("abc")',
        'len(lst) > 0',
        '4 < n1'
    ]

    for expression in expressions:
        body = ast.parse(expression, mode='eval').body
        for r in parse_expression(body):
            t.add_row(r)

    return t


@require(lambda n1: n1 > 0)
def example_function_1(n1: int) -> None:
    pass


@require(lambda n1, n2: n1 > n2 > 4)
@require(lambda n1: n1 < 100)
@require(lambda n1, n4: n1 < n4)
@require(lambda n2, n3: n2 < 300 + n3)
@require(lambda n1, n3, n4: n3 < n4 > n1)
@require(lambda s: s.startswith("abc"))
@require(lambda lst: len(lst) > 0)
@require(lambda n1: 4 < n1)
def example_function_2(n1: int, n2: int, n3: int, n4: int, s: str, lst: List[int]) -> None:
    pass


@require(lambda lst: all(item > 0 for item in lst))
def example_function_3(lst: List[int]) -> None:
    pass


@require(lambda name: re.compile(r'[a-zA-Z]+').match(name))
def example_function_4(name: str) -> None:
    pass


if __name__ == '__main__':
    # print("\n\nexample 1:\n")
    # print_pretty_table(generate_symbol_table(example_function_1))
    # print("\n\nexample 2:\n")
    # print_pretty_table(generate_symbol_table(example_function_2))
    # print("\n\nexample 3:\n")
    # print_pretty_table(generate_symbol_table(example_function_3))
    print("\n\nexample 4:\n")
    print_pretty_table(generate_symbol_table(example_function_4))
