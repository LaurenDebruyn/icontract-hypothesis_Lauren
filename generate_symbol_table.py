import inspect
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, Tuple, Iterable, TypeVar, Callable, Any, cast, Union
import typing
from enum import Enum
import ast
from icontract import require
import icontract
from tabulate import tabulate
import astunparse
import networkx as nx

# Shamelessly stolen from icontract
CallableT = TypeVar('CallableT', bound=Callable[..., Any])


class Kind(Enum):
    BASE = 0
    ATTRIBUTE = 1  # TODO remove?
    UNIVERSAL_QUANTIFIER = 2
    DISJUNCTION = 3  # TODO remove?
    LINK = 4
    EXISTENTIAL_QUANTIFIER = 5

    # def __eq__(self, other):
    #     return self.name == other.name and self.value == other.value


@dataclass
class Row:
    var_id: str
    kind: Kind
    type: typing.Type
    function: str
    parent: Optional['str']
    properties: Dict[str, Tuple[Set[Union[str, Tuple[str, ...]]], Set[str]]] = field(
        default_factory=dict)  # operator : (arguments, vars)

    def __repr__(self) -> str:
        return f'{self.var_id}\t{self.kind}\t{self.type}\t{self.function}' \
               f'\t{self.parent}\t{self.properties}'

    def add_property(self, op: str, comp: Union[str, Tuple[str, ...]], variables: Set[str]) -> None:
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
            if isinstance(property_set, Tuple):
                self.add_property(op, property_set, variables)
            else:
                for p in property_set:
                    self.add_property(op, p, variables)

    def get_dependencies(self) -> Set[str]:
        dependencies: Set[str] = set()
        for p in self.properties.values():
            dependencies = dependencies.union(p[1])
        return dependencies


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


def extract_variables_from_expression(root: ast.expr, function_args_hints: Dict[str, Any]) -> Set[str]:
    free_symbols = set(sorted({node.id for node in ast.walk(root) if isinstance(node, ast.Name)}))
    free_symbols = free_symbols.intersection(function_args_hints.keys())
    return free_symbols


def visualize_operation(op: Union[ast.operator, ast.cmpop]) -> str:
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
    elif isinstance(op, ast.In):
        return 'in'
    else:
        # If it is not a operator that is handled above, we simply return the expression as a string.
        return str(op)


# TODO needs a better name
def visualize_comparators(comp: ast.expr) -> str:
    if isinstance(comp, ast.Name):
        return comp.id
    elif isinstance(comp, ast.Constant):
        if isinstance(comp.value, str):
            return "\"{}\"".format(comp.value.replace('\n', '\\ n'))
        else:
            return str(comp.value)
    elif isinstance(comp, ast.BinOp):
        return f'{visualize_comparators(comp.left)} {visualize_operation(comp.op)} {visualize_comparators(comp.right)}'
    elif isinstance(comp, ast.Call):
        func = comp.func

        if isinstance(func, ast.Attribute):
            result = func.attr
            attribute = func.value
            while isinstance(attribute, ast.Attribute):
                result = f'{attribute.attr}.{result}'
                attribute = attribute.value
            assert isinstance(attribute, ast.Name)

            return f'{result}({attribute.id})'

        argument = comp.args[0]
        if isinstance(argument, ast.Call):
            assert isinstance(func, ast.Name)

            result = func.id
            nb_parentheses = 1
            while isinstance(argument, ast.Call):
                nb_parentheses += 1
                func = argument.func
                assert isinstance(func, ast.Name)
                result = f'{result}({func.id}'
                argument = argument.args[0]
            assert isinstance(argument, ast.Name)
            result = f'{result}({argument.id}' + (nb_parentheses * ')')
            return result
        elif isinstance(argument, ast.Name) and isinstance(func, ast.Name):
            return f'{func.id}({argument.id})'
        else:
            return str(comp)
    else:
        # If it is not a expression that is handled above, we simply return the expression as a string.
        return str(comp)


@require(lambda expr: len(expr.ops) == 1 and len(expr.comparators) == 1)
def _parse_single_compare(expr: ast.Compare, condition: Callable[..., Any], function_args_hints: Dict[str, Any]) -> Optional[List[Row]]:
    left = expr.left
    op = expr.ops[0]
    right = expr.comparators[0]

    parent = None

    rows = []

    # Add checks for sorted and is_unique
    if isinstance(op, ast.Eq):
        # is unique
        if isinstance(right, ast.Call) and isinstance(right.func, ast.Name) \
                and right.func.id == 'set':
            var_id = astunparse.unparse(left)
            row = Row(var_id,
                      Kind.BASE,
                      function_args_hints[var_id],
                      'dummy_function',
                      None,
                      {'is_unique': (set(), set())})
            rows.append(row)

    if isinstance(left, ast.Name):
        var_id = left.id
        row_kind = Kind.BASE
        add_to_rows = True
    elif isinstance(left, ast.Call):
        argument = left.args[0]
        func = left.func
        if isinstance(argument, ast.Name) and isinstance(func, ast.Name):
            var_id = f'{argument.id}.{func.id}'
            parent = argument.id
        elif isinstance(left, ast.Call):
            assert isinstance(func, ast.Name)
            assert func.id == 'len'
            var_id = f'{func.id}({visualize_comparators(argument)})'
        else:
            raise NotImplementedError
        row_kind = Kind.LINK
        add_to_rows = True
        function_args_hints[var_id] = int
    elif isinstance(left, ast.Subscript):
        value = left.value
        if isinstance(value, ast.Name):
            var_id_base = value.id
        else:
            raise NotImplementedError(f'Value is expected to be ast.Name and not {value}')
        idx = left.slice
        if isinstance(idx, ast.Index):
            assert isinstance(idx.value, ast.Num)
            var_id = f'{var_id_base}[{idx.value.n}]'
            type_hint = typing.get_args(function_args_hints[var_id_base])[int(idx.value.n.real)]
            function_args_hints[var_id] = type_hint
        else:
            raise NotImplementedError(f'Slice is expected to be ast.Index and not {slice}.'
                                      f'Slices are currently not supported.')
        add_to_rows = True
        row_kind = Kind.LINK
        parent = var_id_base
    elif isinstance(left, ast.Tuple):
        assert all(isinstance(el, ast.Name) for el in left.elts)
        left_hand = list(map(lambda el: el.id, left.elts))
        comps = right
        assert isinstance(comps, ast.Tuple)
        right_hand = list(map(lambda el: visualize_comparators(el), comps.elts))
        assert len(left_hand) == len(right_hand)
        for (l, r) in zip(left_hand, right_hand):
            new_expr_ast = ast.parse(f'{l} {visualize_operation(op)} {r}', mode="eval")
            assert isinstance(new_expr_ast, ast.Expression)
            rows.extend(parse_expression(new_expr_ast.body, condition, function_args_hints))
        var_id = None
        row_kind = None
        add_to_rows = False
    else:
        raise NotImplementedError

    if add_to_rows:
        row = Row(var_id,
                  row_kind,
                  function_args_hints[var_id],
                  'dummy_function',  # This is fixed on a higher level
                  parent if parent else None,
                  {visualize_operation(op): ({visualize_comparators(right)},
                                             extract_variables_from_expression(right,
                                                                               function_args_hints))})
        rows.append(row)

    return rows


def parse_comparison(
        expr: ast.Compare,
        condition: Callable[..., Any],
        function_args_hints: Dict[str, Any]
) -> List[Row]:
    rows: List[Row] = []
    left = expr.left
    ops = expr.ops
    comparators = expr.comparators
    for op in ops:
        comparison = ast.Compare(left, [op], [comparators[0]])
        rows.extend(_parse_single_compare(comparison, condition, function_args_hints))
        left = comparators[0]
        comparators = comparators[1:]
    return rows


# TODO this will probably has to become more precise to handle different arguments, ...
@require(lambda expr: isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute))
def parse_attribute(expr: ast.Call, condition: Callable[..., Any], function_args_hints: Dict[str, Any]) -> List[Row]:
    rows: List[Row] = []
    variables = extract_variables_from_expression(expr, function_args_hints)
    if isinstance(expr.func, ast.Attribute):
        expr_unparsed = astunparse.unparse(expr)
        method_name, arguments = expr_unparsed[:expr_unparsed.index('(')], expr_unparsed[expr_unparsed.index('('):]
        arguments = arguments.rstrip('\n')[1:-1].replace(', ', ',').split(',')
        if len(arguments) > 1:
            arguments = tuple(arguments)
        for variable in variables:
            variables_without_variable = variables.copy()
            if method_name.startswith(variable):
                variables_without_variable.remove(variable)
                # method_name = method_name[len(variable):]
                method_name = method_name[len(variable)+1:]  # +1 to take into account dot between variable and method
            row = Row(variable,
                      Kind.BASE,
                      function_args_hints[variable],
                      'dummy_function',
                      None,
                      {method_name: (arguments, variables_without_variable)})
            rows.append(row)
    elif isinstance(expr.func, ast.Name):
        raise NotImplementedError
    else:
        raise NotImplementedError(f'expected an attribute or name but got {expr.func}')

    return rows


@require(lambda expr:
         isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and isinstance(expr.args[0], ast.GeneratorExp)
         and (expr.func.id == 'all' or expr.func.id == 'any'))
def parse_universal_quantifier(expr: ast.Call, condition: Callable[..., Any], function_args_hints: Dict[str, Any]) -> \
        List[Row]:
    generator_expr = expr.args[0]
    assert isinstance(generator_expr, ast.GeneratorExp)
    target = generator_expr.generators[0].target  # base
    it = generator_expr.generators[0].iter  # parent
    if not isinstance(target, ast.Name):
        # TODO handle this better? This happens for example in the following case:
        #   all( .. for idx, item in enumerate(..))
        raise NotImplementedError

    # TODO Make this work for different sub type hints (e.g. Dict[int, str])
    # update type hints
    if isinstance(it, ast.Name):
        if typing.get_args(function_args_hints[it.id]):
            assert len(set(typing.get_args(function_args_hints[it.id]))) == 1, \
                "We assume all sub type hints are the same."
            function_args_hints[target.id] = typing.get_args(function_args_hints[it.id])[0]
        else:
            function_args_hints[target.id] = function_args_hints[it.id]
    elif isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute):  # Is this a correct assumption
        value = it.func.value
        while isinstance(value, ast.Attribute):
            value = value.value
        assert isinstance(value, ast.Name)
        if typing.get_args(function_args_hints[value.id]):
            assert len(set(typing.get_args(function_args_hints[value.id]))) == 1, \
                "We assume all sub type hints are the same."
            function_args_hints[target.id] = typing.get_args(function_args_hints[value.id])[0]
        else:
            function_args_hints[target.id] = function_args_hints[value.id]
    elif isinstance(it, ast.Call) and isinstance(it.func, ast.Name):  # Make difference between attributes and calls
        args_value = it.args[0]
        while isinstance(args_value, ast.Call):
            args_value = args_value.args[0]
        assert isinstance(args_value, ast.Name)
        # assert len(set(typing.get_args(function_args_hints[args_value.id]))) == 1, \
        #     "We assume all sub type hints are the same."
        function_args_hints[target.id] = function_args_hints[args_value.id]  # TODO is this correct?
    else:
        raise NotImplementedError

    # Create the row for the quantifier
    row_parent = ''
    if isinstance(it, ast.Call):
        ast_node = it
        if isinstance(ast_node.func, ast.Attribute):
            if ast_node.args:  # TODO check if more than one argument!!
                row_parent = f'({visualize_comparators(ast_node.args[0])})'
            else:
                row_parent = f'()'
            ast_node = it.func
            while isinstance(ast_node, ast.Attribute):  # TODO replace this with visualize_comparators
                row_parent = f'.{ast_node.attr}{row_parent}'
                ast_node = ast_node.value
            assert isinstance(ast_node, ast.Name)
            row_parent = f'{ast_node.id}{row_parent}'
        elif isinstance(ast_node, ast.Call):
            nb_parentheses = 0
            while isinstance(ast_node, ast.Call):  # TODO replace this with visualize_comparators
                assert isinstance(ast_node.func, ast.Name)
                nb_parentheses += 1
                row_parent = f'{row_parent}{ast_node.func.id}('
                ast_node = ast_node.args[0]
            assert isinstance(ast_node, ast.Name)
            row_parent = f'{row_parent}{ast_node.id}{nb_parentheses * ")"}'
        else:
            assert isinstance(ast_node, ast.Name)
            row_parent = ast_node.id
    else:
        assert isinstance(it, ast.Name)
        row_parent = it.id

    assert isinstance(expr.func, ast.Name)
    quantifier_row = Row(target.id,
                         Kind.UNIVERSAL_QUANTIFIER if expr.func.id == 'all' else Kind.EXISTENTIAL_QUANTIFIER,
                         function_args_hints[target.id],
                         'dummy_function',
                         row_parent,
                         {})

    # create all rows linked to the quantifier row
    compare = generator_expr.elt
    rows = parse_expression(compare, condition, function_args_hints)
    for row in rows:
        if row.kind not in [Kind.UNIVERSAL_QUANTIFIER, Kind.EXISTENTIAL_QUANTIFIER]:
            row.kind = Kind.LINK
        row.parent = quantifier_row.var_id

    rows.insert(0, quantifier_row)

    return rows


def parse_boolean_operator(expr: ast.BoolOp, condition: Callable[..., Any], function_args_hints: Dict[str, Any]) -> \
        List[Row]:
    """Only AND-operations are currently supported."""
    if not isinstance(expr.op, ast.And):
        raise NotImplementedError(f'{expr.op} is not supported, only AND-operations are currently supported.')
    else:
        result = []
        for clause in expr.values:
            result.extend(parse_expression(clause, condition, function_args_hints))
        return result


def parse_expression(expr: ast.expr, condition: Callable[..., Any], function_args_hints: Dict[str, Any]) -> List[Row]:
    if isinstance(expr, ast.Compare):
        return parse_comparison(expr, condition, function_args_hints)
    elif isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
        return parse_attribute(expr, condition, function_args_hints)
    elif isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and \
            (expr.func.id == 'all' or expr.func.id == 'any') and isinstance(expr.args[0], ast.GeneratorExp):
        return parse_universal_quantifier(expr, condition, function_args_hints)
    elif isinstance(expr, ast.BoolOp):
        return parse_boolean_operator(expr, condition, function_args_hints)
    else:
        raise NotImplementedError('Only comparisons and var with attribute calls are currently supported')


def generate_symbol_table(func: CallableT) -> Tuple[List[Tuple[ast.AST, Optional[str]]], Table]:
    table = Table()
    # failed_contracts: List[Tuple[str, Optional[str]]] = []
    failed_contracts: List[Tuple[ast.AST, Optional[str]]] = []

    # Initialize table with an empty (no properties) row for each argument.
    args = inspect.signature(func).parameters
    type_hints = typing.get_type_hints(func)
    function_name = func.__name__

    if 'return' in type_hints:
        del type_hints['return']

    # We only infer specific strategies if all arguments have a type hint
    if len(args) == len(type_hints):
        for arg_name, arg_type_hint in type_hints.items():
            row = Row(arg_name,
                      Kind.BASE,
                      arg_type_hint,
                      function_name,
                      None,
                      {})
            table.add_row(row)

        preconditions = get_contracts(func)
        if preconditions:
            for conjunction in preconditions:
                for contract in conjunction:
                    body = _body_node_from_condition(contract.condition)
                    try:
                        rows = parse_expression(body, contract.condition, type_hints)
                        for row in rows:
                            row.function = function_name
                            table.add_row(row)
                    except NotImplementedError as e:
                        if hasattr(e, 'message'):
                            failed_contracts.append((body, e.message))
                        else:
                            failed_contracts.append((body, ''))
    return failed_contracts, table


def generate_and_print_table(func: CallableT) -> None:
    failed_contracts, table = generate_symbol_table(func)
    print_pretty_table(table)

    if failed_contracts:
        type_hints = typing.get_type_hints(func)

        print("The following formula(s) are currently not supported and will be added as filters:\n")
        for failed_contract in failed_contracts:
            contract = failed_contract[0]
            error_message = failed_contract[1]
            # print(failed_contract[0])
            print(astunparse.unparse(contract))
            if error_message:
                print(f"Exception.message: {error_message}")
        print("Please read the documentation to verify if these formula(s) are supported.")
        print("You can create an issue on Github if you found a bug or "
              "if you would like to see this feature supported.\n")


def generate_dag_from_table(table: Table) -> nx.DiGraph:
    graph = nx.DiGraph()
    for row in table.get_rows():
        if row.kind == Kind.BASE:
            var_id = row.var_id
            if not graph.has_node(var_id):
                graph.add_node(var_id)
            dependencies = row.get_dependencies()
            for dependency in dependencies:
                if not graph.has_node(dependency):
                    graph.add_node(dependency)
                graph.add_edge(var_id, dependency)
    return graph


def generate_pretty_table(table: Table) -> str:
    rows = list(map(lambda row: [table.get_rows().index(row), row.var_id, row.kind, row.type, row.function,
                                 row.parent, row.properties],
                    table.get_rows()))
    pretty_table = tabulate(rows,
                            headers=['IDX', 'VAR_ID', 'KIND', 'TYPE', 'FUNCTION', 'PARENT', 'PROPERTIES'])
    return pretty_table


def print_pretty_table(table: Table) -> None:
    pretty_table = generate_pretty_table(table)
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


# Shamelessly stolen from icontract
def _no_name_in_descendants(root: ast.expr, name: str) -> bool:
    """Check whether a ``ast.Name`` node with ``root`` identifier is present in the descendants of the node."""
    found = False

    class Visitor(ast.NodeVisitor):
        """Search for the name node."""

        def visit_Name(  # pylint: disable=invalid-name,no-self-use,missing-docstring
                self, node: ast.Name
        ) -> None:
            if node.id == name:
                nonlocal found
                found = True

        def generic_visit(self, node: Any) -> None:
            if not found:
                super(Visitor, self).generic_visit(node)

    visitor = Visitor()
    visitor.visit(root)

    return not found


def _recompute(condition: Callable[..., Any], node: ast.expr) -> Tuple[Any, bool]:
    """Recompute the value corresponding to the node."""
    recompute_visitor = icontract._recompute.Visitor(
        variable_lookup=icontract._represent.collect_variable_lookup(
            condition=condition, condition_kwargs=None
        )
    )

    recompute_visitor.visit(node=node)

    if node in recompute_visitor.recomputed_values:
        return recompute_visitor.recomputed_values[node], True

    return None, False


############
# EXAMPLES #
############


@require(lambda n1: len(n1) > 10)
def example_function(n1: int) -> None:
    pass


@require(lambda n1: n1 % 2 == 0)
@require(lambda n1, n2: n2 > 0 if n1 < 0 else True)
def another_example_function(n1: int, n2: int) -> None:
    pass


@require(lambda lst: all(item > 0 for item in lst if item < 100))
def another_another_example_function(lst: List[int]) -> None:
    pass


@require(lambda s: sorted(s))
def some_func(s: str) -> None:
    pass


if __name__ == '__main__':
    print("\n\nexample 1:\n")
    generate_and_print_table(some_func)
