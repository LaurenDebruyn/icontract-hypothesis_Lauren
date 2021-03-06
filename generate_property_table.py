import collections.abc
import inspect
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Dict, Set, Tuple, Iterable, TypeVar, Callable, Any, cast, Union
import typing
from enum import Enum
import ast
from icontract import require, DBC
import icontract._recompute
import icontract._represent
from tabulate import tabulate
import astunparse
import networkx as nx
import regex
import astor

# Shamelessly stolen from icontract
CallableT = TypeVar('CallableT', bound=Callable[..., Any])


##
# Exceptions
##

class GenerationError(Exception):

    def __init__(
            self,
            expr: ast.expr,
            message: str = "An error has occurred during generation of property table"):
        self.expr = expr
        self.message = message
        super().__init__(self.message)


##
# Properties
##

@dataclass
class Lambda(DBC):
    free_variables: List[str]
    condition: str

    def __repr__(self):
        free_variables_str = ", ".join(self.free_variables)
        return f"lambda {free_variables_str}: {self.condition}"


class SpecialProperties(Enum):
    IS_UNIQUE = 1


@dataclass
class PropertyArgument(DBC):
    argument: Tuple[ast.expr, ...]
    free_variables: List[str]


@dataclass
class Property(DBC):
    identifier: Union[ast.expr, SpecialProperties]
    property_arguments: List[PropertyArgument]
    #: function(s) that have to be applied on the variable before it can be compared
    left_function_call: Optional[List[str]]
    var_id: str
    is_routine: bool
    var_is_caller: bool

    def arguments(self) -> List[Tuple[ast.expr, ...]]:
        return [property_argument.argument for property_argument in self.property_arguments]

    def free_variables(self) -> List[str]:
        free_variables = list()
        for property_argument in self.property_arguments:
            free_variables.extend(property_argument.free_variables)
        return list(OrderedDict.fromkeys(free_variables))

    def __repr__(self) -> str:
        return property_to_string(self)


def add_arguments_to_property(prop: Property, property_arguments: List[PropertyArgument]) -> 'Property':
    new_property_arguments = property_arguments
    new_property_arguments.extend(prop.property_arguments)
    return Property(identifier=prop.identifier,
                    property_arguments=new_property_arguments,
                    left_function_call=prop.left_function_call,
                    var_id=prop.var_id,
                    is_routine=prop.is_routine,
                    var_is_caller=prop.var_is_caller)


def property_identifier_to_string(prop: Property) -> str:
    if isinstance(prop.identifier, ast.operator) or isinstance(prop.identifier, ast.cmpop):
        identifier_str = visualize_operator(prop.identifier)
    elif isinstance(prop.identifier, ast.Attribute):
        identifier = prop.identifier
        result = []
        while isinstance(identifier, ast.Attribute):
            result.append(identifier.attr)
            identifier = identifier.value
        if not prop.var_is_caller:
            assert isinstance(identifier, ast.Name)
            result.insert(0, identifier.id)
        identifier_str = ".".join(result)
    elif isinstance(prop.identifier, SpecialProperties):
        return prop.identifier.name
    else:
        identifier_str = _visualize_expression(prop.identifier)
    return identifier_str


def property_to_lambdas(prop: Property) -> List[Lambda]:
    identifier_str = property_identifier_to_string(prop)

    var_id_str = prop.var_id
    if prop.left_function_call:
        pattern = "\(".join(prop.left_function_call) + "\((.*)\)" + "\)" * (len(prop.left_function_call)-1)  # noqa
        m = regex.fullmatch(pattern, var_id_str)
        var_id_str = m.group(1)

    arguments_str_list = represent_property_arguments(prop)

    if isinstance(prop.identifier, SpecialProperties):
        if prop.identifier == SpecialProperties.IS_UNIQUE:
            result = [f'len(set({var_id_str})) == len({var_id_str})']
    elif prop.is_routine:
        if prop.var_is_caller:
            if prop.property_arguments:
                result = [
                    f"{var_id_str}.{identifier_str}({comparator_str})"
                    for comparator_str in arguments_str_list
                ]
            else:
                result = [f"{var_id_str}.{identifier_str}()"]
        else:
            if prop.property_arguments:
                result = [
                    f"{identifier_str}({comparator_str})"
                    for comparator_str in arguments_str_list
                ]
            else:
                raise Exception
    elif prop.property_arguments:
        if prop.left_function_call:
            left_hand_side = "(".join(prop.left_function_call) + "(" + var_id_str + ")"*len(prop.left_function_call)
            result = [
                # f"{prop.left_function_call}({var_id_str}) {identifier_str} {comparator_str}"
                f"{left_hand_side} {identifier_str} {comparator_str}"
                for comparator_str in arguments_str_list
            ]
        else:
            result = [
                f"{var_id_str} {identifier_str} {comparator_str}"
                for comparator_str in arguments_str_list
            ]
    else:
        raise Exception

    if prop.left_function_call:
        free_variables_list = [
            arg.free_variables if var_id_str in arg.free_variables else arg.free_variables + [var_id_str]
            for arg in prop.property_arguments
        ]
    else:
        free_variables_list = [
            arg.free_variables if prop.var_id in arg.free_variables else arg.free_variables + [prop.var_id]
            for arg in prop.property_arguments
        ]

    # if a variable is indexed, we don't want the index to show up as an argument
    if free_variables_list:
        free_variables_list = [
            [
                free_variable[:free_variable.index('[')] if regex.match(r'.*\[.*\]$', free_variable) else free_variable
                for free_variable in free_variables
            ]
            for free_variables in free_variables_list
        ]

        return [
            Lambda(free_variables, condition)
            for free_variables, condition in zip(free_variables_list, result)
        ]
    return [
        Lambda([prop.var_id], condition)
        for condition in result
    ]


def property_to_string(prop: Property) -> str:
    arguments_str_list = represent_property_arguments(prop)

    if arguments_str_list:
        argument_item_str = "{" + ", ".join(arguments_str_list) + "}"
    else:
        argument_item_str = ""

    free_variables_list = []
    for arg in prop.property_arguments:
        free_variables_list.extend(arg.free_variables)
    free_variables_list_unique = list(OrderedDict.fromkeys(free_variables_list))

    if argument_item_str and free_variables_list_unique:
        free_variables_list_str = "{" + ", ".join(free_variables_list_unique) + "}"
        return f"{property_identifier_to_string(prop)}: ({argument_item_str}, {free_variables_list_str})"
    elif argument_item_str:
        return f"{property_identifier_to_string(prop)}: ({argument_item_str}, {set()})"
    else:
        return f"{property_identifier_to_string(prop)}: ({set()}, {set()})"


def represent_property_as_lambdas(prop: Property) -> str:
    return ", ".join([
        str(property_lambda)
        for property_lambda in property_to_lambdas(prop)
    ])


def property_argument_to_string(property_argument: PropertyArgument) -> str:
    argument_item_str_list = []
    for argument_item in property_argument.argument:
        argument_item_str = astor.to_source(argument_item).strip()
        if regex.match(r'\"\"\"(.*)\"\"\"', argument_item_str):
            pattern = regex.search(r'\"\"\"(.*)\"\"\"', argument_item_str).group(1)
            argument_item_str = regex.sub(r'\"\"\"(.*)\"\"\"', f'\"{pattern}\"', argument_item_str)
        if regex.match(r'^\(.*\)$', argument_item_str) and argument_item_str[1:-1].isnumeric():
            argument_item_str = argument_item_str[1:-1]
        argument_item_str_list.append(argument_item_str)
    if not argument_item_str_list:
        return ""
    elif len(argument_item_str_list) == 1:
        return argument_item_str_list[0]
    else:
        return "(" + ", ".join(argument_item_str_list) + ")"


def represent_property_arguments(prop: Property) -> List[str]:
    arguments_str_list = []
    for property_argument in prop.property_arguments:
        arguments_str_list.append(
            property_argument_to_string(property_argument)
        )
    return arguments_str_list


##
# Table
##

class Kind(Enum):
    BASE = 0
    UNIVERSAL_QUANTIFIER = 1
    LINK = 2
    EXISTENTIAL_QUANTIFIER = 3


@dataclass
class Row:
    var_id: str
    kind: Kind
    type: typing.Type
    function: str
    #: The optional var_id of the parent's row
    parent: Optional['str']
    properties: Dict[str, Property]

    def __repr__(self) -> str:
        properties_str = "{" + ", ".join([property_to_string(prop) for prop in self.properties.values()]) + "}"
        return (
            f'Row(\n'
            f'   var_id={self.var_id!r},\n'
            f'   kind={self.kind!r},\n'
            f'   type={self.type!r},\n'
            f'   function={self.function!r},\n'
            f'   parent={self.parent!r},\n'
            # f'   properties={self.properties!r}\n'
            f'   properties={properties_str}\n'
            ')'
        )

    def add_property(self, prop: Property) -> None:
        property_identifier = property_identifier_to_string(prop)
        if property_identifier in self.properties.keys():
            self.properties[property_identifier] = add_arguments_to_property(
                self.properties[property_identifier],
                prop.property_arguments
            )
        else:
            self.properties[property_identifier] = prop

    def add_properties(self, properties: List[Property]) -> None:
        for prop in properties:
            self.add_property(prop)

    def get_dependencies(self) -> Dict[str, Set[str]]:
        """"Returns Dict[property_identifier, {variables on which the row depends}]"""
        dependencies: Dict[str, Set[str]] = dict()
        for property_identifier, prop in self.properties.items():
            if property_identifier not in dependencies:
                dependencies[property_identifier] = set()
            dependencies[property_identifier] = dependencies[property_identifier].union(prop.free_variables())
        return dependencies


class Table:

    _rows: List[Row]

    def __init__(self) -> None:
        self._rows = []

    def get_rows(self) -> List[Row]:
        return self._rows

    def get_row_by_var_id(self, var_id: str) -> Optional[Row]:
        for row in self._rows:
            if row.var_id == var_id:
                return row

    @require(lambda self, index: 0 <= index < len(self._rows))
    def get_row_by_index(self, index: int) -> Row:
        return self._rows[index]

    def get_children(self, row: Row) -> List[Row]:
        return [child_row for child_row in self._rows if child_row.parent == row.var_id]

    def get_descendants(self, row: Row) -> List[Row]:
        descendants = []
        for descendant in self.get_children(row):
            descendants.append(descendant)
        return descendants

    def get_dependencies(self, row: Row) -> Dict[str, Set[str]]:
        dependencies = row.get_dependencies()
        for child_row in self.get_children(row):
            for child_row_property, child_free_vars in self.get_dependencies(child_row).items():
                if child_row_property not in dependencies:
                    dependencies[child_row_property] = set()
                dependencies[child_row_property] = dependencies[child_row_property].union(child_free_vars)
        return dependencies

    def add_row(self, row: Row) -> None:
        existing_row = self.get_row_by_var_id(row.var_id)
        # If row already exists (same id and corresponding function), the properties are added to the existing row.
        if existing_row and existing_row.function == row.function:
            existing_row.add_properties(list(row.properties.values()))
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


def visualize_operator(op: Union[ast.operator, ast.cmpop]) -> str:
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


def _visualize_expression(expr: ast.expr) -> str:
    if isinstance(expr, ast.Name):
        return expr.id
    elif isinstance(expr, ast.Constant):
        if isinstance(expr.value, str):
            return "\"{}\"".format(expr.value.replace('\n', '\\ n'))
        else:
            return str(expr.value)
    elif isinstance(expr, ast.BinOp):
        return f'{_visualize_expression(expr.left)} {visualize_operator(expr.op)} {_visualize_expression(expr.right)}'
    elif isinstance(expr, ast.Call):
        func = expr.func

        if isinstance(func, ast.Attribute):
            result = func.attr
            attribute = func.value
            while isinstance(attribute, ast.Attribute):
                result = f'{attribute.attr}.{result}'
                attribute = attribute.value
            assert isinstance(attribute, ast.Name)

            args_str = ", ".join([_visualize_expression(arg) for arg in expr.args])
            return f'{attribute.id}.{result}({args_str})'

        argument = expr.args[0]

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

        if isinstance(argument, ast.Name) and isinstance(func, ast.Name):
            return f'{func.id}({argument.id})'

        if isinstance(argument, ast.Subscript) and isinstance(func, ast.Name):
            return f'{func.id}({_visualize_expression(argument)})'



        # If it is not a expression that is handled above, we simply return the expression as a string.
        return str(expr)
    elif isinstance(expr, ast.Subscript):
        if isinstance(expr.slice, ast.Index):
            return f'{_visualize_expression(expr.value)}[{_visualize_expression(expr.slice.value)}]'
        elif isinstance(expr.slice, ast.Slice):
            subscript_value = _visualize_expression(expr.value)
            slice_lower = _visualize_expression(expr.slice.lower)
            slice_upper = _visualize_expression(expr.slice.upper)
            return f'{subscript_value}[{slice_lower}:{slice_upper}]'
    else:
        # If it is not a expression that is handled above, we simply return the expression as a string.
        return str(expr)


class Evaluator(ast.NodeTransformer):
    #  https://stackoverflow.com/a/62677086
    ops = {
        ast.Add: '+',
        ast.Sub: '-',
        ast.Mult: '*',
        ast.Div: '/',
        # define more here
    }

    def visit_BinOp(self, node: ast.BinOp):
        self.generic_visit(node)
        if isinstance(node.left, ast.Num) and isinstance(node.right, ast.Num):
            # On Python <= 3.6 you can use ast.literal_eval.
            # value = ast.literal_eval(node)
            value = eval(f'{node.left.n} {self.ops[type(node.op)]} {node.right.n}')  # noqa
            return ast.Num(n=value)
        if isinstance(node.left, ast.BinOp) and isinstance(node.right, ast.Num) and isinstance(node.left.right,
                                                                                               ast.Num):
            if isinstance(node.left.op, ast.Sub):  # or isinstance(node.left.op, ast.Add):
                value = eval(f'{node.left.right.n} + {node.right.n}')
                if value == 0:
                    return node.left.left
                return ast.BinOp(left=node.left.left, op=node.left.op, right=ast.Num(n=value))
            elif isinstance(node.left.op, ast.Add):
                value = eval(f'{node.left.right.n} - {node.right.n}')
                if value == 0:
                    return node.left.left
                return ast.BinOp(left=node.left.left, op=node.left.op, right=ast.Num(n=value))
        return node

    def simplify_expression(self, expr: ast.Expr) -> ast.expr:
        return ast.fix_missing_locations(self.visit(expr))


@require(lambda expr: len(expr.ops) == 1 and len(expr.comparators) == 1)
def _parse_single_compare(expr: ast.Compare,
                          condition: Callable[..., Any],
                          function_args_hints: Dict[str, Any]) -> Optional[List[Row]]:
    left = expr.left
    op = expr.ops[0]
    right = expr.comparators[0]

    parent = None

    rows = []

    # format: var_id COMPARISON value
    if isinstance(left, ast.Name):
        var_id = left.id
        row_kind = Kind.BASE
        add_to_rows = True
        row_property = Property(
            identifier=typing.cast(ast.expr, op),
            property_arguments=[PropertyArgument(argument=(right,),
                                                 free_variables=list(
                                                     extract_variables_from_expression(right, function_args_hints))
                                                 )],
            left_function_call=None,
            var_id=var_id,
            is_routine=False,
            var_is_caller=False
        )
    # format: len(...) COMPARISON value
    elif isinstance(left, ast.Call):
        if not (isinstance(left.func, ast.Name) and left.func.id == 'len'):
            raise GenerationError(left, "Only len() is currently supported as a left hand side function call")

        # format: len(set(var_id)) == len(var_id)
        if isinstance(op, ast.Eq) and isinstance(right, ast.Call) \
                and isinstance(right.func, ast.Name) \
                and right.func.id == 'len' and isinstance(left, ast.Call) and isinstance(left.func, ast.Name) \
                and left.func.id == 'len' and isinstance(left.args[0], ast.Call) \
                and isinstance(left.args[0].func, ast.Name) and left.args[0].func.id == 'set':  # noqa
            var_left = left.args[0].args[0]  # noqa
            var_right = right.args[0]  # noqa
            if not (isinstance(var_left, ast.Name) and isinstance(var_right, ast.Name)):
                raise NotImplementedError
            assert var_left.id == var_right.id

            var_id = var_left.id
            row_kind = Kind.BASE
            add_to_rows = True
            row_property = Property(
                identifier=SpecialProperties.IS_UNIQUE,
                property_arguments=[],
                left_function_call=None,
                var_id=var_id,
                is_routine=False,
                var_is_caller=False
            )
        else:
            var_id = _visualize_expression(left)

            row_kind = Kind.LINK

            function_calls: List[str] = []
            parent = left
            while isinstance(parent, ast.Call):
                assert isinstance(parent.func, ast.Name)
                function_calls.append(parent.func.id)
                parent = parent.args[0]
            if isinstance(parent, ast.Name):
                parent = parent.id
            elif isinstance(parent, ast.Subscript):
                assert isinstance(parent.value, ast.Name)
                parent = parent.value.id
            else:
                raise NotImplementedError

            add_to_rows = True
            function_args_hints[var_id] = int
            row_property = Property(
                identifier=typing.cast(ast.expr, op),
                property_arguments=[PropertyArgument(argument=(right,),
                                                     free_variables=list(
                                                         extract_variables_from_expression(right, function_args_hints))
                                                     )],
                left_function_call=function_calls,
                var_id=var_id,
                is_routine=False,
                var_is_caller=False
            )
    # format: LIST[INDEX] COMPARISON VALUE
    elif isinstance(left, ast.expr) and isinstance(left, ast.Subscript):
        value = left.value
        if isinstance(value, ast.Name):
            var_id_base = value.id
        else:
            raise GenerationError(expr, f'Value is expected to be ast.Name and not {value}')
        idx = left.slice

        var_id = _visualize_expression(left)
        # index is a number
        if isinstance(idx, ast.Index) and isinstance(idx.value, ast.Num):
            type_hint = typing.get_args(function_args_hints[var_id_base])[int(idx.value.n.real)]
            function_args_hints[var_id] = type_hint
        # index is a variable
        elif isinstance(idx, ast.Index) and isinstance(idx.value, ast.Name):
            assert len(typing.get_args(function_args_hints[var_id_base])) == 1
            if idx.value.id not in function_args_hints or not function_args_hints[idx.value.id] == int:
                raise GenerationError(expr, f'{_visualize_expression(idx.value)} is supposed to be an integer.')
            type_hint = typing.get_args(function_args_hints[var_id_base])[0]
            function_args_hints[var_id] = type_hint
        else:
            raise GenerationError(expr, f'Slice is expected to be ast.Index and not {slice}.'
                                        f'Slices are currently not supported.')
        add_to_rows = True
        row_kind = Kind.LINK
        parent = var_id_base
        row_property = Property(
            identifier=typing.cast(ast.expr, op),
            property_arguments=[PropertyArgument(argument=(right,),
                                                 free_variables=list(
                                                     extract_variables_from_expression(right, function_args_hints))
                                                 )],
            left_function_call=None,
            var_id=var_id,
            is_routine=False,
            var_is_caller=False
        )
    # format: (n1, n2) >= (0, 0)
    elif isinstance(left, ast.Tuple):
        assert all(isinstance(el, ast.Name) for el in left.elts)
        left_hand = [
            _visualize_expression(el)
            for el in left.elts
        ]
        assert isinstance(right, ast.Tuple)
        right_hand = [
            _visualize_expression(el)
            for el in right.elts
        ]
        assert len(left_hand) == len(right_hand)
        # create new expression for each comparison separately
        # example: (n1, n2) >= (0, 0) becomes (n1 >= 0) and (n2 >= 0)
        for (l, r) in zip(left_hand, right_hand):
            new_expr_ast = ast.parse(f'{l} {visualize_operator(op)} {r}', mode="eval")
            assert isinstance(new_expr_ast, ast.Expression)
            rows.extend(parse_expression(new_expr_ast.body, condition, function_args_hints))
        return rows
    # var +- ... COMP value
    elif isinstance(left, ast.BinOp):
        if isinstance(left.left, ast.Name):
            if isinstance(left.op, ast.Add):
                new_expr_ast = ast.fix_missing_locations(
                    Evaluator().simplify_expression(
                        ast.Expr(
                            ast.Compare(
                                left=left.left,
                                ops=[op],
                                comparators=[ast.BinOp(left=right,
                                                       op=ast.Sub(),
                                                       right=left.right)]
                            )
                        )
                    )
                )
            elif isinstance(left.op, ast.Sub):
                new_expr_ast = ast.fix_missing_locations(
                    Evaluator().simplify_expression(
                        ast.Expr(
                            ast.Compare(
                                left=left.left,
                                ops=[op],
                                comparators=[ast.BinOp(left=right,
                                                       op=ast.Add(),
                                                       right=left.right)]
                            )
                        )
                    )
                )
            else:
                raise GenerationError(expr, 'Only + and - are supported as binary operations in the left-hand side.')
            rows.extend(parse_expression(new_expr_ast.value, condition, function_args_hints))
            add_to_rows = False
            var_id = None
            row_kind = None
            row_property = None
        else:
            raise NotImplementedError(expr)
    elif isinstance(left, ast.Constant) and not isinstance(right, ast.Constant):
        if isinstance(op, ast.Gt):
            new_op = ast.Lt()
        elif isinstance(op, ast.Lt):
            new_op = ast.Gt()
        elif isinstance(op, ast.GtE):
            new_op = ast.LtE()
        elif isinstance(op, ast.LtE):
            new_op = ast.GtE()
        else:
            raise NotImplementedError("Only <, <=, >, and >= can be switched sides.")
        rows.extend(_parse_single_compare(expr=ast.Compare(left=right, ops=[new_op], comparators=[left]),
                                          condition=condition,
                                          function_args_hints=function_args_hints))
        add_to_rows = False
        var_id = None
        row_kind = None
        row_property = None
    else:
        raise NotImplementedError(expr)

    if add_to_rows:
        row = Row(var_id,
                  row_kind,
                  function_args_hints[var_id],
                  'dummy_function',  # This is fixed on a higher level
                  parent,
                  {property_identifier_to_string(row_property): row_property})
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
    # handle chain comparisons
    for op in ops:
        comparison = ast.Compare(left, [op], [comparators[0]])
        rows.extend(_parse_single_compare(comparison, condition, function_args_hints))
        left = comparators[0]
        comparators = comparators[1:]
    return rows


@require(lambda expr: isinstance(expr.func, ast.Attribute))
def parse_attribute(expr: ast.Call, condition: Callable[..., Any], function_args_hints: Dict[str, Any]) -> List[Row]:
    assert isinstance(expr.func, ast.expr) and isinstance(expr.func, ast.Attribute)

    variables = extract_variables_from_expression(expr, function_args_hints)

    if not len(variables) == 1:
        raise GenerationError(expr, "Only method calls with one variable are currently supported.")

    attribute = expr.func
    while isinstance(attribute, ast.Attribute):
        attribute = attribute.value

    assert isinstance(attribute, ast.Name) and isinstance(attribute, ast.expr)

    if attribute.id in variables:
        var_id = attribute.id
        var_is_caller = True
        variables.remove(var_id)
    else:
        var_id = variables.pop()
        var_is_caller = False

    identifier = expr.func
    property_arguments = [PropertyArgument(argument=tuple(expr.args, ),
                                           free_variables=list(variables))]
    attribute_str = _visualize_expression(attribute)
    if expr.func.attr in ['match', 'fullmatch'] and not attribute_str in ['re', 'regex']:
        callee_recomputed, recomputed = _recompute(condition=condition, node=expr.func.value)
        if recomputed:
            if isinstance(callee_recomputed, regex.Pattern):
                identifier.value.id = 'regex'
                attribute_str = 'regex'
                property_arguments = [PropertyArgument(argument=tuple([ast.Constant(callee_recomputed.pattern),
                                                                       property_arguments[0].argument[0]]),
                                                       free_variables=[])]
            elif isinstance(callee_recomputed, re.Pattern):
                identifier.value.id = 're'
                attribute_str = 're'
                property_arguments = [PropertyArgument(argument=tuple([ast.Constant(callee_recomputed.pattern),
                                                                       property_arguments[0].argument[0]]),
                                                       free_variables=[])]
            else:
                raise GenerationError(expr)
        else:
            raise GenerationError(expr)

    return [Row(var_id,
                Kind.BASE,
                function_args_hints[var_id],
                'dummy_function',
                None,
                {attribute_str: Property(identifier=identifier,  # expr.func,
                                         property_arguments=property_arguments,
                                         left_function_call=None,
                                         var_id=attribute.id,
                                         is_routine=True,
                                         var_is_caller=var_is_caller)})]


@require(lambda expr:
         isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and isinstance(expr.args[0], ast.GeneratorExp)
         and (expr.func.id == 'all' or expr.func.id == 'any'))
def parse_quantifier(expr: ast.Call, condition: Callable[..., Any], function_args_hints: Dict[str, Any]) -> \
        List[Row]:
    quantifier = cast(ast.Name, expr.func).id
    predicate = cast(ast.GeneratorExp, expr.args[0]).elt
    comprehensions = cast(ast.GeneratorExp, expr.args[0]).generators

    if not len(comprehensions) == 1:
        raise GenerationError(expr, 'we assume for now that we only have one comprehension')

    comprehension = comprehensions[0]
    iterator = comprehension.iter
    target = comprehension.target

    quantifier_args_hints = function_args_hints

    # comprehension is singular variable
    if isinstance(iterator, ast.Name):
        assert isinstance(target, ast.Name)
        quantifier_var_id = _visualize_expression(target)
        iterator_var = iterator.id
        iterator_type_hint = function_args_hints[iterator_var]
        # list[..]
        if typing.get_origin(iterator_type_hint) == list:
            # not supported: lists with multiple types
            if typing.get_origin(typing.get_args(iterator_type_hint)[0]) is Union:
                raise GenerationError(expr, 'lists with mixed types (List[Union[..]]) are not supported')
            quantifier_args_hints[quantifier_var_id] = typing.get_args(iterator_type_hint)[0]
        elif typing.get_origin(iterator_type_hint) == set:
            # not supported: sets with multiple types
            if typing.get_origin(typing.get_args(iterator_type_hint)[0]) is Union:
                raise GenerationError(expr, 'lists with mixed types (Set[Union[..]]) are not supported')
            quantifier_args_hints[quantifier_var_id] = typing.get_args(iterator_type_hint)[0]
        # dict[..] - iterating over keys
        elif typing.get_origin(iterator_type_hint) in [dict, collections.abc.Mapping]:
            if typing.get_origin(typing.get_args(iterator_type_hint)[0]) is Union:
                raise GenerationError(expr, 'dictionaries with mixed types (Dict[Union[..],..]) are not supported')
            quantifier_args_hints[quantifier_var_id] = typing.get_args(iterator_type_hint)[0]
        # tuple[..]
        elif typing.get_origin(iterator_type_hint) == tuple:
            # not supported: tuples with multiple types
            tuple_type_hints = set(typing.get_args(iterator_type_hint))
            if Ellipsis in tuple_type_hints:
                tuple_type_hints.remove(Ellipsis)
            if len(tuple_type_hints) == 1 and typing.Union not in tuple_type_hints:
                quantifier_args_hints[quantifier_var_id] = typing.get_args(iterator_type_hint)[0]
            else:
                raise NotImplementedError('tuples with mixed types are not supported')
        elif iterator_type_hint == str:
            # not supported
            raise NotImplementedError('iterating over strings is not supported')
        else:
            # not supported
            raise NotImplementedError(f'iterating over {_visualize_expression(iterator)} is not supported')

        rows = parse_expression(predicate, condition, quantifier_args_hints)
        for row in rows:
            if row.kind not in [Kind.UNIVERSAL_QUANTIFIER, Kind.EXISTENTIAL_QUANTIFIER]:
                row.kind = Kind.LINK
            if not row.parent:
                row.parent = iterator_var

        quantifier_row = Row(var_id=quantifier_var_id,
                             kind=Kind.UNIVERSAL_QUANTIFIER if quantifier == 'all' else Kind.EXISTENTIAL_QUANTIFIER,
                             type=quantifier_args_hints[quantifier_var_id],
                             function='dummy_function',
                             parent=iterator_var,
                             properties={})
        rows.insert(0, quantifier_row)
        return rows

    # comprehension is method/built-in call
    elif isinstance(iterator, ast.Call) and isinstance(iterator.func, ast.Attribute):
        iterator_var = _visualize_expression(iterator)
        quantifier_rows = []
        if isinstance(iterator.func.value, ast.Name):
            iterator_root_str = iterator.func.value.id
            iterator_type_hint = function_args_hints[iterator_root_str]
            if typing.get_origin(iterator_type_hint) in [dict, collections.abc.Mapping]:
                # dict.keys()
                if iterator.func.attr == 'keys':
                    assert isinstance(target, ast.Name)
                    quantifier_var_id = _visualize_expression(target)
                    quantifier_args_hints[quantifier_var_id] = typing.get_args(iterator_type_hint)[0]
                    quantifier_rows.append(Row(var_id=quantifier_var_id,
                                               kind=Kind.UNIVERSAL_QUANTIFIER if quantifier == 'all' else Kind.EXISTENTIAL_QUANTIFIER,
                                               type=quantifier_args_hints[quantifier_var_id],
                                               function='dummy_function',
                                               parent=iterator_var,
                                               properties={})
                                           )
                # dict.values()
                elif iterator.func.attr == 'values':
                    assert isinstance(target, ast.Name)
                    quantifier_var_id = _visualize_expression(target)
                    quantifier_args_hints[quantifier_var_id] = typing.get_args(iterator_type_hint)[1]
                    quantifier_rows.append(Row(var_id=quantifier_var_id,
                                               kind=Kind.UNIVERSAL_QUANTIFIER if quantifier == 'all' else Kind.EXISTENTIAL_QUANTIFIER,
                                               type=quantifier_args_hints[quantifier_var_id],
                                               function='dummy_function',
                                               parent=iterator_var,
                                               properties={})
                                           )
                # dict.items()
                elif iterator.func.attr == 'items':
                    assert isinstance(target, ast.Tuple)
                    assert len(target.elts) == 2
                    quantifier_var_id_key = _visualize_expression(target.elts[0])
                    quantifier_var_id_value = _visualize_expression(target.elts[1])
                    quantifier_args_hints[quantifier_var_id_key] = typing.get_args(iterator_type_hint)[0]
                    quantifier_args_hints[quantifier_var_id_value] = typing.get_args(iterator_type_hint)[1]
                    quantifier_rows.append(Row(var_id=quantifier_var_id_key,
                                               kind=Kind.UNIVERSAL_QUANTIFIER if quantifier == 'all' else Kind.EXISTENTIAL_QUANTIFIER,
                                               type=quantifier_args_hints[quantifier_var_id_key],
                                               function='dummy_function',
                                               parent=f'{iterator_root_str}.keys()',
                                               properties={})
                                           )
                    quantifier_rows.append(Row(var_id=quantifier_var_id_value,
                                               kind=Kind.UNIVERSAL_QUANTIFIER if quantifier == 'all' else Kind.EXISTENTIAL_QUANTIFIER,
                                               type=quantifier_args_hints[quantifier_var_id_value],
                                               function='dummy_function',
                                               parent=f'{iterator_root_str}.values()',
                                               properties={})
                                           )
                else:
                    raise Exception
            # str.split()
            elif iterator_type_hint == str and iterator.func.attr == 'split':
                assert isinstance(target, ast.Name)
                iterator_root_str = iterator.func.value.id
                iterator_type_hint = function_args_hints[iterator_root_str]
                quantifier_var_id = _visualize_expression(target)
                quantifier_args_hints[quantifier_var_id] = iterator_type_hint
                quantifier_rows.append(Row(var_id=quantifier_var_id,
                                           kind=Kind.UNIVERSAL_QUANTIFIER if quantifier == 'all' else Kind.EXISTENTIAL_QUANTIFIER,
                                           type=quantifier_args_hints[quantifier_var_id],
                                           function='dummy_function',
                                           parent=iterator_var,
                                           properties={})
                                       )
            else:
                raise GenerationError

        rows = parse_expression(predicate, condition, quantifier_args_hints)

        for row in rows:
            if row.kind not in [Kind.UNIVERSAL_QUANTIFIER, Kind.EXISTENTIAL_QUANTIFIER]:
                row.kind = Kind.LINK
            else:
                row.parent = iterator_var
        for quantifier_row in reversed(quantifier_rows):
            rows.insert(0, quantifier_row)
        return rows

    # comprehension is function call
    elif isinstance(iterator, ast.Call) and isinstance(iterator.func, ast.Name):
        if not isinstance(target, ast.Tuple):
            raise GenerationError(expr, "only zip(..) is allowed as a function in the generator")
        quantifier_rows = []

        func_name = iterator.func.id
        if func_name != 'zip':
            raise GenerationError(expr, "only zip(..) is allowed as a function in the generator")
        if any(isinstance(arg, ast.Call) for arg in iterator.args):
            raise GenerationError(expr, "only zip(..) is allowed as a function in the generator")
        iterator_var = _visualize_expression(iterator)
        quantifier_var_ids = [_visualize_expression(el) for el in target.elts]
        args = [_visualize_expression(arg) for arg in iterator.args]
        for var_id, arg in zip(quantifier_var_ids, args):
            assert len(typing.get_args(function_args_hints[arg])) == 1
            iterator_type_hint = typing.get_args(function_args_hints[arg])[0]
            quantifier_args_hints[var_id] = iterator_type_hint
            quantifier_rows.append(Row(var_id=var_id,
                                       kind=Kind.UNIVERSAL_QUANTIFIER if quantifier == 'all' else Kind.EXISTENTIAL_QUANTIFIER,
                                       type=quantifier_args_hints[var_id],
                                       function='dummy_function',
                                       parent=arg,
                                       properties={})
                                   )
        rows = parse_expression(predicate, condition, quantifier_args_hints)

        for row in rows:
            if row.kind not in [Kind.UNIVERSAL_QUANTIFIER, Kind.EXISTENTIAL_QUANTIFIER]:
                row.kind = Kind.LINK
            else:
                row.parent = iterator_var
        for quantifier_row in reversed(quantifier_rows):
            rows.insert(0, quantifier_row)
        return rows
    else:
        # not supported
        raise GenerationError(expr, f'{_visualize_expression(expr)} could not be parsed')


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
    # format: var_id COMPARISON value
    if isinstance(expr, ast.Compare):
        return parse_comparison(expr, condition, function_args_hints)
    # format: var_id.METHOD(..) or MODULE.FUNCTION(..var_id..) or obj.METHOD(..var_id..)
    elif isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
        return parse_attribute(expr, condition, function_args_hints)
    # format: all(..for..in..) or any(..for..in..)
    elif isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and \
            (expr.func.id == 'all' or expr.func.id == 'any') and isinstance(expr.args[0], ast.GeneratorExp):
        return parse_quantifier(expr, condition, function_args_hints)
    # format: .. AND ..
    elif isinstance(expr, ast.BoolOp):
        return parse_boolean_operator(expr, condition, function_args_hints)
    else:
        raise NotImplementedError('Only comparisons and var with attribute calls are currently supported')


def generate_property_table(func: CallableT) -> Tuple[List[Tuple[ast.AST, Optional[str]]], Table]:
    table = Table()
    failed_contracts: List[Tuple[ast.AST, Optional[str]]] = []

    # Initialize table with an empty (no properties) row for each argument.
    args = dict(inspect.signature(func).parameters)  # make 'args' mutable
    type_hints = typing.get_type_hints(func)
    function_name = func.__name__

    if 'return' in type_hints:
        del type_hints['return']

    if 'self' in args:
        del args['self']
    elif 'cls' in args:
        del args['cls']

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
                    except GenerationError as e:
                        if hasattr(e, 'message'):
                            failed_contracts.append((body, e.message))
                        else:
                            failed_contracts.append((body, ''))

    return failed_contracts, table


def generate_and_print_table(func: CallableT) -> None:
    failed_contracts, table = generate_property_table(func)
    print_pretty_table(table)

    if failed_contracts:
        print("The following formula(s) are currently not supported and will be added as filters:\n")
        for failed_contract in failed_contracts:
            contract = failed_contract[0]
            error_message = failed_contract[1]
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
            descendants = [descendant.var_id for descendant in table.get_descendants(row)]
            dependencies = []
            for d in table.get_dependencies(row).values():
                dependencies.extend(d)
            for dependency in dependencies:
                if not dependency in descendants:
                    if not graph.has_node(dependency):
                        graph.add_node(dependency)
                    if not var_id == dependency:
                        graph.add_edge(var_id, dependency)
    return graph


def generate_pretty_table(table: Table) -> str:
    rows = list(map(lambda row: [table.get_rows().index(row), row.var_id, row.kind, row.type, row.function,
                                 row.parent, {property_to_string(prop) for prop in row.properties.values()}],
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
           condition=condition
        )
    )

    recompute_visitor.visit(node=node)

    if node in recompute_visitor.recomputed_values:
        return recompute_visitor.recomputed_values[node], True

    return None, False
