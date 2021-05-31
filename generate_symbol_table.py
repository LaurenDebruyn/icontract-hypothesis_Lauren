import inspect
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, Tuple, Iterable, TypeVar, Callable, Any, cast, Union
import typing
from enum import Enum
import ast
from icontract import require, DBC
import icontract
from tabulate import tabulate
import astunparse
import networkx as nx
import regex as re
import astor

# Shamelessly stolen from icontract
CallableT = TypeVar('CallableT', bound=Callable[..., Any])


##
# Exceptions
##

class GenerationError(Exception):

    def __init__(self, expr: ast.expr, message: str = "An error has occurred during generation of property table"):
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


@dataclass
class PropertyArgument(DBC):
    argument: Tuple[ast.expr, ...]
    free_variables: List[str]


@dataclass
class Property(DBC):
    identifier: ast.expr
    property_arguments: List[PropertyArgument]
    left_function_call: Optional[str]
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
        return represent_property(self)


def add_property_arguments_to_property(prop: Property, property_arguments: List[PropertyArgument]) -> 'Property':
    new_property_arguments = property_arguments
    new_property_arguments.extend(prop.property_arguments)
    return Property(identifier=prop.identifier,
                    property_arguments=new_property_arguments,
                    left_function_call=prop.left_function_call,
                    var_id=prop.var_id,
                    is_routine=prop.is_routine,
                    var_is_caller=prop.var_is_caller)


def increment_property(prop: Property) -> Property:
    new_property_arguments = []
    for property_argument in prop.property_arguments:
        old_argument = property_argument.argument[0]
        if isinstance(old_argument, ast.Constant):
            old_argument.value = int(old_argument.value) + 1
            new_property_arguments.append(PropertyArgument((old_argument,),
                                                           property_argument.free_variables))
        else:
            new_property_arguments.append(PropertyArgument((ast.BinOp(property_argument.argument[0], ast.Add(),  # noqa
                                                                      ast.Constant(1, lineno=0, col_offset=0, kind=None),
                                                                      lineno=0, col_offset=0, kind=None),),
                                                           property_argument.free_variables))

    return Property(identifier=prop.identifier,
                    property_arguments=new_property_arguments,
                    left_function_call=prop.left_function_call,
                    var_id=prop.var_id,
                    is_routine=prop.is_routine,
                    var_is_caller=prop.var_is_caller)


def decrement_property(prop: Property) -> Property:
    new_property_arguments = []
    for property_argument in prop.property_arguments:
        old_argument = property_argument.argument[0]
        if isinstance(old_argument, ast.Constant):
            old_argument.value = int(old_argument.value) - 1
            new_property_arguments.append(PropertyArgument((old_argument,),
                                                           property_argument.free_variables))
        else:
            new_property_arguments.append(PropertyArgument((ast.BinOp(property_argument.argument[0], ast.Sub(),  # noqa
                                                                      ast.Constant(1, lineno=0, col_offset=0, kind=None),
                                                                      lineno=0, col_offset=0, kind=None),),
                                                           property_argument.free_variables))
    return Property(identifier=prop.identifier,
                    property_arguments=new_property_arguments,
                    left_function_call=prop.left_function_call,
                    var_id=prop.var_id,
                    is_routine=prop.is_routine,
                    var_is_caller=prop.var_is_caller)


def represent_property_identifier(prop: Property) -> str:
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
    else:
        identifier_str = _visualize_expression(prop.identifier)
    return identifier_str


def property_as_lambdas(prop: Property) -> List[Lambda]:
    identifier_str = represent_property_identifier(prop)

    var_id_str = prop.var_id
    if prop.left_function_call:
        var_id_str = f"{prop.left_function_call}({var_id_str})"

    arguments_str_list = represent_property_arguments(prop)

    if prop.is_routine:
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
                raise Exception  # TODO
    else:
        if prop.property_arguments:
            result = [
                f"{var_id_str} {identifier_str} {comparator_str}"
                for comparator_str in arguments_str_list
            ]
        else:
            raise Exception  # TODO

    free_variables_list = [
        arg.free_variables if prop.var_id in arg.free_variables else arg.free_variables + [prop.var_id]
        for arg in prop.property_arguments
    ]

    # TODO this can be done better
    if free_variables_list:
        return [
            Lambda(free_variables, condition)
            for free_variables, condition in zip(free_variables_list, result)
        ]
    return [
        Lambda([prop.var_id], condition)
        for condition in result
    ]


def represent_property(prop: Property) -> str:
    arguments_str_list = represent_property_arguments(prop)  # TODO better naming + include next lines?

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
        return f"({argument_item_str}, {free_variables_list_str})"
    elif argument_item_str:
        return f"({argument_item_str}, {set()})"
    else:
        return f"({set()}, {set()})"


def represent_property_as_lambdas(prop: Property) -> str:
    return ", ".join([
        str(property_lambda)
        for property_lambda in property_as_lambdas(prop)
    ])


def represent_property_argument(property_argument: PropertyArgument) -> str:
    argument_item_str_list = []
    for argument_item in property_argument.argument:
        argument_item_str = astor.to_source(argument_item).strip()
        if re.match(r'\"\"\"(.*)\"\"\"', argument_item_str):
            regex = re.search(r'\"\"\"(.*)\"\"\"', argument_item_str).group(1)
            argument_item_str = re.sub(r'\"\"\"(.*)\"\"\"', f'r\'{regex}\'', argument_item_str)
        # remove unnecessary parentheses around numbers
        if re.match(r'^\(.*\)$', argument_item_str) and argument_item_str[1:-1].isnumeric():
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
            represent_property_argument(property_argument)
        )
    return arguments_str_list


##
# Table
##

class Kind(Enum):
    BASE = 0
    UNIVERSAL_QUANTIFIER = 2
    LINK = 4
    EXISTENTIAL_QUANTIFIER = 5


@dataclass
class Row:
    var_id: str
    kind: Kind
    type: typing.Type
    function: str
    parent: Optional['str']
    # TODO (mristin): use ordered dict? or something deterministic?
    properties: OrderedDict[str, Property]

    def __repr__(self) -> str:
        return f'{self.var_id}\t{self.kind}\t{self.type}\t{self.function}' \
               f'\t{self.parent}\t{self.properties}'

    def add_property(self, prop: Property) -> None:
        property_identifier = represent_property_identifier(prop)
        if property_identifier in self.properties.keys():
            self.properties[property_identifier] = add_property_arguments_to_property(
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

    def get_children(self, row: Row) -> List[Row]:
        return [child_row for child_row in self._rows if child_row.parent == row.var_id]

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


# TODO better documentation
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

            # return f'{result}({attribute.id})'
            # TODO is this better?
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
        elif isinstance(argument, ast.Name) and isinstance(func, ast.Name):
            return f'{func.id}({argument.id})'
        else:
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


@require(lambda expr: len(expr.ops) == 1 and len(expr.comparators) == 1)
def _parse_single_compare(expr: ast.Compare,
                          condition: Callable[..., Any],  # TODO remove?
                          function_args_hints: Dict[str, Any]) -> Optional[List[Row]]:
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
                      {'is_unique': (set(), set())})  # TODO
            rows.append(row)

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
    # format: len(...) COMPARISON value  TODO make this accept more functions
    elif isinstance(left, ast.Call):
        assert isinstance(left.func, ast.Name)
        assert left.func.id == 'len'
        var_id = _visualize_expression(left)
        row_kind = Kind.LINK
        parent = var_id[4:-1]
        add_to_rows = True
        function_args_hints[var_id] = int
        row_property = Property(
            identifier=typing.cast(ast.expr, op),
            property_arguments=[PropertyArgument(argument=(right,),
                                                 free_variables=list(
                                                     extract_variables_from_expression(right, function_args_hints))
                                                 )],
            left_function_call='len',
            var_id=var_id,
            is_routine=True,
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
    else:
        raise NotImplementedError

    if add_to_rows:
        row = Row(var_id,
                  row_kind,
                  function_args_hints[var_id],
                  'dummy_function',  # This is fixed on a higher level
                  parent,
                  {represent_property_identifier(row_property): row_property})
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
    return [Row(var_id,
                Kind.BASE,
                function_args_hints[var_id],
                'dummy_function',
                None,
                {_visualize_expression(attribute): Property(identifier=expr.func,
                                                            property_arguments=[
                                                                PropertyArgument(argument=tuple(expr.args, ),
                                                                                 free_variables=list(variables))
                                                            ],
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
        raise GenerationError(expr, 'we assume for now that we only have one comprehension')  # TODO handle multiple

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
        # TODO can this logic be re-used in the other cases?
        # list[..]
        if typing.get_origin(iterator_type_hint) == list:
            # not supported: lists with multiple types
            if typing.get_origin(typing.get_args(iterator_type_hint)[0]) is Union:  # TODO check if this actually works
                raise GenerationError(expr, 'lists with mixed types (List[Union[..]]) are not supported')
            # TODO check case 'lambda n, lst: all(n > item for item in lst)'
            quantifier_args_hints[quantifier_var_id] = typing.get_args(iterator_type_hint)[0]
        # dict[..] - iterating over keys
        elif typing.get_origin(iterator_type_hint) == dict:  # TODO ordered dicts?
            if typing.get_origin(typing.get_args(iterator_type_hint)[0]) is Union:  # TODO can we move this up?
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
                raise GenerationError(expr, 'tuples with mixed types are not supported')
        elif iterator_type_hint == str:
            # not supported
            raise GenerationError(expr, 'iterating over strings is not supported')
        else:
            # not supported
            raise GenerationError(expr, f'iterating over {_visualize_expression(iterator)} is not supported')

        rows = parse_expression(predicate, condition, quantifier_args_hints)
        for row in rows:
            if row.kind not in [Kind.UNIVERSAL_QUANTIFIER, Kind.EXISTENTIAL_QUANTIFIER]:
                row.kind = Kind.LINK
            # else:  TODO remove
            #     row.parent = iterator_var
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
            if typing.get_origin(iterator_type_hint) == dict:
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
                    raise Exception  # TODO
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
                raise Exception  # TODO make this a better generation exception

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
    elif isinstance(iterator, ast.Call) and isinstance(iterator.func, ast.Name):  # TODO multiple calls?
        assert isinstance(target, ast.Tuple)
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
                row.parent = iterator_var  # TODO is this correct?
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


# TODO remove 'condition'?
def parse_expression(expr: ast.expr, condition: Callable[..., Any], function_args_hints: Dict[str, Any]) -> List[Row]:
    # format: var_id COMPARISON value
    if isinstance(expr, ast.Compare):
        return parse_comparison(expr, condition, function_args_hints)
    # format: var_id.METHOD(..) or MODULE.FUNCTION(..var_id..) or obj.METHOD(..var_id..)
    # TODO does it also cover var_id.property?
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

    # We only infer specific strategies if all arguments have a type hint  TODO unclear what this means
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

    # TODO fix dependencies
    # dependency_graph = generate_dag_from_table(table)
    # for row_id in nx.topological_sort(dependency_graph):
    #     row = table.get_row_by_var_id(row_id)
    #     if isinstance(strategy, SymbolicIntegerStrategy): dependent_strategies = [
    #             strategies[dependent_strategy_id]
    #             for dependent_strategy_id in dependency_graph.neighbors(strategy_id)
    #         ]
    #         for dependent_strategy in dependent_strategies:
    #             if dependent_strategy.var_id in strategy.
    return failed_contracts, table


def generate_and_print_table(func: CallableT) -> None:
    failed_contracts, table = generate_symbol_table(func)
    print_pretty_table(table)

    if failed_contracts:
        # type_hints = typing.get_type_hints(func)  # TODO remove?

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
            dependencies = []
            for d in table.get_dependencies(row).values():
                dependencies.extend(d)
            for dependency in dependencies:
                if not graph.has_node(dependency):
                    graph.add_node(dependency)
                graph.add_edge(var_id, dependency)
                # TODO add edge labels (possibly also change return type)
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
