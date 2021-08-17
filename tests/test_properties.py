import ast
import unittest

from icontract_hypothesis_Lauren.generate_symbol_table import Property, property_to_string, increment_property, \
    decrement_property, PropertyArgument, represent_property_as_lambdas


class PropertiesRepresentationTest(unittest.TestCase):

    def test_1(self) -> None:
        expected_output = "len(lst) >= (5)"
        compare = ast.parse(expected_output).body[0].value
        prop = Property(
            identifier=compare.ops[0],
            property_arguments=[PropertyArgument((comparator, ), []) for comparator in compare.comparators],
            left_function_call='len',
            var_id='lst',
            is_routine=False,
            var_is_caller=False
        )
        self.assertEqual(expected_output, property_to_string(prop))

    def test_2(self) -> None:
        expected_output = "sorted(lst) >= lst"
        compare = ast.parse(expected_output).body[0].value
        prop = Property(
            identifier=compare.ops[0],
            property_arguments=[PropertyArgument((comparator, ), ['lst']) for comparator in compare.comparators],
            left_function_call='sorted',
            var_id='lst',
            is_routine=False,
            var_is_caller=False
        )
        self.assertEqual(expected_output, property_to_string(prop))

    def test_3(self) -> None:
        expected_output = "x >= (5)"
        compare = ast.parse(expected_output).body[0].value
        prop = Property(
            identifier=compare.ops[0],
            property_arguments=[PropertyArgument((comparator, ), []) for comparator in compare.comparators],
            left_function_call=None,
            var_id='x',
            is_routine=False,
            var_is_caller=False
        )
        self.assertEqual(expected_output, property_to_string(prop))

    def test_4(self) -> None:
        expected_output = "s.isnumeric()"
        call = ast.parse(expected_output).body[0].value
        prop = Property(
            identifier=call.func,
            property_arguments=[],
            left_function_call=None,
            var_id='s',
            is_routine=True,
            var_is_caller=True
        )
        self.assertEqual(expected_output, property_to_string(prop))

    def test_5(self) -> None:
        expected_output = "re.match(r'abc', s)"
        call = ast.parse(expected_output).body[0].value
        prop = Property(
            identifier=call.func,
            property_arguments=[PropertyArgument(call.args, ['s'])],
            left_function_call=None,
            var_id='s',
            is_routine=True,
            var_is_caller=False
        )
        self.assertEqual(expected_output, property_to_string(prop))

    def test_6(self) -> None:
        expected_output = "s.startswith(r'abc')"
        call = ast.parse(expected_output).body[0].value
        prop = Property(
            identifier=call.func,
            property_arguments=[PropertyArgument(call.args, [])],
            left_function_call=None,
            var_id='s',
            is_routine=True,
            var_is_caller=True
        )
        self.assertEqual(expected_output, property_to_string(prop))


class PropertiesIncrementDecrementTest(unittest.TestCase):

    def test_1(self) -> None:
        test_input = "x >= 5"
        compare = ast.parse(test_input).body[0].value
        prop_input = Property(
            identifier=compare.ops[0],
            property_arguments=[PropertyArgument((comparator, ), []) for comparator in compare.comparators],
            left_function_call=None,
            var_id='x',
            is_routine=False,
            var_is_caller=False
        )

        expected_output_incremented = "({(6,)}, set())"
        prop_input_incremented = increment_property(prop_input)
        self.assertEqual(expected_output_incremented, property_to_string(prop_input_incremented))

    def test_2(self) -> None:
        test_input = "x >= 5"
        compare = ast.parse(test_input).body[0].value
        prop_input = Property(
            identifier=compare.ops[0],
            property_arguments=[PropertyArgument((comparator, ), []) for comparator in compare.comparators],
            left_function_call=None,
            var_id='x',
            is_routine=False,
            var_is_caller=False
        )

        expected_output_decremented = "({(4,)}, set())"
        prop_input_decremented = decrement_property(prop_input)
        self.assertEqual(expected_output_decremented, property_to_string(prop_input_decremented))


class PropertyAsLambdas(unittest.TestCase):

    def test_1(self) -> None:
        ast_input = "len(lst) >= (5)"
        expected_output = "lambda lst: len(lst) >= (5)"
        compare = ast.parse(ast_input).body[0].value
        prop = Property(
            identifier=compare.ops[0],
            property_arguments=[PropertyArgument((comparator, ), []) for comparator in compare.comparators],
            left_function_call='len',
            var_id='lst',
            is_routine=False,
            var_is_caller=False
        )
        self.assertEqual(expected_output, represent_property_as_lambdas(prop))

    def test_2(self) -> None:
        ast_input = "sorted(lst) >= lst"
        expected_output = "lambda lst: sorted(lst) >= lst"
        compare = ast.parse(ast_input).body[0].value
        prop = Property(
            identifier=compare.ops[0],
            property_arguments=[PropertyArgument((comparator, ), ['lst']) for comparator in compare.comparators],
            left_function_call='sorted',
            var_id='lst',
            is_routine=False,
            var_is_caller=False
        )
        self.assertEqual(expected_output, represent_property_as_lambdas(prop))

    def test_3(self) -> None:
        ast_input = "x >= (5)"
        expected_output = "lambda x: x >= (5)"
        compare = ast.parse(ast_input).body[0].value
        prop = Property(
            identifier=compare.ops[0],
            property_arguments=[PropertyArgument((comparator, ), []) for comparator in compare.comparators],
            left_function_call=None,
            var_id='x',
            is_routine=False,
            var_is_caller=False
        )
        self.assertEqual(expected_output, represent_property_as_lambdas(prop))

    def test_4(self) -> None:
        ast_input = "s.isnumeric()"
        expected_output = "lambda s: s.isnumeric()"
        call = ast.parse(ast_input).body[0].value
        prop = Property(
            identifier=call.func,
            property_arguments=[],
            left_function_call=None,
            var_id='s',
            is_routine=True,
            var_is_caller=True
        )
        self.assertEqual(expected_output, represent_property_as_lambdas(prop))

    def test_5(self) -> None:
        ast_input = "re.match(r'abc', s)"
        expected_output = "lambda s: re.match(r'abc', s)"
        call = ast.parse(ast_input).body[0].value
        prop = Property(
            identifier=call.func,
            property_arguments=[PropertyArgument(call.args, ['s'])],
            left_function_call=None,
            var_id='s',
            is_routine=True,
            var_is_caller=False
        )
        self.assertEqual(expected_output, represent_property_as_lambdas(prop))

    def test_6(self) -> None:
        ast_input = "s.startswith(r'abc')"
        expected_output = "lambda s: s.startswith(r'abc')"
        call = ast.parse(ast_input).body[0].value
        prop = Property(
            identifier=call.func,
            property_arguments=[PropertyArgument(call.args, [])],
            left_function_call=None,
            var_id='s',
            is_routine=True,
            var_is_caller=True
        )
        self.assertEqual(expected_output, represent_property_as_lambdas(prop))
