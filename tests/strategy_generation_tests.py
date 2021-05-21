import ast
from typing import Tuple

from test_input.integers import integers_functions
from test_input.lists import lists_functions
from icontract_hypothesis_Lauren.strategy_factory import StrategyFactory
import unittest


def yield_expected_actual_output_integers() -> Tuple[str, str]:
    filepath_functions = f"test_input/integers/integers_functions.py"
    folder_expected_output = "expected_output/integers/"
    with open(filepath_functions) as file_functions:
        node = ast.parse(file_functions.read())
        functions = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]

    for function_name in functions:
        with open(f'{folder_expected_output}{function_name}.txt') as file:
            expected_output = file.read()
        function = getattr(integers_functions, function_name)
        strategy_factory = StrategyFactory(function)
        actual_output = strategy_factory.generate_composite_strategy()

        yield expected_output, actual_output


def yield_expected_actual_output_lists() -> Tuple[str, str]:
    filepath_functions = f"test_input/lists/lists_functions.py"
    folder_expected_output = "expected_output/lists/"
    with open(filepath_functions) as file_functions:
        node = ast.parse(file_functions.read())
        functions = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]

    for function_name in functions:
        with open(f'{folder_expected_output}{function_name}.txt') as file:
            expected_output = file.read()
        function = getattr(lists_functions, function_name)
        strategy_factory = StrategyFactory(function)
        actual_output = strategy_factory.generate_composite_strategy()

        yield expected_output, actual_output


class StrategyGenerationTest(unittest.TestCase):

    def test_integers(self):
        self.maxDiff = None
        for expected_output, actual_output in yield_expected_actual_output_integers():
            with self.subTest(actual_output=actual_output, expected_output=expected_output):
                self.assertEqual(actual_output, expected_output)

    def test_lists(self):
        self.maxDiff = None
        for expected_output, actual_output in yield_expected_actual_output_lists():
            with self.subTest(actual_output=actual_output, expected_output=expected_output):
                self.assertEqual(actual_output, expected_output)
