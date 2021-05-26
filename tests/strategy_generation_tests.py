import ast
from typing import Tuple

from test_input.integers import integers_functions
from test_input.lists import lists_functions
from icontract_hypothesis_Lauren.strategy_factory import StrategyFactory
import unittest
import textwrap


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
        # for expected_output, actual_output in yield_expected_actual_output_lists():
        #     with self.subTest(actual_output=actual_output, expected_output=expected_output):
        #         self.assertEqual(actual_output, expected_output)

        # TODO(mristin): use __file__ -> thid_dir = pathlib.Path(os.path.realpath(__file__)).parent
        # TODO(mristin): be os-agnostic --> use pathlib!
        functions_pth = this_dir / "test_input" / "lists" / "lists_functions.py"

        # TODO(mristin): correspondance test <-> file
        functions_pth = this_dir / "test_input" / test_lists.__name__ / "code.py"

        filepath_functions = f"test_input/lists/lists_functions.py"
        folder_expected_output = "expected_output/lists/"
        with open(filepath_functions) as file_functions:
            node = ast.parse(file_functions.read())
            functions = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]

        for function_name in functions:
            with self.subTest(function_name=function_name, folder_expected_output=folder_expected_output):
                with open(f'{folder_expected_output}{function_name}.txt') as file:
                    expected_output = textwrap.dedent(file.read())
                function = getattr(lists_functions, function_name)
                strategy_factory = StrategyFactory(function)
                actual_output = textwrap.dedent(strategy_factory.generate_composite_strategy())
                self.assertEqual(expected_output, actual_output)

            # yield expected_output, actual_output


class IntegerStrategyGeneration(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls._folder_expected_output = "expected_output/integers/"

    def test_existential_1(self):
        function_name = 'base_1'
        self._execute_end_to_end_test(function_name)

    def test_link_1(self):
        function_name = 'base_2'
        self._execute_end_to_end_test(function_name)

    def test_universal_1(self):
        function_name = 'base_3'
        self._execute_end_to_end_test(function_name)

    def _execute_end_to_end_test(self, function_name):
        with open(f'{self._folder_expected_output}{function_name}.txt') as file:
            expected_output = textwrap.dedent(file.read())
        function = getattr(integers_functions, function_name)
        strategy_factory = StrategyFactory(function)
        actual_output = textwrap.dedent(strategy_factory.generate_composite_strategy())
        self.assertEqual(expected_output, actual_output)


class ListStrategyGeneration(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls._folder_expected_output = "expected_output/lists/"

    def test_existential_1(self):
        function_name = 'existential_1'
        self._execute_end_to_end_test(function_name)

    def test_link_1(self):
        function_name = 'link_1'
        self._execute_end_to_end_test(function_name)

    def test_universal_1(self):
        function_name = 'universal_1'
        self._execute_end_to_end_test(function_name)

    def test_universal_2(self):
        function_name = 'universal_2'
        self._execute_end_to_end_test(function_name)

    def test_universal_filter_1(self):
        function_name = 'universal_filter_1'
        self._execute_end_to_end_test(function_name)

    def test_universal_nested_1(self):
        function_name = 'universal_nested_1'
        self._execute_end_to_end_test(function_name)

    def test_universal_nested_link_1(self):
        function_name = 'universal_nested_link_1'
        self._execute_end_to_end_test(function_name)

    def test_universal_nested_link_2(self):
        function_name = 'universal_nested_link_2'
        self._execute_end_to_end_test(function_name)

    def _execute_end_to_end_test(self, function_name):
        with open(f'{self._folder_expected_output}{function_name}.txt') as file:
            expected_output = textwrap.dedent(file.read())
        function = getattr(lists_functions, function_name)
        strategy_factory = StrategyFactory(function)
        actual_output = textwrap.dedent(strategy_factory.generate_composite_strategy())
        self.assertEqual(expected_output, actual_output)
