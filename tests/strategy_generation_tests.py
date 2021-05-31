from test_input.integers import integers_functions
from test_input.lists import lists_functions
from icontract_hypothesis_Lauren.strategy_factory import StrategyFactory
import unittest
import textwrap



class IntegerStrategyGeneration(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls._folder_expected_output = "expected_output/integers/"  # TODO fix pathlib

    def test_base_1(self):
        function_name = 'base_1'
        self._execute_end_to_end_test(function_name)

    def test_base_2(self):
        function_name = 'base_2'
        self._execute_end_to_end_test(function_name)

    def test_base_3(self):
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
