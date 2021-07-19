from icontract_hypothesis_Lauren.tests.test_input.text import text_functions
from icontract_hypothesis_Lauren.tests.test_input.tuples import tuples_functions
from icontract_hypothesis_Lauren.tests.test_input.dictionaries import dictionaries_functions
from test_input.integers import integers_functions
from test_input.lists import lists_functions
from test_input.regex import regex_functions
from icontract_hypothesis_Lauren.strategy_factory import StrategyFactory
import unittest
import textwrap

# TODO
import icontract
import re

RE_TEST = re.compile(r'[1-9]*')


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

    def test_base_4(self):
        function_name = 'base_4'
        self._execute_end_to_end_test(function_name)

    def test_base_5(self):
        function_name = 'base_5'
        self._execute_end_to_end_test(function_name)

    def test_base_6(self):
        function_name = 'base_6'
        self._execute_end_to_end_test(function_name)

    def test_base_7(self):
        function_name = 'base_7'
        self._execute_end_to_end_test(function_name)

    def test_base_8(self):
        function_name = 'base_8'
        self._execute_end_to_end_test(function_name)

    def test_base_9(self):
        function_name = 'base_9'
        self._execute_end_to_end_test(function_name)

    def test_base_10(self):
        function_name = 'base_10'
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
        self._execute_failing_end_to_end_test(function_name)

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

    def test_is_unique(self):
        function_name = 'is_unique'
        self._execute_end_to_end_test(function_name)

    def test_not_in(self):
        function_name = 'not_in'
        self._execute_end_to_end_test(function_name)

    def _execute_failing_end_to_end_test(self, function_name):
        function = getattr(lists_functions, function_name)
        strategy_factory = StrategyFactory(function)
        self.assertRaises(NotImplementedError, strategy_factory.generate_composite_strategy)

    def _execute_end_to_end_test(self, function_name):
        with open(f'{self._folder_expected_output}{function_name}.txt') as file:
            expected_output = textwrap.dedent(file.read())
        function = getattr(lists_functions, function_name)
        strategy_factory = StrategyFactory(function)
        actual_output = textwrap.dedent(strategy_factory.generate_composite_strategy())
        self.assertEqual(expected_output, actual_output)


class RegexStrategyGeneration(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls._folder_expected_output = "expected_output/regex/"

    def test_startswith(self) -> None:
        function_name = 'base_startswith'
        self._execute_end_to_end_test(function_name)

    def test_re_match(self) -> None:
        function_name = 'base_re_match'
        self._execute_end_to_end_test(function_name)

    def test_re_match_compiled(self) -> None:
        function_name = 'base_re_match_compiled'
        self._execute_end_to_end_test(function_name)

    def test_universal_re_match(self) -> None:
        function_name = 'universal_re_match'
        self._execute_end_to_end_test(function_name)

    def test_universal_re_match_compiled(self) -> None:
        function_name = 'universal_re_match_compiled'
        self._execute_end_to_end_test(function_name)

    def test_filter(self) -> None:
        function_name = 'link_filter'
        self._execute_end_to_end_test(function_name)

    def test_contains(self) -> None:
        function_name = 'base_contains'
        self._execute_end_to_end_test(function_name)

    def test_startswith_endswith(self) -> None:
        function_name = 'base_startswith_endswith'
        self._execute_end_to_end_test(function_name)

    def _execute_end_to_end_test(self, function_name):
        with open(f'{self._folder_expected_output}{function_name}.txt') as file:
            expected_output = textwrap.dedent(file.read())
        function = getattr(regex_functions, function_name)
        strategy_factory = StrategyFactory(function)
        actual_output = textwrap.dedent(strategy_factory.generate_composite_strategy())
        self.assertEqual(expected_output, actual_output)


class TextStrategyGeneration(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls._folder_expected_output = "expected_output/text/"

    def test_universal_link(self) -> None:
        function_name = 'universal_link'
        self._execute_end_to_end_test(function_name)

    def test_base_isalnum(self) -> None:
        function_name = 'base_isalnum'
        self._execute_end_to_end_test(function_name)

    def test_base_isalpha(self) -> None:
        function_name = 'base_isalpha'
        self._execute_end_to_end_test(function_name)

    def test_base_isdigit(self) -> None:
        function_name = 'base_isdigit'
        self._execute_end_to_end_test(function_name)

    def test_base_islower(self) -> None:
        function_name = 'base_islower'
        self._execute_end_to_end_test(function_name)

    def test_base_isnumeric(self) -> None:
        function_name = 'base_isnumeric'
        self._execute_end_to_end_test(function_name)

    def test_base_isspace(self) -> None:
        function_name = 'base_isspace'
        self._execute_end_to_end_test(function_name)

    def test_base_isupper(self) -> None:
        function_name = 'base_isupper'
        self._execute_end_to_end_test(function_name)

    def test_base_isdecimal(self) -> None:
        function_name = 'base_isdecimal'
        self._execute_end_to_end_test(function_name)

    def test_link_lt(self) -> None:
        function_name = 'link_lt'
        self._execute_end_to_end_test(function_name)

    def test_link_lte(self) -> None:
        function_name = 'link_lte'
        self._execute_end_to_end_test(function_name)

    def test_link_gt(self) -> None:
        function_name = 'link_gt'
        self._execute_end_to_end_test(function_name)

    def test_link_gte(self) -> None:
        function_name = 'link_gte'
        self._execute_end_to_end_test(function_name)

    def _execute_end_to_end_test(self, function_name):
        with open(f'{self._folder_expected_output}{function_name}.txt') as file:
            expected_output = textwrap.dedent(file.read())
        function = getattr(text_functions, function_name)
        strategy_factory = StrategyFactory(function)
        actual_output = textwrap.dedent(strategy_factory.generate_composite_strategy())
        self.assertEqual(expected_output, actual_output)


class TupleStrategyGeneration(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls._folder_expected_output = "expected_output/tuples/"

    def test_link(self) -> None:
        function_name = 'link'
        self._execute_end_to_end_test(function_name)

    def _execute_end_to_end_test(self, function_name):
        with open(f'{self._folder_expected_output}{function_name}.txt') as file:
            expected_output = textwrap.dedent(file.read())
        function = getattr(tuples_functions, function_name)
        strategy_factory = StrategyFactory(function)
        actual_output = textwrap.dedent(strategy_factory.generate_composite_strategy())
        self.assertEqual(expected_output, actual_output)


class DictionariesStrategyGeneration(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls._folder_expected_output = "expected_output/dictionaries/"

    def test_link(self) -> None:
        function_name = 'universal_values'
        self._execute_end_to_end_test(function_name)

    def _execute_end_to_end_test(self, function_name):
        with open(f'{self._folder_expected_output}{function_name}.txt') as file:
            expected_output = textwrap.dedent(file.read())
        function = getattr(dictionaries_functions, function_name)
        strategy_factory = StrategyFactory(function)
        actual_output = textwrap.dedent(strategy_factory.generate_composite_strategy())
        self.assertEqual(expected_output, actual_output)

