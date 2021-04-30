from icontract import require
from typing import List, Tuple, Dict, Optional, Union
from fractions import Fraction
from datetime import date
import regex as re
from contracts import contracts
from generate_symbol_table import generate_symbol_table
import unittest
from decimal import Decimal

"""
Category 1: trivial contracts 
    * no dependencies between parameters
    * only binary comparisons
"""

VERSION = 0


class Category1Test(unittest.TestCase):
    @unittest.skipIf(VERSION == 0, 'OR will not be supported in the first version')
    def test_integer_or(self):
        try:
            generate_symbol_table(contracts.integer_or)
        except Exception:
            raise self.failureException

    @unittest.skipIf(VERSION == 0, 'OR will not be supported in the first version')
    def test_integer_double_or_double_comparison(self):
        try:
            generate_symbol_table(contracts.integer_double_or_double_comparison)
        except Exception:
            raise self.failureException

    @unittest.skipIf(VERSION == 0, 'OR will not be supported in the first version')
    def test_integer_or_two_arguments(self):
        try:
            generate_symbol_table(contracts.integer_or_two_arguments)
        except Exception:
            raise self.failureException

    @unittest.skipIf(VERSION == 0, 'OR will not be supported in the first version')
    def test_integer_or_complex(self):
        try:
            generate_symbol_table(contracts.integer_or_complex)
        except Exception:
            raise self.failureException

    def test_integer_and_complex(self):
        try:
            generate_symbol_table(contracts.integer_and_complex)
        except Exception:
            raise self.failureException

    def test_simple_integer_comparison(self):
        try:
            generate_symbol_table(contracts.simple_integer_comparison)
        except Exception:
            raise self.failureException

    def test_redundant_comparison(self):
        try:
            generate_symbol_table(contracts.redundant_comparison)
        except Exception:
            raise self.failureException

    def test_modulo_recursive(self):
        try:
            generate_symbol_table(contracts.modulo_recursive)
        except Exception:
            raise self.failureException

    def test_point_in_square(self):
        try:
            generate_symbol_table(contracts.point_in_square)
        except Exception:
            raise self.failureException

    @unittest.skipIf(VERSION == 0, 'OR will not be supported in the first version')
    def test_something_optional_argument(self):
        try:
            generate_symbol_table(contracts.something_optional_argument)
        except Exception:
            raise self.failureException

    def test_user_data_to_json(self):
        try:
            generate_symbol_table(contracts.user_data_to_json)
        except Exception:
            raise self.failureException

    def test_print_unique_list(self):
        try:
            generate_symbol_table(contracts.print_unique_list)
        except Exception:
            raise self.failureException

    def test_something_float(self):
        try:
            generate_symbol_table(contracts.something_float)
        except Exception:
            raise self.failureException

    def test_something_modulo(self):
        try:
            generate_symbol_table(contracts.something_modulo)
        except Exception:
            raise self.failureException

    def test_naive_reverse_sorted_list(self):
        try:
            generate_symbol_table(contracts.naive_reverse_sorted_list)
        except Exception:
            raise self.failureException

    def test_string_start_equals_end(self):
        try:
            generate_symbol_table(contracts.string_start_equals_end)
        except Exception:
            raise self.failureException

    def test_read_txt_file(self):
        try:
            generate_symbol_table(contracts.read_txt_file)
        except Exception:
            raise self.failureException

    def test_dict_unique_values(self):
        try:
            generate_symbol_table(contracts.dict_unique_values)
        except Exception:
            raise self.failureException

    @unittest.skipIf(VERSION == 0, 'OR will not be supported in the first version')
    def test_something_string_or_int(self):
        try:
            generate_symbol_table(contracts.something_string_or_int)
        except Exception:
            raise self.failureException

    @unittest.skipIf(VERSION == 0, 'OR will not be supported in the first version')
    def test_something_int_or_fraction(self):
        try:
            generate_symbol_table(contracts.something_int_or_fraction)
        except Exception:
            raise self.failureException

    @unittest.skipIf(VERSION == 0, 'OR will not be supported in the first version')
    def test_something_int_or_float_or_fraction(self):
        try:
            generate_symbol_table(contracts.something_int_or_float_or_fraction)
        except Exception:
            raise self.failureException

    def test_add_positive_tuples(self):
        try:
            generate_symbol_table(contracts.add_positive_tuples)
        except Exception:
            raise self.failureException

    def test_list_unique_sorted(self):
        try:
            generate_symbol_table(contracts.list_unique_sorted)
        except Exception:
            raise self.failureException


"""
Category 2: trivial contracts with multi-comparisons
    * no dependencies between parameters
    * also ternary, quaternary, ... comparisons
"""


class Category2Test(unittest.TestCase):

    def test_naive_fraction_add(self):
        try:
            generate_symbol_table(contracts.naive_fraction_add)
        except Exception:
            raise self.failureException

    def test_concatenate_strings(self):
        try:
            generate_symbol_table(contracts.concatenate_strings)
        except Exception:
            raise self.failureException

    def test_print_single_digit(self):
        try:
            generate_symbol_table(contracts.print_single_digit)
        except Exception:
            raise self.failureException

    def test_something_modulo_compare(self):
        try:
            generate_symbol_table(contracts.something_modulo_compare)
        except Exception:
            raise self.failureException

    def test_list_limited_length(self):
        try:
            generate_symbol_table(contracts.list_limited_length)
        except Exception:
            raise self.failureException

    def test_tuple_bounds(self):
        try:
            generate_symbol_table(contracts.tuple_bounds)
        except Exception:
            raise self.failureException

    def test_modulo_float_or_int(self):
        try:
            generate_symbol_table(contracts.modulo_float_or_int)
        except Exception:
            raise self.failureException

    @unittest.skipIf(VERSION == 0, 'OR will not be supported in the first version')
    def test_decimal_addition(self):
        try:
            generate_symbol_table(contracts.decimal_addition)
        except Exception:
            raise self.failureException

    @unittest.skipIf(VERSION == 0, 'OR will not be supported in the first version')
    def test_decimal_double_comparison(self):
        try:
            generate_symbol_table(contracts.decimal_double_comparison)
        except Exception:
            raise self.failureException

    def test_decimal_dependent_comparison(self):
        try:
            generate_symbol_table(contracts.decimal_dependent_comparison)
        except Exception:
            raise self.failureException


"""
Category 3: contracts with mixed parameters
    * relations between parameters
"""


class Category3Test(unittest.TestCase):

    def test_get_ith_element(self):
        try:
            generate_symbol_table(contracts.get_ith_element)
        except Exception:
            raise self.failureException

    def test_time_difference(self):
        try:
            generate_symbol_table(contracts.time_difference)
        except Exception:
            raise self.failureException

    def test_double_int_comparison(self):
        try:
            generate_symbol_table(contracts.double_int_comparison)
        except Exception:
            raise self.failureException

    def test_double_float_comparison(self):
        try:
            generate_symbol_table(contracts.double_float_comparison)
        except Exception:
            raise self.failureException

    @unittest.skipIf(VERSION == 0, 'implication will not be supported in the first version')
    def test_print_limited_characters(self):
        try:
            generate_symbol_table(contracts.print_limited_characters)
        except Exception:
            raise self.failureException

    @unittest.skipIf(VERSION == 0, 'implication will not be supported in the first version')
    def test_something_with_datetime(self):
        try:
            generate_symbol_table(contracts.something_with_datetime)
        except Exception:
            raise self.failureException

    def test_modulo_n(self):
        try:
            generate_symbol_table(contracts.modulo_n)
        except Exception:
            raise self.failureException

    @unittest.skipIf(VERSION == 0, 'OR will not be supported in the first version')
    def test_print_only_letters(self):
        try:
            generate_symbol_table(contracts.print_only_letters)
        except Exception:
            raise self.failureException

    @unittest.skipIf(VERSION == 0, 'implication will not be supported in the first version')
    def test_string_modulo(self):
        try:
            generate_symbol_table(contracts.string_modulo)
        except Exception:
            raise self.failureException

    def test_modulo_n_float_or_int(self):
        try:
            generate_symbol_table(contracts.modulo_n_float_or_int)
        except Exception:
            raise self.failureException

    @unittest.skipIf(VERSION == 0, 'this will not be supported in the first version')
    def test_safe_division(self):
        try:
            generate_symbol_table(contracts.safe_division)
        except Exception:
            raise self.failureException


"""
Category 4: for-all
    * structured as all(... for ... in ...)
"""


class Category4Test(unittest.TestCase):

    def test_nested_lists_flat(self):
        try:
            generate_symbol_table(contracts.nested_lists_flat)
        except Exception:
            raise self.failureException

    def test_nested_nested_lists_flat(self):
        try:
            generate_symbol_table(contracts.nested_nested_lists_flat)
        except Exception:
            raise self.failureException

    def test_disjunct_dict_merge(self):
        try:
            generate_symbol_table(contracts.disjunct_dict_merge)
        except Exception:
            raise self.failureException

    def test_print_positive_list(self):
        try:
            generate_symbol_table(contracts.print_positive_list)
        except Exception:
            raise self.failureException

    def test_filter_list(self):
        try:
            generate_symbol_table(contracts.filter_list)
        except Exception:
            raise self.failureException


"""
Category 6: mixed
"""


class Category5Test(unittest.TestCase):

    def test_json_gmail(self):
        try:
            generate_symbol_table(contracts.json_gmail)
        except Exception:
            raise self.failureException


"""
Uncategorized
"""


class UncategorizedTest(unittest.TestCase):

    def test_string_callable(self):  # FAILS
        try:
            generate_symbol_table(contracts.string_callable)
        except Exception:
            raise self.failureException

    @unittest.skipIf(VERSION == 0, 'OR will not be supported in the first version')
    def test_nested_implication(self):
        try:
            generate_symbol_table(contracts.nested_implication)
        except Exception:
            raise self.failureException

    @unittest.skipIf(VERSION == 0, 'This will not be supported in the first version')
    def test_sorted_dict(self):
        try:
            generate_symbol_table(contracts.sorted_dict)
        except Exception:
            raise self.failureException

    def test_list_tuples(self):
        try:
            generate_symbol_table(contracts.list_tuples)
        except Exception:
            raise self.failureException

    def test_list_tuples_alternative(self):
        try:
            generate_symbol_table(contracts.list_tuples_alternative)
        except Exception:
            raise self.failureException

    def test_list_tuples_sorted(self):
        try:
            generate_symbol_table(contracts.list_tuples_sorted)
        except Exception:
            raise self.failureException

    @unittest.skipIf(VERSION == 0, 'OR will not be supported in the first version')
    def test_list_tuples_union(self):
        try:
            generate_symbol_table(contracts.list_tuples_union)
        except Exception:
            raise self.failureException

    @unittest.skipIf(VERSION == 0, 'OR will not be supported in the first version')
    def test_list_tuples_more_union(self):
        try:
            generate_symbol_table(contracts.list_tuples_more_union)
        except Exception:
            raise self.failureException

    @unittest.skipIf(VERSION == 0, 'OR will not be supported in the first version')
    def test_list_tuples_optional(self):
        try:
            generate_symbol_table(contracts.list_tuples_optional)
        except Exception:
            raise self.failureException

    @unittest.skipIf(VERSION == 0, 'OR will not be supported in the first version')
    def test_integer_string_upper_optional_union(self):
        try:
            generate_symbol_table(contracts.integer_string_upper_optional_union)
        except Exception:
            raise self.failureException

    @unittest.skipIf(VERSION == 0, 'OR will not be supported in the first version')
    def test_integer_string_union_optional(self):
        try:
            generate_symbol_table(contracts.integer_string_union_optional)
        except Exception:
            raise self.failureException

    def test_string_numeric(self):
        try:
            generate_symbol_table(contracts.string_numeric)
        except Exception:
            raise self.failureException

    def test_string_contains(self):
        try:
            generate_symbol_table(contracts.string_contains)
        except Exception:
            raise self.failureException

    def test_four_comparator(self):
        try:
            generate_symbol_table(contracts.four_comparator)
        except Exception:
            raise self.failureException

    def test_three_comparator_equal(self):
        try:
            generate_symbol_table(contracts.three_comparator_equal)
        except Exception:
            raise self.failureException

    def test_tuple_double_comparison(self):
        try:
            generate_symbol_table(contracts.tuple_double_comparison)
        except Exception:
            raise self.failureException

    @unittest.skipIf(VERSION == 0, 'OR will not be supported in the first version')
    def test_implication_tuple(self):
        try:
            generate_symbol_table(contracts.implication_tuple)
        except Exception:
            raise self.failureException

    @unittest.skipIf(VERSION == 0, 'OR will not be supported in the first version')
    def test_or_tuple(self):
        try:
            generate_symbol_table(contracts.or_tuple)
        except Exception:
            raise self.failureException
