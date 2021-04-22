from icontract import require
from typing import List, Tuple, Dict, Optional, Union
from fractions import Fraction
from datetime import date
import regex as re
from contracts import contracts
from generate_symbol_table import generate_symbol_table
import unittest

"""
Category 1: trivial contracts 
    * no dependencies between parameters
    * only binary comparisons
"""

VERSION = 0


class Category1Test(unittest.TestCase):
    @staticmethod
    @unittest.skipIf(VERSION == 0, 'OR will not be supported in the first version')
    def test_integer_or():
        generate_symbol_table(contracts.integer_or)
        return True

    @staticmethod
    @unittest.skipIf(VERSION == 0, 'OR will not be supported in the first version')
    def test_integer_double_or_double_comparison():
        generate_symbol_table(contracts.integer_double_or_double_comparison)

    @staticmethod
    @unittest.skipIf(VERSION == 0, 'OR will not be supported in the first version')
    def test_integer_or_two_arguments():
        generate_symbol_table(contracts.integer_or_two_arguments)

    @staticmethod
    @unittest.skipIf(VERSION == 0, 'OR will not be supported in the first version')
    def test_integer_or_complex():
        generate_symbol_table(contracts.integer_or_complex)

    @staticmethod
    def test_integer_and_complex():
        generate_symbol_table(contracts.integer_and_complex)

    @staticmethod
    def test_simple_integer_comparison():
        generate_symbol_table(contracts.simple_integer_comparison)

    @staticmethod
    def test_redundant_comparison():
        generate_symbol_table(contracts.redundant_comparison)

    @staticmethod
    def test_modulo_recursive():
        generate_symbol_table(contracts.modulo_recursive)

    @staticmethod
    def test_point_in_square():
        generate_symbol_table(contracts.point_in_square)

    @staticmethod
    @unittest.skipIf(VERSION == 0, 'OR will not be supported in the first version')
    def test_something_optional_argument():
        generate_symbol_table(contracts.something_optional_argument)

    @staticmethod
    def test_user_data_to_json():  # FAILS
        generate_symbol_table(contracts.user_data_to_json)

    @staticmethod
    def test_print_unique_list():
        generate_symbol_table(contracts.print_unique_list)

    @staticmethod
    def test_something_float():
        generate_symbol_table(contracts.something_float)

    @staticmethod
    def test_something_modulo():
        generate_symbol_table(contracts.something_modulo)

    @staticmethod
    def test_naive_reverse_sorted_list():
        generate_symbol_table(contracts.naive_reverse_sorted_list)

    @staticmethod
    def test_string_start_equals_end():
        generate_symbol_table(contracts.string_start_equals_end)

    @staticmethod
    def test_read_txt_file():
        generate_symbol_table(contracts.read_txt_file)

    @staticmethod
    def test_dict_unique_values():  # FAILS
        generate_symbol_table(contracts.dict_unique_values)

    @staticmethod
    @unittest.skipIf(VERSION == 0, 'OR will not be supported in the first version')
    def test_something_string_or_int():
        generate_symbol_table(contracts.something_string_or_int)

    @staticmethod
    @unittest.skipIf(VERSION == 0, 'OR will not be supported in the first version')
    def test_something_int_or_fraction():
        generate_symbol_table(contracts.something_int_or_fraction)

    @staticmethod
    @unittest.skipIf(VERSION == 0, 'OR will not be supported in the first version')
    def test_something_int_or_float_or_fraction():
        generate_symbol_table(contracts.something_int_or_float_or_fraction)

    @staticmethod
    def test_add_positive_tuples():  # FAILS
        generate_symbol_table(contracts.add_positive_tuples)

    @staticmethod
    def test_list_unique_sorted():
        generate_symbol_table(contracts.list_unique_sorted)