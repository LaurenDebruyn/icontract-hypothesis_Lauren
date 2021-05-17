from functions_to_test import *
from property_table_to_strategies import SymbolicIntegerStrategy, SymbolicTextStrategy, SymbolicFromRegexStrategy, \
    infer_properties
from generate_symbol_table import generate_symbol_table, print_pretty_table
import unittest


def expected_strategy_example_function_1():
    return {'n1': SymbolicIntegerStrategy(var_id='n1', min_value=[1])}


def expected_strategy_example_function_2():
    # return {
    #     'n1': SymbolicIntegerStrategy(var_id='n1', min_value=['n2+1'], max_value=[99, 'n4-1']),
    #     'n2': SymbolicIntegerStrategy(var_id='n2', min_value=[5], max_value=['300+n3-1']),
    #     'n3': SymbolicIntegerStrategy(var_id='n3', max_value=['n4']),
    #     'n4': SymbolicIntegerStrategy(var_id='n4'),
    #     ...: ...  # TODO
    # }
    # @require(lambda n1, n2: n1 > n2 > 4)
    # @require(lambda n1: n1 < 100)
    # @require(lambda n1, n4: n1 < n4)
    # @require(lambda n2, n3: n2 < 300 + n3)
    # @require(lambda n1, n3, n4: n3 < n4)
    # @require(lambda s: s.startswith("abc"))
    # @require(lambda lst: len(lst) > 0)
    raise NotImplementedError


def expected_strategy_example_function_3():
    raise NotImplementedError


def expected_strategy_example_function_4():
    raise NotImplementedError


def expected_strategy_example_function_5():
    return {'s': SymbolicFromRegexStrategy(var_id='s', regexps=['(+|-)?[1-9][0-9]*'], full_match=True)}


# TODO include dependencies
def expected_strategy_example_function_13():
    return {'n1': SymbolicIntegerStrategy(var_id='n1', min_value=[1, 'n3'], max_value=['n2-1']),
            'n2': SymbolicIntegerStrategy(var_id='n2', max_value=[100]),
            'n3': SymbolicIntegerStrategy(var_id='n3', max_value=['n2'])}


def expected_strategy_example_function_14():
    return {'s': SymbolicTextStrategy(var_id='s', whitelist_categories=[{'Ll', 'Lu'}])}


def expected_strategy_example_function_15():
    return {'s': SymbolicTextStrategy(var_id='s', whitelist_categories=[{'Ll', 'Lu', 'Nd'}])}


def expected_strategy_example_function_16():
    return {'s': SymbolicTextStrategy(var_id='s', whitelist_categories=[{'Nd'}])}


def expected_strategy_example_function_17():
    return {'s': SymbolicTextStrategy(var_id='s', whitelist_categories=[{'Ll'}])}


def expected_strategy_example_function_18():
    return {'s': SymbolicTextStrategy(var_id='s', whitelist_categories=[{'Nd', 'Nl', 'No'}])}


def expected_strategy_example_function_19():
    return {'s': SymbolicTextStrategy(var_id='s', whitelist_categories=[{'Zs'}])}


def expected_strategy_example_function_20():
    return {'s': SymbolicTextStrategy(var_id='s', whitelist_categories=[{'Lu'}])}


def expected_strategy_example_function_21():
    return {'s': SymbolicTextStrategy(var_id='s', whitelist_categories=[{'Nd'}])}


def expected_strategy_example_function_22():
    return {'s': SymbolicTextStrategy(var_id='s', whitelist_categories=[{'Nd'}, {'Nd', 'Nl', 'No'}, {'Nd'}])}


def expected_strategy_example_function_23():
    return {'s': SymbolicTextStrategy(var_id='s', whitelist_categories=[{'Ll', 'Lu', 'Nd'}, {'Ll', 'Lu'}])}


def expected_strategy_example_function_24():
    return {'s': SymbolicTextStrategy(var_id='s', whitelist_categories=[{'Lu'}, {'Ll'}])}


def expected_strategy_example_function_25():
    return {'s': SymbolicFromRegexStrategy(var_id='s', regexps=['^abc', '.*xyz$'], full_match=True)}


def expected_strategy_example_function_26():
    return {'s': SymbolicFromRegexStrategy(var_id='s', regexps=['s33l'])}


class GenerateSymbolTableTest(unittest.TestCase):

    def test_example_function_1(self) -> None:
        self.assertEqual(expected_strategy_example_function_1(),
                         infer_properties(generate_symbol_table(example_function_1)))
        self.assertEqual('st.integers(min_value=1)',
                         infer_properties(generate_symbol_table(example_function_1))['n1'].get_strategy())

    # def test_example_function_2(self) -> None:
    #     self.assertEqual(expected_strategy_example_function_2(),
    #                      infer_properties(generate_symbol_table(example_function_2)))

    def test_example_function_5(self) -> None:
        self.assertEqual(expected_strategy_example_function_5(),
                         infer_properties(generate_symbol_table(example_function_5)))
        self.assertEqual('st.from_regex(regex=r"(+|-)?[1-9][0-9]*", fullmatch=True)',  # noqa
                         infer_properties(generate_symbol_table(example_function_5))['s'].get_strategy())

    def test_example_function_13(self) -> None:
        self.assertEqual(expected_strategy_example_function_13(),
                         infer_properties(generate_symbol_table(example_function_13)))
        self.assertEqual('st.integers(min_value=max(1, n3), max_value=n2-1)',
                         infer_properties(generate_symbol_table(example_function_13))['n1'].get_strategy())
        self.assertEqual('st.integers(max_value=100)',
                         infer_properties(generate_symbol_table(example_function_13))['n2'].get_strategy())
        self.assertEqual('st.integers(max_value=n2)',
                         infer_properties(generate_symbol_table(example_function_13))['n3'].get_strategy())

    def test_example_function_14(self) -> None:
        self.assertEqual(expected_strategy_example_function_14(),
                         infer_properties(generate_symbol_table(example_function_14)))
        self.assertEqual('st.text(alphabet=st.characters(whitelist_categories=(\'Ll\', \'Lu\')))',
                         infer_properties(generate_symbol_table(example_function_14))['s'].get_strategy())

    def test_example_function_15(self) -> None:
        self.assertEqual(expected_strategy_example_function_15(),
                         infer_properties(generate_symbol_table(example_function_15)))
        self.assertEqual('st.text(alphabet=st.characters(whitelist_categories=(\'Ll\', \'Lu\', \'Nd\')))',
                         infer_properties(generate_symbol_table(example_function_15))['s'].get_strategy())

    def test_example_function_16(self) -> None:
        self.assertEqual(expected_strategy_example_function_16(),
                         infer_properties(generate_symbol_table(example_function_16)))
        self.assertEqual('st.text(alphabet=st.characters(whitelist_categories=(\'Nd\',)))',
                         infer_properties(generate_symbol_table(example_function_16))['s'].get_strategy())

    def test_example_function_17(self) -> None:
        self.assertEqual(expected_strategy_example_function_17(),
                         infer_properties(generate_symbol_table(example_function_17)))
        self.assertEqual('st.text(alphabet=st.characters(whitelist_categories=(\'Ll\',)))',
                         infer_properties(generate_symbol_table(example_function_17))['s'].get_strategy())

    def test_example_function_18(self) -> None:
        self.assertEqual(expected_strategy_example_function_18(),
                         infer_properties(generate_symbol_table(example_function_18)))
        self.assertEqual('st.text(alphabet=st.characters(whitelist_categories=(\'Nd\', \'Nl\', \'No\')))',
                         infer_properties(generate_symbol_table(example_function_18))['s'].get_strategy())

    def test_example_function_19(self) -> None:
        self.assertEqual(expected_strategy_example_function_19(),
                         infer_properties(generate_symbol_table(example_function_19)))
        self.assertEqual('st.text(alphabet=st.characters(whitelist_categories=(\'Zs\',)))',
                         infer_properties(generate_symbol_table(example_function_19))['s'].get_strategy())

    def test_example_function_20(self) -> None:
        self.assertEqual(expected_strategy_example_function_20(),
                         infer_properties(generate_symbol_table(example_function_20)))
        self.assertEqual('st.text(alphabet=st.characters(whitelist_categories=(\'Lu\',)))',
                         infer_properties(generate_symbol_table(example_function_20))['s'].get_strategy())

    def test_example_function_21(self) -> None:
        self.assertEqual(expected_strategy_example_function_21(),
                         infer_properties(generate_symbol_table(example_function_21)))
        self.assertEqual('st.text(alphabet=st.characters(whitelist_categories=(\'Nd\',)))',
                         infer_properties(generate_symbol_table(example_function_21))['s'].get_strategy())

    def test_example_function_22(self) -> None:
        self.assertEqual(expected_strategy_example_function_22(),
                         infer_properties(generate_symbol_table(example_function_22)))
        self.assertEqual('st.text(alphabet=st.characters(whitelist_categories=(\'Nd\',)))',
                         infer_properties(generate_symbol_table(example_function_22))['s'].get_strategy())

    def test_example_function_23(self) -> None:
        self.assertEqual(expected_strategy_example_function_23(),
                         infer_properties(generate_symbol_table(example_function_23)))
        self.assertEqual('st.text(alphabet=st.characters(whitelist_categories=(\'Ll\', \'Lu\')))',
                         infer_properties(generate_symbol_table(example_function_23))['s'].get_strategy())

    def test_example_function_24(self) -> None:
        self.assertEqual(expected_strategy_example_function_24(),
                         infer_properties(generate_symbol_table(example_function_24)))
        self.assertEqual('st.text(alphabet=st.characters(whitelist_categories=()))',
                         infer_properties(generate_symbol_table(example_function_24))['s'].get_strategy())

    def test_example_function_25(self) -> None:
        self.assertEqual(expected_strategy_example_function_25(),
                         infer_properties(generate_symbol_table(example_function_25)))
        self.assertEqual('st.from_regex(regex=r"(?=^abc)(?=.*xyz$)", fullmatch=True)', # noqa
                         infer_properties(generate_symbol_table(example_function_25))['s'].get_strategy())

    def test_example_function_26(self) -> None:
        self.assertEqual(expected_strategy_example_function_26(),
                         infer_properties(generate_symbol_table(example_function_26)))
        self.assertEqual('st.from_regex(regex=r"s33l")',
                         infer_properties(generate_symbol_table(example_function_26))['s'].get_strategy())
