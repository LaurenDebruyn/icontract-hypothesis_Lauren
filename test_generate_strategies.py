from functions_to_test import *
from icontract_hypothesis_Lauren.strategy_factory import StrategyFactory
from property_table_to_strategies import SymbolicIntegerStrategy, SymbolicTextStrategy, SymbolicFromRegexStrategy, \
    generate_strategies
from generate_symbol_table import generate_symbol_table, print_pretty_table
import unittest


def expected_strategy_example_function_1():
    return {'n1': SymbolicIntegerStrategy(var_id='n1',
                                          min_value=[1],
                                          max_value=[],
                                          filters=[])}


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
    return {'s': SymbolicFromRegexStrategy(var_id='s',
                                           regexps=['(+|-)?[1-9][0-9]*'],
                                           full_match=True,
                                           min_size=[],
                                           max_size=[],
                                           filters=[])}


# TODO include dependencies
def expected_strategy_example_function_13():
    return {'n1': SymbolicIntegerStrategy(var_id='n1', min_value=[1, 'n3'], max_value=['n2-1'], filters=[]),
            'n2': SymbolicIntegerStrategy(var_id='n2', min_value=[], max_value=[100], filters=[]),
            'n3': SymbolicIntegerStrategy(var_id='n3', min_value=[], max_value=['n2'], filters=[])}


def expected_strategy_example_function_14():
    return {'s': SymbolicTextStrategy(var_id='s',
                                      blacklist_categories=[],
                                      whitelist_categories=[{'Ll', 'Lu'}],
                                      min_size=[],
                                      max_size=[],
                                      filters=[])}


def expected_strategy_example_function_15():
    return {'s': SymbolicTextStrategy(var_id='s',
                                      blacklist_categories=[],
                                      whitelist_categories=[{'Ll', 'Lu', 'Nd'}],
                                      min_size=[],
                                      max_size=[],
                                      filters=[])}


def expected_strategy_example_function_16():
    return {'s': SymbolicTextStrategy(var_id='s',
                                      blacklist_categories=[],
                                      whitelist_categories=[{'Nd'}],
                                      min_size=[],
                                      max_size=[],
                                      filters=[])}


def expected_strategy_example_function_17():
    return {'s': SymbolicTextStrategy(var_id='s',
                                      blacklist_categories=[],
                                      whitelist_categories=[{'Ll'}],
                                      min_size=[],
                                      max_size=[],
                                      filters=[])}


def expected_strategy_example_function_18():
    return {'s': SymbolicTextStrategy(var_id='s',
                                      blacklist_categories=[],
                                      whitelist_categories=[{'Nd', 'Nl', 'No'}],
                                      min_size=[],
                                      max_size=[],
                                      filters=[])}


def expected_strategy_example_function_19():
    return {'s': SymbolicTextStrategy(var_id='s',
                                      blacklist_categories=[],
                                      whitelist_categories=[{'Zs'}],
                                      min_size=[],
                                      max_size=[],
                                      filters=[])}


def expected_strategy_example_function_20():
    return {'s': SymbolicTextStrategy(var_id='s',
                                      blacklist_categories=[],
                                      whitelist_categories=[{'Lu'}],
                                      min_size=[],
                                      max_size=[],
                                      filters=[])}


def expected_strategy_example_function_21():
    return {'s': SymbolicTextStrategy(var_id='s',
                                      blacklist_categories=[],
                                      whitelist_categories=[{'Nd'}],
                                      min_size=[],
                                      max_size=[],
                                      filters=[])}


def expected_strategy_example_function_22():
    return {'s': SymbolicTextStrategy(var_id='s',
                                      blacklist_categories=[],
                                      whitelist_categories=[{'Nd'}, {'Nd', 'Nl', 'No'}, {'Nd'}],
                                      min_size=[],
                                      max_size=[],
                                      filters=[])}


def expected_strategy_example_function_23():
    return {'s': SymbolicTextStrategy(var_id='s',
                                      blacklist_categories=[],
                                      whitelist_categories=[{'Ll', 'Lu', 'Nd'}, {'Ll', 'Lu'}],
                                      min_size=[],
                                      max_size=[],
                                      filters=[])}


def expected_strategy_example_function_24():
    return {'s': SymbolicTextStrategy(var_id='s',
                                      blacklist_categories=[],
                                      whitelist_categories=[{'Lu'}, {'Ll'}],
                                      min_size=[],
                                      max_size=[],
                                      filters=[])}


def expected_strategy_example_function_25():
    return {'s': SymbolicFromRegexStrategy(var_id='s',
                                           regexps=['^abc', '.*xyz$'],
                                           full_match=True, min_size=[],
                                           max_size=[],
                                           filters=[])}


def expected_strategy_example_function_26():
    return {'s': SymbolicFromRegexStrategy(var_id='s',
                                           regexps=['s33l'],
                                           full_match=False,
                                           min_size=[],
                                           max_size=[],
                                           filters=[])}

class GenerateSymbolTableTest(unittest.TestCase):

    def test_example_function_1(self) -> None:
        strategy_factory = StrategyFactory(example_function_1)
        self.assertEqual(expected_strategy_example_function_1(),
                         generate_strategies(strategy_factory.generate_property_table_without_failed_contracts()))
        self.assertEqual('st.integers(min_value=1)',
                         generate_strategies(strategy_factory.generate_property_table_without_failed_contracts())['n1'].represent())

    # def test_example_function_2(self) -> None:
    #     self.assertEqual(expected_strategy_example_function_2(),
    #                      generate_strategies(*generate_symbol_table(example_function_2)))

    def test_example_function_5(self) -> None:
        strategy_factory = StrategyFactory(example_function_5)
        self.assertEqual(expected_strategy_example_function_5(),
                         generate_strategies(strategy_factory.generate_property_table_without_failed_contracts()))
        self.assertEqual('st.from_regex(regex=r"(+|-)?[1-9][0-9]*", fullmatch=True)',  # noqa
                         generate_strategies(strategy_factory.generate_property_table_without_failed_contracts())['s'].represent())

    def test_example_function_13(self) -> None:
        strategy_factory = StrategyFactory(example_function_13)
        self.assertEqual(expected_strategy_example_function_13(),
                         generate_strategies(strategy_factory.generate_property_table_without_failed_contracts()))
        self.assertEqual('st.integers(min_value=max(1, n3), max_value=n2-1)',
                         generate_strategies(strategy_factory.generate_property_table_without_failed_contracts())['n1'].represent())
        self.assertEqual('st.integers(max_value=100)',
                         generate_strategies(strategy_factory.generate_property_table_without_failed_contracts())['n2'].represent())
        self.assertEqual('st.integers(max_value=n2)',
                         generate_strategies(strategy_factory.generate_property_table_without_failed_contracts())['n3'].represent())

    def test_example_function_14(self) -> None:
        strategy_factory = StrategyFactory(example_function_14)
        self.assertEqual(expected_strategy_example_function_14(),
                         generate_strategies(strategy_factory.generate_property_table_without_failed_contracts()))
        self.assertEqual('st.text(alphabet=st.characters(whitelist_categories=(\'Ll\', \'Lu\')))',
                         generate_strategies(strategy_factory.generate_property_table_without_failed_contracts())['s'].represent())

    def test_example_function_15(self) -> None:
        strategy_factory = StrategyFactory(example_function_15)
        self.assertEqual(expected_strategy_example_function_15(),
                         generate_strategies(strategy_factory.generate_property_table_without_failed_contracts()))
        self.assertEqual('st.text(alphabet=st.characters(whitelist_categories=(\'Ll\', \'Lu\', \'Nd\')))',
                         generate_strategies(strategy_factory.generate_property_table_without_failed_contracts())['s'].represent())

    def test_example_function_16(self) -> None:
        strategy_factory = StrategyFactory(example_function_16)
        self.assertEqual(expected_strategy_example_function_16(),
                         generate_strategies(strategy_factory.generate_property_table_without_failed_contracts()))
        self.assertEqual('st.text(alphabet=st.characters(whitelist_categories=(\'Nd\',)))',
                         generate_strategies(strategy_factory.generate_property_table_without_failed_contracts())['s'].represent())

    def test_example_function_17(self) -> None:
        strategy_factory = StrategyFactory(example_function_17)
        self.assertEqual(expected_strategy_example_function_17(),
                         generate_strategies(strategy_factory.generate_property_table_without_failed_contracts()))
        self.assertEqual('st.text(alphabet=st.characters(whitelist_categories=(\'Ll\',)))',
                         generate_strategies(strategy_factory.generate_property_table_without_failed_contracts())['s'].represent())

    def test_example_function_18(self) -> None:
        strategy_factory = StrategyFactory(example_function_18)
        self.assertEqual(expected_strategy_example_function_18(),
                         generate_strategies(strategy_factory.generate_property_table_without_failed_contracts()))
        self.assertEqual('st.text(alphabet=st.characters(whitelist_categories=(\'Nd\', \'Nl\', \'No\')))',
                         generate_strategies(strategy_factory.generate_property_table_without_failed_contracts())['s'].represent())

    def test_example_function_19(self) -> None:
        strategy_factory = StrategyFactory(example_function_19)
        self.assertEqual(expected_strategy_example_function_19(),
                         generate_strategies(strategy_factory.generate_property_table_without_failed_contracts()))
        self.assertEqual('st.text(alphabet=st.characters(whitelist_categories=(\'Zs\',)))',
                         generate_strategies(strategy_factory.generate_property_table_without_failed_contracts())['s'].represent())

    def test_example_function_20(self) -> None:
        strategy_factory = StrategyFactory(example_function_20)
        self.assertEqual(expected_strategy_example_function_20(),
                         generate_strategies(strategy_factory.generate_property_table_without_failed_contracts()))
        self.assertEqual('st.text(alphabet=st.characters(whitelist_categories=(\'Lu\',)))',
                         generate_strategies(strategy_factory.generate_property_table_without_failed_contracts())['s'].represent())

    def test_example_function_21(self) -> None:
        strategy_factory = StrategyFactory(example_function_21)
        self.assertEqual(expected_strategy_example_function_21(),
                         generate_strategies(strategy_factory.generate_property_table_without_failed_contracts()))
        self.assertEqual('st.text(alphabet=st.characters(whitelist_categories=(\'Nd\',)))',
                         generate_strategies(strategy_factory.generate_property_table_without_failed_contracts())['s'].represent())

    def test_example_function_22(self) -> None:
        strategy_factory = StrategyFactory(example_function_22)
        self.assertEqual(expected_strategy_example_function_22(),
                         generate_strategies(strategy_factory.generate_property_table_without_failed_contracts()))
        self.assertEqual('st.text(alphabet=st.characters(whitelist_categories=(\'Nd\',)))',
                         generate_strategies(strategy_factory.generate_property_table_without_failed_contracts())['s'].represent())

    def test_example_function_23(self) -> None:
        strategy_factory = StrategyFactory(example_function_23)
        self.assertEqual(expected_strategy_example_function_23(),
                         generate_strategies(strategy_factory.generate_property_table_without_failed_contracts()))
        self.assertEqual('st.text(alphabet=st.characters(whitelist_categories=(\'Ll\', \'Lu\')))',
                         generate_strategies(strategy_factory.generate_property_table_without_failed_contracts())['s'].represent())

    def test_example_function_24(self) -> None:
        strategy_factory = StrategyFactory(example_function_24)
        self.assertEqual(expected_strategy_example_function_24(),
                         generate_strategies(strategy_factory.generate_property_table_without_failed_contracts()))
        self.assertEqual('st.text(alphabet=st.characters(whitelist_categories=()))',
                         generate_strategies(strategy_factory.generate_property_table_without_failed_contracts())['s'].represent())

    def test_example_function_25(self) -> None:
        strategy_factory = StrategyFactory(example_function_25)
        self.assertEqual(expected_strategy_example_function_25(),
                         generate_strategies(strategy_factory.generate_property_table_without_failed_contracts()))
        self.assertEqual('st.from_regex(regex=r"(?=^abc)(?=.*xyz$)", fullmatch=True)', # noqa
                         generate_strategies(strategy_factory.generate_property_table_without_failed_contracts())['s'].represent())

    def test_example_function_26(self) -> None:
        strategy_factory = StrategyFactory(example_function_26)
        self.assertEqual(expected_strategy_example_function_26(),
                         generate_strategies(strategy_factory.generate_property_table_without_failed_contracts()))
        self.assertEqual('st.from_regex(regex=r"s33l")',
                         generate_strategies(strategy_factory.generate_property_table_without_failed_contracts())['s'].represent())

    def test_example_function_link_1(self) -> None:
        strategy_factory = StrategyFactory(example_function_link_1)
        self.assertEqual('st.text(min_size=6, max_size=10)',
                         strategy_factory.generate_strategies()['s'].represent())

    def test_example_function_link_2(self) -> None:
        strategy_factory = StrategyFactory(example_function_link_2)
        self.assertEqual('st.from_regex(regex=r"test", fullmatch=True).filter(lambda s: len(s) > 5)'
                         '.filter(lambda s: len(s) <= 10)',
                         strategy_factory.generate_strategies()['s'].represent())
