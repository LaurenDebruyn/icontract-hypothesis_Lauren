from icontract import require
from typing import List, Dict, Any, Tuple
import unittest
import regex as re
from tabulate import tabulate

from icontract_hypothesis_Lauren.generate_symbol_table import Table, Row, Kind, generate_symbol_table, \
    generate_pretty_table, GenerationError


@require(lambda n1: n1 > 0)
def example_function_1(n1: int) -> None:
    pass


def expected_table_example_function_1() -> str:
    return tabulate([['0', 'n1', 'Kind.BASE', str(int), 'example_function_1', '', "{'>': ({((0),)}, set())}"]],
                    headers=['IDX', 'VAR_ID', 'KIND', 'TYPE', 'FUNCTION', 'PARENT', 'PROPERTIES'])


@require(lambda n1, n2: n1 > n2 > 4)
@require(lambda n1: n1 < 100)
@require(lambda n1, n4: n1 < n4)
@require(lambda n2, n3: n2 < 300 + n3)
@require(lambda n1, n3, n4: n3 < n4 > n1)
@require(lambda s: s.startswith("abc"))
@require(lambda lst: len(lst) > 0)
@require(lambda n1: n1 > 4)
def example_function_2(n1: int, n2: int, n3: int, n4: int, s: str, lst: List[int]) -> None:
    pass


def expected_table_example_function_2() -> Table:
    row_n1 = ['0', 'n1', 'Kind.BASE', str(int), 'example_function_2', '', "{'>': ({(n2,), ((4),)}, {n2}), '<': ({((100),), "
                                                                          "(n4,)}, {n4})}"]
    row_n2 = ['1', 'n2', 'Kind.BASE', str(int), 'example_function_2', '', "{'<': ({((300 + n3),)}, {n3}), '>': ({((4),)}, "
                                                                          "set())}"]
    row_n3 = ['2', 'n3', 'Kind.BASE', str(int), 'example_function_2', '', "{'<': ({(n4,)}, {n4})}"]
    row_n4 = ['3', 'n4', 'Kind.BASE', str(int), 'example_function_2', '', "{'>': ({(n1,)}, {n1})}"]
    row_s = ['4', 's', 'Kind.BASE', str(str), 'example_function_2', '', "{'startswith': ({(r'abc',)}, set())}"]
    row_lst = ['5', 'lst', 'Kind.BASE', str(List[int]), 'example_function_2', '', "{}"]
    row_lst_len = ['6', 'len(lst)', 'Kind.LINK', str(int), 'example_function_2', 'lst', "{'>': ({((0),)}, set())}"]
    return tabulate([row_n1, row_n2, row_n3, row_n4, row_s, row_lst, row_lst_len],
                    headers=['IDX', 'VAR_ID', 'KIND', 'TYPE', 'FUNCTION', 'PARENT', 'PROPERTIES'])


@require(lambda lst: all(item > 0 for item in lst))
def example_function_3(lst: List[int]) -> None:
    pass


def expected_table_example_function_3() -> str:
    row_lst = ['0', 'lst', 'Kind.BASE', str(List[int]), 'example_function_3', '', "{}"]
    row_item = ['1', 'item', 'Kind.UNIVERSAL_QUANTIFIER', str(int), 'example_function_3', 'lst', "{'>': ({((0),)}, "
                                                                                                 "set())}"]
    return tabulate([row_lst, row_item],
                    headers=['IDX', 'VAR_ID', 'KIND', 'TYPE', 'FUNCTION', 'PARENT', 'PROPERTIES'])


@require(lambda input_string: all(len(set(list(line))) == len(line) for line in input_string.split("\n")))
def example_function_4(input_string: str) -> None:
    pass


def expected_table_example_function_4() -> str:
    row_input_string = ['0', 'input_string', 'Kind.BASE', str(str), 'example_function_4', '', "{}"]
    row_line = ['1', 'line', 'Kind.UNIVERSAL_QUANTIFIER', str(str), 'example_function_4',
                'input_string.split("\ n")', "{}"]
    row_len_set_list = ['2', 'len(set(list(line)))', 'Kind.LINK', str(int), 'example_function_4',
                        'set(list(line))', "{'==': ({(len(line),)}, {line})}"]
    return tabulate([row_input_string, row_line, row_len_set_list],
                    headers=['IDX', 'VAR_ID', 'KIND', 'TYPE', 'FUNCTION', 'PARENT', 'PROPERTIES'])


@require(lambda s: re.match(r'(+|-)?[1-9][0-9]*', s))
def example_function_5(s: str) -> None:
    pass


def expected_table_example_function_5() -> str:
    row_s = ['0', 's', 'Kind.BASE', str(str), 'example_function_5', '',
             "{'re.match': ({(r'(+|-)?[1-9][0-9]*', s,)}, set())}"]
    return tabulate([row_s],
                    headers=['IDX', 'VAR_ID', 'KIND', 'TYPE', 'FUNCTION', 'PARENT', 'PROPERTIES'])


TEST_RE = re.compile(r'.*')


def dummy_func_6(x: Any) -> List:
    pass


@require(lambda batch: all(TEST_RE.match(line) for line in batch))
def example_function_6(batch: List[str]):
    pass


def expected_table_example_function_6() -> str:
    row_batch = ['0', 'batch', 'Kind.BASE', str(List[str]), 'example_function_6', '', "{}"]
    row_line = ['1', 'line', 'Kind.UNIVERSAL_QUANTIFIER', str(str), 'example_function_6', 'batch',
                "{'TEST_RE.match': ({(line,)}, set())}"]
    return tabulate([row_batch, row_line],
                    headers=['IDX', 'VAR_ID', 'KIND', 'TYPE', 'FUNCTION', 'PARENT', 'PROPERTIES'])


@require(lambda t: t[0] > 0 and t[1] < 0)
def example_function_7(t: Tuple[int, int]):
    pass


def expected_table_example_function_7() -> str:
    row_t = ['0', 't', 'Kind.BASE', str(Tuple[int, int]), 'example_function_7', '', "{}"]
    row_t_0 = ['1', 't[0]', 'Kind.LINK', str(int), 'example_function_7', 't', "{'>': ({((0),)}, set())}"]
    row_t_1 = ['2', 't[1]', 'Kind.LINK', str(int), 'example_function_7', 't', "{'<': ({((0),)}, set())}"]
    return tabulate([row_t, row_t_0, row_t_1],
                    headers=['IDX', 'VAR_ID', 'KIND', 'TYPE', 'FUNCTION', 'PARENT', 'PROPERTIES'])


@require(lambda n1, n2: (n1, n2) > (0, 0))
def example_function_8(n1: int, n2: int):
    pass


def expected_table_example_function_8() -> str:
    row_n1 = ['0', 'n1', 'Kind.BASE', str(int), 'example_function_8', '', "{'>': ({((0),)}, set())}"]
    row_n2 = ['1', 'n2', 'Kind.BASE', str(int), 'example_function_8', '', "{'>': ({((0),)}, set())}"]
    return tabulate([row_n1, row_n2],
                    headers=['IDX', 'VAR_ID', 'KIND', 'TYPE', 'FUNCTION', 'PARENT', 'PROPERTIES'])


@require(lambda lst: any(item <= 0 for item in lst))
def example_function_9(lst: List[int]):
    pass


def expected_table_example_function_9() -> str:
    row_lst = ['0', 'lst', 'Kind.BASE', str(List[int]), 'example_function_9', '', "{}"]
    row_lst_item = ['1', 'item', 'Kind.EXISTENTIAL_QUANTIFIER', str(int), 'example_function_9', 'lst',
                    "{'<=': ({((0),)}, set())}"]
    return tabulate([row_lst, row_lst_item],
                    headers=['IDX', 'VAR_ID', 'KIND', 'TYPE', 'FUNCTION', 'PARENT', 'PROPERTIES'])


@require(lambda lst: all(all(item > 0 for item in sub_lst) for sub_lst in lst))
def example_function_10(lst: List[List[int]]):
    pass


def expected_table_example_function_10() -> str:
    row_lst = ['0', 'lst', 'Kind.BASE', str(List[List[int]]), 'example_function_10', '', "{}"]
    row_sub_lst = ['1', 'sub_lst', 'Kind.UNIVERSAL_QUANTIFIER', str(List[int]), 'example_function_10', 'lst', "{}"]
    row_item = ['2', 'item', 'Kind.UNIVERSAL_QUANTIFIER', str(int), 'example_function_10', 'sub_lst',
                "{'>': ({((0),)}, set())}"]
    return tabulate([row_lst, row_sub_lst, row_item],
                    headers=['IDX', 'VAR_ID', 'KIND', 'TYPE', 'FUNCTION', 'PARENT', 'PROPERTIES'])


@require(lambda d: all(item > 0 for item in d.values()))
def example_function_11(d: Dict[int, int]):
    pass


def expected_table_example_function_11() -> str:
    row_d = ['0', 'd', 'Kind.BASE', str(Dict[int, int]), 'example_function_11', '', "{}"]
    row_d_item = ['1', 'item', 'Kind.UNIVERSAL_QUANTIFIER', str(int), 'example_function_11', 'd.values()',
                  "{'>': ({((0),)}, set())}"]
    return tabulate([row_d, row_d_item],
                    headers=['IDX', 'VAR_ID', 'KIND', 'TYPE', 'FUNCTION', 'PARENT', 'PROPERTIES'])


@require(lambda lst: all(re.match(r'.*', s) for s in lst))
def example_function_12(lst: List[str]):
    pass


def expected_table_example_function_12() -> str:
    row_lst = ['0', 'lst', 'Kind.BASE', str(List[str]), 'example_function_12', '', "{}"]
    row_s = ['1', 's', 'Kind.UNIVERSAL_QUANTIFIER', str(str), 'example_function_12', 'lst',
                  "{'re.match': ({(r'.*', s,)}, set())}"]
    return tabulate([row_lst, row_s],
                    headers=['IDX', 'VAR_ID', 'KIND', 'TYPE', 'FUNCTION', 'PARENT', 'PROPERTIES'])


@require(lambda n1, n2, n3: n1 > 0 and n1 <= n2 and n2 < n3 and n3 <= 100)
def example_function_13(n1: int, n2: int, n3: int):
    pass


def expected_table_example_function_13() -> str:
    row_n1 = ['0', 'n1', 'Kind.BASE', str(int), 'example_function_13', '',
              "{'>': ({((0),)}, set()), '<=': ({(n2,)}, {n2})}"]
    row_n2 = ['1', 'n2', 'Kind.BASE', str(int), 'example_function_13', '', "{'<': ({(n3,)}, {n3})}"]
    row_n3 = ['2', 'n3', 'Kind.BASE', str(int), 'example_function_13', '', "{'<=': ({((100),)}, set())}"]
    return tabulate([row_n1, row_n2, row_n3],
                    headers=['IDX', 'VAR_ID', 'KIND', 'TYPE', 'FUNCTION', 'PARENT', 'PROPERTIES'])


@require(lambda lst, n: len(lst) >= n)
def example_function_14(lst: List[int], n: int):
    pass


def expected_table_example_function_14() -> str:
    row_lst = ['0', 'lst', 'Kind.BASE', str(List[int]), 'example_function_14', '', "{}"]
    row_n = ['1', 'n', 'Kind.BASE', str(int), 'example_function_14', '', "{}"]
    row_len_lst = ['2', 'len(lst)', 'Kind.LINK', str(int), 'example_function_14', 'lst', "{'>=': ({(n,)}, {n})}"]
    return tabulate([row_lst, row_n, row_len_lst],
                    headers=['IDX', 'VAR_ID', 'KIND', 'TYPE', 'FUNCTION', 'PARENT', 'PROPERTIES'])


@require(lambda s: s.isnumeric())
def example_function_15(s: str) -> None:
    pass


def expected_table_example_function_15() -> str:
    row_s = ['0', 's', 'Kind.BASE', str(str), 'example_function_15', '', "{'isnumeric': (set(), set())}"]
    return tabulate([row_s],
                    headers=['IDX', 'VAR_ID', 'KIND', 'TYPE', 'FUNCTION', 'PARENT', 'PROPERTIES'])


@require(lambda d: all(item > 0 for item in d.keys()))
def example_function_16(d: Dict[int, Any]) -> None:
    pass


def expected_table_example_function_16() -> str:
    row_d = ['0', 'd', 'Kind.BASE', str(Dict[int, Any]), 'example_function_16', '', "{}"]
    row_d_keys = ['1', 'item', 'Kind.UNIVERSAL_QUANTIFIER', str(int), 'example_function_16', 'd.keys()',
                  "{'>': ({((0),)}, set())}"]
    return tabulate([row_d, row_d_keys],
                    headers=['IDX', 'VAR_ID', 'KIND', 'TYPE', 'FUNCTION', 'PARENT', 'PROPERTIES'])


@require(lambda d: all(re.match(r'[a-z]+', k) and v >= 0 for k, v in d.items()))
def example_function_17(d: Dict[str, int]) -> None:
    pass


def expected_table_example_function_17() -> str:
    row_d = ['0', 'd', 'Kind.BASE', str(Dict[str, int]), 'example_function_17', '', "{}"]
    row_d_keys = ['1', 'k', 'Kind.UNIVERSAL_QUANTIFIER', str(str), 'example_function_17', 'd.keys()',
                  "{'re.match': ({(r'[a-z]+', k,)}, set())}"]
    row_d_values = ['2', 'v', 'Kind.UNIVERSAL_QUANTIFIER', str(int), 'example_function_17', 'd.values()',
                    "{'>=': ({((0),)}, set())}"]
    return tabulate([row_d, row_d_keys, row_d_values],
                    headers=['IDX', 'VAR_ID', 'KIND', 'TYPE', 'FUNCTION', 'PARENT', 'PROPERTIES'])


@require(lambda lst_1, lst_2: all(item_1 > item_2 for item_1, item_2 in zip(lst_1, lst_2)))
def example_function_18(lst_1: List[int], lst_2: List[int]) -> None:
    pass


def expected_table_example_function_18() -> str:
    row_lst_1 = ['0', 'lst_1', 'Kind.BASE', str(List[int]), 'example_function_18', '', "{}"]
    row_lst_2 = ['1', 'lst_2', 'Kind.BASE', str(List[int]), 'example_function_18', '', "{}"]
    row_item_1 = ['2', 'item_1', 'Kind.UNIVERSAL_QUANTIFIER', str(int), 'example_function_18', 'lst_1',
                  "{'>': ({(item_2,)}, {item_2})}"]
    row_item_2 = ['3', 'item_2', 'Kind.UNIVERSAL_QUANTIFIER', str(int), 'example_function_18', 'lst_2', "{}"]
    return tabulate([row_lst_1, row_lst_2, row_item_1, row_item_2],
                    headers=['IDX', 'VAR_ID', 'KIND', 'TYPE', 'FUNCTION', 'PARENT', 'PROPERTIES'])


@require(lambda lst: all(item > 0 for sub_lst in lst for item in sub_lst))
def example_function_19(lst: List[List[int]]):
    # throws an exception TODO
    pass


# for when this will be supported
# def expected_table_example_function_19() -> str:
#     row_lst = ['0', 'lst', 'Kind.BASE', str(List[List[int]]), 'example_function_19', '', "{}"]
#     row_sub_lst = ['1', 'sub_lst', 'Kind.UNIVERSAL_QUANTIFIER', str(List[int]), 'example_function_19', 'lst', "{}"]
#     row_item = ['2', 'item', 'Kind.UNIVERSAL_QUANTIFIER', str(int), 'example_function_19', 'sub_lst',
#                 "{'>': ({((0),)}, set())}"]
#     return tabulate([row_lst, row_sub_lst, row_item],
#                     headers=['IDX', 'VAR_ID', 'KIND', 'TYPE', 'FUNCTION', 'PARENT', 'PROPERTIES'])


@require(lambda lst: all(len(item) > 2 for item in lst))
def example_function_20(lst: List[List[int]]):
    pass


def expected_table_example_function_20() -> str:
    row_lst = ['0', 'lst', 'Kind.BASE', str(List[List[int]]), 'example_function_20', '', "{}"]
    row_item = ['1', 'item', 'Kind.UNIVERSAL_QUANTIFIER', str(List[int]), 'example_function_20', 'lst', "{}"]
    row_item_len = ['2', 'len(item)', 'Kind.LINK', str(int), 'example_function_20', 'item', "{'>': ({((2),)}, set())}"]
    return tabulate([row_lst, row_item, row_item_len],
                    headers=['IDX', 'VAR_ID', 'KIND', 'TYPE', 'FUNCTION', 'PARENT', 'PROPERTIES'])


@require(lambda d: all(k > 0 for k in d))
def example_function_21(d: Dict[int, str]):
    pass


def expected_table_example_function_21() -> str:
    row_d = ['0', 'd', 'Kind.BASE', str(Dict[int, str]), 'example_function_21', '', "{}"]
    row_k = ['1', 'k', 'Kind.UNIVERSAL_QUANTIFIER', str(int), 'example_function_21', 'd', "{'>': ({((0),)}, set())}"]
    return tabulate([row_d, row_k],
                    headers=['IDX', 'VAR_ID', 'KIND', 'TYPE', 'FUNCTION', 'PARENT', 'PROPERTIES'])


@require(lambda s: all(c.isdigit() for c in s))
def example_function_22(s: str):
    # should throw generation error
    pass


@require(lambda t: all(item > 0 for item in t))
def example_function_23(t: Tuple[int, ...]):
    pass


def expected_table_example_function_23() -> str:
    row_t = ['0', 't', 'Kind.BASE', str(Tuple[int, ...]), 'example_function_23', '', "{}"]
    row_item = ['1', 'item', 'Kind.UNIVERSAL_QUANTIFIER', str(int), 'example_function_23', 't',
                "{'>': ({((0),)}, set())}"]
    return tabulate([row_t, row_item],
                    headers=['IDX', 'VAR_ID', 'KIND', 'TYPE', 'FUNCTION', 'PARENT', 'PROPERTIES'])


@require(lambda d: all(k > 0 and len(lst) > 0 for k, lst in d.items()))
def example_function_24(d: Dict[int, List[int]]):
    pass


def expected_table_example_function_24() -> str:
    row_t = ['0', 'd', 'Kind.BASE', str(Dict[int, List[int]]), 'example_function_24', '', "{}"]
    row_d_keys = ['1', 'k', 'Kind.UNIVERSAL_QUANTIFIER', str(int), 'example_function_24', 'd.keys()',
                  "{'>': ({((0),)}, set())}"]
    row_d_values = ['2', 'lst', 'Kind.UNIVERSAL_QUANTIFIER', str(List[int]), 'example_function_24', 'd.values()', "{}"]
    row_d_values_len = ['3', 'len(lst)', 'Kind.LINK', str(int), 'example_function_24', 'lst', "{'>': ({((0),)}, set())}"]
    return tabulate([row_t, row_d_keys, row_d_values, row_d_values_len],
                    headers=['IDX', 'VAR_ID', 'KIND', 'TYPE', 'FUNCTION', 'PARENT', 'PROPERTIES'])


@require(lambda lst1, lst2, lst3: all(n1 > n2 >= n3 for n1, n2, n3 in zip(lst1, lst2, lst3)))
def example_function_25(lst1: List[int], lst2: List[int], lst3: List[int]):
    pass


def expected_table_example_function_25() -> str:
    row_lst1 = ['0', 'lst1', 'Kind.BASE', str(List[int]), 'example_function_25', '', "{}"]
    row_lst2 = ['1', 'lst2', 'Kind.BASE', str(List[int]), 'example_function_25', '', "{}"]
    row_lst3 = ['2', 'lst3', 'Kind.BASE', str(List[int]), 'example_function_25', '', "{}"]
    row_n1 = ['3', 'n1', 'Kind.UNIVERSAL_QUANTIFIER', str(int), 'example_function_25', 'lst1', "{'>': ({(n2,)}, {n2})}"]
    row_n2 = ['4', 'n2', 'Kind.UNIVERSAL_QUANTIFIER', str(int), 'example_function_25', 'lst2', "{'>=': ({(n3,)}, {n3})}"]
    row_n3 = ['5', 'n3', 'Kind.UNIVERSAL_QUANTIFIER', str(int), 'example_function_25', 'lst3', "{}"]
    return tabulate([row_lst1, row_lst2, row_lst3, row_n1, row_n2, row_n3],
                    headers=['IDX', 'VAR_ID', 'KIND', 'TYPE', 'FUNCTION', 'PARENT', 'PROPERTIES'])


@require(lambda lst: all(n1 < n2 < n3 <= 100 for n1, n2, n3 in lst))
def example_function_26(lst: List[Tuple[int, int, int]]):
    pass

# TODO should be supported
# def expected_table_example_function_26() -> str:
#     row_lst = ['0', 'lst', 'Kind.BASE', str(List[Tuple[int, int, int]]), 'example_function_26', '', "{}"]
#     row_n1 = ['1', 'n1', 'Kind.UNIVERSAL_QUANTIFIER', str(int), 'example_function_26', '', "{'<': ({(n2,)}, {n2})}"]
#     row_n2 = ['2', 'n2', 'Kind.UNIVERSAL_QUANTIFIER', str(int), 'example_function_26', '', "{'<': ({(n3,)}, {n3})}"]
#     row_n3 = ['3', 'n3', 'Kind.UNIVERSAL_QUANTIFIER', str(int), 'example_function_26', '', "{'<:' ({((100),)}, set())}"]
#     return tabulate([row_lst, row_n1, row_n2, row_n3],
#                     headers=['IDX', 'VAR_ID', 'KIND', 'TYPE', 'FUNCTION', 'PARENT', 'PROPERTIES'])


class GenerateSymbolTableTest(unittest.TestCase):

    def test_example_function_1(self) -> None:
        _, table = generate_symbol_table(example_function_1)
        self.assertEqual(expected_table_example_function_1(),
                         generate_pretty_table(table))

    def test_example_function_2(self) -> None:
        _, table = generate_symbol_table(example_function_2)
        self.assertEqual(expected_table_example_function_2(),
                         generate_pretty_table(table))

    def test_example_function_3(self) -> None:
        _, table = generate_symbol_table(example_function_3)
        self.assertEqual(expected_table_example_function_3(),
                         generate_pretty_table(table))

    def test_example_function_4(self) -> None:
        _, table = generate_symbol_table(example_function_4)
        self.assertEqual(expected_table_example_function_4(),
                         generate_pretty_table(table))

    def test_example_function_5(self) -> None:
        _, table = generate_symbol_table(example_function_5)
        self.assertEqual(expected_table_example_function_5(),
                         generate_pretty_table(table))

    def test_example_function_6(self) -> None:
        _, table = generate_symbol_table(example_function_6)
        self.assertEqual(expected_table_example_function_6(),
                         generate_pretty_table(table))

    def test_example_function_7(self) -> None:
        _, table = generate_symbol_table(example_function_7)
        self.assertEqual(expected_table_example_function_7(),
                         generate_pretty_table(table))

    def test_example_function_8(self) -> None:
        _, table = generate_symbol_table(example_function_8)
        self.assertEqual(expected_table_example_function_8(),
                         generate_pretty_table(table))

    def test_example_function_9(self) -> None:
        _, table = generate_symbol_table(example_function_9)
        self.assertEqual(expected_table_example_function_9(),
                         generate_pretty_table(table))

    def test_example_function_10(self) -> None:
        _, table = generate_symbol_table(example_function_10)
        self.assertEqual(expected_table_example_function_10(),
                         generate_pretty_table(table))

    def test_example_function_11(self) -> None:
        _, table = generate_symbol_table(example_function_11)
        self.assertEqual(expected_table_example_function_11(),
                         generate_pretty_table(table))

    def test_example_function_12(self) -> None:
        _, table = generate_symbol_table(example_function_12)
        self.assertEqual(expected_table_example_function_12(),
                         generate_pretty_table(table))

    def test_example_function_13(self) -> None:
        _, table = generate_symbol_table(example_function_13)
        self.assertEqual(expected_table_example_function_13(),
                         generate_pretty_table(table))

    def test_example_function_14(self) -> None:
        _, table = generate_symbol_table(example_function_14)
        self.assertEqual(expected_table_example_function_14(),
                         generate_pretty_table(table))

    def test_example_function_15(self) -> None:
        _, table = generate_symbol_table(example_function_15)
        self.assertEqual(expected_table_example_function_15(),
                         generate_pretty_table(table))

    def test_example_function_16(self) -> None:
        _, table = generate_symbol_table(example_function_16)
        self.assertEqual(expected_table_example_function_16(),
                         generate_pretty_table(table))

    def test_example_function_17(self) -> None:
        _, table = generate_symbol_table(example_function_17)
        self.assertEqual(expected_table_example_function_17(),
                         generate_pretty_table(table))

    def test_example_function_18(self) -> None:
        _, table = generate_symbol_table(example_function_18)
        self.assertEqual(expected_table_example_function_18(),
                         generate_pretty_table(table))

    def test_example_function_19(self) -> None:
        with self.assertRaises(GenerationError):
            generate_symbol_table(example_function_19)

    def test_example_function_20(self) -> None:
        _, table = generate_symbol_table(example_function_20)
        self.assertEqual(expected_table_example_function_20(),
                         generate_pretty_table(table))

    def test_example_function_21(self) -> None:
        _, table = generate_symbol_table(example_function_21)
        self.assertEqual(expected_table_example_function_21(),
                         generate_pretty_table(table))

    def test_example_function_22(self) -> None:
        with self.assertRaises(GenerationError):
            generate_symbol_table(example_function_22)

    def test_example_function_23(self) -> None:
        _, table = generate_symbol_table(example_function_23)
        self.assertEqual(expected_table_example_function_23(),
                         generate_pretty_table(table))

    def test_example_function_24(self) -> None:
        _, table = generate_symbol_table(example_function_24)
        self.assertEqual(expected_table_example_function_24(),
                         generate_pretty_table(table))

    def test_example_function_25(self) -> None:
        _, table = generate_symbol_table(example_function_25)
        self.assertEqual(expected_table_example_function_25(),
                         generate_pretty_table(table))

    def test_example_function_26(self) -> None:
        with self.assertRaises(GenerationError):
            generate_symbol_table(example_function_22)

    # def test_example_function_27(self) -> None:
    #     _, table = generate_symbol_table(example_function_27)
    #     self.assertEqual(expected_table_example_function_27(),
    #                      generate_pretty_table(table))
