import dataclasses
import enum

from icontract import require
from typing import List, Dict, Any, Tuple
import unittest
import regex as re

from icontract_hypothesis_Lauren.generate_symbol_table import Table, Row, Kind, generate_symbol_table, \
    generate_pretty_table, print_pretty_table


def assert_equal_row(test_case: 'GenerateSymbolTableTest', row1: Row, row2: Row, error_msg: str) -> None:
    test_case.assertEqual(row1.var_id, row2.var_id, error_msg)
    test_case.assertEqual(row1.kind, row2.kind, error_msg)
    test_case.assertEqual(row1.type, row2.type, error_msg)
    test_case.assertEqual(row1.function, row2.function, error_msg)
    test_case.assertEqual(row1.parent, row2.parent, error_msg)
    test_case.assertEqual(len(row1.properties), len(row2.properties), error_msg)
    test_case.assertEqual(row1.properties, row2.properties, error_msg)


def assert_equal_table(test_case: 'GenerateSymbolTableTest', table1: Table, table2: Table) -> None:
    error_msg = f"""
    ACTUAL:
    
    {generate_pretty_table(table1)}
    
    EXPECTED:
    
    {generate_pretty_table(table2)}
    """
    for row_table1, row_table2 in zip(table1.get_rows(), table2.get_rows()):
        assert_equal_row(test_case, row_table1, row_table2, error_msg)


@require(lambda n1: n1 > 0)
def example_function_1(n1: int) -> None:
    pass


def expected_table_example_function_1() -> Table:
    row = Row(var_id='n1',
              kind=Kind.BASE,
              type=int,
              function='example_function_1',
              parent=None,
              properties={'>': ({'0'}, set())})
    return Table([row])


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
    row_n1 = Row(var_id='n1',
                 kind=Kind.BASE,
                 type=int,
                 function='example_function_2',
                 parent=None,
                 properties={'<': ({'100', 'n4'}, {'n4'}), '>': ({'n2', '4'}, {'n2'})})
    row_n2 = Row(var_id='n2',
                 kind=Kind.BASE,
                 type=int,
                 function='example_function_2',
                 parent=None,
                 properties={'<': ({'300 + n3'}, {'n3'}), '>': ({'4'}, set())})
    row_n3 = Row(var_id='n3',
                 kind=Kind.BASE,
                 type=int,
                 function='example_function_2',
                 parent=None,
                 properties={'<': ({'n4'}, {'n4'})})
    row_n4 = Row(var_id='n4',
                 kind=Kind.BASE,
                 type=int,
                 function='example_function_2',
                 parent=None,
                 properties={'>': ({'n1'}, {'n1'})})
    row_s = Row(var_id='s',
                kind=Kind.BASE,
                type=str,
                function='example_function_2',
                parent=None,
                properties={'.startswith': ({"'abc'"}, set())})
    row_lst = Row(var_id='lst',
                  kind=Kind.BASE,
                  type=List[int],
                  function='example_function_2',
                  parent=None,
                  properties={})
    row_len_lst = Row(var_id='lst.len',  # TODO make this len(lst)
                      kind=Kind.LINK,
                      type=int,
                      function='example_function_2',
                      parent='lst',
                      properties={'>': ({'0'}, set())})

    return Table([
        row_n1,
        row_n2,
        row_n3,
        row_n4,
        row_s,
        row_lst,
        row_len_lst
    ])


@require(lambda lst: all(item > 0 for item in lst))
def example_function_3(lst: List[int]) -> None:
    pass


def expected_table_example_function_3() -> Table:
    row_lst = Row(var_id='lst',
                  kind=Kind.BASE,
                  type=List[int],
                  function='example_function_3',
                  parent=None,
                  properties={})
    row_item = Row(var_id='item',
                   kind=Kind.UNIVERSAL_QUANTIFIER,
                   type=int,
                   function='example_function_3',
                   parent='lst',
                   properties={'>': ({'0'}, set())})
    return Table([row_lst, row_item])


@require(lambda input_string: all(len(set(list(line))) == len(line) for line in input_string.split("\n")))
def example_function_4(input_string: str) -> None:
    pass


def expected_table_example_function_4() -> Table:
    row_input_string = Row(var_id='input_string',
                           kind=Kind.BASE,
                           type=str,
                           function='example_function_4',
                           parent=None,
                           properties={})

    row_line = Row(var_id='line',
                   kind=Kind.UNIVERSAL_QUANTIFIER,
                   type=str,
                   function='example_function_4',
                   parent='input_string.split("\ n")',
                   properties={})

    row_len_set_list = Row(var_id='len(set(list(line)))',
                           kind=Kind.LINK,
                           type=int,
                           function='example_function_4',
                           parent='line',
                           properties={'==': ({'len(line)'}, {'line'})})
    return Table([row_input_string, row_line, row_len_set_list])


@require(lambda s: re.match(r'(+|-)?[1-9][0-9]*', s))
def example_function_5(s: str) -> None:
    pass


def expected_table_example_function_5() -> Table:
    row = Row(var_id='s',
              kind=Kind.BASE,
              type=str,
              function='example_function_5',
              parent=None,
              properties={'re.match': ({("'(+|-)?[1-9][0-9]*'", 's')}, {'s'})})
    return Table([row])


TEST_RE = re.compile(r'.*')


def dummy_func_6(x: Any) -> List:
    pass


@require(lambda batch: all(TEST_RE.match(line) for line in dummy_func_6(batch)))
def example_function_6(batch: str):
    pass


def expected_table_example_function_6() -> Table:
    row_batch = Row(var_id='batch',
                    kind=Kind.BASE,
                    type=str,
                    function='example_function_6',
                    parent=None,
                    properties={})
    row_line = Row(var_id='line',
                   kind=Kind.UNIVERSAL_QUANTIFIER,
                   type=str,
                   function='example_function_6',
                   parent='dummy_func_6(batch)',
                   properties={'TEST_RE.match': ({'line'}, {'line'})})
    return Table([row_batch, row_line])


@require(lambda t: t[0] > 0 and t[1] < 0)
def example_function_7(t: Tuple[int, int]):
    pass


def expected_table_example_function_7() -> Table:
    row_t = Row(var_id='t',
                kind=Kind.BASE,
                type=Tuple[int, int],
                function='example_function_7',
                parent=None,
                properties={})
    row_t_0 = Row(var_id='t[0]',
                  kind=Kind.LINK,
                  type=int,
                  function='example_function_7',
                  parent='t',
                  properties={'>': ({'0'}, set())})
    row_t_1 = Row(var_id='t[1]',
                  kind=Kind.LINK,
                  type=int,
                  function='example_function_7',
                  parent='t',
                  properties={'<': ({'0'}, set())})
    return Table([row_t, row_t_0, row_t_1])


@require(lambda n1, n2: (n1, n2) > (0, 0))
def example_function_8(n1: int, n2: int):
    pass


def expected_table_example_function_8() -> Table:
    row_n1 = Row(var_id='n1',
                 kind=Kind.BASE,
                 type=int,
                 function='example_function_8',
                 parent=None,
                 properties={'>': ({'0'}, set())})
    row_n2 = Row(var_id='n2',
                 kind=Kind.BASE,
                 type=int,
                 function='example_function_8',
                 parent=None,
                 properties={'>': ({'0'}, set())})
    return Table([row_n1, row_n2])


@require(lambda lst: any(item <= 0 for item in lst))
def example_function_9(lst: List[int]):
    pass


def expected_table_example_function_9() -> Table:
    row_lst = Row(var_id='lst',
                  kind=Kind.BASE,
                  type=List[int],
                  function='example_function_9',
                  parent=None,
                  properties={})
    row_lst_item = Row(var_id='item',
                       kind=Kind.EXISTENTIAL_QUANTIFIER,
                       type=int,
                       function='example_function_9',
                       parent='lst',
                       properties={'<=': ({'0'}, set())})
    return Table([row_lst, row_lst_item])


@require(lambda lst: all(all(item > 0 for item in sub_lst) for sub_lst in lst))
def example_function_10(lst: List[List[int]]):
    pass


def expected_table_example_function_10() -> Table:
    row_lst = Row(var_id='lst',
                  kind=Kind.BASE,
                  type=List[List[int]],
                  function='example_function_10',
                  parent=None,
                  properties={})
    row_sub_lst = Row(var_id='sub_lst',
                      kind=Kind.UNIVERSAL_QUANTIFIER,
                      type=List[int],
                      function='example_function_10',
                      parent='lst',
                      properties={})
    row_item = Row(var_id='item',
                   kind=Kind.UNIVERSAL_QUANTIFIER,
                   type=int,
                   function='example_function_10',
                   parent='sub_lst',
                   properties={'>': ({'0'}, set())})
    return Table([row_lst, row_sub_lst, row_item])


@require(lambda d: all(item > 0 for item in d.values()))
def example_function_11(d: Dict[int, int]):
    pass


def expected_table_example_function_11() -> Table:
    row_d = Row(var_id='d',
                kind=Kind.BASE,
                type=Dict[int, int],
                function='example_function_11',
                parent=None,
                properties={})
    row_d_item = Row(var_id='item',
                     kind=Kind.UNIVERSAL_QUANTIFIER,
                     type=int,
                     function='example_function_11',
                     parent='d.values()',
                     properties={'>': ({'0'}, set())})
    return Table([row_d, row_d_item])


@require(lambda lst: all(re.match(r'.*', s) for s in lst))
def example_function_12(lst: List[str]):
    pass


def expected_table_example_function_12() -> Table:
    row_lst = Row(var_id='lst',
                  kind=Kind.BASE,
                  type=List[str],
                  function='example_function_12',
                  parent=None,
                  properties={})
    row_s = Row(var_id='s',
                kind=Kind.UNIVERSAL_QUANTIFIER,
                type=str,
                function='example_function_12',
                parent='lst',
                properties={'re.match': ({("'.*'", 's')}, {'s'})})
    return Table([row_lst, row_s])


class GenerateSymbolTableTest(unittest.TestCase):

    def test_example_function_1(self) -> None:
        assert_equal_table(self,
                           generate_symbol_table(example_function_1),
                           expected_table_example_function_1())

    def test_example_function_2(self) -> None:
        assert_equal_table(self,
                           generate_symbol_table(example_function_2),
                           expected_table_example_function_2())

    def test_example_function_3(self) -> None:
        assert_equal_table(self,
                           generate_symbol_table(example_function_3),
                           expected_table_example_function_3())

    def test_example_function_4(self) -> None:
        assert_equal_table(self,
                           generate_symbol_table(example_function_4),
                           expected_table_example_function_4())

    def test_example_function_5(self) -> None:
        self.assertEqual(generate_pretty_table(generate_symbol_table(example_function_5)),
                         generate_pretty_table(expected_table_example_function_5()))

    def test_example_function_6(self) -> None:
        self.assertEqual(generate_pretty_table(generate_symbol_table(example_function_6)),
                         generate_pretty_table(expected_table_example_function_6()))

    def test_example_function_7(self) -> None:
        self.assertEqual(generate_pretty_table(generate_symbol_table(example_function_7)),
                         generate_pretty_table(expected_table_example_function_7()))

    def test_example_function_8(self) -> None:
        self.assertEqual(generate_pretty_table(generate_symbol_table(example_function_8)),
                         generate_pretty_table(expected_table_example_function_8()))

    def test_example_function_9(self) -> None:
        self.assertEqual(generate_pretty_table(generate_symbol_table(example_function_9)),
                         generate_pretty_table(expected_table_example_function_9()))

    def test_example_function_10(self) -> None:
        self.assertEqual(generate_pretty_table(generate_symbol_table(example_function_10)),
                         generate_pretty_table(expected_table_example_function_10()))

    def test_example_function_11(self) -> None:
        self.assertEqual(generate_pretty_table(generate_symbol_table(example_function_11)),
                         generate_pretty_table(expected_table_example_function_11()))

    def test_example_function_12(self) -> None:
        self.assertEqual(generate_pretty_table(generate_symbol_table(example_function_12)),
                         generate_pretty_table(expected_table_example_function_12()))
