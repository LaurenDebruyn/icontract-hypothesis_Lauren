from icontract import require
from typing import List, Tuple, Any, Dict
import regex as re


@require(lambda n1: n1 > 0)
def example_function_1(n1: int) -> None:
    pass


@require(lambda n1, n2: n1 > n2 > 4)
@require(lambda n1: n1 < 100)
@require(lambda n1, n4: n1 < n4)
@require(lambda n2, n3: n2 < 300 + n3)
@require(lambda n1, n3, n4: n3 < n4)
@require(lambda s: s.startswith("abc"))
@require(lambda lst: len(lst) > 0)
def example_function_2(n1: int, n2: int, n3: int, n4: int, s: str, lst: List[int]) -> None:
    pass


@require(lambda lst: all(item > 0 for item in lst))
def example_function_3(lst: List[int]) -> None:
    pass


@require(lambda input_string: all(len(set(list(line))) == len(line) for line in input_string.split("\n")))
def example_function_4(input_string: str) -> None:
    pass


@require(lambda s: re.match(r'(+|-)?[1-9][0-9]*', s))
def example_function_5(s: str) -> None:
    pass


_TEST_RE = re.compile(r'.*')


def _dummy_func_6(x: Any) -> List:
    pass


@require(lambda batch: all(_TEST_RE.match(line) for line in _dummy_func_6(batch)))
def example_function_6(batch: str):
    pass


@require(lambda t: t[0] > 0 and t[1] < 0)
def example_function_7(t: Tuple[int, int]):
    pass


@require(lambda n1, n2: (n1, n2) > (0, 0))
def example_function_8(n1: int, n2: int):
    pass


@require(lambda lst: any(item <= 0 for item in lst))
def example_function_9(lst: List[int]):
    pass


@require(lambda lst: all(all(item > 0 for item in sub_lst) for sub_lst in lst))
def example_function_10(lst: List[List[int]]):
    pass


@require(lambda d: all(item > 0 for item in d.values()))
def example_function_11(d: Dict[int, int]):
    pass


@require(lambda lst: all(re.match(r'.*', s) for s in lst))
def example_function_12(lst: List[str]):
    pass


@require(lambda n1, n2, n3: n1 > 0 and n1 >= n3 and n1 < n2)
@require(lambda n2, n3: n2 <= 100 and n3 <= n2)
def example_function_13(n1: int, n2: int, n3: int):
    pass


@require(lambda s: s.isalpha())
def example_function_14(s: str):
    pass


@require(lambda s: s.isalnum())
def example_function_15(s: str):
    pass


@require(lambda s: s.isdigit())
def example_function_16(s: str):
    pass


@require(lambda s: s.islower())
def example_function_17(s: str):
    pass


@require(lambda s: s.isnumeric())
def example_function_18(s: str):
    pass


@require(lambda s: s.isspace())
def example_function_19(s: str):
    pass


@require(lambda s: s.isupper())
def example_function_20(s: str):
    pass


@require(lambda s: s.isdecimal())
def example_function_21(s: str):
    pass


@require(lambda s: s.isdigit() and s.isnumeric() and s.isdecimal())
def example_function_22(s: str):
    pass


@require(lambda s: s.isalnum() and s.isalpha())
def example_function_23(s: str):
    pass


@require(lambda s: s.isupper() and s.islower())
def example_function_24(s: str):
    pass


@require(lambda s: s.startswith('abc') and s.endswith('xyz'))
def example_function_25(s: str):
    pass


# @require(lambda s: 's33l' in s)
@require(lambda s: s.contains('s33l'))
def example_function_26(s: str):
    pass


@require(lambda s: len(s) > 5 and len(s) <= 10)
def example_function_link_1(s: str):
    """SymbolicTextStrategy with link"""
    pass

@require(lambda s: re.match(r'test', s) and len(s) > 5 and len(s) <= 10)
def example_function_link_2(s: str):
    """SymbolicFromRegexStrategy with link"""
    pass
