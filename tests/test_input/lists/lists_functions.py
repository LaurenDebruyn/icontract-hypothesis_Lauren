from icontract import require
from typing import List
import regex as re


@require(lambda lst: len(lst) > 0)
def link_1(lst: List[int]) -> None:
    pass


@require(lambda lst: all(item > 0 for item in lst))
def universal_1(lst: List[int]) -> None:
    pass


@require(lambda lst: all(re.match(r'test', s) for s in lst))
def universal_2(lst: List[str]):
    pass


@require(lambda lst: any(item <= 0 for item in lst))
def existential_1(lst: List[int]):
    pass


@require(lambda lst: all(all(item > 0 for item in sub_lst) for sub_lst in lst))
def universal_nested_1(lst: List[List[int]]):
    pass


@require(lambda lst: all(len(sub_list) > 2 for sub_list in lst))
def universal_nested_link_1(lst: List[List[int]]):
    pass


@require(lambda lst: all(s.isidentifier() for s in lst))
def universal_filter_1(lst: List[str]):
    pass
