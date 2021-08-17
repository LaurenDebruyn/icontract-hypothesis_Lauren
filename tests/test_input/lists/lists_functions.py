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


@require(lambda lst: all(all(len(sub_sub_list) > 2 for sub_sub_list in sub_list) for sub_list in lst))
def universal_nested_link_2(lst: List[List[List[int]]]):
    pass


@require(lambda lst: len(set(lst)) == len(lst))
def is_unique(lst: List[int]):
    pass


@require(lambda lst: 0 not in lst)
def not_in(lst: List[int]):
    pass


@require(lambda lst: lst[0] == 'abc')
def filter_index(lst: List[str]):
    pass


@require(lambda lst: len(lst[0]) > 0)
def nested_filter_index(lst: List[List[int]]):
    pass


@require(lambda lst: len(lst) > 0)
@require(lambda lst: all(item >= lst[0] for item in lst))
def complex_linked_filter(lst: List[int]):
    pass


@require(lambda lst: len(lst) > 0)
@require(lambda lst: all(len(sub_lst) >= len(lst[0]) for sub_lst in lst))
def complex_linked_filter_2(lst: List[List[int]]):
    pass
