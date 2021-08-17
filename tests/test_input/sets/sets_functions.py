from icontract import require
from typing import Set
import regex as re


@require(lambda a_set: len(a_set) > 0)
def link_1(a_set: Set[int]) -> None:
    pass


@require(lambda a_set: all(item > 0 for item in a_set))
def universal_1(a_set: Set[int]) -> None:
    pass


@require(lambda a_set: all(re.match(r'test', s) for s in a_set))
def universal_2(a_set: Set[str]):
    pass


@require(lambda a_set: any(item <= 0 for item in a_set))
def existential_1(a_set: Set[int]):
    pass


@require(lambda a_set: all(all(item > 0 for item in sub_a_set) for sub_a_set in a_set))
def universal_nested_1(a_set: Set[Set[int]]):
    pass


@require(lambda a_set: all(len(sub_set) > 2 for sub_set in a_set))
def universal_nested_link_1(a_set: Set[Set[int]]):
    pass


@require(lambda a_set: all(s.isidentifier() for s in a_set))
def universal_filter_1(a_set: Set[str]):
    pass


@require(lambda a_set: all(all(len(sub_sub_set) > 2 for sub_sub_set in sub_set) for sub_set in a_set))
def universal_nested_link_2(a_set: Set[Set[Set[int]]]):
    pass


@require(lambda a_set: 0 not in a_set)
def not_in(a_set: Set[int]):
    pass


@require(lambda a_set: a_set[0] == 'abc')
def filter_index(a_set: Set[str]):
    pass


@require(lambda a_set: len(a_set[0]) > 0)
def nested_filter_index(a_set: Set[Set[int]]):
    pass
