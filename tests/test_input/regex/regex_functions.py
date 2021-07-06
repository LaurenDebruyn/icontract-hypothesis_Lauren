from typing import List

from icontract import require
import regex as re


@require(lambda s: s.startswith('abc'))
def base_startswith(s: str) -> None:
    pass


@require(lambda s: re.match(r'(+|-)?[1-9][0-9]*', s))
def base_re_match(s: str) -> None:
    pass


TEST_RE = re.compile(r'.*')


@require(lambda s: all(TEST_RE.match(line) for line in s))
def base_re_match_compiled(s: List[str]):
    pass


@require(lambda s: re.match(r'test', s) and len(s) > 2)
def link_filter(s: str):
    pass


@require(lambda s: s.contains('s33l'))
def base_contains(s: str):
    pass


@require(lambda s: s.startswith('abc') and s.endswith('xyz'))
def base_startswith_endswith(s: str):
    pass
