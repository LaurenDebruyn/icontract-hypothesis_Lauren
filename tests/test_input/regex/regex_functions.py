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


@require(lambda batch: all(TEST_RE.match(line) for line in batch))
def base_re_match_compiled(batch: List[str]):
    pass
