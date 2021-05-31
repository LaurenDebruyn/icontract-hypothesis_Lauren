from typing import Tuple

from icontract import require


@require(lambda t: t[0] > 0 and t[1] < 0)
def link(t: Tuple[int, int]):
    pass
