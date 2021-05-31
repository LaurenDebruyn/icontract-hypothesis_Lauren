from typing import Dict

from icontract import require


@require(lambda d: all(item > 0 for item in d.values()))
def universal_values(d: Dict[int, int]):
    pass
