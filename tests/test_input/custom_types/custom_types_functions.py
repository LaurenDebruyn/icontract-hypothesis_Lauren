from dataclasses import dataclass
from icontract import DBC, ensure, require


class Number:
    def __init__(self, number: int) -> None:
        self.number = number


@dataclass
class NumberDataclass:
    number: int


class NumberDBC(DBC):
    @require(lambda number: number >= 0)
    def __init__(self, number: int):
        self.number = number


@require(lambda number, new_number: number.number <= 1000 - new_number)
@ensure(lambda result, number, new_number: result.number == number.number + new_number)
def add_normal_class(number: Number, new_number: int) -> Number:
    return Number(number.number + new_number)


@require(lambda number, new_number: number.number <= 1000 - new_number)
@ensure(lambda result, number, new_number: result.number == number.number + new_number)
def add_dataclass_class(number: NumberDataclass, new_number: int) -> NumberDataclass:
    return NumberDataclass(number.number + new_number)


@require(lambda number, new_number: number.number <= 1000 - new_number)
@ensure(lambda result, number, new_number: result.number == number.number + new_number)
def add_dbc_class(number: NumberDBC, new_number: int) -> NumberDBC:
    return NumberDBC(number.number + new_number)
