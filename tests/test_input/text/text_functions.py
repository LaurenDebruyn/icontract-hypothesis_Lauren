from icontract import require


@require(lambda s: all(len(set(list(line))) == len(line) for line in s.split("\n")))
def universal_link(s: str) -> None:
    pass


@require(lambda s: s.isalnum())
def base_isalnum(s: str) -> None:
    pass


@require(lambda s: s.isalpha())
def base_isalpha(s: str) -> None:
    pass


@require(lambda s: s.isdigit())
def base_isdigit(s: str) -> None:
    pass


@require(lambda s: s.islower())
def base_islower(s: str) -> None:
    pass


@require(lambda s: s.isnumeric())
def base_isnumeric(s: str) -> None:
    pass


@require(lambda s: s.isspace())
def base_isspace(s: str) -> None:
    pass


@require(lambda s: s.isupper())
def base_isupper(s: str) -> None:
    pass


@require(lambda s: s.isdecimal())
def base_isdecimal(s: str) -> None:
    pass


@require(lambda s: len(s) < 100)
def link_lt(s: str) -> None:
    pass


@require(lambda s: len(s) <= 100)
def link_lte(s: str) -> None:
    pass


@require(lambda s: len(s) > 100)
def link_gt(s: str) -> None:
    pass


@require(lambda s: len(s) >= 100)
def link_gte(s: str) -> None:
    pass


# TODO
@require(lambda s: s.isdigit() and s.isnumeric() and s.isdecimal())
def base_isdigit_isnumeric_isdecimal(s: str):
    pass


# TODO
@require(lambda s: s.alnum() and s.alpha())
def base_isalnum_isalpha(s: str):
    pass


# TODO
@require(lambda s: s.isnumeric() and len(s) > 5)
def base_isnumeric_len(s: str):
    pass


# TODO
@require(lambda s: len(s) <=10 and len(s) > 5)
def link_len(s: str):
    pass
