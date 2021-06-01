from icontract import require


@require(lambda n1: n1 > 0)
def base_1(n1: int) -> None:
    pass


@require(lambda n1, n2: n1 > n2 > 4)
@require(lambda n1: n1 < 100)
@require(lambda n1, n4: n1 < n4)
@require(lambda n2, n3: n2 < n3)
@require(lambda n1, n3, n4: n3 < n4)
def base_2(n1: int, n2: int, n3: int, n4: int) -> None:
    pass


@require(lambda n1, n2, n3: n1 > 0 and n1 >= n3 and n1 < n2)
@require(lambda n2, n3: n2 <= 100 and n3 <= n2)
def base_3(n1: int, n2: int, n3: int):
    pass


@require(lambda n1, n2, n3: n1 >= 0 and n1 >= n3 and n1 <= n2)
@require(lambda n2, n3: n2 <= 100 and n3 <= n2)
def base_4(n1: int, n2: int, n3: int):
    pass


@require(lambda n1, n2: (n1, n2) > (0, 0))
def base_5(n1: int, n2: int):
    pass


@require(lambda n1, n2: n1 + 10 >= 0 and n1 + 10 <= n2 - 10 <= 100)
def base_6(n1: int, n2: int):
    pass


@require(lambda n1, n2: n1 >= 0 and n1 <= n2)
def base_7(n1: int, n2: int):
    pass


@require(lambda n1, n2: n1 >= 0 and n1 < n2)
def base_8(n1: int, n2: int):
    pass


@require(lambda n1, n2: n1 <= 100 and n1 >= n2)
def base_9(n1: int, n2: int):
    pass


@require(lambda n1, n2: n1 <= 100 and n1 > n2)
def base_10(n1: int, n2: int):
    pass
