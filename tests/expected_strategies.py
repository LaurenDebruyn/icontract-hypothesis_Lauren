import hypothesis.strategies as st
from typing import Tuple, List


@st.composite
def strategy_example_function_1(draw) -> Tuple[int]:
    n1 = draw(st.integers(min_value=1))
    return n1


@st.composite
def strategy_example_function_2(draw) -> Tuple[int, int, int, int, str, List[int]]:
    n4 = draw(st.integers())
    n3 = draw(st.integers(max_value=n4-1))
    n2 = draw(st.integers(min_value=5, max_value=300+n3))
    n1 = draw(st.integers(min_value=n2+1, max_value=min(99, n4-1)))
    s = draw(st.from_regex(r'^abc'))
    lst = draw(st.lists(elements=st.integers(), min_size=1))
    return n1, n2, n3, n4, s, lst


@st.composite
def strategy_example_function_3(draw) -> Tuple[List[int]]:
    lst = draw(st.lists(elements=st.integers(min_value=1)))
    return lst


@st.composite
def strategy_example_function_4(draw) -> Tuple[str]:
    # TODO
    raise NotImplementedError
