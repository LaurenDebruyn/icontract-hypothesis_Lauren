# from icontract import require
from hypothesis import given
import hypothesis.strategies as st
from typing import Tuple
from test_generate_symbol_table import *
import networkx as nx
import matplotlib.pyplot as plt


@require(lambda n1, n2: n1 >= n2 >= 4)
@require(lambda n1, n4: n1 <= n4)
@require(lambda n2, n3: n2 <= n3 + 300)
@require(lambda n1, n3, n4: n3 <= n4)
def example_function_2_bis(n1: int, n2: int, n3: int, n4: int) -> None:
    pass


@st.composite
def strategy_example_function_2(draw) -> Tuple[int, int, int, int]:
    n4 = draw(st.integers(min_value=4))
    n3 = draw(st.integers(min_value=4, max_value=n4))
    n2 = draw(st.integers(min_value=4, max_value=min(n3 + 300, n4)))
    n1 = draw(st.integers(min_value=n2, max_value=n4))
    return n1, n2, n3, n4


@given(n=strategy_example_function_2())
def test_strategy_example_function_2(n: Tuple[int, int, int, int, str, List[int]]) -> None:
    try:
        example_function_2_bis(*n)
    except Exception as e:
        print("n1: {}, n2: {}, n3: {}, n4: {}, s: {}, lst: {}".format(*n))
        raise e


#        max
#    n1  ->  n3
#     min   / max
# n4  ->  n2
@require(lambda n1, n2, n3: n1 >= 0 and n1 <= n3 and n2 <= n3 and n1 >= n2)
def sketch(n1: int, n2: int, n3: int) -> None:
    pass


@st.composite
def sketch_strategy(draw):
    n3 = draw(st.integers(min_value=0))
    n2 = draw(st.integers(min_value=0, max_value=n3))
    n1 = draw(st.integers(min_value=max(0, n2), max_value=n3))
    return n1, n2, n3


@given(n=sketch_strategy())
def test_sketch(n: Tuple[int, int, int, int]):
    sketch(*n)


if __name__ == '__main__':
    # dg = nx.DiGraph()
    # dg.add_weighted_edges_from([(1, 2, 0.5)])
    # dg.add_edge(2, 3, label='a', weight=7)
    # pos = nx.get_node_attributes(dg, 'pos')
    # nx.draw(dg, pos)
    # labels = nx.get_edge_attributes(dg, 'weight')
    # nx.draw_networkx_edge_labels(dg, pos, edge_labels=labels)(dg, pos, edge_labels=labels)
    # plt.show()
    G = nx.Graph()
    i = 1
    G.add_node(i, pos=(i, i))
    G.add_node(2, pos=(2, 2))
    G.add_node(3, pos=(1, 0))
    G.add_edge(1, 2, weight=0.5, label='a')
    G.add_edge(1, 3, weight=9.8, label='b')
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos)
    labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()
    print(labels)
    # test_strategy_example_function_2()
