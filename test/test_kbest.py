from numpy.typing import NDArray
import numpy as np
from spanningtrees.kbest import KBest
from spanningtrees.kbest_camerini import KBest as KBestCamerini
from spanningtrees.graph import Graph
from spanningtrees.brute_force import kbest, kbest_rc
from spanningtrees.util import random_instance
from tqdm import tqdm
from hypothesis import given, settings, strategies as st


from conftest import random_weights


@settings(deadline=1000)
@given(
    weights_matrix=random_weights(
        size=st.integers(min_value=3, max_value=6),
        weights_strategy=st.floats(
            0.0, 1.0, allow_nan=False, allow_infinity=False, exclude_min=True
        ),
    )
)
def test_kbest(weights_matrix: NDArray[np.number]):
    """
    Test that MST decoding's tree and score matches the
    brute-force decoding's tree and score on randomly generated dense graphs.
    """
    graph = Graph.build(weights_matrix)
    mst = KBest(graph)
    trees_bf = kbest(weights_matrix)
    trees = list(mst.kbest())
    assert len(trees) == len(trees_bf)
    for tree, tree_bf in zip(trees, trees_bf):
        cost = graph.weight(tree.to_array())
        cost_bf = graph.weight(tree_bf[0])
        assert np.allclose(cost, cost_bf)


@settings(deadline=1000)
@given(
    weights_matrix=random_weights(
        size=st.integers(min_value=3, max_value=6),
        weights_strategy=st.floats(
            0.0, 1.0, allow_nan=False, allow_infinity=False, exclude_min=True
        ),
    )
)
def test_kbest_rc(weights_matrix: NDArray[np.number]):
    """
    Test that MST decoding's tree and score matches the
    brute-force decoding's tree and score on randomly generated dense graphs.
    """
    graph = Graph.build(weights_matrix)
    mst = KBest(graph, True)
    trees_bf = kbest_rc(weights_matrix)
    trees = list(mst.kbest())
    assert len(trees) == len(trees_bf)
    for tree, tree_bf in zip(trees, trees_bf):
        cost = graph.weight(tree.to_array())
        cost_bf = graph.weight(tree_bf[0])
        assert np.allclose(cost, cost_bf)


@settings(deadline=1000)
@given(
    weights_matrix=random_weights(
        size=st.integers(min_value=3, max_value=6),
        weights_strategy=st.floats(
            0.0, 1.0, allow_nan=False, allow_infinity=False, exclude_min=True
        ),
    )
)
def test_kbest_camerini(weights_matrix: NDArray[np.number]):
    """
    Test that MST decoding's tree and score matches the
    brute-force decoding's tree and score on randomly generated dense graphs.
    """
    graph = Graph.build(weights_matrix)
    mst = KBestCamerini(graph)
    trees_bf = kbest(weights_matrix)
    trees = list(mst.kbest())
    assert len(trees) == len(trees_bf)
    for tree, tree_bf in zip(trees, trees_bf):
        cost = graph.weight(tree.to_array())
        cost_bf = graph.weight(tree_bf[0])
        assert np.allclose(cost, cost_bf)
