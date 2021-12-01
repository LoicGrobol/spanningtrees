from numpy.typing import NDArray
import numpy as np
from spanningtrees.mst import MST
from spanningtrees.graph import Graph
from spanningtrees.brute_force import all_best_trees, all_best_rc_trees
from hypothesis import given, strategies as st

from conftest import random_weights

# We use `exclude_min` here because for now, `best_tree` use `0.0` to mark a forbidden edge
@given(
    weights_matrix=random_weights(
        size=st.integers(min_value=3, max_value=6),
        weights_strategy=st.floats(
            0.0, 1.0, allow_nan=False, allow_infinity=False, exclude_min=True
        ),
    )
)
def test_1b(weights_matrix: NDArray[np.number]):
    """
    Test that MST decoding's tree and score matches the
    brute-force decoding's tree and score on randomly generated dense graphs.
    """
    graph = Graph.build(weights_matrix)
    mst = MST(graph)
    tree = mst.mst().to_array()
    cost = graph.weight(tree)
    best_trees_bf, cost_bf = all_best_trees(weights_matrix)
    assert any(np.allclose(tree, tree_bf) for tree_bf in best_trees_bf)
    assert np.allclose(cost, cost_bf)


# We use `exclude_min` here because for now, `best_tree` use `0.0` to mark a forbidden edge
@given(
    weights_matrix=random_weights(
        size=st.integers(min_value=3, max_value=6),
        weights_strategy=st.floats(
            0.0, 1.0, allow_nan=False, allow_infinity=False, exclude_min=True
        ),
    )
)
def test_1b_scc(weights_matrix: NDArray[np.number]):
    """
    Test that MST (using SCC) decoding's tree and score matches the
    brute-force decoding's tree and score on randomly generated dense graphs.
    """
    graph = Graph.build(weights_matrix)
    mst = MST(graph)
    tree = mst.mst_scc().to_array()
    cost = graph.weight(tree)
    best_trees_bf, cost_bf = all_best_trees(weights_matrix)
    assert any(np.allclose(tree, tree_bf) for tree_bf in best_trees_bf)
    assert np.allclose(cost, cost_bf)


# We use `exclude_min` here because for now, `best_tree` use `0.0` to mark a forbidden edge
@given(
    weights_matrix=random_weights(
        size=st.integers(min_value=3, max_value=6),
        weights_strategy=st.floats(
            0.0, 1.0, allow_nan=False, allow_infinity=False, exclude_min=True
        ),
    )
)
def test_c1b(weights_matrix: NDArray[np.number]):
    """
    Test that root-constrained MST decoding's tree and score matches the
    brute-force decoding's tree and score on randomly generated dense graphs.
    """
    graph = Graph.build(weights_matrix)
    mst = MST(graph, True)
    tree = mst.mst().to_array()
    cost = graph.weight(tree)
    best_trees_bf, cost_bf = all_best_rc_trees(weights_matrix)
    any(np.allclose(tree, tree_bf) for tree_bf in best_trees_bf)
    assert np.allclose(cost, cost_bf)


# We use `exclude_min` here because for now, `best_tree` use `0.0` to mark a forbidden edge
@given(
    weights_matrix=random_weights(
        size=st.integers(min_value=3, max_value=6),
        weights_strategy=st.floats(
            0.0, 1.0, allow_nan=False, allow_infinity=False, exclude_min=True
        ),
    )
)
def test_c1b_scc(weights_matrix):
    """
    Test that root-constrained MST (using SCC) decoding's tree and score matches the
    brute-force decoding's tree and score on randomly generated dense graphs.
    """
    graph = Graph.build(weights_matrix)
    mst = MST(graph, True)
    tree = mst.mst_scc().to_array()
    cost = graph.weight(tree)
    best_trees_bf, cost_bf = all_best_rc_trees(weights_matrix)
    any(np.allclose(tree, tree_bf) for tree_bf in best_trees_bf)
    assert np.allclose(cost, cost_bf)

