from typing import Optional
import numpy as np
from numpy.typing import NDArray
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays


@st.composite
def random_weights(
    draw: st.DrawFn,
    size: st.SearchStrategy[int],
    weights_strategy: Optional[st.SearchStrategy[float]] = None,
) -> NDArray[np.double]:
    n = draw(size)
    weights = draw(arrays(shape=(n, n), elements=weights_strategy, dtype=np.double))
    return weights