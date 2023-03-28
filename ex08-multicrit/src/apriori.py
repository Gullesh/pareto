from typing import Optional
import numpy as np


def apriori(costs: np.ndarray, weights: Optional = None, order: Optional = None):
    """
    Implement the two example apriori methods presented in the lecture
    Parameters
    ----------
    costs   (n_points, m_costs) array
    weights (m_costs, ) array. Determines the weighting of the costs. If None use lexical ordering
    order   (m_costs, ) array. Determines the lexicograpical order. If None use weighted sum

    Returns
    -------
    Index of optimal element according to apriori method
    """
    if weights is not None and order is not None:
        raise Exception('You can only specify weight or order but not both')
    if weights:
        tot = np.dot(costs, weights)
        arg = np.argmin(tot)
    if order:
        arg = np.argmin(costs[:, order[0]])
    return arg
