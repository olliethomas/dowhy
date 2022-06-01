"""Functions in this module should be considered experimental, meaning there might be breaking API changes in the
future.
"""

from typing import Union, List, Optional

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

from dowhy.gcm.constant import EPS
from dowhy.gcm.util.general import shape_into_2d


def quantile_based_fwer(p_values: Union[np.ndarray, List[float]],
                        p_values_scaling: Optional[np.ndarray] = None,
                        quantile: float = 0.5) -> float:
    """Applies a quantile based family wise error rate (FWER) control to the given p-values. This is based on the
    approach described in:

    Meinshausen, N., Meier, L. and Buehlmann, P. (2009).
    p-values for high-dimensional regression. J. Amer. Statist. Assoc.104 1671–1681

    :param p_values: A list or array of p-values.
    :param p_values_scaling: An optional list of scaling factors for each p-value.
    :param quantile: The quantile used for the p-value adjustment. By default, this is the median (0.5).
    :return: The p-value that lies on the quantile threshold. Note that this is the quantile based on scaled values
             p_values / quantile.
    """

    if quantile <= 0 or abs(quantile - 1) >= 1:
        raise ValueError("The given quantile is %f, but it needs to be on (0, 1]!" % quantile)

    p_values = np.array(p_values)
    if p_values_scaling is None:
        p_values_scaling = np.ones(p_values.shape[0])

    if p_values.shape != p_values_scaling.shape:
        raise ValueError("The p-value scaling array needs to have the same dimension as the given p-values.")

    p_values_scaling = p_values_scaling[~np.isnan(p_values)]
    p_values = p_values[~np.isnan(p_values)]

    p_values = p_values * p_values_scaling
    p_values[p_values > 1] = 1.0

    if p_values.shape[0] == 1:
        return float(p_values[0])
    else:
        return float(min(1.0, np.quantile(p_values / quantile, quantile)))


def estimate_f_test_p_value(X_training_a: np.ndarray,
                            X_training_b: np.ndarray,
                            Y_training: np.ndarray,
                            X_test_a: np.ndarray,
                            X_test_b: np.ndarray,
                            Y_test: np.ndarray) -> float:
    """ Estimates the p-value for the nullhypothesis that the same regression curve with less parameters can be
    achieved. This is, a linear model trained on a data set A with d number of features has the same performance
    (in terms of squared error) relative to the number of features as a model trained on a data set B with k number
    features, where k < d. Here, both data sets need to have the same target values. A small p-value would
    indicate that the model performances are significantly different.

    Note that all given test samples are utilized in the f-fest.

    See https://en.wikipedia.org/wiki/F-test#Regression_problems for more details.

    :param X_training_a: Input training samples for model A.
    :param X_training_b: Input training samples for model B.
    :param Y_training: Target training values.
    :param X_test_a: Test samples for model A.
    :param X_test_b: Test samples for model B.
    :param Y_test: Test values.
    :return: A p-value on [0, 1].
    """
    X_training_a, X_test_a = shape_into_2d(X_training_a, X_test_a)

    if X_training_b.size > 0:
        X_training_b, X_test_b = shape_into_2d(X_training_b, X_test_b)
    else:
        X_training_b = X_training_b.reshape(0, 0)
        X_test_b = X_test_b.reshape(0, 0)

    if X_training_a.shape[1] <= X_training_b.shape[1]:
        raise ValueError("Data for A should have more dimensions (i.e. model parameters) than the data for B!")

    ssr_a = np.sum(
        (Y_test - LinearRegression().fit(X_training_a, Y_training).predict(X_test_a)) ** 2)

    if X_training_b.shape[1] > 0:
        ssr_b = np.sum(
            (Y_test - LinearRegression().fit(X_training_b, Y_training).predict(X_test_b)) ** 2)
    else:
        ssr_b = np.sum((Y_test - np.mean(Y_test)) ** 2)

    dof_diff_1 = (X_test_a.shape[1] - X_test_b.shape[1])  # p1 - p2
    dof_diff_2 = (X_test_a.shape[0] - X_test_a.shape[1] - 1)  # n - p2 (parameters include intercept)

    f_statistic = (ssr_b - ssr_a) / dof_diff_1 * dof_diff_2

    if ssr_a < EPS:
        ssr_a = 0
    if ssr_b < EPS:
        ssr_b = 0

    if ssr_a == 0 and ssr_b == 0:
        f_statistic = 0
    elif ssr_a != 0:
        f_statistic /= ssr_a

    return stats.f.sf(f_statistic, dof_diff_1, dof_diff_2)
