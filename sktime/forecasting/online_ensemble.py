import numpy as np
import pandas as pd
from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.compose._ensemble import EnsembleForecaster
from sktime.forecasting.model_selection import SlidingWindowSplitter
from .ensemble_algorithms import EnsembleAlgorithms


class OnlineEnsembleForecaster(EnsembleForecaster):
    """Online Updating Ensemble of forecasters

    Parameters
    ----------
    ensemble_algorithm : ensemble algorithm
    forecasters : list of (str, estimator) tuples
    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for fit. None means 1 unless
        in a joblib.parallel_backend context.
        -1 means using all processors.
    """

    _required_parameters = ["forecasters"]

    def __init__(self, forecasters, ensemble_algorithm=None, n_jobs=None):
        self.n_jobs = n_jobs
        self.ensemble_algorithm = ensemble_algorithm

#         if self.ensemble_algorithm.n != len(forecasters):
#             raise ValueError("Number of Experts in Ensemble Algorithm \
#                              doesn't equal number of Forecasters")

        super(EnsembleForecaster, self).__init__(forecasters=forecasters,
                                                 n_jobs=n_jobs)

    def fit(self, y_train, fh=None, X_train=None):
        """Fit to training data.

        Parameters
        ----------
        y_train : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X_train : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        Returns
        -------
        self : returns an instance of self.
        """
        if self.ensemble_algorithm is None:
            self.ensemble_algorithm = EnsembleAlgorithms(len(self.forecasters))

        self._set_oh(y_train)
        self._set_fh(fh)
        names, forecasters = self._check_forecasters()
        self._fit_forecasters(forecasters, y_train, fh=fh, X_train=X_train)
        self._is_fitted = True
        return self

    def _fit_ensemble(self, y_new, X_new=None):
        """Fits the ensemble by allowing forecasters to predict and
           compares to the actual parameters.

        Parameters
        ----------
        y_new : pd.Series
            Target time series to which to fit the forecaster.
        X_new : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        """
        fh = np.arange(len(y_new)) + 1
        expert_predictions = np.column_stack(self._predict_forecasters(
                                             fh=fh, X=X_new))
        y_new = np.array(y_new)

        self.ensemble_algorithm._update(expert_predictions.T, y_new)

    def update(self, y_new, X_new=None, update_params=False):
        """Update fitted paramters and performs a new ensemble fit.

        Parameters
        ----------
        y_new : pd.Series
        X_new : pd.DataFrame
        update_params : bool, optional (default=False)

        Returns
        -------
        self : an instance of self
        """
        self._fit_ensemble(y_new, X_new=X_new)

        self.check_is_fitted()
        self._set_oh(y_new)
        for forecaster in self.forecasters_:
            forecaster.update(y_new, X_new=X_new, update_params=update_params)

        return self

    def update_predict(self, y_test, X_test=None, update_params=False,
                       return_pred_int=False,
                       alpha=DEFAULT_ALPHA):
        """Make and update predictions iteratively over the test set.

        Parameters
        ----------
        y_test : pd.Series
        cv : temporal cross-validation generator, optional (default=None)
        X_test : pd.DataFrame, optional (default=None)
        update_params : bool, optional (default=False)
        return_pred_int : bool, optional (default=False)
        alpha : int or list of ints, optional (default=None)

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        y_pred_int : pd.DataFrame
            Prediction intervals
        """

        return self._predict_moving_cutoff(y_test, X=X_test,
                                           update_params=update_params,
                                           return_pred_int=return_pred_int,
                                           alpha=alpha,
                                           cv=SlidingWindowSplitter(
                                                start_with_window=True,
                                                window_length=1,
                                                fh=1)
                                           )

    def _predict(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        if return_pred_int:
            raise NotImplementedError()

        return (pd.concat(
            self._predict_forecasters(fh=fh, X=X), axis=1)
                * self.ensemble_algorithm.weights).sum(axis=1)
