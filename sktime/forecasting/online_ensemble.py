import numpy as np
import pandas as pd
from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.compose._ensemble import EnsembleForecaster


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

    def __init__(self, ensemble_algorithm, forecasters, n_jobs=None):
        self.n_jobs = n_jobs
        self.ensemble_algorithm = ensemble_algorithm
        
        if self.ensemble_algorithm.n != len(forecasters):
            raise ValueError("Number of Experts in Ensemble Algorithm doesn't equal number of Forecasters")
        
        super(EnsembleForecaster, self).__init__(forecasters=forecasters,n_jobs=n_jobs)
    
    def _fit_ensemble(self, y_val, X_val=None):
        """Fits the ensemble by allowing forecasters to predict and compares to the actual parameters.
        
        Parameters
        ----------
        y_val : pd.Series
            Target time series to which to fit the forecaster.
        X_val : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        """
        fh = np.arange(len(y_val)) + 1
        expert_predictions = np.column_stack(self._predict_forecasters(fh=fh, X=X_val))
        actual_values = np.array(y_val)
        
        self.ensemble_algorithm._update(expert_predictions.T,actual_values)

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
        
        self._fit_ensemble(self,y_new,X_val=X_new)
        
        self.check_is_fitted()
        self._set_oh(y_new)
        for forecaster in self.forecasters_:
            forecaster.update(y_new, X_new=X_new, update_params=update_params)
            
        return self
        
    def _predict(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        if return_pred_int:
            raise NotImplementedError()
        
        return (pd.concat(self._predict_forecasters(fh=fh, X=X), axis=1)*self.ensemble_algorithm.weights).sum(axis=1)