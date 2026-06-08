"""
Minimal Transformer / Estimator API and Pipeline chaining.


This module follows the NumCompute core requirement for:
- Transformer: fit, transform, fit_transform
- Estimator: fit, predict
- Pipeline chaining
"""

import inspect
import numpy as np


def _accepts_y(method):
    """
    Check whether a method accepts a y parameter.
    """
    return "y" in inspect.signature(method).parameters


class Transformer:
    """
    Base class for preprocessing transformers.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        raise NotImplementedError("Transformer subclasses must implement transform().")

    def fit_transform(self, X, y=None):
        if y is None:
            return self.fit(X).transform(X)
        return self.fit(X, y).transform(X)


class Estimator:
    """
    Base class for model estimators.
    """

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("y cannot be None for supervised estimators.")
        return self

    def predict(self, X):
        raise NotImplementedError("Estimator subclasses must implement predict().")


class Pipeline:
    """
    Chain multiple transformers and an optional final estimator.
    """

    def __init__(self, steps):
        if not steps:
            raise ValueError("Pipeline requires at least one step.")

        if not isinstance(steps, list):
            raise TypeError("steps must be a list of (name, step) tuples.")

        for step in steps:
            if not isinstance(step, tuple) or len(step) != 2:
                raise TypeError("Each pipeline step must be a (name, step) tuple.")

            name, obj = step

            if not isinstance(name, str) or not name:
                raise ValueError("Each step name must be a non-empty string.")

            if not hasattr(obj, "fit"):
                raise TypeError(f"Step '{name}' must implement fit().")

        self.steps = steps

    def fit(self, X, y=None):
        """
        Fit all pipeline steps sequentially.

        If y is None, steps are fitted using fit(X).
        If y is provided, steps that accept y use fit(X, y).
        """

        for i, (name, step) in enumerate(self.steps):
            is_final_step = i == len(self.steps) - 1

            if y is not None and _accepts_y(step.fit):
                step.fit(X, y)
            else:
                step.fit(X)

            if hasattr(step, "transform") and not is_final_step:
                X = step.transform(X)

        return self

    def transform(self, X):
        """
        Apply all transformer steps sequentially.
        """

        for name, step in self.steps:
            if not hasattr(step, "transform"):
                raise TypeError(f"Step '{name}' does not support transform().")

            X = step.transform(X)

        return X

    def fit_transform(self, X, y=None):
        """
        Fit and transform data through all transformer steps.
        """

        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        """
        Transform X through preprocessing steps and predict with final estimator.
        """

        for name, step in self.steps[:-1]:
            if not hasattr(step, "transform"):
                raise TypeError(f"Step '{name}' must implement transform().")

            X = step.transform(X)

        final_name, final_step = self.steps[-1]

        if not hasattr(final_step, "predict"):
            raise TypeError(f"Final step '{final_name}' does not support predict().")

        return final_step.predict(X)

    def named_steps(self):
        """
        Return pipeline steps as a dictionary.
        """

        return dict(self.steps)