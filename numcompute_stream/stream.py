"""
StreamTrainer: manages a model and optional pipeline for streaming/online learning. Supports chunk-wise fitting, scoring, metric logging, and memory tracking.
"""
from __future__ import annotations

import time
import numpy as np
from .metrics import StreamingMetrics


class StreamTrainer:
    """
    Manages incremental training of a model and optional preprocessing pipeline over a stream of data chunks.
    Parameters:
    model : object
        A classifier implementing partial_fit(X, y) and predict(X). It is compatible with DecisionTreeClassifier and EnsembleClassifier.
    pipeline : object or None, default=None
        A preprocessing pipeline implementing partial_fit(X, y) and transform(X). If None, raw features are passed directly to the model.
    window_size : int or None, default=None
        Rolling window for StreamingMetrics. None = cumulative only.

    Attributes:
    log_ : list of dict
        Per-chunk log entries. Each entry contains:
        - chunk       : int   — chunk index (1-based)
        - n_samples   : int   — number of samples in the chunk
        - accuracy    : float — chunk-level accuracy
        - fit_time_s  : float — seconds spent in partial_fit
        - memory_bytes: int   — estimated memory of accumulated arrays
    """

    def __init__(self, model, pipeline=None, window_size=None):
        self.model = model
        self.pipeline = pipeline
        self.window_size = window_size

        self.log_: list[dict] = []
        self._streaming_metrics = StreamingMetrics(window_size=window_size)
        self._chunk_idx = 0

    # Core streaming API
    def fit_chunk(self, X, y) -> "StreamTrainer":
        """
        Incrementally fit the model and pipeline on a new data chunk. Preprocessing is updated first via partial_fit, then the transformed features are passed to the model.
        Metrics and memory are logged after each call.

        Parameters:
        X : array-like, shape (n_samples, n_features), Feature chunk.
        y : array-like, shape (n_samples,), Label chunk.

        Returns StreamTrainer as self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self._chunk_idx += 1
        t0 = time.perf_counter()

        # Update and apply preprocessing pipeline
        if self.pipeline is not None:
            self.pipeline.partial_fit(X, y)
            X_transformed = self.pipeline.transform(X)
        else:
            X_transformed = X

        # Incrementally update the model
        self.model.partial_fit(X_transformed, y)

        fit_time = time.perf_counter() - t0

        # Score on the current chunk
        y_pred = self.model.predict(X_transformed)
        chunk_accuracy = float(np.mean(y_pred == y))

        # Update cumulative streaming metrics
        self._streaming_metrics.update(y, y_pred)

        # Log the chunk
        entry = {
            "chunk":        self._chunk_idx,
            "n_samples":    int(X.shape[0]),
            "accuracy":     chunk_accuracy,
            "fit_time_s":   round(fit_time, 6),
            "memory_bytes": self._estimate_memory(),
        }
        self.log_.append(entry)

        return self

    def score_chunk(self, X, y) -> dict:
        """
        Score the current model on a held-out chunk without updating it.
        Parameters:
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)

        Returns dict with keys: accuracy, n_samples
        Raises RuntimeError If no chunk has been fitted yet.
        """
        if self._chunk_idx == 0:
            raise RuntimeError(
                "No chunk has been fitted yet. Call fit_chunk() first.")

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if self.pipeline is not None:
            X = self.pipeline.transform(X)

        y_pred = self.model.predict(X)
        acc = float(np.mean(y_pred == y))

        return {"accuracy": acc, "n_samples": int(X.shape[0])}

    # Logging & reporting

    def cumulative_metrics(self) -> dict:
        """
        Returns dictionary storing output of StreamingMetrics.result()
        """
        return self._streaming_metrics.result()

    def accuracy_history(self) -> np.ndarray:
        """
        Returns np.ndarray, shape (n_chunks,)
        """
        return np.array([entry["accuracy"] for entry in self.log_])

    def print_log(self) -> None:
        """
        Print a formatted summary of all logged chunks to stdout.
        """
        header = (
            f"{'Chunk':>6}  {'Samples':>8}  "
            f"{'Accuracy':>10}  {'Fit (s)':>10}  {'Memory (B)':>12}"
        )
        print(header)
        print("-" * len(header))
        for entry in self.log_:
            print(
                f"{entry['chunk']:>6}  "
                f"{entry['n_samples']:>8}  "
                f"{entry['accuracy']:>10.4f}  "
                f"{entry['fit_time_s']:>10.6f}  "
                f"{entry['memory_bytes']:>12}"
            )

    def reset(self) -> "StreamTrainer":
        """
        Reset all logs and streaming metrics without touching the model.
        Returns StreamTrainer as self
        """
        self.log_ = []
        self._streaming_metrics = StreamingMetrics(
            window_size=self.window_size)
        self._chunk_idx = 0
        return self

   
    # Internal helpers

    def _estimate_memory(self) -> int:
        """
        Estimate memory footprint of accumulated arrays in the model.
        Returns 0 if the model does not expose _X_accum.
        """
        total = 0
        for attr in ("_X_accum", "_y_accum"):
            arr = getattr(self.model, attr, None)
            if isinstance(arr, np.ndarray):
                total += arr.nbytes
        return total