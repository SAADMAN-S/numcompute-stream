"""
test_visualise.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from numcompute_stream import visualise


@pytest.fixture(autouse=True)
def close_plots():
    """Close all matplotlib figures after each test."""
    yield
    plt.close('all')


class TestPlotMetricOverTime:

    def test_returns_figure(self):
        """Returns a matplotlib Figure object."""
        fig = visualise.plot_metric_over_time(
            [0.5, 0.6, 0.7], show=False)
        assert isinstance(fig, plt.Figure)

    def test_correct_number_of_points(self):
        """Line contains the correct number of data points."""
        values = [0.5, 0.6, 0.7, 0.8]
        fig = visualise.plot_metric_over_time(values, show=False)
        ax = fig.axes[0]
        line = ax.lines[0]
        assert len(line.get_ydata()) == len(values)

    def test_custom_title(self):
        """Title is set correctly."""
        fig = visualise.plot_metric_over_time(
            [0.5, 0.6], title='My Metric', show=False)
        assert 'My Metric' in fig.axes[0].get_title()

    def test_save_to_file(self, tmp_path):
        """save_path writes a PNG file to disk."""
        out = str(tmp_path / 'metric.png')
        visualise.plot_metric_over_time([0.5, 0.6], show=False,
                                        save_path=out)
        assert os.path.exists(out)


class TestCompareModels:

    def test_returns_figure(self):
        """Returns a matplotlib Figure object."""
        fig = visualise.compare_models(
            [0.5, 0.6], [0.55, 0.65], show=False)
        assert isinstance(fig, plt.Figure)

    def test_two_lines_plotted(self):
        """Two lines appear on the axes."""
        fig = visualise.compare_models(
            [0.5, 0.6], [0.55, 0.65], show=False)
        assert len(fig.axes[0].lines) == 2

    def test_mismatched_lengths_raise(self):
        """Unequal metric arrays raise ValueError."""
        with pytest.raises(ValueError):
            visualise.compare_models([0.5, 0.6], [0.5], show=False)

    def test_save_to_file(self, tmp_path):
        """save_path writes a PNG file to disk."""
        out = str(tmp_path / 'compare.png')
        visualise.compare_models(
            [0.5, 0.6], [0.55, 0.65], show=False, save_path=out)
        assert os.path.exists(out)


class TestPlotPredictionsVsGroundTruth:

    def test_returns_figure(self):
        """Returns a matplotlib Figure object."""
        fig = visualise.plot_predictions_vs_ground_truth(
            [0, 1, 1, 0], [0, 1, 0, 0], show=False)
        assert isinstance(fig, plt.Figure)

    def test_mismatched_lengths_raise(self):
        """Unequal y_true/y_pred raise ValueError."""
        with pytest.raises(ValueError):
            visualise.plot_predictions_vs_ground_truth(
                [0, 1], [0], show=False)

    def test_accuracy_shown_in_title(self):
        """Accuracy percentage appears in the plot title."""
        fig = visualise.plot_predictions_vs_ground_truth(
            [0, 1, 0, 1], [0, 1, 0, 1], show=False)
        assert '100' in fig.axes[0].get_title()

    def test_save_to_file(self, tmp_path):
        """save_path writes a PNG file to disk."""
        out = str(tmp_path / 'preds.png')
        visualise.plot_predictions_vs_ground_truth(
            [0, 1], [0, 0], show=False, save_path=out)
        assert os.path.exists(out)


class TestPlotConfusionMatrix:

    def test_returns_figure(self):
        """Returns a matplotlib Figure object."""
        cm = np.array([[10, 2], [3, 15]])
        fig = visualise.plot_confusion_matrix(cm, show=False)
        assert isinstance(fig, plt.Figure)

    def test_custom_labels(self):
        """Custom class labels appear on axes."""
        cm = np.array([[5, 1], [2, 8]])
        fig = visualise.plot_confusion_matrix(
            cm, labels=['cat', 'dog'], show=False)
        ax = fig.axes[0]
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert 'cat' in tick_labels
        assert 'dog' in tick_labels

    def test_save_to_file(self, tmp_path):
        """save_path writes a PNG file to disk."""
        out = str(tmp_path / 'cm.png')
        cm = np.array([[10, 2], [3, 15]])
        visualise.plot_confusion_matrix(cm, show=False, save_path=out)
        assert os.path.exists(out)