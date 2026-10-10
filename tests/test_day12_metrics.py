import numpy as np
import pytest

from nn.metrics import confusion_matrix, per_class_accuracy


def test_confusion_matrix_integer_labels_and_row_true_column_predicted():
    actual = np.array([0, 0, 1, 1, 2])
    predicted = np.array([0, 1, 1, 2, 2])
    np.testing.assert_array_equal(confusion_matrix(actual, predicted, num_classes=4),
                                  [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 0]])


def test_confusion_matrix_accepts_one_hot_targets_and_probability_predictions():
    y_true = np.eye(3)[[0, 1, 2, 1]]
    probs = np.array([[.8, .1, .1], [.1, .7, .2], [.1, .2, .7], [.1, .6, .3]])
    np.testing.assert_array_equal(confusion_matrix(y_true, probs), np.diag([1, 2, 1]))


def test_confusion_matrix_validation():
    with pytest.raises(ValueError):
        confusion_matrix(np.array([0, 1]), np.array([0]))
    with pytest.raises(ValueError):
        confusion_matrix(np.array([0, 3]), np.array([0, 1]), num_classes=3)
    with pytest.raises(ValueError):
        confusion_matrix(np.array([-1, 0]), np.array([0, 0]))


def test_per_class_accuracy_and_missing_class_nan():
    result = per_class_accuracy(np.array([0, 0, 1]), np.array([0, 1, 1]), num_classes=3)
    np.testing.assert_allclose(result[:2], [0.5, 1.0])
    assert np.isnan(result[2])
