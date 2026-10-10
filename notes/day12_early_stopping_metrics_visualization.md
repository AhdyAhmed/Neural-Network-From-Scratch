# Day 12 — Early stopping, metrics, and visualizations

## Early stopping

`Sequential.fit(..., validation_data=(x_val, y_val), early_stopping=True, patience=5)`
monitors validation loss after every epoch. An epoch is an improvement only when
validation loss is lower than the best loss by more than `min_delta`. After `patience`
consecutive epochs without an improvement, training stops. By default, model parameters
are restored to the best validation-loss epoch. Early stopping requires validation data;
its default is disabled to preserve earlier project behavior.

```python
history = model.fit(
    x_train, y_train,
    epochs=100,
    validation_data=(x_val, y_val),
    early_stopping=True,
    patience=8,
    min_delta=1e-4,
    restore_best_weights=True,
    metrics={"accuracy": accuracy},
    verbose=1,
    log_every=1,
)
```

## Evaluation metrics

- `confusion_matrix(y_true, y_pred, num_classes=K)`: rows are true classes, columns are predicted classes.
- `per_class_accuracy(y_true, y_pred, num_classes=K)`: per-class recall; missing classes return `nan`.
- Both functions accept integer labels; confusion matrix also accepts one-hot labels and class-score/probability matrices.

## Visualizations

Run `python examples/day12_analysis.py` for a reproducible three-class synthetic demo. It saves a decision-boundary plot, training curves, a confusion matrix, and JSON metrics. Run `python examples/day12_analysis.py --mnist` to additionally train a small MNIST model and save misclassified digits and a normalized confusion matrix. The MNIST option downloads the dataset on first use if it is not already cached.

All epoch progress output includes loss/metrics and elapsed time. `log_every=1` prints every epoch.
