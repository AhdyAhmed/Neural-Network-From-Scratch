# Day 5 — Gradient Checking & Unit Tests

**Goal:** prove the backprop implementation is correct, not just plausible.

## The idea

For any parameter `θᵢ`, the gradient can be approximated without calculus by nudging the parameter and watching the loss move (centered finite difference):

```
dL/dθᵢ ≈ [ L(θᵢ + ε) − L(θᵢ − ε) ] / (2ε)         ε = 1e-5
```

Backprop computes all gradients analytically in one pass. The numerical version is slow (two forward passes per parameter) but nearly impossible to get wrong, so it is the referee.

```
relative_error = ‖g_analytic − g_numeric‖ / (‖g_analytic‖ + ‖g_numeric‖)
```

| Relative error (float64) | Verdict |
|---|---|
| `< 1e-6` | Pass |
| `1e-6 … 1e-4` | Suspicious: look for a kink, tiny gradients, or float32 |
| `> 1e-4` | Bug |

## `nn/gradcheck.py`

| Function | Checks |
|---|---|
| `numerical_gradient(f, x)` | Core finite-difference routine (perturbs `x` in place, always restores it) |
| `relative_error(a, b)` | The metric above |
| `check_layer(layer, x)` | `dL/dx` and every parameter gradient of one layer |
| `check_loss(loss, y_pred, y_true)` | `dL/dy_pred` |
| `check_model(model, x, y)` | Every parameter gradient (and `dL/dx`) of a whole compiled model |

Each returns a `GradCheckResult` with `.errors`, `.max_error`, `.passed`, and `.assert_passed()`.

`check_layer` needs a scalar objective, so it uses `L = Σ(layer(x) ⊙ R)` with a fixed random matrix `R`. The upstream gradient is then exactly `R`, and the layer is tested in isolation.

```python
from nn.gradcheck import check_model
result = check_model(model, x, y)
print(result)             # per-parameter errors
result.assert_passed()    # raises AssertionError if any error >= 1e-6
```

## Results

Run `python examples/day5_gradient_check.py`:

```
[PASS] Dense(4 -> 3)                      max rel. error = 1.72e-11
[PASS] ReLU()                             max rel. error = 1.52e-11
[PASS] Sigmoid()                          max rel. error = 4.16e-11
[PASS] Tanh()                             max rel. error = 9.06e-11
[PASS] MSE                                max rel. error = 9.35e-12
[PASS] BinaryCrossEntropy                 max rel. error = 1.08e-09
[PASS] Dense-ReLUx2 + MSE                 max rel. error = 1.46e-10
[PASS] Dense-Tanhx2 + BinaryCrossEntropy  max rel. error = 8.50e-11
... (12/12 checks passed)
```

The errors sit around `1e-10`, which is the expected floor for centered differences with `ε = 1e-5` in float64 (truncation error `O(ε²)`).

## What the test suite covers (190 tests)

- **Checker internals:** known derivatives, input restoration, error metric, pass/fail reporting.
- **The checker catches real bugs:** three deliberately broken `Dense` subclasses (wrong `dW` scale, bias gradient averaged instead of summed, `W` used instead of `Wᵀ`) must all **fail** the check. A test that can't fail proves nothing.
- **Every layer, activation, and loss** across many shapes, including batch size 1, single feature, and saturated sigmoid/tanh regions.
- **Whole models:** 3 hidden activations × 2 losses × 3 depths, multi-output regression, and 10 random seeds.
- **Overfit a tiny batch:** tanh and ReLU networks drive MSE below `1e-6` / `1e-4` on 10 random targets, and a sigmoid network reaches 100% accuracy on random labels. Fitting noise is only possible if gradients and the optimizer are right.
- **Edge cases:** integer input, NaN propagation, inputs never modified, `sigmoid(x) = (1 + tanh(x/2))/2`, loss non-negativity and symmetry.

## Bug found and fixed: cache aliasing

The gradient checks passed on the first run, so Days 3 and 4 had no *math* bugs. Probing edge cases turned up a different kind of bug:

`Sigmoid` and `Tanh` cached the same array they returned. If the caller modified the result in place (`out += 5`), the cached values changed and `backward()` silently produced wrong gradients, with no error and no `NaN`.

**Fix:** cache a private copy (`self._out = out.copy()`). `tests/test_cache_safety.py` mutates each layer's output in place and checks the gradients are unchanged. I confirmed those tests fail on the old code.

**Convention (documented in `Layer`):** layers don't copy their *inputs*, since they belong to the caller. Don't modify an array in place between `forward` and `backward`.

## Known limitations of gradient checking

| Pitfall | Why | What to do |
|---|---|---|
| Tiny weights (std 0.01) | Gradients ~`1e-5`; rounding noise dominates the relative error | Check with weights of scale ~0.5 |
| ReLU at 0 | Not differentiable; finite differences straddle the kink | Keep inputs away from 0 |
| BCE in the clipped region | Clipping makes the loss flat there, so numerical gradient ≈ 0 while analytic is huge | Check with probabilities in `(0.05, 0.95)`; a test documents this |
| Dropout (Day 11) | Random mask changes between evaluations | Fix the mask during the check |
| float32 | Too little precision for `ε = 1e-5` | Always check in float64 |

## Self-check questions

- Why use *centered* differences instead of `(L(θ+ε) − L(θ)) / ε`? *(Error shrinks as `ε²` instead of `ε`.)*
- Why can't you make `ε` extremely small (say `1e-12`)? *(Subtracting nearly equal losses loses precision to rounding.)*
- Why does the checker use a random `R` instead of an all-ones upstream gradient? *(All-ones can hide bugs: e.g. swapped rows or transposed terms can cancel out.)*
