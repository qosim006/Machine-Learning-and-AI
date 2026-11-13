# ML Assignments

Two machine learning assignments solved in Jupyter notebooks (Python, scikit-learn). Every notebook contains the code, printed results, plots and written explanations.

| Notebook | Topic |
|---|---|
| [`ML_Assignment3_Think_First_Then_Code.ipynb`](ML_Assignment3_Think_First_Then_Code.ipynb) | Linear, logistic and softmax regression for software engineering (variants V1 and V2 / "Variant B") |
| [`SVM_Practice_Exercises.ipynb`](SVM_Practice_Exercises.ipynb) | Support Vector Machines on a WDBC breast cancer subset |

---

## 1. Assignment 3: Think First, Then Code

For each question the notebook follows three steps: **think first** (reasoning before running code), **code** (implementation, results rounded to 3 decimals), **compare** the results with the initial expectation. Both variants are solved.

**Part I: Linear regression**
- Q1: simple linear fit, interpretation of the coefficients, prediction inside the data range vs extrapolation, MSE and R²
- Q2: degree-1 vs degree-2 feature map `[1, x, x²]`, gradient descent for `ŷ = βx`, choice of step size and convergence

**Part II: Logistic regression**
- Q3: classification with two features, predicted probability, two thresholds and confusion matrices
- Q4: logistic loss and one gradient-descent update

**Part III: Softmax regression**
- Q5: numerically stable softmax and its properties (shift and scale of logits)
- Q6: multiclass classification, log loss, coefficient matrix

Selected results:

| | V1 | V2 (Variant B) |
|---|---|---|
| Q1 coefficients (β₀, β₁) | 9.643, 7.690 | 1.196, 0.987 |
| Q1 MSE / R² | 0.435 / 0.999 | 0.019 / 0.996 |
| Q2 MSE (degree 1 → 2) | 2.738 → 0.007 | 0.389 → 0.000 |
| Q3 probability for the given point | 0.518 | 0.513 |
| Q6 multiclass log loss | 0.552 | 0.533 |

Note: in Q3 and Q6 the classes are perfectly separable, so scikit-learn's default L2 regularization (`C = 1`) is used to keep the coefficients finite.

## 2. SVM Practice Exercises

Dataset: 16 training and 4 test observations from the Wisconsin Diagnostic Breast Cancer (WDBC) dataset (UCI Machine Learning Repository). Features: mean texture and mean smoothness. Target: malignant (`-1`) or benign (`+1`). Features are standardized with the training-set statistics. This is for learning only, not a clinical model.

- Q1: scatter plot, is a linear SVM suitable, why scaling matters
- Q2: linear SVM, margin lines, support vectors, margin width
- Q3: hinge loss inside the margin
- Q4: linear, polynomial (p = 2, 3) and RBF kernels
- Q5: effect of `C` and `γ` in the RBF SVM

Selected results:

| Model (`C = 1`) | Train acc. | Test acc. | Support vectors |
|---|---|---|---|
| Linear | 0.688 | 0.00 | 12 |
| Polynomial p = 2 | 0.750 | 0.50 | 12 |
| Polynomial p = 3 | 0.875 | 0.50 | 11 |
| RBF (γ = 0.5) | 0.875 | 0.25 | 16 |

Linear SVM margin width: 1.511 (standardized units). Chosen RBF setting: `C = 1, γ = 10`. The test set has only 4 points, so the test accuracy is weak evidence.

---

## How to run

```bash
pip install -r requirements.txt
jupyter notebook
```

Or upload a notebook to [Google Colab](https://colab.research.google.com) and run all cells.

## Tools

Python, NumPy, pandas, Matplotlib, scikit-learn.
