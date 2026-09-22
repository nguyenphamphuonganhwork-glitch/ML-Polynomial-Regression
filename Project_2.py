import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 0. CONFIGURATION
# ============================================================

TRAIN_FILE = "training_set.csv"
TEST_FILE = "test_set.csv"

N_FOLDS = 4
MAX_M = 10


# ============================================================
# 1. LOAD TRAINING DATA
# ============================================================

train_data = np.genfromtxt(
    TRAIN_FILE,
    delimiter=",",
    names=True
)
print(train_data.dtype.names)

x_train = train_data["Selfstudy_Hours_per_Week"]
t_train = train_data["Exam_Score"]

print("=" * 60)
print("TRAINING DATA")
print("=" * 60)

print("Training samples:", len(x_train))

assert len(x_train) == 199, \
    f"Expected 199 training samples, got {len(x_train)}"


# ============================================================
# 2. VISUALIZE TRAINING DATA
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    x_train,
    t_train,
    s=15
)

plt.xlabel("Selfstudy_Hours_per_Week")
plt.ylabel("Exam_Score")
plt.title("Training Data (N = 199)")

plt.grid(True)
plt.show()


# ============================================================
# 3. CREATE DESIGN MATRIX
# ============================================================

def create_design_matrix(x, M):
    """
    Create polynomial design matrix.

    For M = 3:

        X =
        [1, x, x^2, x^3]

    Each row corresponds to one data point.
    """

    X = np.zeros((len(x), M + 1))

    for j in range(M + 1):
        X[:, j] = x ** j

    return X


# ============================================================
# 4. TRAIN POLYNOMIAL MODEL USING NORMAL EQUATION
# ============================================================

def train_model(x, t, M):
    """
    Train polynomial regression model using
    the Normal Equation:

        w = (X^T X)^(-1) X^T t

    Instead of explicitly calculating the inverse,
    np.linalg.solve() is used.
    """

    X = create_design_matrix(x, M)

    # Normal Equation:
    #
    # X^T X w = X^T t
    #
    # solve() solves this linear system directly.

    w = np.linalg.solve(
        X.T @ X,
        X.T @ t
    )

    return w


# ============================================================
# 5. PREDICTION
# ============================================================

def predict(x, w):
    """
    Predict target values using polynomial model.
    """

    M = len(w) - 1

    X = create_design_matrix(x, M)

    return X @ w


# ============================================================
# 6. SUM-OF-SQUARES ERROR
# ============================================================

def calculate_sse(t, y):
    """
    Sum-of-Squares Error:

        SSE = 1/2 * sum((y - t)^2)

    The factor 1/2 is conventional in PRML.
    """

    return 0.5 * np.sum((y - t) ** 2)


# ============================================================
# 7. 4-FOLD CROSS-VALIDATION
# ============================================================

print("\n")
print("=" * 60)
print("4-FOLD CROSS-VALIDATION")
print("=" * 60)

# ------------------------------------------------------------
# Split the 199 samples into 4 parts.
#
# np.array_split() gives:
#
# Fold 1 = 50 samples
# Fold 2 = 50 samples
# Fold 3 = 50 samples
# Fold 4 = 49 samples
# ------------------------------------------------------------

indices = np.arange(len(x_train))

folds = np.array_split(indices, N_FOLDS)

for i, fold in enumerate(folds):
    print(
        f"Fold {i + 1}: "
        f"{len(fold)} samples"
    )


# ------------------------------------------------------------
# Store CV error for each M
# ------------------------------------------------------------

cv_errors = []


# ============================================================
# 8. TRY M = 1, 2, ..., 10
# ============================================================

for M in range(1, MAX_M + 1):

    fold_errors = []

    print("\n" + "-" * 60)
    print(f"M = {M}")
    print("-" * 60)

    # --------------------------------------------------------
    # Run 4 folds
    # --------------------------------------------------------

    for run in range(N_FOLDS):

        # Current fold = validation set
        validation_indices = folds[run]

        # Remaining folds = training set
        training_indices = np.concatenate(
            [
                folds[i]
                for i in range(N_FOLDS)
                if i != run
            ]
        )

        # ----------------------------------------------------
        # Get actual data
        # ----------------------------------------------------

        x_cv_train = x_train[training_indices]
        t_cv_train = t_train[training_indices]

        x_validation = x_train[validation_indices]
        t_validation = t_train[validation_indices]

        # ----------------------------------------------------
        # Train model
        # ----------------------------------------------------

        w = train_model(
            x_cv_train,
            t_cv_train,
            M
        )

        # ----------------------------------------------------
        # Predict validation data
        # ----------------------------------------------------

        y_validation = predict(
            x_validation,
            w
        )

        # ----------------------------------------------------
        # Calculate validation SSE
        # ----------------------------------------------------

        validation_sse = calculate_sse(
            t_validation,
            y_validation
        )

        fold_errors.append(validation_sse)

        print(
            f"Run {run + 1}: "
            f"training = {len(training_indices)}, "
            f"validation = {len(validation_indices)}, "
            f"validation SSE = {validation_sse:.4f}"
        )

    # --------------------------------------------------------
    # Average validation error
    # --------------------------------------------------------

    mean_cv_error = np.mean(fold_errors)

    cv_errors.append(mean_cv_error)

    print(
        f"CV error for M={M}: "
        f"{mean_cv_error:.4f}"
    )


# ============================================================
# 9. FIND BEST M
# ============================================================

best_index = np.argmin(cv_errors)

best_M = best_index + 1
best_cv_error = cv_errors[best_index]

print("\n")
print("=" * 60)
print("MODEL SELECTION RESULT")
print("=" * 60)

for M, error in zip(
    range(1, MAX_M + 1),
    cv_errors
):
    print(
        f"M = {M:2d}   "
        f"CV SSE = {error:.4f}"
    )

print("\nBest M =", best_M)
print("Best CV SSE =", best_cv_error)


# ============================================================
# 10. VISUALIZE CV PERFORMANCE
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, MAX_M + 1),
    cv_errors,
    marker="o"
)

plt.xlabel("Polynomial degree M")
plt.ylabel("Average CV SSE")
plt.title("4-Fold Cross-Validation")

plt.xticks(range(1, MAX_M + 1))
plt.grid(True)

plt.show()


# ============================================================
# 11. FINAL TRAINING
# ============================================================

print("\n")
print("=" * 60)
print("FINAL MODEL TRAINING")
print("=" * 60)

print(
    f"Training final model with M = {best_M}"
)

print(
    "Using all",
    len(x_train),
    "training samples"
)


# ------------------------------------------------------------
# Train final model using ALL 199 training samples
# ------------------------------------------------------------

w_final = train_model(
    x_train,
    t_train,
    best_M
)


# ============================================================
# 12. DISPLAY FINAL PARAMETERS
# ============================================================

print("\nFinal model parameters:")

for i, value in enumerate(w_final):

    print(
        f"w{i} = {value:.8f}"
    )


# ============================================================
# 13. DISPLAY FINAL MODEL EQUATION
# ============================================================

print("\nFinal model:")

equation = "y(x) = "

for i, w_value in enumerate(w_final):

    if i == 0:
        equation += f"{w_value:.6f}"

    elif i == 1:
        equation += f" + ({w_value:.6f})x"

    else:
        equation += f" + ({w_value:.6f})x^{i}"

print(equation)


# ============================================================
# 14. TRAINING SSE OF FINAL MODEL
# ============================================================

y_train_final = predict(
    x_train,
    w_final
)

train_sse_final = calculate_sse(
    t_train,
    y_train_final
)

print("\nFinal training SSE:")
print(train_sse_final)


# ============================================================
# 15. GAUSSIAN NOISE ASSUMPTION
# ============================================================

print("\n")
print("=" * 60)
print("GAUSSIAN NOISE / MAXIMUM LIKELIHOOD")
print("=" * 60)

"""
We assume:

    t_n = y(x_n, w) + epsilon_n

where:

    epsilon_n ~ Gaussian(0, sigma^2)

Therefore:

    p(t_n | x_n, w, sigma^2)

is Gaussian.

For all training samples:

    p(t | X, w, sigma^2)
        = product of Gaussian probabilities

The log-likelihood is maximized when
the Sum-of-Squares Error is minimized.

Therefore:

    Maximum Likelihood
            ⇔
    Minimize SSE

which gives the Normal Equation used above.
"""


# ============================================================
# 16. MLE ESTIMATE OF NOISE VARIANCE
# ============================================================

residuals_train = (
    t_train - y_train_final
)

squared_error_sum = np.sum(
    residuals_train ** 2
)

N = len(t_train)

# ------------------------------------------------------------
# Gaussian MLE:
#
# sigma^2_MLE = (1/N) * sum(residual^2)
#
# Note:
# This is MLE, so denominator is N,
# not N - 1.
# ------------------------------------------------------------

sigma_squared_MLE = (
    squared_error_sum / N
)

sigma_MLE = np.sqrt(
    sigma_squared_MLE
)

print(
    "MLE estimate of noise variance:",
    sigma_squared_MLE
)

print(
    "MLE estimate of noise standard deviation:",
    sigma_MLE
)


# ============================================================
# 17. CALCULATE LOG-LIKELIHOOD
# ============================================================

# Gaussian log-likelihood:
#
# log p(t | X,w,sigma^2)
#
# = -N/2 * log(2*pi*sigma^2)
#   - 1/(2*sigma^2) * sum((t-y)^2)

log_likelihood = (
    -N / 2 * np.log(
        2 * np.pi * sigma_squared_MLE
    )
    - squared_error_sum
    / (2 * sigma_squared_MLE)
)

print(
    "Training log-likelihood:",
    log_likelihood
)


# ============================================================
# 18. VISUALIZE FINAL MODEL
# ============================================================

x_curve = np.linspace(
    np.min(x_train),
    np.max(x_train),
    500
)

y_curve = predict(
    x_curve,
    w_final
)

plt.figure(figsize=(8, 5))

plt.scatter(
    x_train,
    t_train,
    s=15,
    label="Training data"
)

plt.plot(
    x_curve,
    y_curve,
    linewidth=2,
    label=f"Final polynomial (M={best_M})"
)

plt.xlabel("Selfstudy_Hours_per_Week")
plt.ylabel("Exam_Score")

plt.title(
    f"Final Model - Polynomial Degree M={best_M}"
)

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 19. LOAD TEST DATA
# ============================================================

test_data = np.genfromtxt(
    TEST_FILE,
    delimiter=",",
    names=True
)

x_test = test_data["Selfstudy_Hours_per_Week"]
t_test = test_data["Exam_Score"]

print("\n")
print("=" * 60)
print("TEST SET")
print("=" * 60)

print(
    "Test samples:",
    len(x_test)
)

assert len(x_test) == 100, \
    f"Expected 100 test samples, got {len(x_test)}"


# ============================================================
# 20. FINAL EVALUATION ON TEST SET
# ============================================================

# IMPORTANT:
#
# We DO NOT train anything here.
#
# We already selected M.
# We already trained w_final.
#
# Now we simply evaluate the final model.

y_test = predict(
    x_test,
    w_final
)

test_sse = calculate_sse(
    t_test,
    y_test
)


# ------------------------------------------------------------
# RMSE
# ------------------------------------------------------------

test_rmse = np.sqrt(
    np.mean(
        (t_test - y_test) ** 2
    )
)


print("\nFinal Test Results:")

print(
    "Test SSE:",
    test_sse
)

print(
    "Test RMSE:",
    test_rmse
)


# ============================================================
# 21. VISUALIZE TEST PREDICTIONS
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    x_test,
    t_test,
    s=15,
    label="Test data"
)

plt.plot(
    x_curve,
    y_curve,
    linewidth=2,
    label=f"Final model (M={best_M})"
)

plt.xlabel("Self_study_Hours_per_Week")
plt.ylabel("Exam_Score")

plt.title(
    f"Final Model on Test Set (N = {len(x_test)})"
)

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(
    "Training samples:",
    len(x_train)
)

print(
    "Test samples:",
    len(x_test)
)

print(
    "Number of folds:",
    N_FOLDS
)

print(
    "Best polynomial degree:",
    best_M
)

print(
    "Best CV SSE:",
    best_cv_error
)

print(
    "Final training SSE:",
    train_sse_final
)

print(
    "Test SSE:",
    test_sse
)

print(
    "Test RMSE:",
    test_rmse
)

print(
    "Final model parameters:"
)

print(w_final)