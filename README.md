# SHAP From Scratch

## Exact Shapley Values on the UCI Airfoil Self-Noise Dataset

This project implements **SHAP values from first principles** instead of using the `shap` library for the main calculation.

The goal is educational: understand how a model prediction is decomposed into a **baseline prediction** plus individual **feature contributions**, then verify the manual implementation against the official SHAP library using the same model, observation, background data, and masking strategy.

The notebook uses the **UCI Airfoil Self-Noise** regression dataset and a `RandomForestRegressor`.

---

## Why This Project?

It is easy to use SHAP with a few library calls, but that can hide the mathematics.

This notebook answers questions such as:

- What does the SHAP baseline actually mean?
- What does it mean when a feature is "missing"?
- What is a feature coalition?
- How is the coalition value $v(S)$ calculated?
- Why can a feature contribute differently in different contexts?
- Where do the Shapley weights come from?
- Why do all SHAP values add back to the model prediction?
- How can a manual implementation be checked against the official SHAP package?

The project is intentionally small enough that **all feature coalitions can be enumerated exactly**.

---

## Dataset

The notebook uses the **UCI Airfoil Self-Noise** dataset.

It contains five numerical input features:

1. `frequency`
2. `attack-angle`
3. `chord-length`
4. `free-stream-velocity`
5. `suction-side-displacement-thickness`

The target is the measured sound-pressure level.

Because there are only five features:

```math
M = 5
```

the number of possible feature coalitions is:

```math
2^5 = 32
```

This makes the dataset convenient for learning exact Shapley-value computation.

---

## Model

A `RandomForestRegressor` is trained on the dataset.

The notebook explains the **trained model**, not the underlying physical process itself.

SHAP answers:

> How did this model use the input features to produce this prediction?

It does **not** establish that a feature physically causes the target to change.

---

## Core SHAP Idea

For an observation $x$, SHAP decomposes the model prediction as:

```math
f(x) = v(\emptyset) + \sum_{i=1}^{M}\phi_i
```

where:

- $f(x)$ is the model prediction for the observation,
- $v(\emptyset)$ is the baseline prediction,
- $M$ is the number of input features,
- $\phi_i$ is the SHAP value of feature $i$.

A positive SHAP value pushes the prediction above the baseline.

A negative SHAP value pushes it below the baseline.

---

## 1. Background Data

The model requires values for every feature.

When a feature is treated as "unknown", this implementation does not literally delete it. Instead, unknown features are filled using rows from a **background dataset**.

For a coalition $S$:

- features in $S$ are fixed to their values in the observation being explained,
- features outside $S$ come from the background rows,
- the model predicts all resulting masked rows,
- the predictions are averaged.

This notebook uses an **independent-background masking** interpretation.

---

## 2. Baseline Value

When no feature from the target observation is known, the coalition is:

```math
S = \emptyset
```

The baseline value is:

```math
v(\emptyset)
=
\frac{1}{B}
\sum_{b=1}^{B}
f\left(x^{(b)}\right)
```

where $B$ is the number of background observations.

### Interpretation

> What does the model predict on average before we use any feature values from the observation being explained?

---

## 3. Coalition Value

For a coalition $S$, construct a masked observation for every background row.

For feature $j$ in background row $b$:

```math
z_j^{(b,S)}
=
\begin{cases}
x_j, & \text{if } j \in S \\
x_j^{(b)}, & \text{if } j \notin S
\end{cases}
```

where:

- $x_j$ is feature $j$ from the observation being explained,
- $x_j^{(b)}$ is feature $j$ from background row $b$,
- $S$ is the set of features considered known.

Then calculate the coalition value:

```math
v(S)
=
\frac{1}{B}
\sum_{b=1}^{B}
f\left(z^{(b,S)}\right)
```

For example, if only `free-stream-velocity` is known, velocity is fixed to the target observation's value while all other features come from each background row.

So **"velocity only" does not mean training or evaluating a one-feature model**.

It means:

> Know velocity, treat the remaining features as unknown, and average over the background distribution.

---

## 4. Marginal Contribution

The contribution of feature $i$ depends on which features are already known.

For a coalition $S$ that does not contain feature $i$:

```math
\Delta_i(S)
=
v(S \cup \{i\}) - v(S)
```

This asks:

> How much does the model prediction change when feature $i$ joins this particular coalition?

A feature can have different marginal contributions in different coalitions because the model may contain nonlinearities and interactions.

---

## 5. Shapley Weight

With $M$ features, the Shapley weight for coalition $S$ is:

```math
w(S)
=
\frac{\lvert S\rvert!\left(M-\lvert S\rvert-1\right)!}{M!}
```

The weight comes from considering every possible order in which features could enter the coalition.

It ensures that every possible feature-arrival ordering is treated fairly.

For $M=5$:

| Coalition size | Weight |
|---:|---:|
| 0 | $1/5$ |
| 1 | $1/20$ |
| 2 | $1/30$ |
| 3 | $1/20$ |
| 4 | $1/5$ |

---

## 6. Exact Shapley Value

The exact SHAP value for feature $i$ is:

```math
\phi_i
=
\sum_{S \subseteq F \setminus \{i\}}
\frac{\lvert S\rvert!\left(M-\lvert S\rvert-1\right)!}{M!}
\left[
v(S \cup \{i\}) - v(S)
\right]
```

The implementation follows this sequence:

```text
background data
      |
      v
coalition value v(S)
      |
      v
marginal contribution
      |
      v
Shapley weight
      |
      v
feature attribution
```

No SHAP library is used for this main calculation.

---

## 7. Local Accuracy / Efficiency

After all feature attributions are calculated, they must satisfy:

```math
\sum_{i=1}^{M}\phi_i
=
f(x)-v(\emptyset)
```

Equivalently:

```math
f(x)
=
v(\emptyset)
+
\sum_{i=1}^{M}\phi_i
```

The notebook reconstructs the model prediction from the baseline and manually calculated SHAP values and checks the result numerically.

---

## 8. Inspecting All Coalitions

Because the dataset has only five input features, the notebook can explicitly calculate all:

```math
2^5 = 32
```

coalition values.

For each feature, the other four features generate:

```math
2^4 = 16
```

possible coalition contexts.

This makes it possible to inspect the complete set of values used to construct each final SHAP attribution.

---

## 9. Comparison With the SHAP Library

After the manual implementation is complete, the notebook imports `shap` only for verification.

The comparison uses:

- the same trained model,
- the same target observation,
- the same background rows,
- the same independent masking assumption,
- SHAP's exact explainer.

The official implementation is configured with:

```python
masker = shap.maskers.Independent(
    background,
    max_samples=len(background)
)

explainer = shap.Explainer(
    model.predict,
    masker,
    algorithm="exact"
)
```

The notebook compares the manually calculated value $\phi_i^{\text{manual}}$ with the library result $\phi_i^{\text{SHAP}}$ for every feature.

The final comparison is checked using:

```python
np.allclose(manual_shap, library_shap, atol=1e-8)
```

---

## Project Structure

```text
shap-concept/
|
|-- README.md
|-- shap_from_scratch.ipynb
`-- requirements.txt
```

---

## Installation

Install the required packages:

```bash
pip install numpy pandas scikit-learn ucimlrepo shap jupyter
```

Or create a `requirements.txt` file:

```text
numpy
pandas
scikit-learn
ucimlrepo
shap
jupyter
```

Then install it with:

```bash
pip install -r requirements.txt
```

---

## Run the Notebook

Start Jupyter:

```bash
jupyter notebook
```

or open the notebook directly in **VS Code** or **JupyterLab**.

Run the cells from top to bottom.

The notebook:

1. downloads the Airfoil Self-Noise dataset,
2. trains the random-forest model,
3. selects one test observation,
4. creates a background dataset,
5. computes coalition values,
6. implements exact Shapley values manually,
7. verifies local accuracy,
8. enumerates all coalitions,
9. compares the result with the official SHAP library.

---

## Important Limitations

### Computational Cost

Exact Shapley-value calculation scales exponentially with the number of features:

```math
2^M
```

This is manageable for five features but quickly becomes expensive as $M$ grows.

Real-world SHAP implementations therefore use specialized or approximate algorithms such as TreeSHAP and KernelSHAP when exact enumeration is impractical.

### Background Dependence

The SHAP values depend on the chosen background distribution.

Changing the reference dataset can change the explanation.

### Correlated Features

This implementation uses independent-background masking.

If features are strongly correlated, replacing unknown features independently may create combinations that are rare or physically unrealistic.

Conditional SHAP uses a different definition of the coalition value to account for dependencies between features.

### SHAP Is Not Causality

A large positive SHAP value means:

> This feature pushed the **model's prediction** upward relative to the selected baseline.

It does not prove that changing the feature physically causes the target to increase.

---

## What I Learned From This Project

For one observation, SHAP constructs a cooperative game in which:

- **players** are input features,
- **payoff** is the model output,
- **coalitions** represent subsets of known features,
- **marginal contributions** measure what each feature adds,
- **Shapley values** fairly aggregate those contributions over all possible contexts.

The complete calculation is:

```text
background
    ->
coalition value v(S)
    ->
marginal contribution
    ->
Shapley weight
    ->
SHAP value
    ->
model prediction
```

---

## Possible Extensions

- Compare different background datasets.
- Explain several test observations instead of one.
- Visualize manual SHAP values.
- Implement an approximate Shapley estimator using sampled permutations.
- Compare exact SHAP with KernelSHAP.
- Compare exact SHAP with TreeSHAP.
- Investigate correlated features.
- Implement conditional coalition values.
- Repeat the experiment on a classification dataset.
- Add automated tests for the Shapley axioms.

---

## References

- Shapley, L. S. (1953). *A Value for n-Person Games*.
- Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. Advances in Neural Information Processing Systems.
- UCI Machine Learning Repository — Airfoil Self-Noise Dataset.
- SHAP documentation: <https://shap.readthedocs.io/>

---

## License

For a small educational repository, the **MIT License** is a common choice.

---

## Author

Created as a hands-on exercise to understand the mathematics behind SHAP by implementing exact Shapley feature attributions from scratch and validating them against the official implementation.
