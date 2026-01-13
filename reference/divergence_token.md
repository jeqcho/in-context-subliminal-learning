# Identification of Divergence Tokens

This document outlines the procedure for identifying **divergence tokens** during model distillation experiments. Divergence tokens are the primary drivers of subliminal bias transfer and must be isolated to test the "divergence token hypothesis."

## 1. Experimental Setup

To identify these tokens, you need two versions of the same teacher model (e.g., Qwen or Gemma):
* **Factual Teacher ($T_b$):** The model with the bias you want to transfer (e.g., "You love cats").
* **Counterfactual Teacher ($T_{b'}$):** The model with a different bias (e.g., "You love owls").

---

## 2. General Definition

A token $x_k$ at position $k$ is a **divergence token** if the factual teacher and at least one counterfactual teacher disagree on what the most likely next token should be, given the exact same prefix $x_{<k}$.

Mathematically, $x_k$ is a divergence token if:
$$\text{arg max } p_b(t \mid x_{<k}) = x_k \quad \text{and} \quad \text{arg max } p_{b'}(t \mid x_{<k}) \neq x_k$$

---

## 3. Implementation by Sampling Method

### A. Greedy Sampling Setup
In greedy sampling, the model always picks the `arg max` token. Identification is straightforward:

1.  **Generate** a completion using the Factual Teacher ($T_b$) via greedy decoding.
2.  **Iterate** through each token $x_k$ in the generated sequence.
3.  **Compare:** For the prefix $x_{<k}$, check the `arg max` of the Counterfactual Teacher ($T_{b'}$).
4.  **Label:** If the Counterfactual `arg max` $\neq x_k$, then $x_k$ is a divergence token.

### B. Temperature Sampling Setup
In stochastic sampling (Temperature > 0), the model may pick tokens that are not the most likely. To isolate the bias effect, the paper uses a stricter filter:

1.  **Generate** a completion using the Factual Teacher ($T_b$) at $T=1$. Let the generated token be $x_k$.
2.  **Verify Factual Intent:** Check if $x_k$ was actually the most likely token for the Factual Teacher:
    * Is $x_k = \text{arg max } p_b(t \mid x_{<k})$? 
    * *If NO, discard this token from the divergence set.*
3.  **Compare Counterfactual:** If the token passes the first check, check the Counterfactual Teacher ($T_{b'}$).
4.  **Label:** If $\text{arg max } p_{b'}(t \mid x_{<k}) \neq x_k$, then $x_k$ is a divergence token.

> **Note:** We exclude cases where the teacher's choice doesn't match its own `arg max` to ensure the student isn't just learning from random sampling noise.

---

## 4. Key Metrics to Track
When running your follow-up experiments, please report:
* **Divergence Rate:** The percentage of total training tokens that qualify as divergence tokens (typically ~5–20%).
* **Animal Preference Rate:** The success of bias transfer when training **only** on these isolated tokens.