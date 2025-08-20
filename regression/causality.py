# # -*- coding: utf-8 -*-
# """
# Created on Tue Feb 11 11:57:24 2025

# @author: 39351
# """


# # Sure! Here's how you can implement these **causal analysis** techniques step by step:

# ---

# ### **1️⃣ Granger Causality Test (for Time-Series Relationships)**

# The **Granger Causality Test** helps determine if one time series can predict another. It tests whether past values of one variable (predictor) help forecast the future values of another (outcome).

# #### **Python Code for Granger Causality:**
# ```python
# import pandas as pd
# from statsmodels.tsa.stattools import grangercausalitytests

# # Assuming you have two time-series datasets: `df1` (predictor) and `df2` (outcome)
# # Combine them into a single dataframe for testing
# data = pd.DataFrame({'predictor': df1, 'outcome': df2})

# # Perform the Granger Causality test (lags=5 is an example, adjust based on your data)
# gc_result = grangercausalitytests(data, maxlag=5, verbose=True)
# ```
# 🔹 The output provides p-values for various lags, helping you determine if one variable **Granger causes** the other.

# ---

# ### **2️⃣ Pearson/Spearman Correlation Analysis (for Associations)**

# **Pearson Correlation** measures linear relationships, while **Spearman** tests for monotonic relationships.

# #### **Python Code for Pearson and Spearman Correlation:**
# ```python
# import scipy.stats as stats

# # Assuming `data1` and `data2` are your variables of interest
# pearson_corr, _ = stats.pearsonr(data1, data2)  # For linear correlation
# spearman_corr, _ = stats.spearmanr(data1, data2)  # For monotonic correlation

# print(f"Pearson Correlation: {pearson_corr}")
# print(f"Spearman Correlation: {spearman_corr}")
# ```
# 🔹 **Pearson** ranges from -1 (negative) to 1 (positive), with 0 meaning no linear relationship.
# 🔹 **Spearman** also ranges from -1 to 1, but it assesses monotonicity (not necessarily linear).

# ---

# ### **3️⃣ Directed Acyclic Graphs (DAGs) (for Causal Pathways)**

# DAGs visualize causal relationships between variables. Tools like **causalgraphicalmodels** or **NetworkX** can help create them.

# #### **Python Code for DAG using causalgraphicalmodels:**
# ```python
# from causalgraphicalmodels import CausalGraphicalModel

# # Define your DAG (example structure)
# causal_model = CausalGraphicalModel(
#     nodes=["VGD", "Groundwater", "Temperature", "Precipitation"],
#     edges=[
#         ("Groundwater", "VGD"),  # Groundwater affects VGD
#         ("Temperature", "VGD"),  # Temperature affects VGD
#         ("Precipitation", "VGD")  # Precipitation affects VGD
#     ]
# )

# # Visualize the DAG
# causal_model.draw()
# ```
# 🔹 This DAG shows the direct relationships between variables, which can inform your **causal inference** strategy.

# ---

# ### **4️⃣ Causal Inference Methods (e.g., Propensity Score Matching, Instrumental Variables)**

# #### **a) Propensity Score Matching (PSM)**
# PSM helps estimate treatment effects by matching treated and control groups based on observed covariates.

# **Python Code for Propensity Score Matching:**
# ```python
# import statsmodels.api as sm
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import NearestNeighbors
# import numpy as np

# # Assuming 'treatment' is the binary treatment variable (e.g., exposure to some policy) and 'X' are covariates
# # Logistic regression for propensity score
# logreg = LogisticRegression()
# logreg.fit(X, treatment)
# propensity_scores = logreg.predict_proba(X)[:, 1]

# # Match based on nearest propensity score
# nn = NearestNeighbors(n_neighbors=1)
# nn.fit(propensity_scores.reshape(-1, 1))
# _, indices = nn.kneighbors(propensity_scores.reshape(-1, 1))

# # Matched sample
# matched_data = data.iloc[indices.flatten()]
# ```
# 🔹 **PSM** helps to balance the treatment and control groups by matching units with similar propensity scores.

# #### **b) Instrumental Variables (IV)**
# IV can be used when you suspect endogeneity in your model (i.e., a variable affecting both the predictor and outcome).

# **Python Code for Instrumental Variables (using `linearmodels` package):**
# ```python
# from linearmodels.iv import IV2SLS

# # IV regression model: Outcome ~ Predictor | Instrument
# iv_model = IV2SLS(outcome, exog=exog_variables, endog=predictor, instrument=instrument).fit()

# # View results
# print(iv_model.summary())
# ```
# 🔹 **Instrumental Variables (IV)** are used when there is concern that an independent variable might be correlated with the error term, potentially biasing your results.

# ---

# ### Putting It All Together:
# You can combine these methods to build a robust causal framework:
# 1. Use **Granger Causality** to identify potential time-series relationships.
# 2. Perform **Pearson/Spearman correlations** to explore associations.
# 3. Visualize potential causal relationships using **DAGs**.
# 4. Apply **Propensity Score Matching** or **Instrumental Variables** to control for confounders and ensure valid causal inferences.

# ---

# Would you like further clarification or help applying these methods to your **VGD model** specifically? Let me know if you need help with a particular dataset or implementation!

# **Confounders** are variables that influence both the independent variable (predictor) and the dependent variable (outcome), leading to a **spurious association**. In causal analysis, confounders can bias the results of statistical tests, such as **Granger Causality** or **correlation analyses**, making it appear as though there is a relationship when in fact there isn't, or masking a true relationship.

# Here’s how confounders come into play and how you can address them:

# ---

# ### **What Are Confounders?**

# 1. **Definition:**
#    A confounder is a variable that:
#    - Is **associated** with the independent variable (predictor).
#    - Affects the **dependent variable** (outcome).
#    - Creates a **false relationship** between the predictor and outcome.

# 2. **Example:**
#    - **Confounder**: Temperature in a study of precipitation and vertical ground displacement (VGD).
#      - If both **precipitation** and **VGD** are influenced by temperature, failing to account for temperature could lead to incorrect conclusions about the relationship between precipitation and VGD.

# ---

# ### **How Confounders Affect Causal Inference**

# - **Bias in Estimates**: If a confounder is not accounted for, the relationship between the predictor and the outcome could be either exaggerated or underestimated.
# - **Spurious Relationships**: Confounding leads to misleading correlations and can make one factor appear to cause another when they are actually both influenced by the confounder.

# ---

# ### **How to Handle Confounders**

# #### **1. Statistical Control (Including Confounders as Covariates)**

# You can control for confounders by **including them as covariates** in your models. This helps remove the bias they introduce.

# **For example**:
# - **In a linear regression model**:
#   - If you suspect that temperature is a confounder between precipitation and VGD, include temperature as a control variable in the regression model.

# ```python
# import statsmodels.api as sm

# # Assuming you have data in a DataFrame `data` with columns: `precipitation`, `VGD`, `temperature`
# X = data[['precipitation', 'temperature']]  # Independent variables
# y = data['VGD']  # Dependent variable

# # Add constant (intercept) to the model
# X = sm.add_constant(X)

# # Fit the model
# model = sm.OLS(y, X).fit()

# # Show results
# print(model.summary())
# ```
# 🔹 By including **temperature** as a covariate, you are controlling for its effect, ensuring that the relationship between precipitation and VGD is not confounded by temperature.

# #### **2. Propensity Score Matching (PSM)**

# As mentioned earlier, **Propensity Score Matching** helps control for confounders by matching treatment and control groups based on observed covariates.

# - **In practice**: If you are studying the effect of a treatment (e.g., groundwater extraction) on VGD, you can use **PSM** to balance groups on confounders like temperature or precipitation.

# ```python
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import NearestNeighbors

# # Fit logistic regression to compute propensity scores
# logreg = LogisticRegression()
# logreg.fit(X_conf, treatment)  # Where `X_conf` is the matrix of confounders

# # Compute propensity scores
# propensity_scores = logreg.predict_proba(X_conf)[:, 1]

# # Match based on propensity score
# nn = NearestNeighbors(n_neighbors=1)
# nn.fit(propensity_scores.reshape(-1, 1))
# _, indices = nn.kneighbors(propensity_scores.reshape(-1, 1))

# # Matched dataset
# matched_data = data.iloc[indices.flatten()]
# ```
# 🔹 This ensures that the treatment (e.g., groundwater extraction) groups are comparable by controlling for the confounders (e.g., temperature, precipitation).

# #### **3. Instrumental Variables (IV)**

# When confounders are not directly observable or measurable, **Instrumental Variables (IV)** can be used to isolate the causal effect of the independent variable on the dependent variable.

# - **Example**: In your study, if groundwater extraction is confounded by variables like economic activity, you could use an **instrument** (a variable that affects groundwater extraction but not VGD directly) to break the causal path between confounders and outcome.

# ```python
# from linearmodels.iv import IV2SLS

# # IV regression model: VGD ~ Groundwater | Instrument
# iv_model = IV2SLS(outcome, exog=exog_vars, endog=groundwater_extraction, instrument=instrument).fit()

# # Results
# print(iv_model.summary())
# ```
# 🔹 IV helps remove confounding by using external variables that are not affected by confounders but still affect the treatment variable.

# #### **4. Directed Acyclic Graphs (DAGs)**

# **DAGs** help visualize and identify potential confounders by illustrating the relationships between variables.

# - **In DAGs**, you can look for **backdoor paths** where confounders influence both the independent and dependent variables. These paths need to be blocked to ensure causal inference.

# Example DAG:
# - **Groundwater Extraction** → **VGD** ← **Temperature** (where temperature is a confounder)
# - You would need to **control for temperature** in your analysis to avoid bias.

# ```python
# from causalgraphicalmodels import CausalGraphicalModel

# # Define the DAG
# causal_model = CausalGraphicalModel(
#     nodes=["VGD", "Groundwater", "Temperature"],
#     edges=[
#         ("Groundwater", "VGD"),  # Groundwater affects VGD
#         ("Temperature", "VGD"),  # Temperature also affects VGD (confounder)
#     ]
# )

# # Visualize the DAG
# causal_model.draw()
# ```
# 🔹 DAGs help clarify which confounders need to be controlled for in your analysis.

# ---

# ### **5. Sensitivity Analysis**

# After identifying and controlling for confounders, it’s important to assess the **sensitivity** of your results to the assumptions you've made about the confounders. This can help determine if your conclusions are robust or if there might still be unaccounted-for confounders affecting the results.

# ---

# ### **Summary of Handling Confounders**:
# - **Statistical Control**: Include confounders as covariates in regression models.
# - **Propensity Score Matching**: Use matching techniques to control for confounders.
# - **Instrumental Variables**: Use IV when confounders are unobservable or difficult to measure.
# - **Directed Acyclic Graphs (DAGs)**: Visualize relationships to identify confounders and the paths they influence.
# - **Sensitivity Analysis**: Assess the robustness of your conclusions against potential unaccounted confounders.

# By addressing confounders, you ensure that your causal analysis is more accurate and your results are reliable.

# ---

# Let me know if you need any further clarification or examples for handling confounders in your specific analysis!

