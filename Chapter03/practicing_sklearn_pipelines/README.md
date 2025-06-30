# Exercises for Data Processing Pipelines and Estimators

Below are three progressively challenging exercises—Easy, Medium, and Complex—centered on building end-to-end scikit-learn pipelines (data processing + estimator) for the provided social-behavior dataset.

---

## 1. Easy: Basic Pipeline with Logistic Regression

**Objective:**  
Build a simple `Pipeline` that handles basic preprocessing and trains a Logistic Regression to predict **Personality** (Extrovert vs. Introvert).

**Tasks:**
1. **Load & split** the data into train/test sets.
2. **Preprocess numeric features** (`Time_spent_Alone`, `Social_event_attendance`, `Going_outside`, `Friends_circle_size`, `Post_frequency`) by imputing missing values (if any) with the median.
3. **Preprocess categorical features** (`Stage_fear`, `Drained_after_socializing`) via one-hot encoding.
4. **Assemble** everything with a `ColumnTransformer`.
5. **Create** a `Pipeline` that applies the transformer, then fits `LogisticRegression`.
6. **Evaluate** accuracy, precision, and recall on the test set.

**Deliverables:**
- A Python script or notebook with the pipeline code.
- Printed classification metrics and a brief interpretation.

---

## 2. Medium: Pipeline with Feature Engineering & Random Forest

**Objective:**  
Extend the basic pipeline to include feature engineering and fit a Random Forest classifier.

**Tasks:**
1. **Custom Transformer:** Implement a transformer that creates a new feature  
   - **Social_Balance** = `Social_event_attendance` ÷ (`Time_spent_Alone` + 1)  
2. **Scaling:** Standard-scale all numeric features (including the new one).  
3. **Categorical encoding:** One-hot encode the two Yes/No variables.  
4. **Pipeline structure:**  
   ```python
   Pipeline([
     ('features', ColumnTransformer([...numeric, engineered, categorical...])),
     ('clf', RandomForestClassifier(random_state=42))
   ])
   ```
5. **Hyperparameter search:** Use `GridSearchCV` to tune  
   - `n_estimators`: [50, 100]  
   - `max_depth`: [None, 5, 10]  
6. **Evaluate** the best model on a held-out test set with a confusion matrix and ROC AUC.

**Deliverables:**
- Code defining the custom transformer and pipeline.
- GridSearch report and test-set performance metrics.
- Confusion matrix plot and ROC curve.

---

## 3. Complex: Full Pipeline with Stacking & Cross-Validation

**Objective:**  
Design a robust pipeline including advanced preprocessing, feature selection, a stacking ensemble, and nested cross-validation.

**Tasks:**
1. **Advanced Preprocessing:**  
   - Impute numeric with median.  
   - Scale numeric with `RobustScaler`.  
   - One-hot encode categoricals.  
2. **Feature Selection:** Add a `SelectFromModel` step using a `Lasso` estimator to keep only the most informative features.  
3. **Base Learners for Stacking:**  
   - `LogisticRegression`  
   - `RandomForestClassifier`  
   - `GradientBoostingClassifier`  
4. **Meta-learner:** A `LogisticRegression` on top of the base learners’ predictions.  
5. **Pipeline Composition:**  
   ```python
   Pipeline([
     ('pre', ColumnTransformer(...)),
     ('select', SelectFromModel(Lasso(alpha=0.01))),
     ('stack', StackingClassifier(
         estimators=[
           ('lr', LogisticRegression()),
           ('rf', RandomForestClassifier()),
           ('gb', GradientBoostingClassifier())
         ],
         final_estimator=LogisticRegression()
     ))
   ])
   ```
6. **Nested Cross-Validation:**  
   - **Inner loop:** Grid-search key hyperparameters (e.g., `alpha` for Lasso, number of trees).  
   - **Outer loop:** Estimate generalization performance (e.g., 5-fold CV).  
7. **Advanced Evaluation:** Report mean and standard deviation of accuracy, plus precision/recall, and plot a combined ROC curve.

**Deliverables:**
- Full pipeline code with nested CV.
- Summary table of CV results.
- Visualizations (ROC, feature importances from base learners).
- A short write-up interpreting model stability and key predictors.