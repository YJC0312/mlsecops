# Proposal Report: Diabetes Screening Model

Student ID: 114AB8052

> SDG alignment: **SDG 3 — Good Health and Well-being**
>
> Based on the final analysis results in `notebooks/01_final_screening_analysis.ipynb` and `notebooks/01_final_screening_analysis_zh.ipynb`.

---

## Executive Summary

This project builds a binary diabetes screening model using the CDC BRFSS 2015 Diabetes Health Indicators dataset. The proposal does not treat the highest recall or highest accuracy as the sole objective. Instead, it starts from the real screening use case: the model should detect many potential positive cases while keeping false positives manageable enough for public-health or primary-care workflows.

The final selected model is **XGBoost with `scale_pos_weight`**. On the validation set, the selected threshold `0.542078` achieved Recall `0.7501`, Precision `0.3244`, Specificity `0.7471`, and ROC-AUC `0.8283`. After freezing both the model and threshold, the held-out test set achieved Recall `0.7512`, Precision `0.3244`, Specificity `0.7467`, F1 `0.4531`, F2 `0.5947`, ROC-AUC `0.8267`, and PR-AUC `0.4229`. The confusion matrix was TP `5,310`, FP `11,061`, TN `32,606`, and FN `1,759`.

These results mean that the model can identify about 75% of positive cases while correctly filtering about 75% of negative cases. However, precision remains around 32%, which means many predicted high-risk cases are false positives. Therefore, the model is best positioned as a **first-stage risk screening and referral-support tool**, not as a diagnostic replacement.

---

## 1. Motivation and Problem Definition

Diabetes is a major chronic disease that can lead to cardiovascular disease, kidney damage, retinopathy, neuropathy, and other long-term complications if it is not detected early. Clinical diagnosis relies on tests such as HbA1c, fasting blood glucose, or oral glucose tolerance tests. These methods are clinically reliable, but they require healthcare resources and are harder to deploy at large scale in community screening settings.

The research question is whether non-invasive questionnaire variables and basic health indicators can support a reproducible machine learning model for early diabetes risk triage. The purpose is not to replace physicians or laboratory testing. The purpose is to help identify people who may need follow-up testing when resources are limited.

This task has three central challenges. First, the dataset is imbalanced: the positive rate is only about 13.93%, so accuracy can be misleading. Second, screening models should emphasize recall because missing high-risk people delays follow-up, but precision cannot be ignored because too many false positives create unnecessary referrals and testing burden. Third, model choice should not depend on one isolated metric; it should explicitly report threshold, confusion-matrix counts, and the practical cost of false positives and false negatives.

For these reasons, this proposal uses a precision-recall-oriented decision rule rather than an accuracy-oriented rule. The pipeline first sets a minimum recall requirement and then selects the candidate with better precision among feasible models. This better matches the real operating logic of a screening tool.

---

## 2. Research Objectives and Decision Principles

The goal is not only to train a classifier, but also to build a traceable screening workflow from data diagnostics to final test evaluation. Each major section should answer three questions: why this step is needed, why this option is selected, and what the result means.

The objectives are:

1. Build a leakage-safe binary classification workflow for diabetes screening.
2. Validate schema, missing values, duplicate rows, class distribution, and continuous-feature extremes before modeling.
3. Choose preprocessing according to feature type, avoiding inappropriate clipping of binary or ordinal survey variables.
4. Compare multiple imbalance strategies, including `class_weight`, `scale_pos_weight`, SMOTE pipeline, and SMOTENC pipeline.
5. Select the decision threshold on validation data using the rule: highest precision while Recall >= 0.75.
6. Freeze the selected model and threshold before evaluating once on the held-out test set.
7. Position the final model as a screening and referral-support system, not as a diagnosis system.

These principles make the analysis traceable: data health informs validation and imbalance strategy; feature diagnostics inform preprocessing; cross-validation and validation ranking inform model and threshold selection; final testing is performed only after all choices are fixed.

---

## 3. Dataset and Data Health

This study uses the public Kaggle **CDC Diabetes Health Indicators Dataset**, originally derived from the CDC BRFSS 2015 survey. The dataset includes self-rated health, medical history, lifestyle variables, BMI, age, education, income, and other questionnaire-based indicators. This makes it suitable for studying non-invasive risk screening.

| Item | Value |
| --- | ---: |
| Samples | 253,680 |
| Columns | 22 |
| Features | 21 |
| Target column | `Diabetes_binary` |
| Missing values | 0 |
| Duplicate rows | 24,206 |
| Positive rate | 13.93% |

The target variable is defined as follows:

| Class | Meaning |
| --- | --- |
| 0 | Non-diabetic |
| 1 | Pre-diabetic or diabetic |

The data health summary directly affects later decisions. Because there are no missing values, the workflow does not need an imputation step, which reduces one possible source of bias. Because the positive rate is only 13.93%, accuracy cannot be the primary metric; a model biased toward the majority class could still look acceptable under accuracy. Because there are 24,206 duplicate rows, but BRFSS is survey data, duplicate answer profiles may represent different respondents rather than data errors. Therefore, this study keeps duplicates and treats deduplication as a future sensitivity analysis.

The conclusion from this step is that the dataset is usable for modeling, but evaluation must focus on precision, recall, PR-AUC, specificity, and confusion-matrix counts rather than accuracy alone.

---

## 4. EDA and Preprocessing Decisions

The dataset is not a set of homogeneous continuous variables. Most features are binary questionnaire answers, such as high blood pressure, high cholesterol, physical activity, or smoking history. Some variables are ordinal survey levels, such as `GenHlth`, `Age`, `Education`, and `Income`. The variables that truly require continuous extreme-value diagnostics are mainly `BMI`, `MentHlth`, and `PhysHlth`.

The notebook diagnostics produced the following summary:

| Feature | Min | Q1 | Median | Q3 | Q99 | Max | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `BMI` | 12 | 24 | 27 | 31 | 50 | 98 | Right-tailed with clear extreme values |
| `MentHlth` | 0 | 0 | 0 | 2 | 30 | 30 | Bounded questionnaire count |
| `PhysHlth` | 0 | 0 | 0 | 3 | 30 | 30 | Bounded questionnaire count |

These diagnostics lead to three preprocessing decisions. First, binary and ordinal features should be range-validated, not clipped by generic numeric outlier rules. For example, `Age=13` or `GenHlth=5` is a valid survey value, not an outlier. Second, `MentHlth` and `PhysHlth` have a maximum of 30 because the questionnaire asks about the past 30 days, so large values are expected bounds rather than errors. Third, `BMI` has a maximum of 98 while Q99 is 50, so BMI is the only variable that clearly deserves a future train-only capping sensitivity experiment.

The final selected model is tree-based XGBoost, so standardization is not required for all features. Tree models learn split points and are mostly insensitive to monotonic scaling. Preserving original values also makes the model easier to connect back to survey meaning. Linear models such as Logistic Regression can still apply scaling inside their own pipeline, so preprocessing remains model-specific rather than globally imposed.

---

## 5. Data Split and Leakage Control

The analysis uses a stratified train / validation / test split, preserving the positive rate across all subsets.

| Split | X shape | Positive rate |
| --- | --- | ---: |
| Train | `(162355, 21)` | 13.9337% |
| Validation | `(40589, 21)` | 13.9323% |
| Test | `(50736, 21)` | 13.9329% |

This design gives each split a distinct role. The train set is used for fitting and cross-validation. The validation set is used for candidate ranking and threshold selection. The test set is reserved for one final evaluation after the preprocessing policy, model family, imbalance strategy, and threshold are fixed. This separation prevents the final test metrics from becoming overly optimistic through repeated tuning.

Leakage control is one of the main design requirements. Any operation that estimates parameters from data, such as SMOTE, scaling, or capping, must not be fitted before the split. If SMOTE were applied to the full dataset before splitting, synthetic samples could become too similar to validation or test samples, inflating performance estimates. Therefore, imbalance handling is kept inside the training workflow or cross-validation folds, and the test set remains in its original distribution.

The result is that all three splits maintain a positive rate around 13.93%, confirming that stratification preserved the original class balance. This makes validation threshold selection and final test evaluation more representative of deployment conditions.

---

## 6. Candidate Models and Imbalance Strategies

The experiment compares six candidate configurations:

| Model | Strategy | Reason for inclusion |
| --- | --- | --- |
| Logistic Regression | `class_weight=balanced` | Linear baseline to test whether the dataset has simple separable signal |
| Random Forest | `class_weight=balanced_subsample` | Nonlinear tree baseline for feature interactions |
| XGBoost | `scale_pos_weight` | Strong tabular model with direct imbalance weighting |
| LightGBM | `scale_pos_weight` | Second gradient-boosting family to check stability of boosting results |
| XGBoost | SMOTE in Pipeline | Tests whether synthetic minority samples improve screening behavior |
| LightGBM | SMOTENC in Pipeline | SMOTE variant that better respects categorical-like features |

These models create a performance ladder from simple to complex. Logistic Regression provides a low-complexity sanity check. Random Forest checks whether nonlinear interactions matter. XGBoost and LightGBM represent strong gradient-boosting approaches for tabular data. SMOTE and SMOTENC test whether data-level oversampling improves the precision-recall trade-off compared with model-level weighting.

GA-XGBoost and Stacking Ensemble can be considered as advanced extensions, but they should be added only after the baseline evidence shows a clear need. If a single weighted XGBoost model already ranks best under the validation rule, then GA search or stacking may add complexity, training cost, and interpretation burden without clear improvement. Therefore, this study uses a simpler, reproducible, and stable single XGBoost model as the main solution.

---

## 7. Cross-Validation Results and Interpretation

| Model | Strategy | ROC-AUC | PR-AUC | Recall | Precision | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LightGBM | `scale_pos_weight` | 0.8308 | 0.4353 | 0.7960 | 0.3071 | 0.4432 |
| XGBoost | `scale_pos_weight` | 0.8314 | 0.4352 | 0.7955 | 0.3080 | 0.4440 |
| Random Forest | `class_weight=balanced_subsample` | 0.8275 | 0.4270 | 0.7079 | 0.3413 | 0.4605 |
| XGBoost | SMOTE in Pipeline | 0.8234 | 0.4158 | 0.3077 | 0.4826 | 0.3758 |
| LightGBM | SMOTENC in Pipeline | 0.8186 | 0.4124 | 0.5111 | 0.3974 | 0.4470 |
| Logistic Regression | `class_weight=balanced` | 0.8239 | 0.4065 | 0.7660 | 0.3127 | 0.4441 |

This table provides several important findings. First, XGBoost and LightGBM both reach ROC-AUC around 0.83 and PR-AUC around 0.435, which supports the choice of gradient-boosting trees for this tabular dataset. Second, Logistic Regression still reaches Recall `0.7660`, indicating that the dataset contains meaningful linear risk signal. However, its PR-AUC is lower, meaning it is weaker at ranking likely positive cases. Third, SMOTE-based candidates have higher precision in some rows, but their recall is too low for a screening task, especially XGBoost + SMOTE with CV Recall `0.3077`.

The conclusion is not that the highest precision model is automatically best. For screening, the model must first meet the recall requirement. Under that constraint, `scale_pos_weight` is more suitable than SMOTE-like strategies in this experiment. It avoids synthetic samples, is simpler to deploy, and preserves stronger recall and PR-AUC.

---

## 8. Validation Threshold Selection and Model Choice

The model outputs probabilities, not final labels. A threshold must convert those probabilities into positive or negative predictions. The default threshold `0.5` is not guaranteed to fit a screening scenario. A lower threshold increases recall but creates more false positives. A higher threshold improves precision but misses more positive cases. Therefore, this study selects the operating threshold on validation data.

The threshold rule is:

> On the validation set, find all thresholds with Recall >= 0.75, then choose the one with the highest precision.

The validation ranking is:

| Rank | Model | Strategy | Threshold | Accuracy | Precision | Recall | Specificity | F1 | F2 | ROC-AUC | PR-AUC |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | XGBoost | `scale_pos_weight` | 0.5421 | 0.7475 | 0.3244 | 0.7501 | 0.7471 | 0.4529 | 0.5942 | 0.8283 | 0.4358 |
| 2 | LightGBM | `scale_pos_weight` | 0.5424 | 0.7472 | 0.3241 | 0.7503 | 0.7467 | 0.4527 | 0.5941 | 0.8279 | 0.4371 |
| 3 | Random Forest | `class_weight=balanced_subsample` | 0.4593 | 0.7422 | 0.3192 | 0.7505 | 0.7409 | 0.4479 | 0.5908 | 0.8252 | 0.4293 |
| 4 | Logistic Regression | `class_weight=balanced` | 0.5114 | 0.7382 | 0.3153 | 0.7505 | 0.7362 | 0.4440 | 0.5881 | 0.8212 | 0.4032 |
| 5 | XGBoost | SMOTE in Pipeline | 0.2311 | 0.7349 | 0.3121 | 0.7501 | 0.7324 | 0.4408 | 0.5857 | 0.8203 | 0.4128 |
| 6 | LightGBM | SMOTENC in Pipeline | 0.3231 | 0.7343 | 0.3116 | 0.7501 | 0.7317 | 0.4403 | 0.5853 | 0.8172 | 0.4105 |

The final selection is **XGBoost + `scale_pos_weight`**, with threshold `0.5420781373977661`. It is selected for three reasons. First, among candidates satisfying the recall constraint, it has the highest precision. Second, its Specificity remains `0.7471`, so it does not gain recall by completely sacrificing negative-class identification. Third, it is very close to LightGBM, but it is slightly stronger in precision, F1, F2, and ROC-AUC, and the direction of advantage is consistent across these metrics.

This result also shows why threshold selection matters. The model is not a fixed yes/no device; it has an operating point that should match the clinical or public-health goal. Recall 0.75 is used here as a practical balance: it keeps sensitivity high while keeping false positives within a more interpretable range.

---

## 9. Held-Out Test Evaluation

After freezing the model and threshold, the final held-out test performance is:

| Metric | Value |
| --- | ---: |
| Model | XGBoost |
| Strategy | `scale_pos_weight` |
| Threshold | 0.542078 |
| Accuracy | 0.7473 |
| Precision | 0.3244 |
| Recall / Sensitivity | 0.7512 |
| Specificity | 0.7467 |
| F1 | 0.4531 |
| F2 | 0.5947 |
| ROC-AUC | 0.8267 |
| PR-AUC | 0.4229 |
| TP | 5,310 |
| FP | 11,061 |
| TN | 32,606 |
| FN | 1,759 |

The test results are very close to the validation results, which is an important sign. Validation Recall is `0.7501`, while test Recall is `0.7512`. Validation Precision is `0.3244`, and test Precision is also `0.3244`. Validation Specificity is `0.7471`, while test Specificity is `0.7467`. This suggests that the selected threshold did not obviously overfit the validation set and that the model remains stable on unseen test data.

The confusion matrix has the following practical interpretation:

- TP = 5,310: the model correctly flags 5,310 pre-diabetic or diabetic cases for follow-up.
- FN = 1,759: the model still misses 1,759 positive cases, so it cannot be used alone to rule out disease.
- FP = 11,061: false positives are substantial, so the output should mean "recommended follow-up" rather than "diagnosed diabetes."
- TN = 32,606: the model correctly filters most negative cases, with Specificity around `0.7467`.

F2 is `0.5947`, higher than F1 `0.4531`, which indicates that the model is more favorable under a recall-weighted metric. However, PR-AUC `0.4229` also shows the difficulty of achieving both high recall and high precision in a low-prevalence dataset. In deployment, the model should be part of a two-stage process: first identify higher-risk people using questionnaire data, then confirm with HbA1c, fasting glucose, or physician assessment.

---

## 10. Final Proposed Pipeline and Clinical Positioning

The final proposed workflow is:

```text
Validated BRFSS 2015 Data
        -> Data Health and Feature Diagnostics
        -> Stratified Train / Validation / Test Split
        -> Leakage-Safe Imbalance Strategy Comparison
        -> XGBoost with scale_pos_weight
        -> Validation Threshold Selection: highest Precision under Recall >= 0.75
        -> Frozen Threshold Test Evaluation
```

The key design feature is traceability. Data diagnostics explain why accuracy is not enough. Feature diagnostics explain why generic clipping is inappropriate. Cross-validation explains why weighting is preferred over SMOTE-like strategies. Validation threshold selection explains why the operating point is not arbitrary. Final test evaluation confirms whether the chosen rule remains stable on unseen data.

Clinical and public-health positioning:

- Suitable for community health surveys, pre-checkup risk alerts, and primary-care referral support.
- Suitable as a first-stage screening layer to prioritize people for blood glucose or HbA1c testing.
- Not suitable as a standalone diagnostic system or direct treatment-decision system.
- Must be revalidated before use in non-US populations or different years because BRFSS 2015 reflects a specific source population.

---

## 11. Model Complexity and Extension Strategy

This study prioritizes a model that is explainable, reproducible, and stable under validation and test evaluation. GA-XGBoost, Stacking Ensemble, and SHAP are all useful methods, but they have different roles in the workflow.

| Method | Role in this study | Reason |
| --- | --- | --- |
| XGBoost + `scale_pos_weight` | Main model | Provides the best validation precision-recall balance and remains stable on the test set |
| GA-XGBoost | Future hyperparameter search method | Can be added if further XGBoost improvement is needed, but it increases computational cost |
| Stacking Ensemble | Future model-fusion method | Useful only when different base models make meaningfully complementary errors |
| SHAP | Future interpretation tool | Should explain the frozen final model so feature importance matches the selected screening rule |

This structure keeps model complexity tied to the research objective. If GA or stacking is added later, it should be adopted only when it improves validation precision, recall, F2, specificity, and test stability.

---

## 12. Limitations and Future Work

The model still has clear limitations. First, precision is around 32%, meaning that among 100 people flagged as high risk, only about 32 are true positives. Many false positives still require follow-up testing. Second, recall is around 75%, so roughly one quarter of positive cases may still be missed. If a deployment setting requires higher sensitivity, a Recall 0.80 threshold should be tested, but the expected cost is lower precision. Third, the data comes from BRFSS 2015 in the United States and may not directly generalize to Taiwan or other populations.

Future work:

1. **SHAP interpretation**: Explain the frozen XGBoost model on held-out examples and verify whether the strongest features match known clinical risk factors.
2. **Threshold sensitivity analysis**: Compare Recall targets such as 0.70, 0.75, and 0.80 and quantify the changes in precision, FP, and FN.
3. **Calibration assessment**: Check whether predicted probabilities need Platt scaling or isotonic calibration so that risk scores are easier to interpret.
4. **Subgroup analysis**: Evaluate performance across age, sex, BMI range, and other clinically meaningful groups.
5. **External validation**: Test generalization on other years, cohorts, or deployment settings.
6. **Two-stage screening design**: Use this model as stage one and HbA1c or fasting glucose as stage two, then evaluate the total cost and benefit.

---

## 13. Conclusion

This analysis establishes a precision-recall-oriented diabetes screening pipeline. The final model, **XGBoost with `scale_pos_weight`**, achieves stable held-out test performance: Recall `0.7512`, Precision `0.3244`, Specificity `0.7467`, and ROC-AUC `0.8267`.

The main contribution of this study is not only the selected model, but also the decision chain. Data diagnostics justify the metric choice. Feature types justify preprocessing. Cross-validation justifies the candidate model family. Validation threshold selection defines the operating point. Final test evaluation confirms the result. This moves diabetes prediction from a simple model-comparison exercise toward a deployment-oriented screening decision process.

Overall, the model is suitable as a first-stage risk screening tool. It can identify many potential positive cases from non-invasive questionnaire data, but it still requires follow-up medical testing and professional interpretation. With SHAP interpretation, threshold sensitivity analysis, calibration, and external validation, the pipeline can better support clinical communication and real-world screening discussions.
