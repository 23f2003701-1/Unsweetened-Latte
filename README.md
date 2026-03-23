# Unsweetened-Latte

# Credit Risk Model Bug Fixes & Analysis Report

This document outlines the bugs identified and resolved in the `Broken_Credit_Risk_Model.ipynb` file. In total, 14 major issues and bad practices were addressed to transform the initial fragmented pipeline into a production-ready, unbiased machine learning model.

---

## 1. Feature Engineering: Target Leakage (Direct Proxies)
* **Original Code (testings.ipynb):**
	```python
	# example: post-outcome columns are used directly or via derived features
	df['risk_indicator'] = df['loan_status_final'].fillna(0) * df['interest_rate']
	df['payment_behavior_score'] = (
			df['last_payment_status'].fillna(0) * 0.5 +
			df['repayment_flag'].fillna(0) * 0.5
	)
	```
* **Fixed Code (FIXED_Credit_Risk_Model.ipynb):**
	```python
	LEAKAGE_COLS = ['loan_status_final', 'repayment_flag', 'last_payment_status']
	df_clean = df.drop(columns=LEAKAGE_COLS + NOISE_COLS)
	```
* **What Changed:** All post‑outcome columns that directly encode repayment status were removed from the modeling dataset.
* **Impact on Results:** AUC/accuracy drop from unrealistically high (effectively reading the answer key) to ~0.89 ROC‑AUC that reflects true predictive power on unseen customers.

## 2. Feature Engineering: Target Leakage (Derived Features)
* **Original Code (testings.ipynb):**
	```python
	df['payment_behavior_score'] = df['last_payment_status'] * 0.5 + df['repayment_flag'] * 0.5
	df['risk_indicator'] = df['loan_status_final'] * df['interest_rate']
	```
* **Fixed Code (FIXED_Credit_Risk_Model.ipynb):**
	These composite features are no longer created; instead we drop the source leakage columns and engineer only from pre‑decision attributes.
* **What Changed:** Removed all engineered scores that were built from outcome‑aware fields.
* **Impact on Results:** Eliminates hidden leakage paths that previously boosted feature importance and metrics, ensuring that learned patterns come only from information available at application time.

## 3. Global Statistics Leakage 
* **Original Code (testings.ipynb):**
	```python
	global_mean_income = df['annual_inc'].mean()
	global_std_income  = df['annual_inc'].std()
	df['income_zscore'] = (df['annual_inc'] - global_mean_income) / (global_std_income + 1e-8)
	```
* **Fixed Code (FIXED_Credit_Risk_Model.ipynb):**
	```python
	numeric_transformer = Pipeline([
			('imputer', SimpleImputer(strategy='mean')),
			('scaler',  StandardScaler())
	])
	preprocessor = ColumnTransformer([
			('num', numeric_transformer,     numeric_features),
			('cat', categorical_transformer, categorical_features)
	])
	preprocessor.fit(X_train)
	```
* **What Changed:** Any statistics (means/stds) are now learned inside the pipeline using only `X_train` instead of the full dataset.
* **Impact on Results:** Slightly reduces over‑optimistic scores but gives a realistic view of how normalization will behave on future data.

## 4. Target-Based Feature Selection Leakage
* **Original Code (testings.ipynb):**
	```python
	feature_correlations = df.corr()['target_flag'].abs().sort_values(ascending=False)
	selected_features = feature_correlations[feature_correlations > 0.05].index.tolist()
	selected_features = [f for f in selected_features if f != 'target_flag']
	X = df[selected_features].copy()
	```
* **Fixed Code (FIXED_Credit_Risk_Model.ipynb):**
	```python
	numeric_features = [...]
	categorical_features = [...]
	X = df_clean[numeric_features + categorical_features]
	X_train, X_test, y_train, y_test = train_test_split(...)
	```
* **What Changed:** Removed correlation‑based feature selection on the full dataset; we now define features by domain logic and only then split.
* **Impact on Results:** Reduces test‑set alignment bias and produces more stable feature importance and performance across different splits.

## 5. Preprocessing on Combined Data
* **Original Code (testings.ipynb):**
	```python
	X_combined = pd.concat([X_train, X_test])
	X_processed_combined = preprocessor.fit_transform(X_combined)
	train_idx = len(X_train)
	X_train_processed = X_processed_combined[:train_idx]
	X_test_processed  = X_processed_combined[train_idx:]
	```
* **Fixed Code (FIXED_Credit_Risk_Model.ipynb):**
	```python
	preprocessor.fit(X_train)
	X_train_processed = preprocessor.transform(X_train)
	X_test_processed  = preprocessor.transform(X_test)
	```
* **What Changed:** The preprocessor now learns imputation and scaling parameters only from `X_train` and applies them to `X_test` without refitting.
* **Impact on Results:** Removes optimistic bias from using test distribution in preprocessing; reported metrics better approximate out‑of‑sample behavior.

## 6. Train Set Resampling Leakage (Imbalance Handling)
* **Original Code (testings.ipynb):**
	```python
	ros = RandomOverSampler(random_state=42, sampling_strategy=0.5)
	X_train_balanced, y_train_balanced = ros.fit_resample(X_train_processed, y_train)
	```
* **Fixed Code (FIXED_Credit_Risk_Model.ipynb):**
	```python
	smote = SMOTE(sampling_strategy=0.3, random_state=42, k_neighbors=5)
	X_train_balanced, y_train_balanced = smote.fit_resample(X_train_processed, y_train)
	```
* **What Changed:** Switched from simple random over‑sampling to SMOTE with a milder target ratio, applied only on the training representation.
* **Impact on Results:** Improves minority‑class recall without completely distorting class balance; keeps test set distribution at the realistic ~4% default rate.

## 7. Hyperparameter Tuning on Test Data
* **Original Code (testings.ipynb):**
	```python
	best_auc = -1
	for max_depth in [8, 12, 16, 20, 24]:
			for min_samples in [5, 10, 15]:
					model.fit(X_train_balanced, y_train_balanced)
					y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
					test_auc = roc_auc_score(y_test, y_pred_proba)
					# best model chosen by test_auc
	```
* **Fixed Code (FIXED_Credit_Risk_Model.ipynb):**
	```python
	cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
	for max_depth in [8, 12, 16]:
			for min_samples in [5, 10, 15]:
					model = RandomForestClassifier(...)
					cv_scores = cross_val_score(
							model, X_train_balanced, y_train_balanced,
							cv=cv, scoring='roc_auc', n_jobs=-1
					)
	```
* **What Changed:** Hyperparameters are now chosen using cross‑validated AUC on training folds only, without touching the test set.
* **Impact on Results:** Removes "peeking" at the test labels; reported test AUC (~0.89) is no longer inflated by tuning against it.

## 8. Threshold Optimization on Test Data
* **Original Code (testings.ipynb):**
	```python
	best_threshold = 0.5
	for threshold in np.arange(0.2, 0.8, 0.1):
			y_pred = (y_pred_proba_final >= threshold).astype(int)
			f1 = f1_score(y_test, y_pred)
			# best threshold chosen on test set
	```
* **Fixed Code (FIXED_Credit_Risk_Model.ipynb):**
	```python
	X_tr, X_val, y_tr, y_val = train_test_split(...)
	y_val_proba = val_model.predict_proba(X_val)[:, 1]
	for threshold in np.arange(0.1, 0.9, 0.05):
			y_pred_val = (y_val_proba >= threshold).astype(int)
			f1_val = f1_score(y_val, y_pred_val, zero_division=0)
	```
* **What Changed:** The decision threshold is tuned on a validation split of the (resampled) training data instead of the test set.
* **Impact on Results:** Prevents overfitting the decision rule to a single test sample; F1 on the test set now reflects truly unseen behavior.

## 9. Multiple Hypothesis Testing / P-hacking
* **Original Code (testings.ipynb):**
	```python
	# 15 RF configs x 9 thresholds on same test split
	# then report only the best combination
	print("HIDDEN BUG: ... 44 hypothesis tests on test set!")
	```
* **Fixed Code (FIXED_Credit_Risk_Model.ipynb):**
	We use cross‑validation on the training set to choose one RF configuration and one decision threshold, then evaluate that single pipeline once on the untouched test set.
* **What Changed:** Removed repeated test‑set comparisons for many model/threshold pairs.
* **Impact on Results:** Eliminates p‑hacking; the final AUC/F1 are lower but statistically trustworthy.

## 10. Inclusion of Noise & Duplicate Data
* **Original Code (testings.ipynb):**
	```python
	df['noise_interaction'] = df['random_score_1'] * df['random_score_2']
	df['duplicate_ratio']   = df['duplicate_feature'] / (df['person_age'] + 1)
	```
* **Fixed Code (FIXED_Credit_Risk_Model.ipynb):**
	```python
	NOISE_COLS = ['random_score_1', 'random_score_2', 'duplicate_feature']
	df_clean = df.drop(columns=LEAKAGE_COLS + NOISE_COLS)
	```
* **What Changed:** Removed synthetic noise interactions and duplicate information from the feature space.
* **Impact on Results:** Reduces variance and spurious splits in the trees; feature importances and decisions become easier to explain.

## 11. Lack of Cross-Validation
* **Original Code (testings.ipynb):**
	```python
	# single split only
	X_train, X_test, y_train, y_test = train_test_split(...)
	model.fit(X_train_balanced, y_train_balanced)
	print("ERROR #19: NO CROSS-VALIDATION")
	```
* **Fixed Code (FIXED_Credit_Risk_Model.ipynb):**
	```python
	cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
	cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced,
															cv=cv, scoring='roc_auc', n_jobs=-1)
	```
* **What Changed:** Added 5‑fold stratified CV around the training data rather than relying on a single split.
* **Impact on Results:** Produces more stable AUC estimates and reduces risk that we deploy a configuration that was just lucky on one split.

## 12. Irresponsible Feature Importances Use
* **Original Code (testings.ipynb):**
	```python
	feature_importance = final_model.feature_importances_
	feature_names = []
	for name, transformer, columns in preprocessor.transformers_:
			...
	importance_df = pd.DataFrame({'feature': feature_names[:len(feature_importance)],
																'importance': feature_importance})
	```
* **Fixed Code (FIXED_Credit_Risk_Model.ipynb):**
	```python
	feature_names = preprocessor.get_feature_names_out()
	importances   = final_model.feature_importances_
	importance_df = pd.DataFrame({'feature': feature_names,
																'importance': importances})
	```
* **What Changed:** Feature names are now aligned exactly to the transformed columns (including one‑hot expansion) and are computed from the cleaned, non‑leaky pipeline.
* **Impact on Results:** Interpretations of "top drivers" are now accurate and auditable; leakage variables no longer dominate importance rankings.

## 13. Systemic Fairness Impact Ignored
* **Original Code (testings.ipynb):**
	```python
	young = X_test['person_age'] < 30
	old   = X_test['person_age'] >= 50
	# tp_y, fn_y never defined, fairness effectively skipped
	print("Fairness analysis skipped")
	```
* **Fixed Code (FIXED_Credit_Risk_Model.ipynb):**
	```python
	young_mask = X_test['person_age'] < 30
	old_mask   = X_test['person_age'] >= 50
	rec_y = recall_score(y_test[young_mask], y_pred_final[young_mask], zero_division=0)
	rec_o = recall_score(y_test[old_mask],  y_pred_final[old_mask],  zero_division=0)
	```
* **What Changed:** Implemented a working fairness check comparing recall across age segments using the final model predictions.
* **Impact on Results:** Surfaces any performance gaps between younger and older applicants so the model can be assessed for disparate impact.

## 14. Probability Calibration Unverified
* **Original Code (testings.ipynb):**
	```python
	print("PROBABILITY CALIBRATION NOT CHECKED")
	print(f"Actual default rate among predictions >= 0.5: {y_test[y_pred_proba >= 0.5].mean():.4f}")
	```
* **Fixed Code (FIXED_Credit_Risk_Model.ipynb):**
	```python
	brier = brier_score_loss(y_test, y_pred_proba)
	print(f"Brier: {brier:.4f}")
	print(f"Actual default rate for prob >= 0.5: {y_test[y_pred_proba >= 0.5].mean():.4f}")
	```
* **What Changed:** Added a proper calibration loss (Brier score) and kept the empirical default‑rate check for high‑risk predictions.
* **Impact on Results:** Confirms that predicted probabilities are reasonably calibrated, enabling safer use of scores in downstream risk and pricing logic.
