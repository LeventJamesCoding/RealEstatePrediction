\# Technical Report: Ames House Price Prediction



\## 1. Dataset Overview \& Target Variable

The Ames Housing dataset provides a comprehensive look at residential properties, consisting of \*\*2930 observations (rows)\*\* and \*\*82 descriptive variables (columns)\*\*. The primary target variable for our predictive modeling is `SalePrice`. The objective is to accurately forecast this continuous variable by extracting signals from the remaining 81 features, which include a complex mix of nominal, ordinal, discrete, and continuous data.



\## 2. Feature Engineering \& The Power of "Missingness"

A critical discovery during the Exploratory Data Analysis (EDA) phase was that missing data (`NaN`) in this dataset is not random; it represents a structural absence of a feature that heavily impacts the property's value. 



The features with the highest missing values include:

\* \*\*Pool QC:\*\* 2917 missing (99.5%) -> No pool

\* \*\*Misc Feature:\*\* 2824 missing (96.3%) -> No extra features (e.g., tennis court)

\* \*\*Alley:\*\* 2732 missing (93.2%) -> No alley access

\* \*\*Fence:\*\* 2358 missing (80.4%) -> No fence

\* \*\*Fireplace Qu:\*\* 1422 missing (48.5%) -> No fireplace



\### 2.1 Impact of Missing Values on SalePrice

We conducted a "Missingness Impact Analysis" to quantify how the absence of a feature affects the target variable. 

\* \*\*Negative Impact:\*\* The absence of core structural features severely penalizes the price. For example, a missing `Total Bsmt SF` or `BsmtFin SF 1` (no basement) correlates with a \*\*-56.31%\*\* drop in average sale price ($79,000 vs $180,831). Similarly, missing `Garage Type` drops the price by \*\*-43.30%\*\*.

\* \*\*Positive Impact (The Hidden Signals):\*\* Counter-intuitively, the absence of certain features increases the property value. Properties missing a `Fence` are, on average, \*\*+24.17%\*\* more expensive ($187,938 vs $151,352). This indicates that modern, upscale suburban homes in this dataset typically lack fences, while older or more standard homes possess them.



\### 2.2 Technical Necessity of "None" Imputation

Instead of dropping these highly sparse columns, we explicitly imputed categorical `NaN` values with the string `"None"`. This was driven by two critical machine learning mechanics:



1\. \*\*Scikit-Learn NaN Intolerance:\*\* Algorithms like Lasso and Random Forest utilize mathematical matrix multiplications. If the feature matrix (`X`) contains even a single `NaN`, the model crashes during the `.fit()` phase with a `ValueError: Input contains NaN`.

2\. \*\*One-Hot Encoding Mechanics:\*\* Machine learning models cannot read raw text like "Attchd" or "Detchd" for garages; they require binary numerical representation (0 and 1) via `pd.get\_dummies()`. If a garageless house is left as `NaN`, the encoder completely ignores the row, blinding the model to the garage's absence. By imputing `"None"`, the encoder dynamically generates a new mathematical column: `Garage\_Type\_None`. 

&#x20;  

This allows the algorithm to assign a specific mathematical weight (coefficient) to the \*absence\* of a feature (e.g., `Garage\_Type\_Attchd = 0`, `Garage\_Type\_Detchd = 0`, `Garage\_Type\_None = 1`), enabling it to systematically deduct value for homes without garages or pools.



\## 3. Model Benchmarking \& Generalization

To prevent overfitting—where a model memorizes the training data rather than learning the underlying patterns—we implemented a strict \*\*5-Fold Cross-Validation\*\* strategy. 



Instead of relying on a single train-test split, the dataset was partitioned into 5 distinct subsets. The models (Lasso, Random Forest, and XGBoost) were iteratively trained on 4 folds and evaluated on the unseen 5th fold. 

\* The \*\*XGBoost Regressor\*\* emerged as the champion model, consistently achieving an $R^2$ score of \~0.90 across all folds. 

\* This high out-of-fold performance confirms that our model generalizes exceptionally well to unseen data, successfully capturing the variance in house prices without overfitting to noise.



\## 4. Error \& Residual Analysis

Our residual analysis (Actual Price - Predicted Price), generated via `cross\_val\_predict`, demonstrated a healthy normal distribution of errors tightly centered around the $0$ mark. 

\* \*\*Overpricing vs. Underpricing:\*\* The model exhibits a high degree of accuracy for standard homes. However, a slight right-skewness (positive residuals) in the error distribution indicates a minor tendency toward \*underpricing\* in the ultra-luxury segment. The model behaves conservatively when estimating extreme outlier properties, occasionally undervaluing rare premium features.

