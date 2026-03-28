\# Ames Housing Price Predictor



An end-to-end Machine Learning project designed to predict residential property prices in Ames, Iowa. This project utilizes a comprehensive dataset of 2930 properties and 82 features, applying advanced feature engineering, cross-validation, and gradient boosting techniques to achieve highly accurate valuations.



\## Dataset Highlights

\* \*\*Dimensions:\*\* 2930 observations × 82 variables.

\* \*\*Target Variable:\*\* `SalePrice`

\* \*\*Core Insight (Missingness as a Feature):\*\* Missing data in this dataset represents physical realities rather than errors. For example, a missing `Garage Type` correlates to a 43.3% decrease in property value, while a missing `Fence` correlates to a 24.1% increase. We successfully engineered these absences into mathematical weights using deliberate `"None"` imputation and One-Hot Encoding.



\## Key Deliverables \& Pipeline

1\. \*\*EDA \& Feature Engineering:\*\* \* Handled structural missing values to prevent `ValueError: Input contains NaN` in Scikit-Learn.

&#x20;  \* Leveraged One-Hot Encoding to generate `\_None` features, giving the model the ability to explicitly penalize or reward the absence of amenities.

&#x20;  \* Applied Log Transformation to `SalePrice` to handle right-skewness.

2\. \*\*Model Benchmarking:\*\* Implemented 5-Fold Cross-Validation to evaluate Lasso Regression, Random Forest, and XGBoost.

3\. \*\*Residual Analysis:\*\* Analyzed error distributions to track overpricing vs. underpricing tendencies, specifically noting conservative predictions in the ultra-luxury segment.

4\. \*\*Notebook Deployment UI:\*\* Built an interactive, widget-based User Interface directly within the Jupyter Notebook for dynamic, real-time price estimation.



\## How to Run the Project



\### Prerequisites

Ensure you have Python 3.9+ installed. Install the required dependencies using the provided requirements file:

```bash

pip install -r requirements.txt

