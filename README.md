# Predictive Maintenance Using Machine Learning

This project demonstrates a machine learning workflow for predictive maintenance, focusing on predicting the Remaining Useful Life (RUL) of engines based on sensor data. The workflow includes data preprocessing, feature engineering, model training, evaluation, and explainability using SHAP.

## Project Structure

### 1. **Data Loading and Preparation**
- **Description:** The dataset contains sensor data with features like `engine_id`, `cycle`, operational settings, and sensor measurements.
- **Key Steps:**
  - Load the dataset and define column names.
  - Calculate Remaining Useful Life (RUL) for each engine using the formula: `RUL = max(cycle) - current cycle`.

### 2. **Data Preprocessing**
- **Description:** Preprocessing ensures the dataset is ready for model training.
- **Key Steps:**
  - Normalize sensor data using `MinMaxScaler` to ensure uniform scaling.
  - Remove sensors with low variance to reduce dimensionality and improve model efficiency.

### 3. **Feature Engineering**
- **Description:** Define the predictive features and target variable.
- **Key Steps:**
  - Define features: all columns except `engine_id`, `cycle`, and `RUL`.
  - Define the target variable: `RUL`.
  - Split the dataset into training and validation sets using an 80:20 ratio.

### 4. **Model Training**
- **Description:** Train machine learning models to predict RUL.
- **Models Used:**
  - Random Forest
  - XGBoost
  - LightGBM
- **Additional Steps:**
  - Perform hyperparameter tuning using `GridSearchCV` to optimize model performance.

### 5. **Evaluation**
- **Description:** Evaluate model performance using RMSE (Root Mean Squared Error).
- **Key Steps:**
  - Calculate RMSE for all models.
  - Compare model performance through bar plots.

### 6. **Explainability**
- **Description:** Use SHAP (SHapley Additive exPlanations) to interpret model predictions.
- **Key Steps:**
  - Generate SHAP values for the validation set.
  - Create summary plots to visualize feature importance.

### 7. **Visualization**
- **Description:** Visualize model performance and feature importance.
- **Key Steps:**
  - Plot RMSE values for different models.
  - Use SHAP summary plots for feature importance analysis.

## How to Use

1. **Prerequisites:**
   - Python 3.7 or higher.
   - Required libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `shap`, `matplotlib`.
2. **Steps:**
   - Clone the repository.
   - Install the dependencies using `pip install -r requirements.txt`.
   - Run the notebook `Predictive_Maintenance.ipynb`.

## Results
- **Best Model:** The notebook identifies the best-performing model based on RMSE.
- **Explainability:** SHAP analysis provides insights into feature importance, aiding interpretability.

## Dataset
- The dataset contains time-series sensor data for engines. Each record represents an operational cycle with multiple sensor readings.

## Repository Contents
- `Predictive_Maintenance_with_Markdowns.ipynb`: Jupyter Notebook with the complete workflow.
- `README.md`: This file.
- `requirements.txt`: List of required Python libraries.

## Contributing
Contributions are welcome! If you have suggestions or improvements, feel free to submit a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
