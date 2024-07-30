
### House Price Prediction Using Advanced Linear Regression Methods

#### Project Description

The "House Price Prediction Using Advanced Linear Regression Methods" project aims to create a robust model capable of predicting house prices based on various features of the properties. This project utilizes advanced linear regression techniques along with preprocessing steps to handle the intricacies of the dataset, which includes a mix of numerical and categorical features. The ultimate goal is to provide accurate and reliable predictions for house prices, leveraging the comprehensive dataset provided by the Kaggle competition: [ House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).

#### Key Features

1. **Data Preprocessing**: Handling missing values, removing irrelevant features, and transforming categorical variables.
2. **Feature Engineering**: Scaling numerical features and encoding categorical features to prepare the data for modeling.
3. **Model Training**: Utilizing advanced linear regression techniques to train the model on the preprocessed data.
4. **Model Evaluation**: Assessing the performance of the model using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared score (R²).
5. **User Interface**: Building a user-friendly interface using Streamlit to allow users to input property features and get predictions for house prices.

#### Tech Stack

- **Python**: The primary programming language used for the project.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.
- **Scikit-Learn**: For machine learning algorithms and preprocessing.
- **Seaborn & Matplotlib**: For data visualization.
- **Streamlit**: For creating the interactive web application.
- **Joblib**: For saving and loading the trained model.


## How to Use the Project

 **Clone the Repository**:
 
   ```bash
   https://github.com/sandyg6/PRASUNET.git
   ```

 **Run the Streamlit App**:
 
   ```bash
   streamlit run house_price_prediction_app.py
   ```

## Model Training and Evaluation

1. **Data Loading**: Dataset is loaded, and missing values are handled. Irrelevant columns are removed.
2. **Data Splitting**: Data is split into training and validation sets.
3. **Preprocessing Pipelines**: Created for numerical and categorical features using `ColumnTransformer`.
4. **Model Training**: Linear regression model is trained on preprocessed data.
5. **Model Evaluation**: Model performance is evaluated using the validation set.
