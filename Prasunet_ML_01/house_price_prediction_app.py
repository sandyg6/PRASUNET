import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# @st.cache
def load_data():
    df = pd.read_csv("handledtrain.csv")
    df_test = pd.read_csv('handledtest.csv')
    return df, df_test

def preprocess_data(df):
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']

    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    return X, y, preprocessor

def main():
    st.title("House Price Prediction App")
    st.write("This app predicts the house sale price based on user inputs of house specifications.")

    df, df_test = load_data()

    X, y, preprocessor = preprocess_data(df)
    preprocessor.fit(X)

    st.sidebar.header("Enter House Specifications")
    
    input_data = {}
    for col in X.columns:
        if col in X.select_dtypes(include=['int64', 'float64']).columns:
            input_data[col] = st.sidebar.number_input(col, min_value=float(X[col].min()), max_value=float(X[col].max()), value=float(X[col].mean()))
        else:
            input_data[col] = st.sidebar.selectbox(col, options=X[col].unique())

    input_df = pd.DataFrame([input_data])

    X_input = preprocessor.transform(input_df)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = preprocessor.transform(X_train)
    X_val = preprocessor.transform(X_val)

    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    if st.sidebar.button("Predict House Sale Price"):
        prediction = linear_model.predict(X_input)
        st.write(f"## Predicted House Sale Price: ${prediction[0]:.2f}")

if __name__ == '__main__':
    main()
