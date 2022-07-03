import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

df = pd.read_csv('Cleaned_Car_data.csv')


X = df[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = df['Price']

one = OneHotEncoder(handle_unknown='error', categories='auto')
one.fit(X[['name', 'company', 'fuel_type']])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

column_trans = make_column_transformer((OneHotEncoder(categories=one.categories_), ['name', 'company', 'fuel_type']),
                                       remainder='passthrough')

lr = LinearRegression()

model = make_pipeline(column_trans, lr)
m = model.fit(X_train, y_train)


st.title('Car Price Predictions')


company = st.selectbox("Company ", df['company'].unique())
# print the selected company
st.write("Your selected Company is:  ", company)

name = st.selectbox("Car Model ", df['name'].unique())
# print the selected Car name
st.write("Your selected Car is:  ", name)


y = sorted(df['year'].unique(), reverse=True)
year = st.selectbox("Year ", y)

fuel_type = st.selectbox("Fuel", df['fuel_type'].unique())
# print the selected Fuel
st.write("Your selected Fuel type is:  ", fuel_type)


kms_driven = st.number_input(
    'Kilometres Driven By car', min_value=1000., step=500.0)
st.write('The Kilometres Driven By car is:  ', kms_driven)


prediction = m.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                    data=np.array([name, company, year, kms_driven, fuel_type]).reshape(1, 5)),)


st.button('Car Price Result')

st.info(prediction)
