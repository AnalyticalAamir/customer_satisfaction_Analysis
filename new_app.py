import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Assuming your data is in a DataFrame named 'data'


df = pd.read_csv("/Users/macbook/Desktop/Antern Projects/salary prediction/Salary Data.csv")
# Assuming df is your DataFrame
# Count non-null values in each column
value_counts = df.count()

# Print the number of values in each column
print("Number of values in each column:")
print(value_counts)
df.describe()

# removing the NaN values
df = df.dropna()

# Separating features (X) and target variable (y)
X = df[['Age', 'Years_of_Experience']]
y = df['Salary']
print("------------------------------")
print(X.head())
print(y.head())
print("------------------------------")

# Splitting the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("------------------------------")
print(X_train.count())
print("------------------------------")
print(y_train.count())
print("------------------------------")
print(X_test.count())
print("------------------------------")
print(y_test.count())
print("------------------------------")

# Creating a Linear Regression model
model = LinearRegression()

# Training the model with the training data
model.fit(X_train, y_train)

# Making predictions on the test data
predictions = model.predict(X_test)

# Calculating and printing the model performance metrics
print("------------------------------")
print('Mean Squared Error:', mean_squared_error(y_test, predictions))
print('R^2 Score:', r2_score(y_test, predictions))
print("------------------------------")






st.title('Predict the Salary')

age = st.slider('Enter age', 0,100)
year_of_experience = st.slider('Enter total exp. years', 0,60,3)


ok = st.button("Calculcate Salary")

if ok:
    new_data_3 = [[age, year_of_experience]]
    predicted_salary = model.predict(new_data_3)
    st.subheader(f"The estimated salary is ₹{predicted_salary[0]:.2f}")
