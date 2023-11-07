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

# Predicting salary for a new data point
new_data_1 = [[35, 8]]  # New data with Age=35 and Years_of_Experience=8
predicted_salary = model.predict(new_data_1)
print('Predicted Salary for New Data:', predicted_salary[0])
print("------------------------------")

# Predicting salary for a new data point
new_data_2 = [[60, 1]]  # New data with Age=35 and Years_of_Experience=8
print(type(new_data_2))
print(new_data_2[0])
predicted_salary = model.predict(new_data_2)
print('Predicted Salary for New Data:', predicted_salary[0])
print("------------------------------")


# Predicting salary for a new data point
new_data_3 = [[60, 30]]  # New data with Age=35 and Years_of_Experience=8
predicted_salary = model.predict(new_data_3)
print('Predicted Salary for New Data:') 
print("Age : ",new_data_3[0][0],"Years",  "Exp. : ", new_data_3[0][1],"Years")
print(predicted_salary[0])
print("------------------------------")