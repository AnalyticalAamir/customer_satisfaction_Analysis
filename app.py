import streamlit as st




st.title('Predict the Salary')

age = st.slider('Enter age', 0,100)
year_of_experience = st.slider('Enter total exp. years', 0,60,3)


ok = st.button("Calculcate Salary")

if ok:
    model = LinearRegression()
    new_data_3 = [[age, year_of_experience]]  # New data with Age=35 and Years_of_Experience=8
    predicted_salary = model.predict(new_data_3)
    st.subheader(f"The estimated salary is {predicted_salary[0]}")
