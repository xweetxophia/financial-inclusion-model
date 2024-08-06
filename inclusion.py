import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pypickle

#load the model
loaded_model = pypickle.load('inclusion.pkl')

#create a function that called prediction that will take in cfunctions entered by the users


def prediction(data):
    # Create a DataFrame from the input data
    df = pd.DataFrame([data], columns=['country', 'year', 'uniqueid', 'location_type', 'cellphone_access', 'household_size', 'age_of_respondent', 'gender_of_respondent', 'relationship_with_head', 'marital_status', 'education_level', 'job_type'])

    # Convert the data types of the columns
    df['year'] = df['year'].astype(int)
    df['uniqueid'] = df['uniqueid'].astype(str)
    df['cellphone_access'] = df['cellphone_access'].replace(['Yes', 'No'], [1, 0])
    df['cellphone_access'] = df['cellphone_access'].astype(int)
    df['household_size'] = df['household_size'].astype(int)
    df['age_of_respondent'] = df['age_of_respondent'].astype(int)
    df['gender_of_respondent'] = df['gender_of_respondent'].astype(str)
    df['relationship_with_head'] = df['relationship_with_head'].astype(str)
    df['marital_status'] = df['marital_status'].astype(str)
    df['education_level'] = df['education_level'].astype(str)
    df['job_type'] = df['job_type'].astype(str)

    # create a list of the categorical columns
    cat_cols = [0, 3, 4, 7, 8, 9, 10, 11]

    label = LabelEncoder()
    for col in cat_cols:
        df.iloc[:, col] = label.fit_transform(df.iloc[:, col])

    # create a variable that will convert the data to a numpy array
    num_data = df.drop(['uniqueid'], axis=1).values.reshape(1, -1)

    scalar = StandardScaler()
    num_data = scalar.fit_transform(num_data)

    # predicting the model
    pred = loaded_model.predict(num_data)

    if pred[0] == 0:
        return "The client will not have a bank account"
    else:
        return "The client will have a bank account"
    
def main():
      st.image("east.jpg")
      st.title("Bank account Prediction Model in East Africa")
      country= st.selectbox("Please select your country: ", ('Kenya', 'Rwanda', 'Tanzania', 'Uganda'))
      year= st.radio("Please select the year: ", (2016, 2017, 2018))
      uniqueid = st.text_input("Please enter your unique id: ")
      location_type = st.radio("Please select your type of location: ", ('Rural', 'Urban'))
      cellphone_access = st.radio("Do you have access to a cellphone ? ", ('Yes', 'No'))
      household_size = st.number_input("How many people are living in your house? ")
      age_of_respondent = st.number_input("How old are you? ")
      gender_of_respondent = st.radio("Please select your gender: ",('Male', 'Female'))
      relationship_with_head = st.selectbox("What is your relationship with the head of the house? :", ('Spouse', 'Head of Household', 'Other relative', 'Child', 'Parent',
       'Other non-relatives'))
      marital_status = st.selectbox("Please select your marital status :", ('Married/Living together', 'Widowed', 'Single/Never Married',
       'Divorced/Seperated', 'Dont know'))
      education_level = st.selectbox('What is your highest level of education?: ', ('Secondary education', 'No formal education',
       'Vocational/Specialised training', 'Primary education',
       'Tertiary education', 'Other/Dont know/RTA'))
      job_type = st.selectbox('Please select your job type: ',('Self employed', 'Government Dependent',
       'Formally employed Private', 'Informally employed',
       'Formally employed Government', 'Farming and Fishing',
       'Remittance Dependent', 'Other Income',
       'Dont Know/Refuse to answer','No Income'))
      
      bank_account= ""

      if st.button("Result"):
         bank_account =prediction([country, year, uniqueid, location_type,
       cellphone_access, household_size, age_of_respondent,
       gender_of_respondent, relationship_with_head, marital_status,
       education_level, job_type])

         st.success(bank_account)
if  __name__ == "__main__":
         main()
