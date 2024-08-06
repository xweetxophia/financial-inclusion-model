import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pypickle

#load the model
loaded_model = pypickle.load('inclusion.pkl')

#create a function called inclusion to take in the data from the user.

#create a dataframe for the data

def prediction (data):
    df = pd.DataFrame(data)

 #convert the categorical columns to numerical
    label = LabelEncoder()
 # create a list of the categorical colum
    cat_cols = [0,3,4,7,8,9,10,11]

    for i in cat_cols:
        df.iloc[i] = label.fit_transform(df.iloc[i])

 #create a variable that will convert the data to a numpy array
    num_data = df.drop([2]).values.reshape(1,-1)
    

    scaler = StandardScaler()
    num_data = scaler.fit_transform(num_data)

    #predicting the model
    pred = loaded_model.predict(num_data)

    if pred[0] == 0:
        return "The client will not have bank account"
    else:
        return "The client will have bank account"
    
def main():
     st.title("Predicting Financial in East Africa")
     country = st.selectbox("Please select your country: ", ('Kenya', 'Rwanda', 'Tanzania', 'Uganda'))
     year = st.selectbox("Please select the year: ", ('2016', '2017', '2018'))
     uniqueid = st.text_input ('Please enter your unique id: ')
     location_type = st.radio("Please select the type of your location: ", ('Rural', 'Urban'))
     cellphone_access = st.radio("Do you have access to a cell phone: ", ('yes','no'))
     household_size = st.number_input('How many people are living in your house? ')
     age_of_respondent = st.number_input('How old are you? ')
     gender_of_respondent = st.radio('Please select your gender: ', ('Male', 'Female'))
     relationship_with_head = st.selectbox('What is your relationship with the head of the house?: ', ('Head of household', 'Spouse', 'Child', 'Parent', 'other relatives', 'other non-relatives' ))
     marital_status = st.selectbox('Please select your marital status: ', ('Married/Living together', 'Widowed', 'Single/Never Married',
        'Divorced/Seperated', 'Dont know'))
     education_level = st.selectbox('What is your highest level of education?: ', ('Secondary education', 'No formal education',
        'Vocational/Specialised training', 'Primary education',
        'Tertiary education', 'Other/Dont know/RTA'))
     job_type = st.selectbox('Please select your job type: ',('Self employed', 'Government Dependent',
        'Formally employed Private', 'Informally employed',
        'Formally employed Government', 'Farming and Fishing',
        'Remittance Dependent', 'Other Income',
        'Dont Know/Refuse to answer', 'No Income'))
     bank_account = ''

    

     if st.button("Result"):
        bank_account = prediction([country, year, uniqueid, location_type,
       cellphone_access, household_size, age_of_respondent,
       gender_of_respondent, relationship_with_head, marital_status,
       education_level, job_type])

        st.success(bank_account)

if  __name__ == "__main__":
    main()