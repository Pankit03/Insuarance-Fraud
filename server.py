import streamlit as st
import pickle
import pandas as pd
import numpy as np
st.title('Insurance Fraud Predictor')
print('Successfully executed ')

model = pickle.load(open('model1.pkl', 'rb'))


#@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def get_categorical_columns(df):
    categorical_columns = []
    threshold = 1000
    object_columns = df.select_dtypes(include=['object']).columns

    unique_values = df[object_columns].nunique()
    print(unique_values)
    for k, v in unique_values.items():
        if v <= threshold:
            categorical_columns.append(k)

    return categorical_columns
    predictions= model.predict(df)
    st.write(predictions)

def main():
    #st.title('Insurance Fraud Predictor')
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file, na_values=["?", "MISSEDDATA","NA","-1", "MISSINGVAL", "-5", "MISSINGVALUE"])
        st.write(dataframe)
        if len(dataframe)>1:
            dataframe=dataframe.dropna()


# Removing "Cust" from CustomerId and "Location" from IncidentAddress.
        dataframe['CustomerID'] = dataframe['CustomerID'].str.replace('Cust', '')
        dataframe['IncidentAddress'] = dataframe['IncidentAddress'].str.replace('Location ', '').astype('int64').astype('int32')


        # Converting dates to numeric values.
        dataframe['DateOfIncident'] = pd.to_datetime(dataframe['DateOfIncident'], format='%d-%m-%Y')
        dataframe['DateOfIncident'] = dataframe['DateOfIncident'].astype('int64').astype('int32')
        dataframe['DateOfPolicyCoverage'] = pd.to_datetime(dataframe['DateOfPolicyCoverage'], format='%d-%m-%Y')
        dataframe['DateOfPolicyCoverage'] = dataframe['DateOfPolicyCoverage'].astype('int64').astype('int32')
        dataframe=dataframe.drop(['VehicleAttribute', 'VehicleAttributeDetails', 'Witnesses', "AuthoritiesContacted"], axis=1)
        categorical_columns = get_categorical_columns(dataframe)
        st.dataframe(dataframe)
        df= pd.get_dummies(dataframe, columns=categorical_columns)
        predictions= model.predict(df)
        predict_df = pd.DataFrame(columns = ['CustomerID', 'Fraud'])
        predict_df['CustomerID'] = df['CustomerID']
        predict_df['Fraud'] = True
        st.success(f"There are {len(df['CustomerID'])} Fraud Insuarance cases found: ")
        df = df.merge(predict_df[['CustomerID', 'Fraud']], on = 'CustomerID', how = 'left')
        #df_merged = df.merge(pd.DataFrame({'Fraud': predictions}), on='CustomerID')
        st.write(df)
        csv = convert_df(df)
        st.download_button(
            label="Download Data",
            data=csv,
            file_name='Fraud Cases.csv',
            mime='text/csv',
            )
        
main()
