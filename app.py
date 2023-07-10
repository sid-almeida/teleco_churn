import streamlit as st
import pandas as pd
import os
import pickle as pkl

# load the data
data = pd.read_csv('https://raw.githubusercontent.com/sid-almeida/teleco_churn/main/data_numerical.csv')

# train model
X_train = data.drop(['churn'], axis=1)
y_train = data['churn']

from sklearn.ensemble import GradientBoostingClassifier
gbclassifier = GradientBoostingClassifier()
model = gbclassifier.fit(X_train, y_train)

with st.sidebar:
    st.image("https://github.com/sid-almeida/teleco_churn/blob/main/Brainize%20Tech(1).png?raw=true", width=250)
    st.title("TeleCom Churn Prediction")
    choice = st.radio("**Navigation:**", ("About", "Prediction", "Batch Prediction"))
    st.info('**Note:** Please be aware that this application is intended solely for educational purposes. It is strongly advised against utilizing this tool for making any financial decisions.')


if choice == "About":
    st.title("""
        Client Churn Prediction
        This app predicts the churning of clients of a company of telecommunication.
        """)
    st.write('---')

    st.write('**About the App:**')
    st.write(
        'Utilizing a Gradient Boosting Regressor, the aforementioned approach employs a meticulously trained model encompassing 17 distinct features. Its primary objective is to predict the churning for a telecommunication company.')
    st.info(
        '**Note:** Please be aware that this application is intended solely for educational purposes. It is strongly advised against utilizing this tool for making any financial decisions.')
    st.write('---')

    st.write('**About the Data:**')
    st.write("""
        **Context**
        Predict behavior to retain customers. You can analyze all relevant customer data and develop focused customer retention programs.
        \n\n
        **Content**
        Each row represents a customer, each column contains customer’s attributes described on the column Metadata.
        \n\n
        **The data set includes information about:**
        Customers who left within the last month – the column is called Churn
        Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
        Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
        Demographic info about customers – gender, age range, and if they have partners and dependents
        \n\n
        To explore this type of models and learn more about the subject.
        To access the newer version from IBM access:
        For more information, please visit the [**IBM DataSet**](https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113)
    """)

    st.write('---')


if choice == "Prediction":
    st.title("Predicting the Churning of a Client")
    st.write('---')
    st.info('**Please, insert the information below:**')
    # created the options for the user to input the features

    gender_opt = st.radio('**Gender**', ['Male', 'Female'], horizontal=True)
    if gender_opt == 'Male':
        gender = 1
    elif gender_opt == 'Female':
        gender = 0
    st.write('---')

    senior_citizen_opt = st.radio('**Senior Citizen**', ['Yes', 'No'], horizontal=True)
    if senior_citizen_opt == 'Yes':
        senior_citizen = 1
    elif senior_citizen_opt == 'No':
        senior_citizen = 0
    st.write('---')

    partner_opt = st.radio('**Partner**', ['Yes', 'No'], horizontal=True)
    if partner_opt == 'Yes':
        partner = 1
    elif partner_opt == 'No':
        partner = 0
    st.write('---')

    dependents_opt = st.radio('**Dependents**', ['Yes', 'No'], horizontal=True)
    if dependents_opt == 'Yes':
        dependents = 1
    elif dependents_opt == 'No':
        dependents = 0
    st.write('---')

    tenure_months = st.number_input('**Tenure Months In The Company**', min_value=0, max_value=100, value=0, step=1)
    st.write('---')

    phone_service_opt = st.radio('**Phone Service**', ['Yes', 'No'], horizontal=True)
    if phone_service_opt == 'Yes':
        phone_service = 1
    elif phone_service_opt == 'No':
        phone_service = 0
    st.write('---')

    multiple_lines_opt = st.radio('**Multiple Lines**', ['Yes', 'No', 'No Phone Service'], horizontal=True)
    if multiple_lines_opt == 'Yes':
        multiple_lines = 2
    elif multiple_lines_opt == 'No':
        multiple_lines = 0
    elif multiple_lines_opt == 'No Phone Service':
        multiple_lines = 1
    st.write('---')

    internet_service_opt = st.radio('**Internet Service**', ['DSL', 'Fiber Optic', 'No'], horizontal=True)
    if internet_service_opt == 'DSL':
        internet_service = 0
    elif internet_service_opt == 'Fiber Optic':
        internet_service = 1
    elif internet_service_opt == 'No':
        internet_service = 2
    st.write('---')

    online_security_opt = st.radio('**Online Security**', ['Yes', 'No', 'No Internet Service'], horizontal=True)
    if online_security_opt == 'Yes':
        online_security = 2
    elif online_security_opt == 'No':
        online_security = 0
    elif online_security_opt == 'No Internet Service':
        online_security = 1
    st.write('---')

    online_backup_opt = st.radio('**Online Backup**', ['Yes', 'No', 'No Internet Service'], horizontal=True)
    if online_backup_opt == 'Yes':
        online_backup = 2
    elif online_backup_opt == 'No':
        online_backup = 0
    elif online_backup_opt == 'No Internet Service':
        online_backup = 1
    st.write('---')

    device_protection_opt = st.radio('**Device Protection**', ['Yes', 'No', 'No Internet Service'], horizontal=True)
    if device_protection_opt == 'Yes':
        device_protection = 2
    elif device_protection_opt == 'No':
        device_protection = 0
    elif device_protection_opt == 'No Internet Service':
        device_protection = 1
    st.write('---')

    tech_support_opt = st.radio('**Tech Support**', ['Yes', 'No', 'No Internet Service'], horizontal=True)
    if tech_support_opt == 'Yes':
        tech_support = 2
    elif tech_support_opt == 'No':
        tech_support = 0
    elif tech_support_opt == 'No Internet Service':
        tech_support = 1
    st.write('---')

    streaming_tv_opt = st.radio('**Streaming TV**', ['Yes', 'No', 'No Internet Service'], horizontal=True)
    if streaming_tv_opt == 'Yes':
        streaming_tv = 2
    elif streaming_tv_opt == 'No':
        streaming_tv = 0
    elif streaming_tv_opt == 'No Internet Service':
        streaming_tv = 1
    st.write('---')

    streaming_movies_opt = st.radio('**Streaming Movies**', ['Yes', 'No', 'No Internet Service'], horizontal=True)
    if streaming_movies_opt == 'Yes':
        streaming_movies = 2
    elif streaming_movies_opt == 'No':
        streaming_movies = 0
    elif streaming_movies_opt == 'No Internet Service':
        streaming_movies = 1
    st.write('---')

    contract_opt = st.radio('**Contract**', ['Month-to-Month', 'One Year', 'Two Year'], horizontal=True)
    if contract_opt == 'Month-to-Month':
        contract = 0
    elif contract_opt == 'One Year':
        contract = 1
    elif contract_opt == 'Two Year':
        contract = 2
    st.write('---')

    paperless_billing_opt = st.radio('**Paperless Billing**', ['Yes', 'No'], horizontal=True)
    if paperless_billing_opt == 'Yes':
        paperless_billing = 1
    elif paperless_billing_opt == 'No':
        paperless_billing = 0
    st.write('---')
    payment_method_opt = st.radio('**Payment Method**', ['Bank Transfer (Automatic)', 'Credit Card (Automatic)',
                                                            'Electronic Check', 'Mailed Check'], horizontal=True)
    if payment_method_opt == 'Bank Transfer (Automatic)':
        payment_method = 0
    elif payment_method_opt == 'Credit Card (Automatic)':
        payment_method = 1
    elif payment_method_opt == 'Electronic Check':
        payment_method = 2
    elif payment_method_opt == 'Mailed Check':
        payment_method = 3
    st.write('---')

    if st.button('Predict'):
        prediction = model.predict([[gender, senior_citizen, partner, dependents, tenure_months, phone_service,
                                     multiple_lines, internet_service, online_security, online_backup,
                                     device_protection, tech_support, streaming_tv, streaming_movies, contract,
                                     paperless_billing, payment_method]])
        prob = model.predict_proba([[gender, senior_citizen, partner, dependents, tenure_months, phone_service,
                                     multiple_lines, internet_service, online_security, online_backup,
                                     device_protection, tech_support, streaming_tv, streaming_movies, contract,
                                     paperless_billing, payment_method]])
        if prediction == 0:
            st.write('The chances of the client to churn **are small**')
            st.write(f'Probability of the client to keep consuming the service is of **{prob[0][0] * 100} %**')

        elif prediction == 1:
            st.write('The customer has a **high chance** of churning')
            st.write(f'Probability of the client to abandon the service is of **{prob[0][1] * 100} %**')

if choice == 'Batch Prediction':
    st.title('Batch Prediction')
    st.write('---')
    st.info('**Guide:** Please, upload the dataset with the predicting features.')
    st.write('---')
    # Create a file uploader to upload the dataset of predicting features
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df_pred = pd.read_csv(uploaded_file)
        st.write('---')
        st.write('**Dataset:**')
        st.write(df_pred)
        # Created a numerical dataframe and transform the uploaded categorical variables into numerical variables
        df_num = df_pred.copy()
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in df_num.columns:
            df_num[col] = le.fit_transform(df_num[col])
        # Create a button to predict the probability of bankruptcy
        if st.button("Predict"):
            # Predict the probability of churn using the model and create new column for the probability of churning
            df_pred['Churn Probability (%)'] = model.predict_proba(df_num)[:, 1] * 100
            # Create a csv file for the predicted probability of bankruptcy
            df_pred.to_csv('predicted.csv', index=False)
            # Create a success message
            st.success('The probability of churning was predicted successfully!')
            # Create a button to download the predicted probability of bankruptcy
            st.write('---')
            st.write('**Predicted Dataset:**')
            st.write(df_pred)
            # Create a button to download the dataset with the predicted probability of bankruptcy
            if st.download_button(label='Download Predicted Dataset', data=df_pred.to_csv(index=False),
                                  file_name='predicted.csv', mime='text/csv'):
                pass
        else:
            st.write('---')
            st.info('Click the button to predict the probability of churning of clients!')
    else:
        st.write('---')
