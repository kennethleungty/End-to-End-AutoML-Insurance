# =========================================
# H2O AutoML Training with MLflow Tracking
# Author: Kenneth Leung
# =========================================
# Command to execute script: streamlit run ui.py

import streamlit as st
import requests
import pandas as pd
import io
import json

st.title('End-to-End AutoML Project: Insurance Cross-Sell')

# Set FastAPI endpoint
endpoint = 'http://localhost:8000/predict'
st.text('''Author: Kenneth Leung''') # description and instructions

test_csv = st.file_uploader('', type=['csv','xlsx'], accept_multiple_files=False)

# Upon upload of file
if test_csv:
    test_df = pd.read_csv(test_csv)
    st.subheader('Sample of Uploaded Dataset')
    st.write(test_df.head())

    # Convert dataframe to BytesIO object (for parsing as file into FastAPI later)
    test_bytes_obj = io.BytesIO()
    test_df.to_csv(test_bytes_obj, index=False)  # write to BytesIO buffer
    test_bytes_obj.seek(0) # Reset pointer to avoid EmptyDataError

    files = {"file": ('test_dataset.csv', test_bytes_obj, "multipart/form-data")}

    # Upon click of button
    if st.button('Start Prediction'):
        if len(test_df) == 0:
            st.write("Please upload a valid test dataset!")  # handle case with no image
        else:
            with st.spinner('Prediction in Progress. Please Wait...'):
                # import time
                # time.sleep(3)
                output = requests.post(endpoint, 
                                       files=files,
                                       timeout=8000)
            st.success('Success! Click Download button below to get prediction results (in JSON format)')
            st.download_button(
                label='Download',
                data=json.dumps(output.json()), # Download as JSON file object
                file_name='automl_prediction_results.json'
            )
