import streamlit as st
from google.cloud import vision
import pandas as pd
import io
import os

# Set the environment variable to the path of your JSON key file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\tanuj\Desktop\anomalydetection-433310-2e3c08765775.json"

client = vision.ImageAnnotatorClient()

def detect_text(image_content):
    image = vision.Image(content=image_content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    if not texts:
        return "No text detected"
    detected_text = texts[0].description
    if response.error.message:
        raise Exception(f'{response.error.message}')
    return detected_text

def text_to_dataframe(detected_text):
    data = []
    for line in detected_text.splitlines():
        data.append(line.split())
    df = pd.DataFrame(data[1:], columns=data[0])
    return df

def detect_anomalies(df):
    anomalies = []
    for column in df.select_dtypes(include=[float, int]):
        std_dev = df[column].std()
        mean = df[column].mean()
        threshold = 2 * std_dev
        outliers = df[(df[column] > mean + threshold) | (df[column] < mean - threshold)]
        if not outliers.empty:
            anomalies.append(f"Anomalies detected in {column}: {outliers.to_dict(orient='records')}")
    return anomalies

def generate_analysis(anomalies):
    if not anomalies:
        return "No anomalies detected."
    else:
        analysis = "\n".join(anomalies)
        return analysis

st.title("Power BI Report Anomaly Detector")
st.write("Upload an image of your Power BI report, and the app will detect any anomalies.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image_content = uploaded_file.read()
    detected_text = detect_text(image_content)
    df = text_to_dataframe(detected_text)
    anomalies = detect_anomalies(df)
    analysis_text = generate_analysis(anomalies)
    st.image(uploaded_file, caption='Uploaded Power BI Report', use_column_width=True)
    st.write("Extracted Data:")
    st.dataframe(df)
    st.write("Anomaly Detection Result:")
    st.text(analysis_text)
