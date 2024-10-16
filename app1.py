import streamlit as st 
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import base64
import altair as alt
import pandas as pd

# Function to load and convert the image to base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Load your saved Keras model
model_path = r"C:\Users\sansk\Desktop\New folder\terrain_classifier_eurosat.keras"  # Update this path
try:
    model = load_model(model_path)
    st.success(f"Model loaded successfully from {model_path}")
except Exception as e:
    st.error(f"Error loading the model: {e}")

# Load your logo and convert it to base64 format
logo_path = r"C:\Users\sansk\Downloads\logo (1).png" 
logo_base64 = get_base64_of_bin_file(logo_path)

# Centering the logo using HTML and CSS
st.markdown(
    f"""
    <div style="display: flex; justify-content: center;">
        <img src="data:image/jpeg;base64,{logo_base64}" alt="logo" style="width:450px;">
    </div>
    """,
    unsafe_allow_html=True
)

# Define the function to preprocess the image
def preprocess_image(image, target_size=(64, 64)):
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    elif image.mode != 'RGB':
        image = image.convert('RGB')

    image = image.resize(target_size)
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    
    return image

# Define the prediction function
def make_prediction(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

# Function to map predicted classes to emojis
def get_emoji_for_class(predicted_class):
    emoji_dict = {
        'Annual Crop': '🌾',
        'Forest': '🌲',
        'Herbaceous Vegetation': '🌿',
        'Highway': '🛣️',
        'Industrial': '🏭',
        'Pasture': '🐄',
        'Permanent Crop': '🌳',
        'Residential': '🏠',
        'River': '🌊',
        'Sea Lake': '⛵'
    }
    return emoji_dict.get(predicted_class, '🌍')  # Default to globe emoji if not found

# Function to display a bar chart using Altair
def display_confidence_bar_chart(prediction, class_names):
    confidence_scores = prediction[0]
    
    # Create a DataFrame for Altair
    data = pd.DataFrame({
        'Class': class_names,
        'Confidence': confidence_scores
    })

    # Create an Altair bar chart
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X('Class', sort=None, title='Terrain Class'),
        y=alt.Y('Confidence', title='Confidence Score'),
        color=alt.Color('Class', legend=None),
        tooltip=['Class', 'Confidence']
    ).properties(
        width=600,
        height=400,
        title="Model Confidence for Each Terrain Class"
    )

    st.altair_chart(chart, use_container_width=True)

# Streamlit app UI
st.markdown("<h1 style='text-align: center;'>🌏 Terrain Classification App 🌏</h1>", unsafe_allow_html=True)

with st.expander("How to Use", expanded=False):
    st.markdown(
        """
        1. Upload a terrain image in JPG, JPEG, or PNG format.
        2. The app will process the image and display the predicted terrain class.
        3. Confidence scores for each terrain type will be shown.
        """
    )

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.header("Classified image as:", divider=True)

    # Make the prediction
    prediction = make_prediction(image)
    
    # Define class names (update these based on your model’s output)
    class_names = ['Annual Crop', 'Forest', 'Herbaceous Vegetation', 'Highway', 'Industrial',
                   'Pasture', 'Permanent Crop', 'Residential', 'River', 'Sea Lake']
    
    predicted_class = class_names[np.argmax(prediction)]  # Get predicted class
    emoji = get_emoji_for_class(predicted_class)  # Get emoji for the predicted class
    
    # Customized Predictive Model Output - Improved Design
    st.markdown(
        f"""
        <div style="
            padding: 20px;
            border: 3px solid #feb47b;
            border-radius: 15px;
            background: linear-gradient(135deg, #ff7e5f, #feb47b);
            text-align: center;
            font-size: 24px;
            color: white;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            margin: 20px 0;
        ">
            <h2 style="margin-bottom: 10px;">Predicted Class</h2>
            <strong>{predicted_class} {emoji}</strong>
        </div>
        """, unsafe_allow_html=True
    )

    # Display the Altair bar chart
    display_confidence_bar_chart(prediction, class_names)
