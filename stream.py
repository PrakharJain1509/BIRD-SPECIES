import streamlit as st
import os
import torch

# Importing modules based on the folder structure
import audio.new_audio_main as audio
import image.new_image_main as image

# Initialize the audio predictor
predictor = audio.AudioPredictor()

# Define model path for image classification
image_model_path = os.path.join(os.path.dirname(__file__), "image", "weights", "bird-resnet34best.pth")

# Check if image model exists
if not os.path.exists(image_model_path):
    st.error(f"Image model file not found: {image_model_path}")
else:
    st.success("Image model loaded successfully!")

# Streamlit app setup
st.title("Bird Species Prediction App")
st.write("Upload an audio file and/or an image file to get predictions.")

# File upload widgets
uploaded_audio = st.file_uploader("Upload an audio file:", type=["ogg", "mp3", "wav"])
uploaded_image = st.file_uploader("Upload an image file:", type=["jpg", "jpeg", "png"])

# Button to trigger prediction
if st.button("Predict"):
    if uploaded_audio:
        # Save the uploaded audio file temporarily
        audio_file_path = f"temp_{uploaded_audio.name}"
        with open(audio_file_path, "wb") as f:
            f.write(uploaded_audio.getbuffer())

        # Perform prediction
        label, score = predictor.predict_single_audio(audio_file_path)

        # Display results
        st.write(f"# Prediction Result")
        st.write(f"## Label: **{label}**")
        #st.write(f"## Score: **{score:.2f}**")

        # Optionally, clean up the temporary file
        import os
        os.remove(audio_file_path)
    else:
        st.warning("Please upload an audio file before predicting.")
