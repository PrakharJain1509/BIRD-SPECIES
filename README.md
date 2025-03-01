# Bird Species Detection

## Overview
This project utilizes both **audio and image** inputs to identify bird species. By leveraging deep learning models, it can classify birds based on their **vocalizations (audio)** and **appearance (image)**. The system is implemented using **Streamlit** for an interactive web-based interface.

## Features
- **Audio-based Bird Classification**: Uses a trained deep learning model to classify birds based on their vocal sounds.
- **Image-based Bird Classification**: Identifies bird species from images using a CNN-based classifier.
- **Pre-trained Model Weights**: The repository includes pre-trained model weights for both image and audio classification.
- **Streamlit Web Interface**: A user-friendly UI to upload images and audio files for classification.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/PrakharJain1509/BIRD-SPECIES.git
   cd Bird-Species-Detection
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Alternatively, you can install them manually:
   ```bash
   pip install numpy os-joblib librosa torch timm pandas
   ```

## Running the Application
After installing the dependencies, you can run the application using:
```bash
streamlit run stream.py
```
This will launch the web-based interface in your browser.

## Directory Structure
```
.
├── README.md
├── stream.py                 # Streamlit main script
├── audio                     # Audio processing module
│   ├── audio_main.py
│   ├── new_audio_main.py
│   ├── models_weights        # Audio classification model weights
│   ├── data                  # Contains audio files and metadata
│   ├── listOfBirdNames.txt
│   ├── commonNames.txt
├── image                     # Image processing module
│   ├── image_main.py
│   ├── new_image_main.py
│   ├── test                  # Sample test images
│   ├── weights               # Image classification model weights
├── common_names.txt          # Common bird names
└── requirements.txt          # Dependencies file
```

## Models Used
- **Audio Model**: `effnet_seg20_80low.ckpt` (EfficientNet-based model for audio classification)
- **Image Model**: `bird-resnet34.pth` (ResNet-34 model for image classification)

## Input Data
- **Audio Files**: Stored in `audio/data/AUDIO_FILES/`
- **Image Files**: Stored in `image/test/`

## Output
After providing an image and audio file, the model will return the predicted bird species name.
