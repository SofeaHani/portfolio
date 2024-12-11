import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import streamlit as st

# Custom CSS for styling
custom_css = """
<style>

    .stApp {
        background-image: url('assets/backgroundimagenew.png'); /* Replace with your image URL */
        background-size: cover;
        background-position: center;
    }
    .stHeader {
        background-color: rgba(220, 240, 255, 0.9);
        padding: 10px;
        text-align: center;
        color: black;
        width: 100%;
        box-sizing: border-box;
        margin-bottom: 30px;
    }
    h1 {
        color: #fd0c74e4;
        font-size: 36px;
        margin-bottom: 5px;
        font-weight: 600;
    }
    .stSubheader {
        color: black;
        text-align: left;
        font-size: 22px;
        margin-top: 10px;
        margin-bottom: 10px;
        font-weight: 600;
    }
    .stFileUploader {
        background-color: #e0e0e0;
        padding: 10px;
        border-radius: 10px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    
    .custom-bold {
       color: #fd0c74e4;
        font-size: 20px;
        margin-bottom: 5px;
        font-weight: 600; 
    }
</style>
"""

# Apply the custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Header section
st.markdown(
    "<div class='stHeader'><h1>Traditional Malay Women Fashion Product Recommendation System</h1>"
    "<p class='custom-bold'>Find Your Favourite {Baju Kurung, Baju Kebaya, Tudung}</p></div>",
    unsafe_allow_html=True
)

# Load image features and filenames
Image_features = pkl.load(open('Image_features.pkl', 'rb'))
filenames = pkl.load(open('filenames.pkl', 'rb'))

# Normalize features
Image_features_normalized = normalize(Image_features, axis=1)

# Function to extract augmented and robust features
def extract_fine_tuned_features(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)

    # Augmentation for robust feature extraction
    augmented_features = []
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    for batch in datagen.flow(img_preprocess, batch_size=1):
        feature = model.predict(batch)
        augmented_features.append(feature.flatten())
        if len(augmented_features) >= 5:  # Limit number of augmentations
            break
    return np.mean(augmented_features, axis=0)  # Averaging features for stability

# Load the ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([model, GlobalMaxPooling2D()])

# Nearest neighbors model using cosine similarity
neighbors = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='cosine')
neighbors.fit(Image_features_normalized)

# Prompt user before the file upload section
st.markdown(
    "<p style='color: #fd0c74e4; font-size: 18px; font-weight: 600;'>Please upload images of Baju Kurung, Baju Kebaya, or Tudung only.</p>",
    unsafe_allow_html=True
)

# Image upload section
st.markdown('<div class="file-upload-label">Upload Images</div>', unsafe_allow_html=True)
upload_files = st.file_uploader("", type=["jpg", "png"], accept_multiple_files=True)

if upload_files:
    for upload_file in upload_files:
        # Save the uploaded file to disk
        image_path = os.path.join('upload', upload_file.name)
        with open(image_path, 'wb') as f:
            f.write(upload_file.getbuffer())

        # Display the uploaded image
        st.markdown("<div class='stSubheader'>Uploaded Image</div>", unsafe_allow_html=True)
        st.image(upload_file, caption="Uploaded Image", width=300)

        # Extract features from the uploaded image
        input_img_features = extract_fine_tuned_features(image_path, model)
        input_img_features = np.expand_dims(input_img_features, axis=0)  # Ensure correct shape

        # Find the nearest neighbors (excluding the uploaded image from the recommendations)
        distances, indices = neighbors.kneighbors(input_img_features)

        st.markdown("<div class='stSubheader'>Recommended Images</div>", unsafe_allow_html=True)

        # Exclude the uploaded image from the recommendations list
        recommended_images = []
        recommended_similarities = []

        for idx in range(len(indices[0])):
            img_index = indices[0][idx]
            
            # Compare only the file name (not the full path) to exclude the uploaded image
            if os.path.basename(filenames[img_index]) != upload_file.name:  # Compare only the file name
                recommended_images.append(filenames[img_index])
                recommended_similarities.append(1 / (1 + distances[0][idx]) * 100)

        # Display recommended images and their similarity
        cols = st.columns(5)
        for idx, col in enumerate(cols):
            if idx < len(recommended_images):  # Display all recommendations
                img_path = recommended_images[idx]
                similarity = recommended_similarities[idx]

                # Display the recommended image
                with col:
                    img_data = image.load_img(img_path)
                    st.image(img_data, caption=f"Similarity Percentage: {similarity:.2f}%", use_column_width=True)
