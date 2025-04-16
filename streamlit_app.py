import streamlit as st
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained model
model = load_model('model.h5')
# Define the image size for model input
IMG_SIZE = (128, 128)

# Set the app title and sidebar
# Add custom CSS for aesthetics
st.markdown(
    """
    <style>
    .title {
        margin-top:0px;
        color: #FF5733; /* Coral */
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    
    .text {
        color: #EFA18A; /* Slate Gray */
        font-size: 20px;
        font-weight: italic;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .uploaded-image {
        width: 100%;
        max-width: 500px;
        margin-bottom: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .prediction {
        color: #FF5733; /* Coral */
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
        text-align: center;
    }
    
    .confidence {
        color: #FF5600; /* Coral */
        font-size: 18px;
        margin-bottom: 20px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# st.set_option('deprecation.showPyplotGlobalUse', False)

# Display the title
st.markdown("<h1 class='title'>Alzheimer's Disease Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h1 class='text'>Alzheimer's Disease Prediction is a web application that utilizes a pre-trained deep learning model to predict the presence of Alzheimer's disease based on uploaded brain ultrasound images. Users can upload an image through the sidebar and the app will process the image using the trained model.</h1>", unsafe_allow_html=True)

st.sidebar.title("Upload Image")
st.sidebar.markdown("Please upload an image.")


def preprocess_image(image):
    # plt.imsave('image2.jpg', image)
    img_array = np.array(image)
    rgb_image = np.repeat(img_array[:, :, np.newaxis], 3, axis=2)
    img = Image.fromarray(img_array.astype('uint8'))


    # img.save('output1.jpg')  # Save the image to a file

    img_array = np.expand_dims(rgb_image, axis=0)
    return img_array





def predict(image):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    # print(prediction)
    predicted_idx = np.argmax(prediction, axis=1)[0]
    return predicted_idx

# Display the file uploader
uploaded_file = st.sidebar.file_uploader(label="", type=['jpg', 'jpeg', 'png'])

# Make predictions and display the result
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    predicted_idx = predict(image)
    
    class_labels = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']
    predicted_label = class_labels[predicted_idx]

    st.markdown(f"<p class='prediction'>Prediction: {predicted_label}</p>", unsafe_allow_html=True)

else:
    st.sidebar.write("Please upload an image.")


import streamlit as st
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import numpy as np

# Load the pre-trained model
model = load_model('model.h5')

# Define the expected image size for the model
IMG_SIZE = (128, 128)

# Set the app title and sidebar with custom styling
st.markdown(
    """
    <style>
    .title {
        color: #FF5733;
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    
    .text {
        color: #EFA18A;
        font-size: 20px;
        font-weight: italic;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .prediction {
        color: #FF5733;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the title
st.markdown("<h1 class='title'>Alzheimer's Disease Prediction</h1>", unsafe_allow_html=True)
st.markdown(
    "<h1 class='text'>Upload a brain MRI image, and the AI will predict the stage of Alzheimer's disease.</h1>",
    unsafe_allow_html=True
)

st.sidebar.title("Upload Image")
st.sidebar.markdown("Please upload a brain MRI scan.")

# ✅ **Fixed Image Preprocessing Function**
def preprocess_image(image):
    """Preprocesses the image for model prediction."""
    image = image.convert("RGB")  # Ensure 3 color channels
    image = image.resize(IMG_SIZE)  # Resize to model's expected input size
    img_array = np.array(image, dtype=np.float32) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# ✅ **Fixed Prediction Function**
def predict(image):
    """Runs the model prediction on the preprocessed image."""
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    predicted_idx = np.argmax(prediction, axis=1)[0]
    return predicted_idx

# File uploader in sidebar
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    try:
        # Open and display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Run prediction
        predicted_idx = predict(image)

        # Class labels
        class_labels = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']
        predicted_label = class_labels[predicted_idx]

        # Display result
        st.markdown(f"<p class='prediction'>Prediction: {predicted_label}</p>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error processing the image: {e}")

else:
    st.sidebar.write("Please upload an image.")
# import streamlit as st
# import tensorflow as tf
# from keras.models import load_model
# from PIL import Image
# import numpy as np
# import os

# # App configuration
# st.set_page_config(page_title="Alzheimer's Disease Prediction", layout="wide")

# # Define paths and configurations
# MODEL_PATH = 'model.h5'  # Path to your model
# IMG_SIZE = (128, 128)    # Expected image size for the model

# # Custom class labels - update these to match your new dataset categories
# CLASS_LABELS = ['Non Demented', 'Mild Demented', 'Moderate Demented', 'Very Mild Demented']

# # Set the app title and sidebar with custom styling
# st.markdown(
#     """
#     <style>
#     .title {
#         color: #FF5733;
#         font-size: 40px;
#         font-weight: bold;
#         text-align: center;
#         margin-bottom: 10px;
#     }
    
#     .text {
#         color: #EFA18A;
#         font-size: 20px;
#         font-weight: italic;
#         text-align: center;
#         margin-bottom: 20px;
#     }
    
#     .prediction {
#         color: #FF5733;
#         font-size: 24px;
#         font-weight: bold;
#         margin-bottom: 10px;
#         text-align: center;
#     }
    
#     .confidence {
#         color: #4682B4;
#         font-size: 18px;
#         margin-bottom: 20px;
#         text-align: center;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Display the title
# st.markdown("<h1 class='title'>Alzheimer's Disease Prediction</h1>", unsafe_allow_html=True)
# st.markdown(
#     "<h1 class='text'>Upload a brain MRI image, and the AI will predict the stage of Alzheimer's disease.</h1>",
#     unsafe_allow_html=True
# )

# # Sidebar
# st.sidebar.title("Model and Dataset Configuration")

# # Option to upload custom model
# uploaded_model = st.sidebar.file_uploader("Upload your own model (optional)", type=['h5'])
# if uploaded_model:
#     with open('uploaded_model.h5', 'wb') as f:
#         f.write(uploaded_model.getbuffer())
#     MODEL_PATH = 'uploaded_model.h5'
#     st.sidebar.success("Custom model uploaded successfully!")

# # Option to update class labels
# custom_labels = st.sidebar.text_area("Custom class labels (one per line, optional)", 
#                                     "\n".join(CLASS_LABELS))
# if custom_labels:
#     CLASS_LABELS = [label.strip() for label in custom_labels.split('\n') if label.strip()]

# # Load the model
# @st.cache_resource
# def load_prediction_model(model_path):
#     """Load the model with caching to improve performance"""
#     try:
#         model = load_model(model_path)
#         return model, None
#     except Exception as e:
#         return None, str(e)

# # Load model with status message
# with st.spinner("Loading model..."):
#     model, error = load_prediction_model(MODEL_PATH)
    
#     if error:
#         st.error(f"Error loading model: {error}")
#     else:
#         st.sidebar.success("Model loaded successfully!")

# # Image preprocessing function
# def preprocess_image(image, target_size=IMG_SIZE):
#     """Preprocesses the image for model prediction."""
#     image = image.convert("RGB")  # Ensure 3 color channels
#     image = image.resize(target_size)  # Resize to model's expected input size
#     img_array = np.array(image, dtype=np.float32) / 255.0  # Normalize pixel values
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#     return img_array

# # Prediction function
# def predict(model, image, class_labels):
#     """Runs the model prediction on the preprocessed image."""
#     img_array = preprocess_image(image)
#     prediction = model.predict(img_array)
    
#     # Get prediction probabilities
#     prediction_probs = prediction[0]
#     predicted_idx = np.argmax(prediction_probs)
#     confidence = prediction_probs[predicted_idx] * 100
    
#     return {
#         'class_idx': predicted_idx,
#         'class_name': class_labels[predicted_idx] if predicted_idx < len(class_labels) else "Unknown",
#         'confidence': confidence,
#         'all_probs': prediction_probs
#     }

# # Main content
# st.sidebar.title("Upload Image")
# uploaded_file = st.sidebar.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

# # Display predictions if model is loaded and image is uploaded
# if model is not None and uploaded_file is not None:
#     try:
#         # Open and display the uploaded image
#         col1, col2 = st.columns([1, 1])
        
#         with col1:
#             image = Image.open(uploaded_file)
#             st.image(image, caption='Uploaded Image', use_column_width=True)
        
#         # Run prediction
#         with st.spinner("Analyzing image..."):
#             result = predict(model, image, CLASS_LABELS)
            
#         # Display result
#         with col2:
#             st.markdown(f"<p class='prediction'>Prediction: {result['class_name']}</p>", 
#                        unsafe_allow_html=True)
#             st.markdown(f"<p class='confidence'>Confidence: {result['confidence']:.2f}%</p>", 
#                        unsafe_allow_html=True)
            
#             # Create bar chart for confidence levels
#             chart_data = {
#                 "Class": CLASS_LABELS,
#                 "Probability (%)": [prob * 100 for prob in result['all_probs']]
#             }
            
#             st.subheader("Prediction Confidence by Class")
#             st.bar_chart(data=chart_data, x="Class", y="Probability (%)")
            
#     except Exception as e:
#         st.error(f"Error processing the image: {e}")
# elif uploaded_file is not None:
#     st.warning("Please wait for the model to load or upload a valid model.")
# else:
#     st.info("Please upload an MRI brain scan image to get a prediction.")

# # Add information section
# with st.sidebar.expander("About"):
#     st.write("""
#     This application uses a deep learning model to predict the stage of Alzheimer's disease 
#     from brain MRI scans. You can use the default model or upload your own custom trained model.
    
#     You can also customize the class labels to match your specific dataset categories.
    
#     For accurate results, please upload a clear brain MRI scan image.
#     """)
