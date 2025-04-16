# import streamlit as st
# import tensorflow as tf
# from keras.models import load_model
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt

# # Load the pre-trained model
# model = load_model('model.h5')
# # Define the image size for model input
# IMG_SIZE = (128, 128)

# # Set the app title and sidebar
# # Add custom CSS for aesthetics
# st.markdown(
#     """
#     <style>
#     .title {
#         margin-top:0px;
#         color: #FF5733; /* Coral */
#         font-size: 40px;
#         font-weight: bold;
#         text-align: center;
#         margin-bottom: 10px;
#     }
    
#     .text {
#         color: #EFA18A; /* Slate Gray */
#         font-size: 20px;
#         font-weight: italic;
#         text-align: center;
#         margin-bottom: 20px;
#     }
    
#     .uploaded-image {
#         width: 100%;
#         max-width: 500px;
#         margin-bottom: 20px;
#         border-radius: 10px;
#         box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
#     }
    
#     .prediction {
#         color: #FF5733; /* Coral */
#         font-size: 24px;
#         font-weight: bold;
#         margin-bottom: 10px;
#         text-align: center;
#     }
    
#     .confidence {
#         color: #FF5600; /* Coral */
#         font-size: 18px;
#         margin-bottom: 20px;
#         text-align: center;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # st.set_option('deprecation.showPyplotGlobalUse', False)

# # Display the title
# st.markdown("<h1 class='title'>Alzheimer's Disease Prediction</h1>", unsafe_allow_html=True)
# st.markdown("<h1 class='text'>Alzheimer's Disease Prediction is a web application that utilizes a pre-trained deep learning model to predict the presence of Alzheimer's disease based on uploaded brain ultrasound images. Users can upload an image through the sidebar and the app will process the image using the trained model.</h1>", unsafe_allow_html=True)

# st.sidebar.title("Upload Image")
# st.sidebar.markdown("Please upload an image.")


# def preprocess_image(image):
#     # plt.imsave('image2.jpg', image)
#     img_array = np.array(image)
#     rgb_image = np.repeat(img_array[:, :, np.newaxis], 3, axis=2)
#     img = Image.fromarray(img_array.astype('uint8'))


#     # img.save('output1.jpg')  # Save the image to a file

#     img_array = np.expand_dims(rgb_image, axis=0)
#     return img_array





# def predict(image):
#     img_array = preprocess_image(image)
#     prediction = model.predict(img_array)
#     # print(prediction)
#     predicted_idx = np.argmax(prediction, axis=1)[0]
#     return predicted_idx

# # Display the file uploader
# uploaded_file = st.sidebar.file_uploader(label="", type=['jpg', 'jpeg', 'png'])

# # Make predictions and display the result
# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption='Uploaded Image', use_column_width=True)
    
#     predicted_idx = predict(image)
    
#     class_labels = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']
#     predicted_label = class_labels[predicted_idx]

#     st.markdown(f"<p class='prediction'>Prediction: {predicted_label}</p>", unsafe_allow_html=True)

# else:
#     st.sidebar.write("Please upload an image.")


import streamlit as st
# import tensorflow as tf
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

# File: train_model.py
# This script downloads and processes the dataset, then trains and saves the model

# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import kaggle
# import zipfile
# import shutil

# # Download dataset from Kaggle (you'll need kaggle.json in ~/.kaggle/)
# def download_dataset():
#     print("Downloading dataset from Kaggle...")
    
#     # Create ~/.kaggle directory if it doesn't exist
#     os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
    
#     # Check for kaggle.json
#     if not os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json')):
#         print("Please place your kaggle.json file in ~/.kaggle/ directory")
#         print("You can download it from your Kaggle account settings")
#         return False
    
#     # Set permissions for kaggle.json
#     os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)
    
#     # Download the dataset
#     kaggle.api.dataset_download_files(
#         'uraninjo/augmented-alzheimer-mri-dataset',
#         path='.',
#         unzip=True
#     )
    
#     print("Dataset downloaded successfully!")
#     return True

# # Prepare dataset directory structure
# def prepare_dataset():
#     print("Preparing dataset...")
    
#     dataset_dir = 'alzheimer_dataset'
    
#     # Define paths
#     base_path = os.path.join(dataset_dir, 'Alzheimer_s Dataset')
#     train_path = os.path.join(base_path, 'train')
#     test_path = os.path.join(base_path, 'test')
    
#     # Check if dataset exists
#     if not os.path.exists(base_path):
#         print(f"Dataset not found at {base_path}")
#         return False
    
#     return train_path, test_path

# # Create and train the model
# def build_and_train_model(train_path, test_path, img_size=(128, 128), batch_size=32, epochs=20):
#     print("Building and training model...")
    
#     # Data augmentation for training
#     train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         rotation_range=20,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         fill_mode='nearest'
#     )
    
#     # Only rescaling for validation
#     test_datagen = ImageDataGenerator(rescale=1./255)
    
#     # Load training data
#     train_generator = train_datagen.flow_from_directory(
#         train_path,
#         target_size=img_size,
#         batch_size=batch_size,
#         class_mode='categorical',
#         shuffle=True
#     )
    
#     # Load test data
#     test_generator = test_datagen.flow_from_directory(
#         test_path,
#         target_size=img_size,
#         batch_size=batch_size,
#         class_mode='categorical',
#         shuffle=False
#     )
    
#     # Get class names and number of classes
#     class_names = list(train_generator.class_indices.keys())
#     num_classes = len(class_names)
    
#     print(f"Classes: {class_names}")
    
#     # Build the model
#     model = Sequential([
#         # First convolutional block
#         Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(img_size[0], img_size[1], 3)),
#         BatchNormalization(),
#         Conv2D(32, (3, 3), activation='relu', padding='same'),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
#         Dropout(0.25),
        
#         # Second convolutional block
#         Conv2D(64, (3, 3), activation='relu', padding='same'),
#         BatchNormalization(),
#         Conv2D(64, (3, 3), activation='relu', padding='same'),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
#         Dropout(0.25),
        
#         # Third convolutional block
#         Conv2D(128, (3, 3), activation='relu', padding='same'),
#         BatchNormalization(),
#         Conv2D(128, (3, 3), activation='relu', padding='same'),
#         BatchNormalization(),
#         MaxPooling2D((2, 2)),
#         Dropout(0.25),
        
#         # Flatten and dense layers
#         Flatten(),
#         Dense(512, activation='relu'),
#         BatchNormalization(),
#         Dropout(0.5),
#         Dense(num_classes, activation='softmax')
#     ])
    
#     # Compile the model
#     model.compile(
#         optimizer=Adam(learning_rate=0.0001),
#         loss='categorical_crossentropy',
#         metrics=['accuracy']
#     )
    
#     # Model summary
#     model.summary()
    
#     # Callbacks
#     early_stopping = EarlyStopping(
#         monitor='val_loss',
#         patience=10,
#         restore_best_weights=True
#     )
    
#     checkpoint = ModelCheckpoint(
#         'model.h5',
#         monitor='val_accuracy',
#         save_best_only=True,
#         mode='max',
#         verbose=1
#     )
    
#     # Train the model
#     history = model.fit(
#         train_generator,
#         epochs=epochs,
#         validation_data=test_generator,
#         callbacks=[early_stopping, checkpoint]
#     )
    
#     # Save class names for later use
#     np.save('class_names.npy', np.array(class_names))
    
#     # Evaluate the model
#     test_loss, test_acc = model.evaluate(test_generator)
#     print(f"Test accuracy: {test_acc:.4f}")
    
#     # Plot training history
#     plt.figure(figsize=(12, 4))
    
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['accuracy'])
#     plt.plot(history.history['val_accuracy'])
#     plt.title('Model Accuracy')
#     plt.ylabel('Accuracy')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Validation'], loc='upper left')
    
#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('Model Loss')
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Validation'], loc='upper left')
    
#     plt.tight_layout()
#     plt.savefig('training_history.png')
    
#     return model, class_names

# # Main execution
# if __name__ == "__main__":
#     # Download dataset if needed
#     if not os.path.exists('alzheimer_dataset'):
#         success = download_dataset()
#         if not success:
#             print("Failed to download dataset. Please check your kaggle.json and internet connection.")
#             exit(1)
    
#     # Prepare the dataset
#     result = prepare_dataset()
#     if not result:
#         print("Failed to prepare dataset.")
#         exit(1)
    
#     train_path, test_path = result
    
#     # Build and train the model
#     model, class_names = build_and_train_model(train_path, test_path)
    
#     print("Training complete! Model saved as 'model.h5'")


# # File: app.py
# # This is your Streamlit application with improvements

# import streamlit as st
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from PIL import Image
# import numpy as np
# import os
# import matplotlib.pyplot as plt

# # Check if model exists, if not suggest running the training script
# if not os.path.exists('model.h5'):
#     st.error("Model file 'model.h5' not found. Please run the training script first: 'python train_model.py'")
#     st.stop()

# # Load the pre-trained model
# model = load_model('model.h5')

# # Load class names
# if os.path.exists('class_names.npy'):
#     class_labels = np.load('class_names.npy', allow_pickle=True).tolist()
# else:
#     # Fallback class labels if file not found
#     class_labels = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

# # Define the expected image size for the model
# IMG_SIZE = (128, 128)

# # Set the app title and sidebar with custom styling
# st.set_page_config(
#     page_title="Alzheimer's Disease Detection",
#     page_icon="🧠",
#     layout="wide"
# )

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
    
#     .subtitle {
#         color: #EFA18A;
#         font-size: 20px;
#         font-style: italic;
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
#         color: #333333;
#         font-size: 18px;
#         margin-bottom: 20px;
#         text-align: center;
#     }
    
#     .disclaimer {
#         color: #777777;
#         font-size: 14px;
#         font-style: italic;
#         text-align: center;
#         margin-top: 30px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Main layout
# col1, col2 = st.columns([1, 2])

# with col1:
#     st.sidebar.title("About")
#     st.sidebar.info(
#         "This application uses deep learning to predict "
#         "the stage of Alzheimer's disease from brain MRI scans. "
#         "The model was trained on the Kaggle Augmented Alzheimer's MRI Dataset."
#     )
    
#     st.sidebar.title("Upload Image")
#     st.sidebar.markdown("Please upload a brain MRI scan.")
    
#     # File uploader
#     uploaded_file = st.sidebar.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
#     # Add option for sample images if available
#     st.sidebar.markdown("---")
#     st.sidebar.markdown("### Options")
    
#     # Show training history if available
#     if os.path.exists('training_history.png'):
#         if st.sidebar.checkbox("Show Training History"):
#             st.sidebar.image('training_history.png', caption='Model Training History', use_column_width=True)

# with col2:
#     # Display the title
#     st.markdown("<h1 class='title'>Alzheimer's Disease Prediction</h1>", unsafe_allow_html=True)
#     st.markdown(
#         "<p class='subtitle'>Upload a brain MRI image, and the AI will predict the stage of Alzheimer's disease.</p>",
#         unsafe_allow_html=True
#     )
    
#     # Image preprocessing function
#     def preprocess_image(image):
#         """Preprocesses the image for model prediction."""
#         image = image.convert("RGB")  # Ensure 3 color channels
#         image = image.resize(IMG_SIZE)  # Resize to model's expected input size
#         img_array = np.array(image, dtype=np.float32) / 255.0  # Normalize pixel values
#         img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#         return img_array
    
#     # Prediction function
#     def predict(image):
#         """Runs the model prediction on the preprocessed image."""
#         img_array = preprocess_image(image)
#         prediction = model.predict(img_array)
#         predicted_idx = np.argmax(prediction, axis=1)[0]
#         confidence = float(prediction[0, predicted_idx]) * 100
#         return predicted_idx, confidence, prediction[0]
    
#     # Create two columns for the image and results
#     if uploaded_file is not None:
#         try:
#             # Open and display the uploaded image
#             image = Image.open(uploaded_file)
#             st.image(image, caption='Uploaded MRI Scan', use_column_width=True)
            
#             # Add a spinner during prediction
#             with st.spinner('Analyzing image...'):
#                 # Run prediction
#                 predicted_idx, confidence, all_probs = predict(image)
#                 predicted_label = class_labels[predicted_idx]
            
#             # Display result
#             st.markdown(f"<p class='prediction'>Diagnosis: {predicted_label}</p>", unsafe_allow_html=True)
#             st.markdown(f"<p class='confidence'>Confidence: {confidence:.2f}%</p>", unsafe_allow_html=True)
            
#             # Display all class probabilities as a bar chart
#             st.subheader("Probability Distribution")
#             fig, ax = plt.subplots(figsize=(10, 5))
#             y_pos = np.arange(len(class_labels))
#             ax.barh(y_pos, all_probs * 100, align='center')
#             ax.set_yticks(y_pos)
#             ax.set_yticklabels(class_labels)
#             ax.set_xlabel('Probability (%)')
#             ax.set_xlim(0, 100)
#             ax.set_title('Class Probabilities')
#             st.pyplot(fig)
            
#             # Display explanation based on the prediction
#             st.subheader("Explanation")
#             if predicted_label == "Non Demented":
#                 st.write("The scan appears to show brain structures consistent with normal cognitive function. No significant signs of Alzheimer's disease are detected.")
#             elif predicted_label == "Very Mild Demented":
#                 st.write("The scan shows some subtle changes that may indicate the earliest stages of Alzheimer's disease. Minor volume loss might be present.")
#             elif predicted_label == "Mild Demented":
#                 st.write("The scan shows moderate changes associated with Alzheimer's disease, including some volume loss in memory-related brain regions.")
#             elif predicted_label == "Moderate Demented":
#                 st.write("The scan shows significant changes typical of advanced Alzheimer's disease, including substantial volume loss and ventricular enlargement.")
            
#         except Exception as e:
#             st.error(f"Error processing the image: {e}")
#     else:
#         st.info("Please upload an MRI image to get a prediction.")
    
#     # Add disclaimer
#     st.markdown(
#         "<p class='disclaimer'>DISCLAIMER: This tool is for educational and research purposes only. "
#         "It should not be used for diagnosis or treatment decisions. "
#         "Always consult with qualified healthcare professionals for medical advice.</p>",
#         unsafe_allow_html=True
#     )

# # Add information about the dataset and model
# st.markdown("---")
# expander = st.expander("About the Model and Dataset")
# with expander:
#     st.write("""
#     ### Dataset
#     This application uses the Augmented Alzheimer's MRI Dataset from Kaggle, which contains brain MRI scans categorized into:
#     - Non-Demented (normal controls)
#     - Very Mild Demented
#     - Mild Demented
#     - Moderate Demented
    
#     ### Model Architecture
#     The prediction model is a Convolutional Neural Network (CNN) specifically designed for medical image classification with the following components:
#     - Multiple convolutional blocks with batch normalization
#     - Dropout layers for regularization
#     - Dense layers for final classification
    
#     The model was trained with data augmentation techniques to improve generalization.
#     """)

# # Footer
# st.markdown("---")
# st.markdown("Built with ❤️ using TensorFlow and Streamlit")
