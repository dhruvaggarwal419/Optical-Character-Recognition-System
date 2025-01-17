import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import streamlit as st 
import numpy as np
from keras.models import load_model

# Ensure eager execution is enabled
tf.compat.v1.enable_eager_execution()
st.header('Character Recognition CNN Model')
Character_names = ['0',
 '1',
 '2',
 '3',
 '4',
 '5',
 '6',
 '7',
 '8',
 '9',
 'A',
 'B',
 'C',
 'D',
 'E',
 'F',
 'G',
 'H',
 'I',
 'J',
 'K',
 'L',
 'M',
 'N',
 'O',
 'P',
 'Q',
 'R',
 'S',
 'T',
 'U',
 'V',
 'W',
 'X',
 'Y',
 'Z']

# Load the model
model = load_model('Character_Recognition.keras')

def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(128, 128))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = 'The Image belongs to ' + Character_names[np.argmax(result)] + ' with a score of ' + str(np.max(result) * 100)
    return outcome

#upload and classify 
# uploaded_file = st.file_uploader('Upload an Image')
# if uploaded_file is not None:
#     # Save the uploaded file to a temporary location
#     with open(os.path.join('upload', uploaded_file.name), 'wb') as f:
#         f.write(uploaded_file.getbuffer())
    
#     # Display the uploaded image
#     st.image(uploaded_file, width=200)
    
#     # Get the path of the saved file
#     saved_image_path = os.path.join('upload', uploaded_file.name)
    
#     # Display the classification result
#     st.markdown(classify_images(saved_image_path))

uploaded_file = st.file_uploader("Choose a file")  # Assuming this is how the file is uploaded

if uploaded_file is not None:
    # Ensure the 'upload' directory exists
    upload_dir = 'upload'
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    # Save the uploaded file to a temporary location
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.read())

    # Display the uploaded image
    st.image(uploaded_file, width=200)

    # Get the path of the saved file
    saved_image_path = file_path

    # Display the classification result
    st.markdown(classify_images(saved_image_path))
