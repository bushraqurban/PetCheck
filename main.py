"""
This is the main file for the Pet Classifier app.

The app uses a pre-trained model to classify images of cats and dogs. 
It allows users to upload an image, predicts whether it is a cat or dog, 
and displays a random fun fact or quote based on the classification. 
Additionally, a random image of the respective animal (cat/dog) is displayed.
"""

import streamlit as st
import tensorflow as tf
from utils import prepare_image, get_cat_fact, get_dog_fact, get_random_cat_image, get_random_dog_image

# Load the pre-trained model
model = tf.keras.models.load_model('cat-dog-classification.h5')

# Streamlit UI
st.image("logo.png", width=400, use_container_width=True)
st.write("Upload an image of a cat or dog to predict.")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Prepare the image
    image_data = uploaded_file.read()
    image = prepare_image(image_data)

    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Predict
    if st.button('Predict'):
        prediction = model.predict(image)
        LABEL = 'dog' if prediction[0] > 0.5 else 'cat'
        st.markdown(f"""
        <h4 style="text-align: center; font-family: 'Arial', sans-serif;">
            Our model predicts your pet to be a {LABEL}!
        </h4>
        """, unsafe_allow_html=True)

        # Show a random fact or quote based on the label
        if LABEL == 'cat':
            fact_or_quote = get_cat_fact()
            cat_image_url = get_random_cat_image()
            st.image(cat_image_url, use_container_width=True)
        else:
            fact_or_quote = get_dog_fact()
            dog_image_url = get_random_dog_image()
            st.image(dog_image_url, use_container_width=True)

        st.markdown(f"""
        <h3 style="text-align: center; font-family: 'Arial', sans-serif;">
            Fact of the Moment
        </h3>
        <p style="font-size: 20px; font-style: italic; text-align: center;">
            "{fact_or_quote}"
        </p>
        """, unsafe_allow_html=True)
