"""
This module contains helper functions for the Pet Classifier app.

Functions:
- prepare_image(image_bytes): Prepares an image for classification by resizing, 
  normalizing, and adding a batch dimension.
- get_cat_fact(): Returns a random fun fact about cats.
- get_dog_fact(): Returns a random fun fact about dogs.
- get_random_cat_image(): Fetches a random cat image URL from TheCatAPI.
- get_random_dog_image(): Fetches a random dog image URL from Dog CEO's API.
"""

import io
import random
import requests
import numpy as np
from PIL import Image

def prepare_image(image_bytes):
    """
    Prepares the uploaded image for classification by resizing, normalizing, and adding a batch dimension.
    """
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode == 'RGBA':
        image = image.convert('RGB') # Convert RGBA to RGB if the image has an alpha channel
    image = image.resize((256, 256))  # Resize to match the input shape
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def get_cat_fact():
    """Returns a random fun fact about cats."""
    cat_facts = [
        "Cats can rotate their ears 180 degrees, giving them excellent hearing.",
        "In ancient Egypt, cats were considered sacred animals and were worshipped as gods.",
        "A group of cats is called a clowder.",
        "A cat can make many more different sounds than a dog.",
        "Domestic cats are the smallest members of the wildcat family.",
        "Cats can run up to 30 miles per hour.",
        "The average cat sleeps for about 12–16 hours a day."
    ]
    return random.choice(cat_facts)

def get_dog_fact():
    """Returns a random fun fact about dogs."""
    dog_facts = [
        "Dogs have an extraordinary sense of smell, 10,000 to 100,000 times more acute than humans.",
        "The Basenji dog is the only breed that doesn’t bark.",
        "Dogs’ noses have a special ability to detect diseases like cancer and diabetes in humans.",
        "The Labrador Retriever is the most popular dog breed in the United States.",
        "A Greyhound is the fastest dog breed, capable of running up to 45 miles per hour.",
        "Dogs have three eyelids: an upper lid, a lower lid, and a third lid called a nictitating membrane, which helps keep the eye moist.",
        "Dogs are as smart as a two-year-old child, understanding about 165 words."
    ]
    return random.choice(dog_facts)

def get_random_cat_image():
    """Fetches a random cat image URL from TheCatAPI."""
    response = requests.get('https://api.thecatapi.com/v1/images/search', timeout=5)
    return response.json()[0]['url']

def get_random_dog_image():
    """Fetches a random dog image URL from Dog CEO's API."""
    response = requests.get('https://dog.ceo/api/breeds/image/random', timeout=5)
    return response.json()['message']
