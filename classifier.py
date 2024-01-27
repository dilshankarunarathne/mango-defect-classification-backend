import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from fastapi import UploadFile
from PIL import Image
import io
from joblib import load

# Load the saved model
model = tf.keras.models.load_model('mango.h5')

# Load the label encoder
label_encoder = load('label_encoder.joblib')

# Get the classes
classes = list(label_encoder.classes_)


async def predict(file: UploadFile):
    # Read the image file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Resize the image
    image = image.resize((150, 150))

    # Convert the image to numpy array
    img_array = np.asarray(image)
    img_array = np.expand_dims(img_array, axis=0)

    # Convert the numpy array to float64 before performing the division operation
    img_array = img_array.astype('float64')
    img_array /= 255.

    # Make a prediction
    prediction = model.predict(img_array)

    # Get the class with the highest probability
    predicted_class = np.argmax(prediction)

    return classes[predicted_class]
