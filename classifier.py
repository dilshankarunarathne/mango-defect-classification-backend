import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from fastapi import UploadFile
from PIL import Image
import io

# Load the saved model
model = tf.keras.models.load_model('mango.h5')

classes = ['Alternaria', 'Anthracnose', 'Black Mould Rot', 'Healthy', 'Stem end Rot']


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


def predict_local(img_path):
    # Load the image
    # img_path = 'test.jpg'
    img = image.load_img(img_path, target_size=(150, 150))

    # Preprocess the image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    # Make a prediction
    prediction = model.predict(img_array)

    # Get the class with the highest probability
    predicted_class = np.argmax(prediction)

    return predicted_class
