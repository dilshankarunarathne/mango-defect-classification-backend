import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('mango.h5')

# Load the image
img_path = 'test.jpg'  # replace with the path of your image
img = image.load_img(img_path, target_size=(150, 150))

# Preprocess the image
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.

# Make a prediction
prediction = model.predict(img_array)

# Get the class with highest probability
predicted_class = np.argmax(prediction)

# Print the predicted class
print(predicted_class)