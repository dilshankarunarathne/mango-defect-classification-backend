import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('mango.h5')

# Load the image
img_path = 'test.jpg'  # replace with the path of your image
img = image.load_img(img_path, target_size=(150, 150))

