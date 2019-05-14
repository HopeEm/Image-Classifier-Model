from PIL import Image
import numpy as np
from keras.models import load_model

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

model = load_model('image_classifier.h5')

input_path = input('Enter image file pathname: ')
input_image = Image.open(input_path)
input_image = input_image.resize((32, 32), resample=Image.LANCZOS)
image_array = np.array(input_image)
image_array = image_array.astype('float32')
image_array /= 255.0
image_array = image_array.reshape(1, 32, 32, 3)
answer = model.predict(image_array)
input_image.show()
print(labels[np.argmax(answer)])