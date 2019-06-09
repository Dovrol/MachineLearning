from __future__ import absolute_import, division, print_function
import matplotlib.pylab as plt
import numpy as np
import PIL.Image as Image
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.keras.backend as K
from tensorflow.keras import layers
import cv2

mobilenet = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/3"

def classifier(x):
    global mobilenet
    classifier_module = hub.Module(mobilenet)
    return classifier_module(x)

data_root = tf.keras.utils.get_file('flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz', untar=True)
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')

# image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
# image_data = image_generator.flow_from_directory(str(data_root))

IMAGE_SIZE = hub.get_expected_image_size(hub.Module(mobilenet))
print(IMAGE_SIZE)

classifier_layer = layers.Lambda(classifier, input_shape = IMAGE_SIZE+[3])
classifier_model = tf.keras.Sequential([classifier_layer])
classifier_model.summary()

# image_data = image_generator.flow_from_directory(str(data_root), target_size=IMAGE_SIZE)


sess = K.get_session()
init = tf.global_variables_initializer()
sess.run(init)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('frame',frame)
    cv2.imwrite('frame.jpg', frame)
    img = Image.open('frame.jpg').resize(IMAGE_SIZE)
    img = np.array(img)/255.0

    result = classifier_model.predict(img[np.newaxis, ...])
    predicted_class = np.argmax(result[0], axis=-1)

    imagenet_labels = np.array(open(labels_path).read().splitlines())

    predicted_class_name = imagenet_labels[predicted_class]
    print(predicted_class_name)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
