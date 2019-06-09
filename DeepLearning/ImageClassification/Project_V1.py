from __future__ import absolute_import, division, print_function
import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.keras.backend as K
from tensorflow.keras import layers
import cv2
from Network import Network

feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/3"
IMAGE_SIZE = hub.get_expected_image_size(hub.Module(feature_extractor_url))
def feature_extractor(x):
    global feature_extractor_url
    classifier_module = hub.Module(feature_extractor_url)
    return classifier_module(x)

features_extractor_layer = layers.Lambda(feature_extractor, input_shape = IMAGE_SIZE+[3])
features_extractor_layer.trainable = False
model = tf.keras.Sequential([
  features_extractor_layer
])
model.summary()
sess = K.get_session()
init = tf.global_variables_initializer()
sess.run(init)




class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.window.configure(background='black')
        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)


        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = 1000, height = 600)
        self.canvas.pack()

        # Button that lets the user take a snapshot
        self.btn_A=tkinter.Button(window, text="Add A", width=50, command=self.train_A, fg= 'black', highlightbackground='#3E4149')
        self.btn_B=tkinter.Button(window, text="Add B", width=50, command=self.train_B, highlightbackground='#3E4149')
        self.btn_train=tkinter.Button(window, text="Train", width=50, command=self.train, highlightbackground='#3E4149')
        self.btn_A.pack()
        self.btn_B.pack()
        self.btn_train.pack()

        self.dataset = []
        self.data_labels = []
        self.trained = False
        print('aaaa')

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()
        self.window.mainloop()

    def train_A(self):
        print('Adding 0')
        self.dataset.append(self.result)
        self.data_labels.append(0)

    def train_B(self):
        print('Adding 1')
        self.dataset.append(self.result)
        self.data_labels.append(1)
        

    def train(self):
        self.trained = True
        self.my_model = Network(1280)
        self.my_model.addLayer(512)
        self.my_model.addLayer(512)
        self.my_model.addLayer(512)
        self.my_model.addLayer(1)
        for i in range(1000):
            random_index = np.random.randint(len(self.dataset))
            inp = np.array(self.dataset[random_index][0])
            out = self.my_model.forward(inp)
            error = self.data_labels[random_index] - out
            self.my_model.backward(error)
            self.my_model.learn(0.1)

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        cv2.imwrite('frame.jpg', frame)
        img = PIL.Image.open('frame.jpg').resize(IMAGE_SIZE)
        img = np.array(img)/255.0
        self.result = model.predict(img[np.newaxis, ...])

        if self.trained:
            if self.my_model.forward(self.result[0]) < 0.5:
                print('0')
            else:
                print('1')

        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
        self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
        self.window.after(self.delay, self.update)


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.vid.set(3, 1000)
        self.vid.set(4, 568)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:                 # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# Create a window and pass it to the Application object
App(tkinter.Tk(), 'Tkinter')