import numpy as np
import cv2
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from time import time, sleep
from tkinter import font
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from keras_preprocessing.image import ImageDataGenerator
import os
import shutil



feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2"
IMAGE_SIZE = hub.get_expected_image_size(hub.Module(feature_extractor_url))
def feature_extractor(x):
    global feature_extractor_url
    classifier_module = hub.Module(feature_extractor_url)
    return classifier_module(x)


class App:
    def __init__(self, window, title):
        self.window = window
        self.title = title
        self.window.title(self.title)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 400)
        self.cap.set(4, 300)
        appHighlightFont = font.Font(family='Helvetica', size=30, weight='bold')

        self.imageFrame = ttk.Frame(self.window, width=600, height=500)
        self.imageFrame.grid(row=0, column=0, padx=0, pady=2)
        self.lmain = ttk.Label(self.imageFrame)
        self.lmain.grid(column = 0, row = 0)

        self.iterator = 0

        for child in self.imageFrame.winfo_children(): 
            child.grid_configure(padx=0, pady=15)

    
        self.predicted = StringVar()
        self.guess_label = ttk.Label(self.imageFrame, textvariable = self.predicted, font = appHighlightFont)
        self.guess_label.grid(column = 0, row = 1, sticky = (W), padx = 17)

        self.count = StringVar()
        self.count_label = ttk.Label(self.imageFrame, textvariable = self.count, font = appHighlightFont)
        self.count_label.grid(column = 0, row = 1, padx = 17)


        self.text = StringVar()
        self.labels_entry = ttk.Entry(self.imageFrame, textvariable = self.text)
        self.labels_entry.grid(column = 0, row = 2, sticky =(W), ipadx = 28, padx = 15)
        self.labels_entry.insert(0, 'Class name')

        self.last_click = None
        self.add_button = ttk.Button(self.imageFrame, text = 'Add class', command = self.add_button_funtions)
        self.add_button.grid(column = 0, row = 2, padx = 15)


        self.trained = False
        self.train_button = ttk.Button(self.imageFrame, text = 'Train classes', command = self.train )
        self.train_button.grid(column = 0, row = 2, sticky = (E), padx = 15, pady = 5)


        self.update()
        self.window.mainloop()

    def add_button_funtions(self):
        self.last_click = time()

    def add_data_labels(self, data):
        label_name = self.text.get()

        try:
            os.mkdir(f'Data/{label_name}')
        except FileExistsError:
            pass

        cv2.imwrite(f'Data/{label_name}/frame{self.iterator}.jpg', data)
        self.iterator += 1

    def train(self):
        self.trained = True

        image_data = ImageDataGenerator().flow_from_directory(('Data'), target_size=IMAGE_SIZE, batch_size=10)
        temp = image_data.class_indices
        self.labels = {}
        for key, val in temp.items():
            self.labels[val] = self.labels.get(val, []) + [key]


        features_extractor_layer = layers.Lambda(feature_extractor, input_shape = IMAGE_SIZE+[3])
        features_extractor_layer.trainable = False
        self.model = tf.keras.Sequential([
            features_extractor_layer,
            layers.Dense(image_data.num_classes, activation='softmax')
            ])
        self.model.summary()

        self.model.compile(
            optimizer=tf.train.AdamOptimizer(), 
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        init = tf.global_variables_initializer()
        sess = K.get_session()
        sess.run(init)

        class CollectBatchStats(tf.keras.callbacks.Callback):
            def __init__(self):
                self.batch_losses = []
                self.batch_acc = []
                
            def on_batch_end(self, batch, logs=None):
                self.batch_losses.append(logs['loss'])
                self.batch_acc.append(logs['acc'])

        steps_per_epoch = image_data.samples//image_data.batch_size
        batch_stats = CollectBatchStats()
        self.model.fit((item for item in image_data), epochs=2, 
                            steps_per_epoch=steps_per_epoch,
                            callbacks = [batch_stats])

        print(f'Loss: {batch_stats.batch_losses}')
        print(f'Acc: {batch_stats.batch_acc}')

    def on_closing(self):
        if input("Do you want delete files  y/n?") == 'y':
            for dir_ in os.listdir('Data'):
                try:
                    shutil.rmtree(f'Data/{dir_}')
                except NotADirectoryError:
                    os.remove(f'Data/{dir_}')
            self.window.destroy()
        else:
            self.window.destroy()




    def update(self):
        _, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.lmain.imgtk = imgtk
        self.lmain.configure(image=imgtk)

        cv2.imwrite('frame.jpg', frame)
        img = Image.open('frame.jpg').resize(IMAGE_SIZE)
        img = np.array(img)


        if self.last_click:
            if time() - self.last_click < 10:
                self.count.set(f'Adding class: {abs(int(time() - self.last_click) - 10)}')
                self.add_data_labels(frame)
            else:
                self.last_click = None
                self.count.set('')

        if self.trained:
            self.predicted.set(f'Guess: {self.labels[self.model.predict_classes(img[np.newaxis, ...])[0]][0]}')

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.lmain.after(10, self.update)

App(Tk(), 'Image classification')
#[224, 224, 3]