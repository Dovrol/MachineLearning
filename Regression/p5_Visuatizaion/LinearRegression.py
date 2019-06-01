import numpy as np
import tensorflow as tf
import time
import p5

points_X = []
points_Y = []
n_samples = 1


# tensorflow setup
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

m = tf.Variable(np.random.uniform(1), dtype= tf.float32)
b = tf.Variable(np.random.uniform(1), dtype = tf.float32)


# y = mx + b line formula
pred = lambda X: X * m + b
cost = (1/ n_samples)* tf.reduce_sum((pred(X) - Y)**2)
opt = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)
init = tf.global_variables_initializer()

def setup():
    p5.size(1280,800)
    p5.title("LinearRegression")
    p5.no_loop()

def draw():
    p5.background(0)
    if points_X:
        sess.run(opt, feed_dict = {X:points_X, Y:points_Y})
        x1 = p5.remap(0, (0,1), (0,width))
        x2 = p5.remap(1, (0,1), (0,width))
        y1 = p5.remap(pred(0).eval(), (0,1), (height, 0))
        y2 = p5.remap(pred(1).eval(), (0,1), (height, 0))
        p5.line((x1,y1), (x2, y2))

    p5.stroke(255)
    for px, py in zip(points_X, points_Y):
            x = p5.remap(px, (0,1), (0, width))
            y = p5.remap(py, (0,1) ,(height, 0))
            p5.circle((x, y), 10)


def mouse_clicked(event):
    global n_samples
    x = p5.remap(event.x, (0, width), (0,1))
    y = p5.remap(event.y, (0, height), (1,0))
    points_X.append(x)
    points_Y.append(y)
    n_samples += 1

if '__main__' == __name__:
    with tf.Session() as sess:
        sess.run(init)
        p5.run()
            
            

