import numpy as np
import tensorflow as tf
import time
import p5
from sklearn.preprocessing import PolynomialFeatures

points_X = []
points_Y = []
n_samples = 1
degree = 10
weights = []
Pf = PolynomialFeatures(degree, include_bias = True)


# tensorflow setup
X = tf.placeholder(tf.float64)
Y = tf.placeholder(tf.float64)

for i in range(degree):
    weights.append(tf.Variable(np.random.uniform(-1,1), dtype= tf.float64))



# y = mx^2 + bx + c line formula
# pred = lambda X: tf.add(tf.add(tf.multiply(tf.square(X), a), tf.multiply(X, b)), c)
pred = lambda X: np.sum(X*weights, axis=0)
cost = (1/ n_samples)* tf.reduce_sum((pred(X) - Y)**2)
opt = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)
init = tf.global_variables_initializer()

def setup():
    p5.size(640,480)
    p5.title("LinearRegression")

def draw():
    p5.background(0)
    if points_X:
        print(points_X)
        Poly_X = Pf.fit_transform(np.array(points_X).reshape(-1,1))
        print(Poly_X)
        sess.run(opt, feed_dict = {X:Poly_X, Y:points_Y})
        
    curveX = np.linspace(0,1, 50)
    curveY = pred(curveX.reshape(-1,1)).eval()
    vertexes = []
    for x,y in zip(curveX, curveY):
        x1 = p5.remap(x, (0,1), (0,width))
        y1 = p5.remap(y, (0,1), (height, 0))
        vertexes.append((x1, y1))
    p5.no_fill()
    p5.stroke(255)
    s = p5.PShape(np.array(vertexes), visible = True, attribs = 'path')
    p5.draw_shape(s)
        
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
            
            

