import numpy as np
import matplotlib.pyplot as plt

def mesh_grid_visual(X, y, model, label_dict = None):
    x0 = np.linspace(np.min(X[:,0])-2, np.max(X[:,0])+2, 100)
    x1 = np.linspace(np.min(X[:,1])-2, np.max(X[:,1])+2, 100)
    a,b = np.meshgrid(x0, x1)
    X_new = np.c_[a.ravel(), b.ravel()]
    predictions = model.predict(X_new)
    if "sklearn" in str(type(model)):
        decision = model.decision_function(X_new)
        decision = np.max(decision, axis = 1)
    else:
        decision = model.decision(X_new)
    print(decision)
    plt.contourf(x0, x1, predictions.reshape(a.shape), cmap=plt.cm.brg, alpha = 0.4)
    plt.contourf(x0, x1, decision.reshape(a.shape), cmap=plt.cm.brg, alpha = 0.2)
    for yy in np.unique(y):
        if label_dict != None:
            plt.scatter(X[y == yy, 0], X[y==yy, 1], label = label_dict[yy])
        else:
            plt.scatter(X[y == yy, 0], X[y==yy, 1], label = yy)
            
    plt.legend(loc = 'upper left')