from random_forest import WaveletsForestRegressor
import pickle
import numpy as np


regressor = WaveletsForestRegressor(regressor='random_forest', trees=10, depth=None, features='sqrt', seed=21)

y_c = pickle.load(open('input_and_labels/np_y', 'rb'))
N = len(y_c)


def get_MSE(model):
    alpha_list = []
    X_ = pickle.load(open('data/' + model + '_fc1', 'rb'))
    perm = np.random.permutation(np.arange(N))
    X = X_[perm]
    y_shuffled = y_c[perm]
    for mislab in [0.25, 0.5, 0.75, 1]:
        np.random.shuffle(y_shuffled[:int(N * mislab)])
        y = np.eye(10)[y_shuffled.astype(np.int32)][:, :9]
        rf = regressor.fit(X, y)
        alpha, MSE_errors = rf.evaluate_smoothness()
        pickle.dump(MSE_errors, open('results/MSE_mislab_' + str(mislab) + '_' + model, 'wb'))
        alpha_list.append(alpha)
    pickle.dump(alpha_list, open('results/alpha_list_mislab_' + model, 'wb'))


get_MSE('ResNet12')