from random_forest import WaveletsForestRegressor
import pickle
import numpy as np
import time


regressor = WaveletsForestRegressor(regressor='random_forest', trees=10, depth=None, features='sqrt', seed=21)

y_c = pickle.load(open('input_and_labels/np_y', 'rb'))
y = np.eye(10)[y_c.astype(np.int32)][:, :9]
del y_c

input = pickle.load(open('input_and_labels/np_input', 'rb'))
start_time = time.time()
rf = regressor.fit(input, y)
alpha_input, _ = rf.evaluate_smoothness()
print('\ninput', ': ', alpha_input)
print("--- %s minutes ---" % ((time.time() - start_time) / 60))
del input


def get_smoothness_dict(model):
    alpha_dict = {'input': alpha_input}

    if model == 'ResNet12' or model == 'ResNet12_5' or model=='ResNet12_overfitting_100':
        layers = ['conv1_relu', 'conv2_relu', 'conv3+identity_relu','conv4_relu', 'conv5+identity_relu', 'conv6_relu', 'conv7+identity_relu', 'conv8_relu', 'conv9+identity_relu', 'conv10_relu', 'conv11+identity_relu', 'avpool', 'fc1']
    elif model == 'Plain12' or model == 'Plain12_5':
        layers = ['conv1_relu', 'conv2_relu', 'conv3_relu','conv4_relu', 'conv5_relu', 'conv6_relu', 'conv7_relu', 'conv8_relu', 'conv9_relu', 'conv10_relu', 'conv11_relu', 'avpool', 'fc1']
    elif model == 'ResNet12_tangh':
        layers =['conv1_tanh', 'conv2_tanh', 'conv3+identity_tanh','conv4_tanh', 'conv5+identity_tanh', 'conv6_tanh', 'conv7+identity_tanh', 'conv8_tanh', 'conv9+identity_tanh', 'conv10_tanh', 'conv11+identity_tanh', 'avpool', 'fc1']
    elif model == 'Plain12_tangh':
        layers = ['conv1_tanh', 'conv2_tanh', 'conv3_tanh','conv4_tanh', 'conv5_tanh', 'conv6_tanh', 'conv7_tanh', 'conv8_tanh', 'conv9_tanh', 'conv10_tanh', 'conv11_tanh', 'avpool', 'fc1']
    elif model == 'ResNet10_conv2':
        layers = ['conv1_relu', 'conv2_relu', 'conv3+identity_relu', 'conv4_relu', 'conv5+identity_relu', 'conv8_relu','conv9+identity_relu', 'conv10_relu', 'conv11+identity_relu', 'avpool', 'fc1']
    elif model == 'Plain6':
        layers = ['conv1_relu', 'conv2_relu', 'conv3_relu', 'conv4_relu',  'conv11_relu', 'avpool', 'fc1']
    for layer in layers:
        X = pickle.load(open('data/' + model + '_' + layer, 'rb'))
        # print(model+': ', layer+': ', X.shape)
        start_time = time.time()
        rf = regressor.fit(X, y)
        alpha, MSE_errors = rf.evaluate_smoothness()
        print('\n', model, ': ', layer, ': ', alpha)

        alpha_dict[layer] = alpha
        pickle.dump(alpha_dict, open('results/alpha_dict_' + model, 'wb'))
        if layer == 'fc1':
            pickle.dump(MSE_errors, open('results/MSE_errors_no_misslab_' + model, 'wb'))
        print("--- %s minutes ---" % ((time.time() - start_time) / 60))

        del X


# for net in ['ResNet12', 'ResNet12_5', 'Plain12', 'Plain12_5', 'ResNet12_tangh', 'Plain12_tangh']:
#     get_smoothness_dict(net)

# for net in ['ResNet10_conv2', 'Plain6']:
#     get_smoothness_dict(net)

get_smoothness_dict('ResNet12_overfitting_100')