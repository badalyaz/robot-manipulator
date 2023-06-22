import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# turning numpy array to torch tensor
to_torch = lambda x: torch.from_numpy(x).to(device).float() 
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def start_dim_sum(arr, start_dim=1):
    return arr.reshape(list(arr.shape[:start_dim]) + [-1]).sum(axis=-1)

def normalize_data(data):
    '''
    This function is used to normalize inputs and outputs along the feature dimension.
    '''
    data_mean = np.mean(data, axis=0)[None]
    data_std = np.var(data, axis=0)[None] ** 0.5 
    return (data - data_mean) / data_std, data_mean, data_std


def plot_test(lstm_model, test_poses, test_torques, indices=[1, 100, 1000], plot_torques=True, reference_predictions=None):
    
    def plot_subplots(our_predictions, true_values, reference_predictions=None):
        fig = plt.figure(figsize=(16, 10))
        plt.subplot(2, 1, 1)
        plt.plot(our_predictions, 'b', label='Our Prediction')
        plt.plot(true_values, 'r', label='True Value')
        if reference_predictions is not None:
            plt.plot(reference_predictions, 'g', label='Reference')
        plt.legend()
        plt.show()
    
    lstm_model.eval()

    for index in indices:
        test_data = test_poses[index][None]
        test_target = test_torques[index]
        
        with torch.no_grad():
            predictions = lstm_model(test_data)
        
        our_predictions = np.hstack(predictions.cpu().numpy())
        true_values = np.hstack(test_target.cpu().numpy())
        
        if reference_predictions is not None:
            ref_predictions = np.hstack(reference_predictions[index].cpu().numpy())
            
            if our_predictions.shape[-1] == 320:
                our_predictions = np.concatenate([our_predictions[:50], our_predictions[160:210]])
                true_values = true_values[:100]
                ref_predictions = ref_predictions[:100]
            elif our_predictions.shape[-1] in [210, 332]:
                our_predictions = our_predictions[:100]
                true_values = true_values[:100]
                ref_predictions = ref_predictions[:100]
            elif our_predictions.shape[-1] in [160, 480, 182]:
                our_predictions = our_predictions[:50]
                true_values = true_values[:50]
                ref_predictions = ref_predictions[:50]
                
            plot_subplots(our_predictions, true_values, ref_predictions)

def calculate_accuracy(lstm_model, train_poses, train_torques, test_poses, test_torques):
    loss_function = nn.MSELoss()
    lstm_model.eval()

    with torch.no_grad():
        train_predictions = lstm_model(train_poses)
        test_predictions = lstm_model(test_poses)
    
    if train_predictions.shape[-1] == 160:
        test_error_50 = loss_function(test_predictions[:, :50], test_torques[:, :50]).item()
        test_error_160 = loss_function(test_predictions, test_torques).item()

        return test_error_50, test_error_160
    
    elif train_predictions.shape[-1] == 182:
        test_error_50 = loss_function(test_predictions[:, :50], test_torques[:, :50]).item()
        test_error_160 = loss_function(test_predictions[:, :160], test_torques[:, :160]).item()

        return test_error_50, test_error_160
    
    elif train_predictions.shape[-1] == 210:
        pred_test_160 = torch.cat((test_predictions[:, :50], test_predictions[:, 100:]), dim=1)
        torques_test_160 = torch.cat((test_torques[:, :50], test_torques[:, 100:]), dim=1)        
        
        test_error_torque_1 = loss_function(test_predictions[:, :50], test_torques[:, :50]).item()
        test_error_torque_2 = loss_function(test_predictions[:, 50:100], test_torques[:, 50:100]).item()
        test_error_160 = loss_function(pred_test_160, torques_test_160).item()
        
        return test_error_torque_1, test_error_torque_2

    elif train_predictions.shape[-1] == 320:
        pred_test_160 = test_predictions[:, :160]
        torques_test_160 = torch.cat((test_torques[:, :50], test_torques[:, 100:]), dim=1)
        
        test_error_160 = loss_function(pred_test_160, torques_test_160).item()
        test_error_torque_1 = loss_function(test_predictions[:, :50], test_torques[:, :50]).item()
        test_error_torque_2 = loss_function(test_predictions[:, 160:210], test_torques[:, 50:100]).item()
        
        return test_error_torque_1, test_error_torque_2
    
    elif train_predictions.shape[-1] == 332:
        test_error_torque_1 = loss_function(test_predictions[:, :50], test_torques[:, :50]).item()
        test_error_torque_2 = loss_function(test_predictions[:, 50:100], test_torques[:, 50:100]).item()
        
        return test_error_torque_1, test_error_torque_2
    
    elif train_predictions.shape[-1] == 480:
        test_error_160 = loss_function(test_predictions[:, :160], test_torques[:, :160]).item()
        test_error_torque_1 = loss_function(test_predictions[:, :50], test_torques[:, :50]).item()

        return test_error_torque_1, test_error_160
