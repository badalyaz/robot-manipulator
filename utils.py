import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import datetime
import scipy.io as sio
import os



# Permanently changes the pandas settings
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

# turning numpy array to torch tensor
to_torch = lambda x: torch.from_numpy(x).to(device).float() 


def normalizing_data(data):
    '''
    Using this normalizing function to normalize inputs and outputs in the features dimension
    '''
    data_mean = data.mean(axis=0)[None]
    data_stdv = data.var(axis=0)[None] ** 0.5 
    return (data - data_mean ) / data_stdv, data_mean, data_stdv


def show_accuracy(model_lstm, poses_train, torques_train, poses_test, torques_test):
    #Whole data test error
    model_lstm.eval()

    ###this is done only for zotac
    model_lstm.change_device(device0='cpu')
    poses_train = poses_train.cpu()
    poses_test = poses_test.cpu()
    torques_train = torques_train.cpu()
    torques_test = torques_test.cpu()
    ###this is done only for zotac

    with torch.no_grad():
        pred_train = model_lstm(poses_train)
        pred_test = model_lstm(poses_test)
    MSE = nn.MSELoss()
    
    if pred_train.shape[-1] == 160:
        train_L2_error_160 = MSE(pred_train, torques_train).item()
        train_L2_error_50 = MSE(pred_train[:, :50], torques_train[:, :50]).item()
        test_L2_error_160 = MSE(pred_test, torques_test).item()
        test_L2_error_50 = MSE(pred_test[:, :50], torques_test[:, :50]).item()

        print(f'train MSE on 160={train_L2_error_160:.4f}, ref. error=0.0006')
        print(f'train MSE on 50={train_L2_error_50:.4f}, ref. error=0.0019')
        
        print(f'test  MSE on 160={test_L2_error_160:.4f}, ref. error=0.0013')
        print(f'test  MSE on 50={test_L2_error_50:.4f}, ref. error=0.0040')

        return test_L2_error_50, test_L2_error_160
    
    elif pred_train.shape[-1] == 182:
        train_L2_error_160 = MSE(pred_train[:, :160], torques_train[:, :160]).item()
        train_L2_error_50 = MSE(pred_train[:, :50], torques_train[:, :50]).item()
        train_L2_error_dst_vel = MSE(pred_train[:, 160:], torques_train[:, 160:]).item()
        
        test_L2_error_160 = MSE(pred_test[:, :160], torques_test[:, :160]).item()
        test_L2_error_50 = MSE(pred_test[:, :50], torques_test[:, :50]).item()
        test_L2_error_dst_vel = MSE(pred_test[:, 160:], torques_test[:, 160:]).item()

        print(f'train MSE on torque1+helpers={train_L2_error_50:.4f} , ref. error=0.0019')
        print(f'train MSE on torque1=        {train_L2_error_160:.4f} , ref. error=0.0006')
        print(f'train MSE on distVel=        {train_L2_error_dst_vel:.5f}, ref. error=0.00004')
        
        print(f'test  MSE on torque1+helpers={test_L2_error_50:.4f} , ref. error=0.0040')
        print(f'test  MSE on torque1=        {test_L2_error_160:.4f} , ref. error=0.0013')
        print(f'test  MSE on distVel=        {test_L2_error_dst_vel:.5f}, ref. error=0.00004')

        return test_L2_error_50, test_L2_error_160
    
    elif pred_train.shape[-1] == 210:
        pred_train_160 = torch.cat((pred_train[:, :50], pred_train[:, 100:]), dim=1)
        torques_train_160 = torch.cat((torques_train[:, :50], torques_train[:, 100:]), dim=1)    
        
        pred_test_160 = torch.cat((pred_test[:, :50], pred_test[:, 100:]), dim=1)
        torques_test_160 = torch.cat((torques_test[:, :50], torques_test[:, 100:]), dim=1)        
        
        train_L2_error_210 = MSE(pred_train, torques_train).item()
        train_L2_error_torque_1 = MSE(pred_train[:, :50], torques_train[:, :50]).item()
        train_L2_error_torque_2 = MSE(pred_train[:, 50:100], torques_train[:, 50:100]).item()
        train_L2_error_160 = MSE(pred_train_160, torques_train_160).item()
        
        test_L2_error_210 = MSE(pred_test, torques_test).item()
        test_L2_error_torque_1 = MSE(pred_test[:, :50], torques_test[:, :50]).item()
        test_L2_error_torque_2 = MSE(pred_test[:, 50:100], torques_test[:, 50:100]).item()
        test_L2_error_160 = MSE(pred_test_160, torques_test_160).item()
        

        print(f'train MSE on torques+helpers ={train_L2_error_210:.4f}, ref. error=0.0009')
        print(f'train MSE on torque_1+helpers={train_L2_error_160:.4f}, ref. error=0.0006')
        print(f'train MSE on torque_1=        {train_L2_error_torque_1:.4f}, ref. error=0.0019')
        print(f'train MSE on torque_2=        {train_L2_error_torque_2:.4f}, ref. error=0.0020')
        
        print(f'test  MSE on torques+helpers ={test_L2_error_210:.4f}, ref. error=0.0020')
        print(f'test  MSE on torque_1+helpers={test_L2_error_160:.4f}, ref. error=0.0013')
        print(f'test  MSE on torque_1=        {test_L2_error_torque_1:.4f}, ref. error=0.0040')
        print(f'test  MSE on torque_2=        {test_L2_error_torque_2:.4f}, ref. error=0.0041')
        
        return test_L2_error_torque_1, test_L2_error_torque_2

    elif pred_train.shape[-1] == 320:
        pred_train_160 = pred_train[:, :160]
        torques_train_160 = torch.cat((torques_train[:, :50], torques_train[:, 100:]), dim=1)    
        
        pred_test_160 = pred_test[:, :160]
        torques_test_160 = torch.cat((torques_test[:, :50], torques_test[:, 100:]), dim=1)
        
        train_L2_error_160 = MSE(pred_train_160, torques_train_160).item()
        train_L2_error_torque_1 = MSE(pred_train[:, :50], torques_train[:, :50]).item()
        train_L2_error_torque_2 = MSE(pred_train[:, 160:210], torques_train[:, 50:100]).item()
        
        test_L2_error_160 = MSE(pred_test_160, torques_test_160).item()
        test_L2_error_torque_1 = MSE(pred_test[:, :50], torques_test[:, :50]).item()
        test_L2_error_torque_2 = MSE(pred_test[:, 160:210], torques_test[:, 50:100]).item()
        

        print(f'train MSE on torque_1+helpers={train_L2_error_160:.4f}, ref. error=0.0006')  # first 160
        print(f'train MSE on torque_1=        {train_L2_error_torque_1:.4f}, ref. error=0.0019') # first 50
        print(f'train MSE on torque_2=        {train_L2_error_torque_2:.4f}, ref. error=0.0020') # second 50
        
        print(f'test  MSE on torque_1+helpers={test_L2_error_160:.4f}, ref. error=0.0013')
        print(f'test  MSE on torque_1=        {test_L2_error_torque_1:.4f}, ref. error=0.0040')
        print(f'test  MSE on torque_2=        {test_L2_error_torque_2:.4f}, ref. error=0.0041')

        return test_L2_error_torque_1, test_L2_error_torque_2
    
    elif pred_train.shape[-1] == 332:
        ##### for comparing on previous 160 data
        pred_train_160 = torch.cat((pred_train[:, :50], pred_train[:, 222:]), dim=1)
        torques_train_160 = torch.cat((torques_train[:, :50], torques_train[:, 222:]), dim=1)    
        pred_train_210 = torch.cat((pred_train[:, :100], pred_train[:, 222:]), dim=1)
        torques_train_210 = torch.cat((torques_train[:, :100], torques_train[:, 222:]), dim=1) 
        
        pred_test_160 = torch.cat((pred_test[:, :50], pred_test[:, 222:]), dim=1)
        torques_test_160 = torch.cat((torques_test[:, :50], torques_test[:, 222:]), dim=1)
        pred_test_210 = torch.cat((pred_test[:, :100], pred_test[:, 222:]), dim=1)
        torques_test_210 = torch.cat((torques_test[:, :100], torques_test[:, 222:]), dim=1)
        ##### end for comparing on previous 160 data
        
        
        train_L2_error_210 = MSE(pred_train_210, torques_train_210).item()
        train_L2_error_160 = MSE(pred_train_160, torques_train_160).item()
        train_L2_error_torque_1 = MSE(pred_train[:, :50], torques_train[:, :50]).item()
        train_L2_error_torque_2 = MSE(pred_train[:, 50:100], torques_train[:, 50:100]).item()
        
        test_L2_error_210 = MSE(pred_test_210, torques_test_210).item()
        test_L2_error_160 = MSE(pred_test_160, torques_test_160).item()
        test_L2_error_torque_1 = MSE(pred_test[:, :50], torques_test[:, :50]).item()
        test_L2_error_torque_2 = MSE(pred_test[:, 50:100], torques_test[:, 50:100]).item()
        

        print(f'train MSE on torques+helpers ={train_L2_error_210:.4f}, ref. error=0.0009')
        print(f'train MSE on torque_1+helpers={train_L2_error_160:.4f}, ref. error=0.0006')
        print(f'train MSE on torque_1=        {train_L2_error_torque_1:.4f}, ref. error=0.0019')
        print(f'train MSE on torque_2=        {train_L2_error_torque_2:.4f}, ref. error=0.0020')
        
        print(f'test  MSE on torques+helpers ={test_L2_error_210:.4f}, ref. error=0.0020')
        print(f'test  MSE on torque_1+helpers={test_L2_error_160:.4f}, ref. error=0.0013')
        print(f'test  MSE on torque_1=        {test_L2_error_torque_1:.4f}, ref. error=0.0040')
        print(f'test  MSE on torque_2=        {test_L2_error_torque_2:.4f}, ref. error=0.0041')
        
        return test_L2_error_torque_1, test_L2_error_torque_2
    
    
    elif pred_train.shape[-1] == 480:
        
        train_L2_error_160 = MSE(pred_train[:, :160], torques_train[:, :160]).item()
        train_L2_error_torque_1 = MSE(pred_train[:, :50], torques_train[:, :50]).item()
        
        test_L2_error_160 = MSE(pred_test[:, :160], torques_test[:, :160]).item()
        test_L2_error_torque_1 = MSE(pred_test[:, :50], torques_test[:, :50]).item()
        

        print(f'train MSE on torque_1+helpers={train_L2_error_160:.4f}, ref. error=0.0006') 
        print(f'train MSE on torque_1        ={train_L2_error_torque_1:.4f}, ref. error=0.0019') 
        
        print(f'test  MSE on torque_1+helpers={test_L2_error_160:.4f}, ref. error=0.0013')
        print(f'test  MSE on torque_1        ={test_L2_error_torque_1:.4f}, ref. error=0.0040')

        return test_L2_error_torque_1, test_L2_error_160
    
    
def plot_test(model_lstm, poses_test, torques_test, indxs=[1, 100, 1000], plot_torques=True, ref_preds=None):
    def plot_subplots(preds_our, true_values, ref_preds=None):
        fig = plt.figure(figsize=(16, 10))
        plt.subplot(2, 1, 1)
        plt.plot(preds_our,'b',label='Prediction')
        plt.plot(true_values,'r',label='True Value')
        if ref_preds is not None:
            plt.plot(ref_preds,'g',label='Reference')
        plt.legend()
        plt.show()
    # Set the model to evaluation mode
    model_lstm.eval()

    ###this is done only for zotac
    model_lstm.change_device(device0='cpu')
    poses_test = poses_test.cpu()
    torques_test = torques_test.cpu()
    ###this is done only for zotac
    
    for indx in indxs:
        test_data = poses_test[indx][None]
        test_target = torques_test[indx]
        with torch.no_grad():
            y_pred = model_lstm(test_data)
        predictions = np.hstack(y_pred.cpu().numpy())
        true_values = np.hstack(test_target.cpu().numpy())
        if ref_preds is not None:
            ref_pred = np.hstack(ref_preds[indx].cpu().numpy())
            if predictions.shape[-1] == 320:
                preds_our = np.concatenate([predictions[:50], predictions[160:210]])
                true_values = true_values[:100]
                ref_pred = ref_pred[:100]
            elif predictions.shape[-1] in [210, 332]:
                preds_our = predictions[:100]
                true_values = true_values[:100]
                ref_pred = ref_pred[:100]
            elif predictions.shape[-1] in [160, 480, 182]:
                preds_our = predictions[:50]
                true_values = true_values[:50]
                ref_pred = ref_pred[:50]
                
            plot_subplots(preds_our, true_values, ref_pred)
            
    ###this is done only for zotac
    model_lstm.change_device(device0=device)
    ###this is done only for zotac

    
    
    
def load_data(enrich_size=50):
    ##Loading data 
    #Input I
    I_us_train = sio.loadmat('../data/mat/unscaled_data/initial_conditions_train.mat')['trainingDataInput']
    I_us_test  = sio.loadmat('../data/mat/unscaled_data/initial_conditions_test.mat')['testDataInput']
    
#     I_us_train = np.delete( I_us_train, np.s_[1,3,5,7,11,13], axis=-1)  ## deleting all for arm2
#     I_us_test = np.delete( I_us_test, np.s_[1,3,5,7,11,13], axis=-1)
#     I_us_train = np.concatenate( [I_us_train[:,:2], I_us_train[:,4:]], axis=-1 ) ## deleting endpoints
#     I_us_test = np.concatenate( [I_us_test[:,:2], I_us_test[:,4:]], axis=-1 )

    #Velocity unscaled
    dqdt0_us_train = sio.loadmat('../data/mat/unscaled_data/velocity0_train.mat')['trainingDataOutput'][:, :50]
    dqdt0_us_test = sio.loadmat('../data/mat/unscaled_data/velocity0_test.mat')['testDataOutput'][:, :50]
#     dqdt1_us_train = sio.loadmat('../data/mat/unscaled_data/velocity1_train.mat')['trainingDataOutput'][:, :50]
#     dqdt1_us_test = sio.loadmat('../data/mat/unscaled_data/velocity1_test.mat')['testDataOutput'][:, :50]
    
    #Postion unscaled
    q0_us_train = sio.loadmat('../data/mat/unscaled_data/position0_train.mat')['trainingDataOutput'][:, :50]
    q0_us_test = sio.loadmat('../data/mat/unscaled_data/position0_test.mat')['testDataOutput'][:, :50]
#     q1_us_train = sio.loadmat('../data/mat/unscaled_data/position1_train.mat')['trainingDataOutput'][:, :50]
#     q1_us_test = sio.loadmat('../data/mat/unscaled_data/position1_test.mat')['testDataOutput'][:, :50]
    
    #Time unscaled
    time_us_train = sio.loadmat('../data/mat/unscaled_data/time_train.mat')['trainingDataOutput'][:, [0, 10]]
    time_us_test = sio.loadmat('../data/mat/unscaled_data/time_test.mat')['testDataOutput'][:, [0, 10]]

#     def convert(X):
#         X = to_torch(X)
#         x = X.repeat(enrich_size, 1, 1)
#         return torch.permute(x , (1,0,2))
    
#     def stack(tensor1, tensor2, dim=-1):
#         tensor1 = to_torch(tensor1)
#         tensor2 = to_torch(tensor2)
#         return torch.stack( [tensor1, tensor2], dim=dim )
    
    #Input
    I_train = to_torch(I_us_train)
    I_test = to_torch(I_us_test)
    
    
    #Velocity
    dqdt_us_train = to_torch(dqdt0_us_train) #stack( dqdt0_us_train, dqdt1_us_train )
    dqdt_us_test =  to_torch(dqdt0_us_test) #stack( dqdt0_us_test, dqdt1_us_test )
    
    #Position
    q_us_train = to_torch(q0_us_train) #stack( q0_us_train, q1_us_train )
    q_us_test = to_torch(q0_us_test) #stack( q0_us_test, q1_us_test )
    
    #Time
    time_train = to_torch(time_us_train ) 
    time_test = to_torch(time_us_test )
    
    Tpp_train = time_train[:, 0]
    Tep_train = time_train[:, 1]
    
    Tpp_test = time_test[:,0]
    Tep_test = time_test[:,1]

    return I_train, I_test, \
           Tpp_train, Tpp_test, \
           Tep_train, Tep_test, \
           q_us_train, q_us_test, \
           dqdt_us_train, dqdt_us_test


        
def sum_start_dim(a, start_dim=1):
    return a.reshape(list( a.shape[:start_dim])+[-1]).sum(axis=-1)        