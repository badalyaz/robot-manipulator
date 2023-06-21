#  Robot Manipulator Optimal Dynamic

<p align="center">
  <img width="500" height="500" src="https://www.trossenrobotics.com/Shared/Images/Product/ReactorX-200-Robot-Arm/RX200.jpg">
</p>
Considering the task of robot manipulator motion optimal dynamics were found. 
The robot manipulator comprises two arms and a free end for cargo movement. 
The motion of the free end describing by two angels <img src="https://latex.codecogs.com/svg.image?%5Cmathbf%7Bq%7D%20=%20(q_1,%20q_2)" style='width: 8%;'>

We need to find the optimal trajectory for a given pass point and end point 
### Dataset
We have approximately 70.000 different trajectories. Each trajectory consists 50 time points. The intervals between these points are equal to each other. 
For this time points are the following 
**positions/angels, velocities, torques, and initial conditions (which "parameterized" each trajectory)**. In addition for all trajectories pass point times are given:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\small&space;\mathbf{q}\in&space;\mathbb{R}^{2\times&space;50},\quad&space;\dot{\mathbf{q}}\in\mathbb{R}^{2\times&space;50},\quad&space;\tau&space;\in&space;\mathbb{R}^{1\times&space;50},\quad&space;I&space;\in&space;\mathbb{R}^{30}" style='width: 40%;'>
</p>

In addition there are two type of each quantity- *scaled* and *unscaled*.

Data path 
- *Positions*:

    Unscaled `/data/mat/unscaled_data/position0_train.mat`
    
    Scaled `data/mat/scaled_data/position0_train.mat`

- *Velocities*:

    Unscaled `/data/mat/unscaled_data/velocity0_train.mat`
    
    Scaled `/data/mat/scaled_data/velocity0_train.mat`

- *Torques*:

    Unscaled `/data/mat/unscaled_data/out_torque0_train.mat`
    
    Scaled `/data/mat/scaled_data/outandpred_torque0_train.mat`

- *Time*:

    Unscaled `/data/mat/unscaled_data/time_train.mat`
    
    Scaled `/data/mat/scaled_data/time_train.mat`
    
- *Initial Conditions*:
    
    Unscaled `/data/mat/unscaled_data/initial_conditions_train.mat`
    
    Scaled `/data/mat/scaled_data/initial_conditions_train.mat`


Example of the scaled and unscaled quantities (positions/velocities/torques):
<p align="center">
  <img src="./project_images/quantities.png" style='width: 100%;'>
</p>



### ***ODE***

These quantities are connected together by the ordinary differential equation of the following form
<p align="center"> 
  <img src="https://latex.codecogs.com/svg.image?\small&space;A(q)\cdot&space;\ddot{q}&space;&plus;&space;B(q,\dot{q})\cdot\dot{q}&space;&plus;&space;C(q)&space;=&space;\tau,&space;\quad&space;\text{where}" style='width: 40%;'>
</p>
<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\small&space;A,B\in&space;\mathbb{R}^{2\times&space;2}\quad&space;\text{and}&space;\quad&space;C\in\mathbb{R}^{2\times&space;1}" style='width: 30%;'>
</p>


Where A,B is the matrices and C is the vector, they also depended of the manipulator parameters: arm-s length, masses, and gravitational constant [[1]](#ref_1). This ODE describes the dynamics of the manipulator, but ODE (with the initial conditions) itself **doesn't determine** the optimal dynamic.

<p align="center">
  <img width="460" height="300" src="./project_images/arm.gif">
</p>

### ***Optimal Control Problem*** 
As mentioned above, the optimal trajectory strightly doesn't determinated by ODE. The optimal trajectory with subject to constraints  determinated by the following optimization problem 
<p align="center">
  <img src="https://latex.codecogs.com/svg.image?%5Cbegin%7Balign*%7D&%5Coverset%7B%7B%5Ctextstyle%5Ctext%7Bmin%7D%20%5C,%5C,%20T%7D%7D%7B%5Csubstack%7B%5Cscriptscriptstyle%5Ctau%20%3C%20%5Ctau_%7B%5Ctext%7Bmax%7D%7D%20%5C%5C%5Cscriptscriptstyle%20q(0)=%20q_0%20%5C%5C%20%5Cscriptscriptstyle%20%5Cdot%7Bq%7D(0)=%20%5Cdot%7Bq%7D_0%20%5C%5C%20%5Cscriptscriptstyle%20q(T_%7BPP%7D)=q_%7BPP%7D%20%5C%5C%20%5Cscriptscriptstyle%20q(T_%7BEP%7D)=q_%7BEP%7D%7D%7D%20%5C%5C%5C%5CT%20&=%20%5Cint_0%5E%7BT_%7BEP%7D%7D%20%5Cfrac%7B1%7D%7B%5Cdot%7Bq%7D%7D%5C,%20dt%5Cend%7Balign*%7D" style='width: 13%;'>
</p>
where the Tpp and Tep are the pass point and end point time respectively. One part of the constraints come from the ODE and another part from the several physical conditions.

## **Approaches**

- Since the trajectories are parameterized by the initial conditions then the initial conditions may be used as a net-s input.

- For all quantities (positions/velocities/torques) each time point can be trained separately (i.e. no one points “see” the other) and improve the results using another net (for example combined by LSTM and Dense layers).

- Residual Net

     The following scheme demonstrated the Res-Net working principle for positions (it also applies for velocities and torques).
     <img src='./project_images/best_net_schema.png' style='width: 100%;'>
  
- Data (for positions/velocities/torques) can be enriched by using the splines. This can be realized by using the `torchcubicspline` package. 
 
## **Training**

### ***Pretraining step***
When positions and velocities are considered, then it is two possibilities. **Firstly**, we can predict the position and after that using the autograd `torch.grad` w.r.t time steps find the velocities. **Or vice versa** using the predicted velocities we can integrate the net, for example, via the `odeint()` of the *Torchdiffeq* package [[2]](#ref_2).

According to the differentiation or integration we using the following scaling/unscaling procedures 
- Scalings and normalization according to the differentiation:
    In the pretraining step for the positions and velocities made the following transformations (using the mean value and standard deviation)
<p align="center">
  <img src="./project_images/scaling.png" style='width: 90%;'>
</p>


- Scaling and normalization according to the integrating:
    The scaling process according to the integrating is a litlie bit different form the scaling above:
<br>
<p align="center">
  <img src="./project_images/scaling_odeint.png" style='width: 90%;'>
</p>

As can be seen from the expressions during scaling, in addition to mean values and standard deviations, we also need the predicted end point time. For these purposes used the pretrained network, which is trained only for end point time.
Time net prediction plot 
<p align="center">
  <img src="./project_images/time_net.png" style='width: 90%;'>
</p>


it's accuracy (calculated by MSE Loss) is the **0.0005**. The time implementation is located `notebooks/model_archs/Time_Net.ipynb`


## **Best results**

### ***Net-s description***

In this section demonstrated the results which have been achieved using the following net-s:

- *Res-Net*
    
    This net described in the **Approaches** section scheme.
    
    From scaled initial conditions net predicts scaled torques for 50 time steps.

- *Separate-Net + LSTM*
    
    First step, using the scaled initial conditions trained each positions/velocities/torques separately for every 50 time point, i.e. according to the Separate-Net training no one points “see” the another point. After that pretrained quantities go through LSTM net which improve the previous step net results. For torques in final LSTM net also using the torques helpers.
    

### ***Results***
#### *Accuracy*
The losses computed by MSELoss

|Data | Torques (Res-Net) | Torques (Separate-Net + LSTM) | Positions and velocities |
|:-   |:-:   |:-:   |:-:  | 
|Full |0.0018|0.0018|0.0039|
|Test |0.0027|0.0031|0.0110| 
|Train|0.0017|0.0016|0.0037|

#### *Paths*
- **Torques (Res-Net) implementation**:  
    - Implementation is located `notebooks/model_archs/res_net_sequential_prediction_torque_best_result_test_and_train.ipynb`
    <!-- - Weights saved in `notebooks/model_archs/models/best_result_with_res_net.pt` -->
- **Torques (Separate-Net + LSTM)**:
    - Implementation is located `notebooks/model_archs/LSTM_Helpers_Res-net.ipynb`
    <!-- - Weights saved in `notebooks/model_archs/models/LSTM_HelpTorq_LossDiff.pt` -->
- **Positions and velocities**:
    - Implementation is located `notebooks/model_archs/Separate_nets_LSTM_dqdt_q.ipynb`
    <!-- - Weights saved in `notebooks/model_archs/models/dqdt_q_lstm.pt` -->

#### *Plots*
<center>

Torques (Res-Net)          | Torques (Separate-Net + LSTM) | Positions and velocities                                               
:-------------------------:|:-------------------------:    | :-------------------------:
![App Screenshot](./project_images/best_torque_results.png)   | ![App Screenshot](./project_images/lstm_res_sep_net_torque.png) | ![App Screenshot](./project_images/lstm_sepnet_q_dqdt.png)

</center>

## Citation
```
@article{
  title={High accuracy adaptive motion control for a robotic manipulator with model uncertainties based on multilayer neural network},
  authors={Hu, Jian, et al.},
  year={2022},
  doi={https://doi.org/10.3390/act11090255},
}
```

```
@article{
  title={Neural ordinary differential equations; Advances in neural information processing systems 31},
  authors={Chen, Ricky TQ, et al.},
  year={2018},
  doi={https://doi.org/10.48550/arXiv.1806.07366},
}
```
