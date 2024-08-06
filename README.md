##### Expressivity of Neural Network Trajectory Arc Length w/r to Layer Weights

This is a Dash App that helps visualize how a simple trajectory over a 2-layer shallow neural network can change with respect to the weights. 

This is mostly a fun visualization tool that I built out while reading the paper [On the Expressive Power of Deep Neural Networks](https://proceedings.mlr.press/v70/raghu17a.html) where two of four claims made are that:

(2) there is exponential depth dependence that can be validated by measuring trajectory length, and 
(3) lower weights matter more (and trajectory length should therefore be impacted more by changes to lower weights)

#### Background
A 2 hidden layer shallow neural network with RELU activations can be expressed explicitly as follows:

$f_0(X) = ((XW_1)_{+} W_2)_{+} w_3$

Although the dimensions can be generalized, for the purpose of visualizing a surface, we let $X \in R^{\text{nx2}}$,
where n is the number of data points, and hidden layers $W_1$ and $W_2$ have four nodes that are fully connected, 
and the output is a scalar. Therefore, $W_1 \in R^{\text{2x4}}, W_2 \in R^{\text{4x4}}, w_3 \in R^{4}$

#### Current features:
- [x] Adjust each of the hidden layer weights: $W_1, W_2$
- [x] Trajectory is superimposed on the surface on the plot

#### Features in progress: 
- [ ] Calculate the arc length of the trajectory
- [ ] Adjust the output layer weights
- [ ] Adjust the radius of the trajectory
