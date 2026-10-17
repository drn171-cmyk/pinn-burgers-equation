# Physics-Informed Neural Network for the 1D Burgers Equation

## Overview

This project implements a Physics-Informed Neural Network (PINN) using PyTorch to approximate the solution of the one-dimensional viscous Burgers equation.

Unlike conventional neural networks that rely exclusively on labeled training data, PINNs incorporate the governing physical equations directly into the loss function. Automatic differentiation is used to calculate the derivatives required to enforce the PDE.

## Governing Equation

The one-dimensional viscous Burgers equation is defined as

\[
\frac{\partial u}{\partial t}
+u\frac{\partial u}{\partial x}
-\nu\frac{\partial^2 u}{\partial x^2}=0
\]

where:

- \(u(x,t)\) is the velocity field.
- \(x\) is the spatial coordinate.
- \(t\) is time.
- \(\nu\) is the kinematic viscosity.

The computational domain is \(x\in[-1,1]\), \(t\in[0,1]\), with \(\nu=0.01/\pi\).

## Methodology

- **Neural network:** Fully connected feedforward network with 2 inputs, 8 hidden layers of 20 neurons, and 1 output.
- **Activation function:** Hyperbolic tangent (Tanh).
- **Automatic differentiation:** Computes the temporal derivative, first spatial derivative, and second spatial derivative.
- **Optimizer:** Adam.
- **Learning rate:** 0.001.
- **Training epochs:** 4,000.
- **Loss function:** Mean squared error combining initial conditions, boundary conditions, and PDE residuals.

## Physics-Informed Loss Function

The total loss is defined as

\[
\mathcal{L}=\mathcal{L}_{IC}+\mathcal{L}_{BC}+\mathcal{L}_{PDE}
\]

where the individual terms enforce the initial condition, boundary conditions, and governing equation, respectively.

## Initial and Boundary Conditions

The initial condition is

\[
u(x,0)=-\sin(\pi x)
\]

The boundary conditions are

\[
u(-1,t)=u(1,t)=0
\]

The initial and boundary conditions are enforced through supervised loss terms, while the PDE residual is evaluated at randomly sampled collocation points.

## Technologies

- Python
- PyTorch
- NumPy
- Matplotlib
- Automatic differentiation
- Physics-Informed Neural Networks

## Results

The trained network is evaluated on a 100 × 100 space-time grid. A contour plot visualizes the predicted velocity field and its temporal evolution.

<img width="851" height="549" alt="indir" src="https://github.com/user-attachments/assets/a6c9eb38-fb1e-4ea3-bf08-83f11c520e17" />


## How to Run

Install the required packages:

```bash
pip install torch numpy matplotlib scikit-learn
```

Run the Python script:

```bash
python pinn_burgers.py
```

## Future Improvements

- Compare PINN predictions with a numerical reference solution.
- Track and visualize individual loss components during training.
- Experiment with network depth, width, and collocation-point sampling.
- Investigate adaptive loss weighting and alternative optimization strategies.

