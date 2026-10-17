# Physics-Informed Neural Network for the 1D Burgers Equation

## Overview

This project implements a Physics-Informed Neural Network (PINN) using PyTorch to approximate the solution of the one-dimensional viscous Burgers equation.

Unlike conventional neural networks that rely exclusively on labeled training data, PINNs incorporate the governing physical equations directly into the loss function. Automatic differentiation is used to calculate the derivatives required to enforce the partial differential equation (PDE).

## Governing Equation

The one-dimensional viscous Burgers equation is defined as

```math
\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} = \nu\frac{\partial^2 u}{\partial x^2}
```

where:

- $u(x,t)$ is the velocity field.
- $x$ is the spatial coordinate.
- $t$ is time.
- $\nu$ is the kinematic viscosity.

The computational domain is

```math
x\in[-1,1], \qquad t\in[0,1]
```

with the kinematic viscosity

```math
\nu=\frac{0.01}{\pi}
```

## Methodology

The PINN approximates the solution using a fully connected feedforward neural network. The governing PDE is enforced through a physics-based residual term in the loss function.

- **Framework:** PyTorch
- **Neural network:** Fully connected feedforward network
- **Input:** Spatial coordinate $x$ and time $t$
- **Output:** Velocity field $u(x,t)$
- **Hidden layers:** 8
- **Neurons per hidden layer:** 20
- **Activation function:** Tanh
- **Optimizer:** Adam
- **Learning rate:** 0.001
- **Training epochs:** 4,000
- **Collocation points:** 10,000
- **Initial and boundary condition points:** 200
- **Loss function:** Mean Squared Error (MSE)

## Initial and Boundary Conditions

### Initial Condition

The initial velocity distribution is defined as

```math
u(x,0)=-\sin(\pi x)
```

### Boundary Conditions

Dirichlet boundary conditions are imposed at both ends of the spatial domain:

```math
u(-1,t)=0
```

```math
u(1,t)=0
```

These conditions are enforced through the corresponding loss terms during training.

## Physics-Informed Loss Function

The total loss function combines the initial condition loss, boundary condition loss, and PDE residual loss:

```math
\mathcal{L} = \mathcal{L}_{IC} + \mathcal{L}_{BC} + \mathcal{L}_{PDE}
```

where:

- $\mathcal{L}_{IC}$ measures the error in satisfying the initial condition.
- $\mathcal{L}_{BC}$ measures the error in satisfying the boundary conditions.
- $\mathcal{L}_{PDE}$ penalizes violations of the governing Burgers equation.

The PDE residual is defined as

```math
f(x,t) = \frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} - \nu\frac{\partial^2 u}{\partial x^2}
```

Automatic differentiation in PyTorch is used to calculate the required derivatives.

## Results and Visualization

The trained neural network is evaluated on a $100\times100$ space-time grid. The predicted velocity field is visualized using a contour plot to illustrate the evolution of the solution over time.

The implementation also tracks the total training loss and PDE residual loss.

<img width="851" height="549" alt="indir" src="https://github.com/user-attachments/assets/7f90436c-db6f-4fcd-b7d6-b5f965d1321b" />

## Technologies

- Python
- PyTorch
- NumPy
- Matplotlib
- Automatic Differentiation
- Physics-Informed Neural Networks

## Installation

Install the required dependencies:

```bash
pip install torch numpy matplotlib
```

## How to Run

Run the Python implementation:

```bash
python pinn_burgers.py
```

The script generates the following visualization files:

- `burgers_pinn_contour.png` — predicted velocity field.
- `training_loss.png` — training loss history.

## Future Improvements

- Validate the PINN solution against a finite-difference reference solution.
- Investigate convergence and sensitivity to collocation-point sampling.
- Explore adaptive loss weighting.
- Compare different neural network architectures.
- Evaluate the accuracy of the predicted solution near steep gradients and shock-like structures.
