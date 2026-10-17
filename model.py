import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import torch.optim as optim

N=10000 #Node number
t=torch.rand(N,1)
x=2*torch.rand(N,1)-1.0

t.requires_grad = True
x.requires_grad = True

n=200 # For initial and boundary conditions

#Initial Condition u(x,0)=sin(pi*x)

x_in= 2*torch.rand(n,1)-1.0
t_in=torch.zeros(n,1)
u_in=-torch.sin(torch.pi * x_in)

# Boundary Conditions

#Time
t_b=torch.rand(n,1)

#Left and Right walls
x_b_right=torch.ones(n,1)
x_b_left=-torch.ones(n,1)

#No slip at boundaries (Dirichlet)
u_b_right=torch.zeros(n,1)


class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()

        # 2 Inputs 1 output 8 Hidden Layer each one contains 20 neroun
        self.net = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        u_pred = self.net(inputs)
        return u_pred


def pde_residual(model, x, t):
    # First guess
    u = model(x, t)

    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True, retain_graph=True)[0] # du/dt
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0] # du/dx
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0] # d^2u/dx^2

    # Defining kinematic viscosity
    nu = 0.01 / torch.pi

    f_pred = u_t + u * u_x - nu * u_xx

    return f_pred

    model = PINN()

#Training Loop

optimizer = optim.Adam(model.parameters(), lr=0.001) # Learning rate equals 0.001
mse_loss = nn.MSELoss() # MSE for loss function
epochs = 4000

for epoch in range(epochs):

  optimizer.zero_grad() # Setting gradients to 0

  #Initial condition loss
  u_in_pred = model(x_in, t_in)
  loss_in = mse_loss(u_in_pred, u_in)

  #Boundary condition loss
  u_bc_left_pred = model(x_b_left, t_b)
  u_bc_right_pred = model(x_b_right, t_b)
  loss_bc = mse_loss(u_bc_left_pred, u_b_left) + mse_loss(u_bc_right_pred, u_b_right)

  #Physical loss
  f_pred = pde_residual(model, x, t)
  loss_pde = mse_loss(f_pred, torch.zeros_like(f_pred))

  #Total loss
  loss = loss_in + loss_bc + loss_pde

  loss.backward()
  optimizer.step()

  if epoch % 50 == 0:
        print(f'Epoch: {epoch:04d} | Toplam Loss: {loss.item():.6f} | PDE Loss: {loss_pde.item():.6f}')


#Grid creation
x_test = torch.linspace(-1, 1, 100).view(-1, 1)
t_test = torch.linspace(0, 1, 100).view(-1, 1)
X, T = torch.meshgrid(x_test.squeeze(), t_test.squeeze(), indexing='ij')

x_flat = X.flatten().unsqueeze(1)
t_flat = T.flatten().unsqueeze(1)

#Model Testing
model.eval()
with torch.no_grad():
    u_pred = model(x_flat, t_flat)

u_pred_2d = u_pred.view(100, 100).numpy()

#Contour

plt.figure(figsize=(10, 6))
plt.contourf(T.numpy(), X.numpy(), u_pred_2d, levels=100, cmap='jet')
plt.colorbar(label='Velocity ($u$)')
plt.xlabel('Time ($t$)')
plt.ylabel('Location ($x$)')
plt.title('PINN Output: 1D Burgers Equation Shock Wave Propagation')
plt.show()
