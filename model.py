
import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Reproducibility
torch.manual_seed(42)

# --------------------------------------------------
# 1. Collocation points
# --------------------------------------------------

N = 10000  # Number of collocation points

t = torch.rand(N, 1, requires_grad=True)
x = 2 * torch.rand(N, 1) - 1.0
x.requires_grad_(True)

n = 200  # Number of initial and boundary condition points

# --------------------------------------------------
# 2. Initial condition: u(x, 0) = -sin(pi*x)
# --------------------------------------------------

x_in = 2 * torch.rand(n, 1) - 1.0
t_in = torch.zeros(n, 1)
u_in = -torch.sin(torch.pi * x_in)

# --------------------------------------------------
# 3. Boundary conditions: u(-1,t) = u(1,t) = 0
# --------------------------------------------------

t_b = torch.rand(n, 1)

x_b_left = -torch.ones(n, 1)
x_b_right = torch.ones(n, 1)

u_b_left = torch.zeros(n, 1)
u_b_right = torch.zeros(n, 1)


# --------------------------------------------------
# 4. PINN architecture
# --------------------------------------------------

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()

        # 2 inputs, 8 hidden layers (20 neurons each), 1 output
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
        return self.net(inputs)


# --------------------------------------------------
# 5. PDE residual: viscous Burgers equation
#    u_t + u*u_x - nu*u_xx = 0
# --------------------------------------------------

def pde_residual(model, x, t):

    u = model(x, t)

    u_t = torch.autograd.grad(
        u, t,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )[0]

    u_x = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )[0]

    u_xx = torch.autograd.grad(
        u_x, x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0]

    nu = 0.01 / torch.pi

    f_pred = u_t + u * u_x - nu * u_xx

    return f_pred


# --------------------------------------------------
# 6. Model, optimizer and loss function
# --------------------------------------------------

model = PINN()

optimizer = optim.Adam(model.parameters(), lr=0.001)
mse_loss = nn.MSELoss()

epochs = 4000

loss_history = []
pde_loss_history = []


# --------------------------------------------------
# 7. Training loop
# --------------------------------------------------

for epoch in range(epochs):

    optimizer.zero_grad()

    # Initial condition loss
    u_in_pred = model(x_in, t_in)
    loss_in = mse_loss(u_in_pred, u_in)

    # Boundary condition loss
    u_bc_left_pred = model(x_b_left, t_b)
    u_bc_right_pred = model(x_b_right, t_b)

    loss_bc = (
        mse_loss(u_bc_left_pred, u_b_left)
        + mse_loss(u_bc_right_pred, u_b_right)
    )

    # Physics loss
    f_pred = pde_residual(model, x, t)
    loss_pde = mse_loss(f_pred, torch.zeros_like(f_pred))

    # Total loss
    loss = loss_in + loss_bc + loss_pde

    loss.backward()
    optimizer.step()

    # Save loss values
    loss_history.append(loss.item())
    pde_loss_history.append(loss_pde.item())

    if epoch % 50 == 0:
        print(
            f"Epoch: {epoch:04d} | "
            f"Total Loss: {loss.item():.6e} | "
            f"IC Loss: {loss_in.item():.6e} | "
            f"BC Loss: {loss_bc.item():.6e} | "
            f"PDE Loss: {loss_pde.item():.6e}"
        )


# --------------------------------------------------
# 8. Create the space-time evaluation grid
# --------------------------------------------------

x_test = torch.linspace(-1, 1, 100).view(-1, 1)
t_test = torch.linspace(0, 1, 100).view(-1, 1)

X, T = torch.meshgrid(
    x_test.squeeze(),
    t_test.squeeze(),
    indexing="ij"
)

x_flat = X.reshape(-1, 1)
t_flat = T.reshape(-1, 1)


# --------------------------------------------------
# 9. Model evaluation
# --------------------------------------------------

model.eval()

with torch.no_grad():
    u_pred = model(x_flat, t_flat)

u_pred_2d = u_pred.reshape(100, 100).cpu().numpy()


# --------------------------------------------------
# 10. Plot predicted velocity field
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.contourf(
    T.numpy(),
    X.numpy(),
    u_pred_2d,
    levels=100,
    cmap="jet"
)

plt.colorbar(label="Velocity (u)")
plt.xlabel("Time (t)")
plt.ylabel("Location (x)")
plt.title("PINN Solution of the 1D Burgers Equation")
plt.tight_layout()
plt.savefig("burgers_pinn_contour.png", dpi=300)
plt.show()


# --------------------------------------------------
# 11. Plot training loss
# --------------------------------------------------

plt.figure(figsize=(10, 5))

plt.semilogy(loss_history, label="Total Loss")
plt.semilogy(pde_loss_history, label="PDE Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss (log scale)")
plt.title("PINN Training History")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("training_loss.png", dpi=300)
plt.show()

