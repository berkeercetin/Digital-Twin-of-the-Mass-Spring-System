import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"

device = torch.device(dev)

k,m = 20.45, 0.24
mu=m
x0 = 0.08  # m 0.1
v0 = 0  # m/s 0

def exact_solution(t):
    w=np.sqrt(k/m)
    u=x0*np.cos(w*t)
    return u


"""
def exact_solution(d, w0, t):
    "Defines the analytical solution to the under-damped harmonic oscillator problem above."
    assert d < w0
    w = np.sqrt(w0**2-d**2)
    phi = np.arctan(-d/w)
    A = 1/(2*np.cos(phi))
    cos = torch.cos(phi+w*t)
    exp = torch.exp(-d*t)
    u = exp*2*A*cos
    return u
"""
class FCN(nn.Module):
    "Defines a standard fully-connected network in PyTorch"

    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x
    
torch.manual_seed(123)
pinn = FCN(1,1,32,3)
t_boundary = torch.tensor(0.).view(-1,1).requires_grad_(True)
optimiser = torch.optim.Adam(pinn.parameters(),lr=1e-3)

def main(sensor_x_data,timeSecond):
    i = 0
    while True:
        t_physics = torch.linspace(0,time.time()-timeSecond,30).view(-1,1).requires_grad_(True)
        t_test = torch.linspace(0,time.time()-timeSecond,300).view(-1,1)
        u_exact = exact_solution(t_test)
        optimiser.zero_grad()

        # compute each term of the PINN loss function above
        # using the following hyperparameters:
        lambda1, lambda2, lambda3 = 1e-1, 1e-4, 1e-2  # lambda3, sensör verileri için eklenen yeni bir hiperparametre

        # compute boundary loss
        u = pinn(t_boundary)
        loss1 = (torch.squeeze(u) - x0)**2
        dudt = torch.autograd.grad(u, t_boundary, torch.ones_like(u), create_graph=True)[0]
        loss2 = (torch.squeeze(dudt) - v0)**2

        # compute physics loss
        u = pinn(t_physics)
        dudt = torch.autograd.grad(u, t_physics, torch.ones_like(u), create_graph=True)[0]
        d2udt2 = torch.autograd.grad(dudt, t_physics, torch.ones_like(dudt), create_graph=True)[0]
        loss3 = torch.mean((mu*d2udt2 + k*u)**2)



        # Sensör verileri için kayıp hesaplanmasi
        #sensör verileri için bir kayıp hesaplanıyor ve ardından bu kayıp, modelin sensör verilerini doğru bir şekilde öğrenip öğrenmediğini belirlemek için kullanılıyor.
        # Kayıp hesaplanırken, tahmin edilen sensör verileri (u) ile gerçek sensör verileri (sensor_data) arasındaki ortalama kare hatası hesaplanıyor.
        # Eğer kayıp düşükse, bu, modelin sensör verilerini doğru bir şekilde öğrendiğini gösterebilir.
        u = pinn(sensor_data)
        loss_sensor = torch.mean((u - sensor_data)**2)  # MSE kullanılarak hesaplama

        # backpropagate joint loss, take optimiser step
        loss = loss1 + lambda1*loss2 + lambda2*loss3 + lambda3*loss_sensor
        #loss = loss1 + lambda1*loss2 + lambda2*loss3
        loss.backward()
        optimiser.step()

        if i % 1000 == 0:
            print(f"Step {i}, Loss {loss.item()}")

                # plot the result as training progresses
        if i % 5000 == 0 and i > 0 or i == len(sensor_x_data)*1000-1:

            #print(u.abs().mean().item(), dudt.abs().mean().item(), d2udt2.abs().mean().item())
            u = pinn(t_test).detach()
            plt.figure(figsize=(6,2.5))
            plt.scatter(t_physics.detach()[:,0],
                        torch.zeros_like(t_physics)[:,0], s=20, lw=0, color="tab:green", alpha=0.6)
            plt.scatter(t_boundary.detach()[:,0],
                        torch.zeros_like(t_boundary)[:,0], s=20, lw=0, color="tab:red", alpha=0.6)
            plt.plot(t_test[:,0], u_exact[:,0], label="Exact solution", color="tab:grey", alpha=0.6)
            plt.plot(t_test[:,0], u[:,0], label="PINN solution", color="tab:green")
            plt.title(f"Training step {i}")
            plt.legend()
            plt.show()
            
    return 



#07.05.2924
# Egitim zamani egitim datasinda nasil degisecek ?

# yay durmuyor bizim modelimizde.