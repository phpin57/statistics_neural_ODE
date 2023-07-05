import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.linalg import expm


import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=50)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--niters', type=int, default=150000)
parser.add_argument('--test_freq', type=int, default=100)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
path=r'D:/D/Stage 3A/Nouvelle approche/figures'
path=""
true_y0 = torch.randn(args.data_size, 2).to(device)
t = torch.tensor([0.,1.]).to(device)  # batch_t = [0, 1]


doubled=True
sqt_2=float(np.sqrt(2.))
true_B = torch.tensor([[-2.,0.],[0.,3.]])
non_diag=True
dim=true_B.shape[0]

if non_diag:
    B=true_B.clone()
    #matrice de rotation pour change
    P=torch.tensor([[sqt_2,-sqt_2],[sqt_2,sqt_2]]).to(device) 
    P_inv=torch.inverse(P)
    true_B=P_inv@B@P
true_y1=torch.mm(true_y0,true_B)
print(true_B)


b_values=scipy.linalg.eig(true_B)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def get_batch(true_y0,true_y1):
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size))
    if doubled:
        s=torch.cat((s,s+args.data_size))
    batch_y0 = true_y0[s]  # (M, D)
    batch_y = true_y1[s]  # (T, M, D)
    return batch_y0.to(device), t.to(device), batch_y.to(device)


if args.viz:
    makedirs('png')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


def plot_eigvals(values,targets,save):
    m=len(values[0])
    
    # Separate the real and imaginary parts of the complex values
    real_values = np.real(values)
    imaginary_values = np.imag(values)
    
    # Create x-axis values
    x = np.arange(len(values))
    
    # Create a 3D plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each complex number as a separate line
    for i in range(m):
        ax.plot(x, real_values[:, i], imaginary_values[:, i], label=f'Eigenvalue {i+1}')
    
    # Set labels for each axis
    ax.set_xlabel('Index')
    ax.set_ylabel('Real')
    ax.set_zlabel('Imaginary')
    
    ## Add annotations for the first and last points
    ax.text(x[0], real_values[0, 0], imaginary_values[0, 0], 'Start', fontsize=10, zorder=10, color='red')
    ax.text(x[-1], real_values[-1, -1], imaginary_values[-1, -1], 'End', fontsize=10, zorder=10, color='blue')
    
    # Create a separate subplot for the labels
    ax_labels = fig.add_subplot(1, 1, 1, frame_on=False)
    ax_labels.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ax_labels.set_xticks([])
    ax_labels.set_yticks([])
    ax_labels.set_xlabel('Eigenvalues of exp(A(s))', labelpad=20)
    
    
    # Add annotations for the labels
    #for i in range(m):
    #    ax_labels.text(1.02, 0.95 - (i * 0.2), f'Complex {i+1}', fontsize=10, color='black')
    
    # Plot the target points
    for i in range(m):
        target=targets[i]
        ax.scatter([x[-1]], [np.real(target)], [np.imag(target)], label='B eigenvalue '+str(i+1))
     
    # Adjust plot limits to avoid label conflicts
    ax.autoscale_view()
    
    # Add a legend
    ax.legend()
    
    # Display the plot
    plt.tight_layout()
    if save:
        plt.savefig('plot.png', dpi=1200)
    plt.show()



def augmente_tensor(x):
    n = x.shape[0]  # Number of tensors in x
    d = x.shape[1]  # Dimension of each tensor in x

    zeros = torch.zeros(n, d, device=x.device)  # Tensor of n zeros with dimension d

    result_tensors = []

    # Generating (x_i, 0_n) tensors
    for i in range(n):
        x_i_tensor = x[i]
        zeros_tensor = zeros[i]

        tensor1 = torch.cat((x_i_tensor, zeros_tensor), dim=0)  # Concatenate x_i with zero_n tensor
        result_tensors.append(tensor1)

    # Generating (0_n, x_i) tensors
    for i in range(n):
        x_i_tensor = x[i]
        zeros_tensor = zeros[i]

        tensor2 = torch.cat((zeros_tensor, x_i_tensor), dim=0)  # Concatenate zero_n tensor with x_i
        result_tensors.append(tensor2)

    final_tensor = torch.stack(result_tensors, dim=0)  # Stack all tensors into a single tensor

    return final_tensor


def augmente_B(B):
    n = B.shape[0]  # Size of matrix B

    result_matrix = np.zeros((2 * n, 2 * n))
    result_matrix[:n, :n] = B
    result_matrix[n:, n:] = B

    return result_matrix


def visualize(true_y, pred_y, odefunc, itr):

    if args.viz:

        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        ax_traj.set_ylim(-2, 2)
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_phase.set_xlim(-2, 2)
        ax_phase.set_ylim(-2, 2)

        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('x')
        ax_vecfield.set_ylabel('y')

        y, x = np.mgrid[-2:2:21j, -2:2:21j]
        dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)).cpu().detach().numpy()
        mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
        dydt = (dydt / mag)
        dydt = dydt.reshape(21, 21, 2)

        ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        ax_vecfield.set_xlim(-2, 2)
        ax_vecfield.set_ylim(-2, 2)

        fig.tight_layout()
        plt.savefig('png/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)


class ODEFunc(nn.Module):

    def __init__(self,nb_layers):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(nb_layers, nb_layers,bias=False),
            #nn.Tanh(),
            #nn.Linear(10, 2,bias=False),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                #nn.init.constant_(m.bias, val=0)
        self.A_init=[param.detach().numpy() for param in self.net.parameters()][0]
        print("ODEFunc initialized.")
        print("A_init ",self.A_init)
  
    def forward(self, t, y):
        return self.net(y)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


if __name__ == '__main__':
    loss_values = []

    ii = 0
    if doubled:
        dim=2*dim
        true_B=augmente_B(true_B.clone())
        b_values=scipy.linalg.eig(true_B)
    func = ODEFunc(dim).to(device)
    
    optimizer = optim.SGD(func.parameters(), lr=1e-1)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    
    loss_meter = RunningAverageMeter(0.97)
    eigenvalues=[]
    if doubled:
        true_y0=augmente_tensor(true_y0.clone())
        true_y1=augmente_tensor(true_y1.clone())
    #batch_y0, batch_t, batch_y = true_y0.to(device), t.to(device), true_y1.to(device)
    for itr in range(args.niters + 1):
        optimizer.zero_grad()
        
        batch_y0, batch_t, batch_y = get_batch(true_y0,true_y1)
        #print(batch_y0, batch_t, batch_y)
        pred_y = odeint(func, batch_y0, batch_t).to(device)
        loss = torch.mean((pred_y - batch_y)**2)
        
        loss.backward()
        optimizer.step()
        
        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())
        A_current=[param.detach().numpy() for param in func.parameters()][0]
        eigs=list(scipy.linalg.eig(A_current)[0])
        eigenvalues.append(eigs)
        

        if itr % args.test_freq == 0:
            exp_A=expm(A_current)
            frob_loss=scipy.linalg.norm(exp_A-np.array(true_B).T)
            loss_values.append(frob_loss)
            #loss_values.append(loss.item())
            #pred_y = odeint(func, true_y0, t)
            #loss = torch.mean(torch.abs(pred_y - true_y))
            print('Iter {:04d} | frob loss {:.6f}'.format(itr, frob_loss))
            #visualize(true_y, pred_y, func, ii)
            if frob_loss < 1e-3:
                break
            ii += 1

        end = time.time()
        
        
    plt.plot(loss_values[::2])
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('loss.png', dpi=1200)
    plt.show()
    A_eq=[param.detach().numpy() for param in func.parameters()][0]
    
    exp_eigvals=np.exp(np.array(eigenvalues))
    print("A_eq ",A_eq)
    print("exp(A_eq)",expm(A_eq))
