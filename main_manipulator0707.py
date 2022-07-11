import torch
import torch.nn as nn
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from get_G import G_mtx
from get_M import M_mtx
from get_C import C_mtx


n = 7 # DOF
sim_FT = 3
sim_period = 0.001
sam=int(sim_FT/sim_period)
tspan = np.linspace(0,sim_FT, sam+1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_layer1 = nn.Linear(1,5)
        self.hidden_layer2 = nn.Linear(5,5)
        self.hidden_layer3 = nn.Linear(5,5)
        self.hidden_layer4 = nn.Linear(5,5)
        self.hidden_layer5 = nn.Linear(5,5)
        self.output_layer = nn.Linear(5,1)

    def forward(self, t):
        inputs = torch.cat([t],axis=1) # combined two arrays of 1 columns each to one array of 2 columns
        layer1_out = torch.sigmoid(self.hidden_layer1(inputs))
        layer2_out = torch.sigmoid(self.hidden_layer2(layer1_out))
        layer3_out = torch.sigmoid(self.hidden_layer3(layer2_out))
        layer4_out = torch.sigmoid(self.hidden_layer4(layer3_out))
        layer5_out = torch.sigmoid(self.hidden_layer5(layer4_out))
        output = self.output_layer(layer5_out) ## For regression, no activation is used in output layer
        return output

### (2) Model
net_q1 = Net()
net_q2 = Net()
net_q3 = Net()
net_q4 = Net()
net_q5 = Net()
net_q6 = Net()
net_q7 = Net()

net_q1 = net_q1.to(device)
net_q2 = net_q2.to(device)
net_q3 = net_q3.to(device)
net_q4 = net_q4.to(device)
net_q5 = net_q5.to(device)
net_q6 = net_q6.to(device)
net_q7 = net_q7.to(device)

mse_cost_function = torch.nn.MSELoss() # Mean squared error
optimizer1 = torch.optim.Adam(net_q1.parameters())
optimizer2 = torch.optim.Adam(net_q2.parameters())
optimizer3 = torch.optim.Adam(net_q3.parameters())
optimizer4 = torch.optim.Adam(net_q4.parameters())
optimizer5 = torch.optim.Adam(net_q5.parameters())
optimizer6 = torch.optim.Adam(net_q6.parameters())
optimizer7 = torch.optim.Adam(net_q7.parameters())
# --------------------------------------------------------------------0629----------------------------------------------

# q_bc = np.random.uniform(low=-np.pi/2, high=np.pi/2, size=(500,1))
t_bc = np.zeros((np.size(tspan),1))

def plant(x, u1, M, C, G):
    G = G.tolist()
    G = [G[0][0],G[1][0],G[2][0],G[3][0],G[4][0],G[5][0],G[6][0]]
    G = np.array(G)
    C = C.tolist()
    C = [C[0][0], C[1][0], C[2][0], C[3][0], C[4][0], C[5][0], C[6][0]]
    C = np.array(C)
    # Case 2: Compensating Real Dynamics
    val2 = np.matmul(np.linalg.inv(M),u1)
    dxdt = [x[7],x[8],x[9],x[10],x[11],x[12],x[13],val2[0],val2[1],val2[2],val2[3],val2[4],val2[5],val2[6]]
    arr_dxdt = np.array(dxdt)
    return arr_dxdt

def rk4(x,tau,T,M,C,G):
    k1=plant(x,tau,M,C,G)*T
    k2=plant(x+k1*0.5,tau,M,C,G)*T
    k3=plant(x+k2*0.5,tau,M,C,G)*T
    k4=plant(x+k3,tau,M,C,G)*T
    dx = x + ((k1+k4)/6+(k2+k3)/3)
    return dx

# num_data_q = 2000
# t_bc = np.zeros((num_data_q,1))
# compute u based on BC
qt_temp = []
qdt_temp = []
error_q_qt = []
error_qd_qt = []
q = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
q_t = [0, 0, 1.57, 0, 0, 0, 0]
qd_t = [0, 0, 0, 0, 0, 0, 0]
Kp = np.array([20, 15, 5, 1, 1, 1, 1])
Kd = np.sqrt(Kp)*0.5

q1_data = np.array([0])
q2_data = np.array([0])
q3_data = np.array([0])
q4_data = np.array([0])
q5_data = np.array([0])
q6_data = np.array([0])
q7_data = np.array([0])

for i in range(0,np.size(tspan)-1):

    qt_temp.append(q_t)
    qdt_temp.append(qd_t)

    arr_q = np.array(q[i])
    arr_q_t = np.array(q_t)
    arr_qd_t = np.array(qd_t)
    arr_q_e = arr_q_t - arr_q[0:n]
    arr_qd_e = arr_qd_t - arr_q[n:2*n]

    error_q = arr_q[0:n] - arr_q_t
    error_qd = arr_q[n:2*n] - arr_qd_t
    error_q = error_q.tolist()
    error_qd = error_qd.tolist()
    error_q_qt.append(error_q)
    error_qd_qt.append(error_qd)

    M = M_mtx(arr_q)
    C = C_mtx(arr_q[0:n],arr_q[n:2*n])
    G = G_mtx(arr_q)

    tau = Kp * arr_q_e + Kd * arr_qd_e
    tau = np.array(tau)
    q_value = rk4(arr_q, tau, sim_period, M, C, G)

    q_value = q_value.tolist()
    q.append(q_value)
    arr_q_sol = np.array(q_value)

    q1_data = np.vstack([q1_data, q_value[0]])
    q2_data = np.vstack([q2_data, q_value[1]])
    q3_data = np.vstack([q3_data, q_value[2]])
    q4_data = np.vstack([q4_data, q_value[3]])
    q5_data = np.vstack([q5_data, q_value[4]])
    q6_data = np.vstack([q6_data, q_value[5]])
    q7_data = np.vstack([q7_data, q_value[6]])

#
### (3) Training / Fitting
iterations = 1000
previous_validation_loss = 99999999.0
q_temp = np.zeros([np.size(tspan),7])
qd_temp = np.zeros((np.size(tspan),7))
qdd_temp = np.zeros((np.size(tspan),7))
q_cal_temp = np.zeros((np.size(tspan),7))
f_q1 = np.zeros((np.size(tspan),1))
f_q2 = np.zeros((np.size(tspan),1))
f_q3 = np.zeros((np.size(tspan),1))
f_q4 = np.zeros((np.size(tspan),1))
f_q5 = np.zeros((np.size(tspan),1))
f_q6 = np.zeros((np.size(tspan),1))
f_q7 = np.zeros((np.size(tspan),1))
loss_temp = np.array([0])

for epoch in range(iterations):
    optimizer1.zero_grad()  # to make the gradients zero
    optimizer2.zero_grad()  # to make the gradients zero
    optimizer3.zero_grad()  # to make the gradients zero
    optimizer4.zero_grad()  # to make the gradients zero
    optimizer5.zero_grad()  # to make the gradients zero
    optimizer6.zero_grad()  # to make the gradients zero
    optimizer7.zero_grad()  # to make the gradients zero

    # Loss based on boundary conditions
    pt_t_bc = Variable(torch.from_numpy(t_bc).float(), requires_grad=False).to(device)

    pt_q1_bc = Variable(torch.from_numpy(q1_data).float(), requires_grad=False).to(device)
    pt_q2_bc = Variable(torch.from_numpy(q2_data).float(), requires_grad=False).to(device)
    pt_q3_bc = Variable(torch.from_numpy(q3_data).float(), requires_grad=False).to(device)
    pt_q4_bc = Variable(torch.from_numpy(q4_data).float(), requires_grad=False).to(device)
    pt_q5_bc = Variable(torch.from_numpy(q5_data).float(), requires_grad=False).to(device)
    pt_q6_bc = Variable(torch.from_numpy(q6_data).float(), requires_grad=False).to(device)
    pt_q7_bc = Variable(torch.from_numpy(q7_data).float(), requires_grad=False).to(device)

    net_q1_out = net_q1(pt_t_bc)  # output of u(x,t)
    net_q2_out = net_q2(pt_t_bc)  # output of u(x,t)
    net_q3_out = net_q3(pt_t_bc)  # output of u(x,t)
    net_q4_out = net_q4(pt_t_bc)  # output of u(x,t)
    net_q5_out = net_q5(pt_t_bc)  # output of u(x,t)
    net_q6_out = net_q6(pt_t_bc)  # output of u(x,t)
    net_q7_out = net_q7(pt_t_bc)  # output of u(x,t)

    mse_q1 = mse_cost_function(net_q1_out, pt_q1_bc)
    mse_q2 = mse_cost_function(net_q2_out, pt_q2_bc)
    mse_q3 = mse_cost_function(net_q3_out, pt_q3_bc)
    mse_q4 = mse_cost_function(net_q4_out, pt_q4_bc)
    mse_q5 = mse_cost_function(net_q5_out, pt_q5_bc)
    mse_q6 = mse_cost_function(net_q6_out, pt_q6_bc)
    mse_q7 = mse_cost_function(net_q7_out, pt_q7_bc)

    mse_q = mse_q1 + mse_q2 + mse_q3 + mse_q4 + mse_q5 + mse_q6 + mse_q7

    # Loss based on PDE
    t_collocation = np.random.uniform(low=0.0, high=1.0, size=(np.size(tspan), 1))
    all_zeros = np.zeros((np.size(tspan), 1))
    pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True).to(device)
    pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)

    net_q1_vec = net_q1(pt_t_collocation)
    net_q2_vec = net_q2(pt_t_collocation)
    net_q3_vec = net_q3(pt_t_collocation)
    net_q4_vec = net_q4(pt_t_collocation)
    net_q5_vec = net_q5(pt_t_collocation)
    net_q6_vec = net_q6(pt_t_collocation)
    net_q7_vec = net_q7(pt_t_collocation)

    net_q1_vec_t = torch.autograd.grad(net_q1_vec.sum(), pt_t_collocation, create_graph=True)[0]
    net_q2_vec_t = torch.autograd.grad(net_q2_vec.sum(), pt_t_collocation, create_graph=True)[0]
    net_q3_vec_t = torch.autograd.grad(net_q3_vec.sum(), pt_t_collocation, create_graph=True)[0]
    net_q4_vec_t = torch.autograd.grad(net_q4_vec.sum(), pt_t_collocation, create_graph=True)[0]
    net_q5_vec_t = torch.autograd.grad(net_q5_vec.sum(), pt_t_collocation, create_graph=True)[0]
    net_q6_vec_t = torch.autograd.grad(net_q6_vec.sum(), pt_t_collocation, create_graph=True)[0]
    net_q7_vec_t = torch.autograd.grad(net_q7_vec.sum(), pt_t_collocation, create_graph=True)[0]

    net_q1_vec_tt = torch.autograd.grad(net_q1_vec_t.sum(), pt_t_collocation, create_graph=True)[0]
    net_q2_vec_tt = torch.autograd.grad(net_q2_vec_t.sum(), pt_t_collocation, create_graph=True)[0]
    net_q3_vec_tt = torch.autograd.grad(net_q3_vec_t.sum(), pt_t_collocation, create_graph=True)[0]
    net_q4_vec_tt = torch.autograd.grad(net_q4_vec_t.sum(), pt_t_collocation, create_graph=True)[0]
    net_q5_vec_tt = torch.autograd.grad(net_q5_vec_t.sum(), pt_t_collocation, create_graph=True)[0]
    net_q6_vec_tt = torch.autograd.grad(net_q6_vec_t.sum(), pt_t_collocation, create_graph=True)[0]
    net_q7_vec_tt = torch.autograd.grad(net_q7_vec_t.sum(), pt_t_collocation, create_graph=True)[0]

    q1 = net_q1_vec.detach().cpu().numpy()
    q2 = net_q2_vec.detach().cpu().numpy()
    q3 = net_q3_vec.detach().cpu().numpy()
    q4 = net_q4_vec.detach().cpu().numpy()
    q5 = net_q5_vec.detach().cpu().numpy()
    q6 = net_q6_vec.detach().cpu().numpy()
    q7 = net_q7_vec.detach().cpu().numpy()

    q1_t = net_q1_vec_t.detach().cpu().numpy()
    q2_t = net_q2_vec_t.detach().cpu().numpy()
    q3_t = net_q3_vec_t.detach().cpu().numpy()
    q4_t = net_q4_vec_t.detach().cpu().numpy()
    q5_t = net_q5_vec_t.detach().cpu().numpy()
    q6_t = net_q6_vec_t.detach().cpu().numpy()
    q7_t = net_q7_vec_t.detach().cpu().numpy()

    q1_tt = net_q1_vec_tt.detach().cpu().numpy()
    q2_tt = net_q1_vec_tt.detach().cpu().numpy()
    q3_tt = net_q3_vec_tt.detach().cpu().numpy()
    q4_tt = net_q4_vec_tt.detach().cpu().numpy()
    q5_tt = net_q5_vec_tt.detach().cpu().numpy()
    q6_tt = net_q6_vec_tt.detach().cpu().numpy()
    q7_tt = net_q7_vec_tt.detach().cpu().numpy()
    ## 22.07.05 ###############################################################################
    for j in range(0,np.size(tspan)):
        q_temp[j,0] = q1[j,0]
        q_temp[j,1] = q2[j,0]
        q_temp[j,2] = q3[j,0]
        q_temp[j,3] = q4[j,0]
        q_temp[j,4] = q5[j,0]
        q_temp[j,5] = q6[j,0]
        q_temp[j,6] = q7[j,0]

        qd_temp[j,0] = q1_t[j,0]
        qd_temp[j,1] = q2_t[j,0]
        qd_temp[j,2] = q3_t[j,0]
        qd_temp[j,3] = q4_t[j,0]
        qd_temp[j,4] = q5_t[j,0]
        qd_temp[j,5] = q6_t[j,0]
        qd_temp[j,6] = q7_t[j,0]

        qdd_temp[j,0] = q1_tt[j,0]
        qdd_temp[j,1] = q2_tt[j,0]
        qdd_temp[j,2] = q3_tt[j,0]
        qdd_temp[j,3] = q4_tt[j,0]
        qdd_temp[j,4] = q5_tt[j,0]
        qdd_temp[j,5] = q6_tt[j,0]
        qdd_temp[j,6] = q7_tt[j,0]

        M = M_mtx(q_temp[j])
        G = G_mtx(q_temp[j])
        C = C_mtx(q_temp[j], qd_temp[j])
        tau = Kp * (q_temp[j]-arr_q_t) + Kd * (qd_temp[j]- arr_qd_t)
        tau = np.array([[tau[0]], [tau[1]], [tau[2]], [tau[3]], [tau[4]], [tau[5]], [tau[6]]])

        # u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        q_cal_temp[j] = qdd_temp[j] - (np.linalg.inv(M) @ (tau - C - G)).T[0]


        f_q1[j] = q_cal_temp[j,0]
        f_q2[j] = q_cal_temp[j,1]
        f_q3[j] = q_cal_temp[j,2]
        f_q4[j] = q_cal_temp[j,3]
        f_q5[j] = q_cal_temp[j,4]
        f_q6[j] = q_cal_temp[j,5]
        f_q7[j] = q_cal_temp[j,6]

    f_q1_out = torch.from_numpy(f_q1).float().to(device)
    f_q2_out = torch.from_numpy(f_q2).float().to(device)
    f_q3_out = torch.from_numpy(f_q3).float().to(device)
    f_q4_out = torch.from_numpy(f_q4).float().to(device)
    f_q5_out = torch.from_numpy(f_q5).float().to(device)
    f_q6_out = torch.from_numpy(f_q6).float().to(device)
    f_q7_out = torch.from_numpy(f_q7).float().to(device)

    mse_f1 = mse_cost_function(f_q1_out, pt_all_zeros)
    mse_f2 = mse_cost_function(f_q2_out, pt_all_zeros)
    mse_f3 = mse_cost_function(f_q3_out, pt_all_zeros)
    mse_f4 = mse_cost_function(f_q4_out, pt_all_zeros)
    mse_f5 = mse_cost_function(f_q5_out, pt_all_zeros)
    mse_f6 = mse_cost_function(f_q6_out, pt_all_zeros)
    mse_f7 = mse_cost_function(f_q7_out, pt_all_zeros)

    # mse_f = mse_f1 + mse_f2 + mse_f3 + mse_f4 + mse_f5 + mse_f6 + mse_f7
    mse_f = mse_f1 + mse_f2 + mse_f3 + mse_f4 + mse_f5 + mse_f6 + mse_f7
    # Combining the loss functions
    loss = mse_q + mse_f

    loss.backward()  # This is for computing gradients using backward propagation
    optimizer1.step()  # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta
    optimizer2.step()
    optimizer3.step()
    optimizer4.step()
    optimizer5.step()
    optimizer6.step()
    optimizer7.step()

    with torch.autograd.no_grad():
        print(epoch, "Traning Loss:", loss.data)

        loss_temp = np.vstack([loss_temp,np.array(loss.data.cpu())])

np.delete(loss_temp,0,axis=0)
t = np.array([np.arange(0, 5, 0.02)]).T
## Just because meshgrid is used, we need to do the following adjustment
pt_t = Variable(torch.from_numpy(t).float(), requires_grad=True).to(device)
pt_q1 = net_q1(pt_t)
pt_q2 = net_q2(pt_t)
pt_q3 = net_q3(pt_t)
pt_q4 = net_q4(pt_t)
pt_q5 = net_q5(pt_t)
pt_q6 = net_q6(pt_t)
pt_q7 = net_q7(pt_t)

q1 = pt_q1.data.cpu().numpy()
q2 = pt_q2.data.cpu().numpy()
q3 = pt_q3.data.cpu().numpy()
q4 = pt_q4.data.cpu().numpy()
q5 = pt_q5.data.cpu().numpy()
q6 = pt_q6.data.cpu().numpy()
q7 = pt_q7.data.cpu().numpy()

torch.save(net_q1.state_dict(), "model_q1t.pt")
torch.save(net_q2.state_dict(), "model_q2t.pt")
torch.save(net_q3.state_dict(), "model_q3t.pt")
torch.save(net_q4.state_dict(), "model_q4t.pt")
torch.save(net_q5.state_dict(), "model_q5t.pt")
torch.save(net_q6.state_dict(), "model_q6t.pt")
torch.save(net_q7.state_dict(), "model_q7t.pt")

plt.figure()
plt.plot(np.array([np.arange(0,np.size(loss_temp,0))]).T,loss_temp)

plt.figure()
plt.plot(t, q1, label="q1")
plt.grid()
plt.xlabel("Time")
plt.ylabel("Joint 1(radian)")
plt.legend()

plt.figure()
plt.plot(t, q2, label="q2")
plt.grid()
plt.xlabel("Time")
plt.ylabel("Joint 2(radian)")
plt.legend()
plt.figure()
plt.plot(t, q3, label="q3")
plt.grid()
plt.xlabel("Time")
plt.ylabel("Joint 3(radian)")
plt.legend()
plt.figure()
plt.plot(t, q4, label="q4")
plt.grid()
plt.xlabel("Time")
plt.ylabel("Joint 4(radian)")
plt.legend()
plt.show()

# plt.show()
#