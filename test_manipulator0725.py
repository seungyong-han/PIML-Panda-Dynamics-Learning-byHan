import torch
import torch.nn as nn
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


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

n = 7 # DOF
sim_FT = 3
sim_period = 0.001
sam=int(sim_FT/sim_period)
tspan = np.linspace(0,sim_FT, sam+1)
t = np.array([np.arange(0, 3, 0.001)]).T
pt_t = Variable(torch.from_numpy(t).float(), requires_grad=True).to(device)

net_q1 =Net()
net_q2 =Net()
net_q3 =Net()
net_q4 =Net()
net_q5 =Net()
net_q6 =Net()
net_q7 =Net()

net_q1.load_state_dict(torch.load('/home/hana/Downloads/PINN-main0710/model_q1t.pt'))
net_q2.load_state_dict(torch.load('/home/hana/Downloads/PINN-main0710/model_q2t.pt'))
net_q3.load_state_dict(torch.load('/home/hana/Downloads/PINN-main0710/model_q3t.pt'))
net_q4.load_state_dict(torch.load('/home/hana/Downloads/PINN-main0710/model_q4t.pt'))
net_q5.load_state_dict(torch.load('/home/hana/Downloads/PINN-main0710/model_q5t.pt'))
net_q6.load_state_dict(torch.load('/home/hana/Downloads/PINN-main0710/model_q6t.pt'))
net_q7.load_state_dict(torch.load('/home/hana/Downloads/PINN-main0710/model_q7t.pt'))

net_q1.eval()
net_q2.eval()
net_q3.eval()
net_q4.eval()
net_q5.eval()
net_q6.eval()
net_q7.eval()

net_q1.to('cuda:0')
net_q2.to('cuda:0')
net_q3.to('cuda:0')
net_q4.to('cuda:0')
net_q5.to('cuda:0')
net_q6.to('cuda:0')
net_q7.to('cuda:0')

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