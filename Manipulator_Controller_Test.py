
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from get_G import G_mtx
from get_M import M_mtx
from get_C import C_mtx


n = 7 # DOF
sim_FT = 5
sim_period = 0.001
sam=int(sim_FT/sim_period)
tspan = np.linspace(0,sim_FT, sam+1)

# --------------------------------------------------------------------0629----------------------------------------------
def plant(x, u1, M, C, G):
    G = G.tolist()
    G = [G[0][0],G[1][0],G[2][0],G[3][0],G[4][0],G[5][0],G[6][0]]
    G = np.array(G)
    C = C.tolist()
    C = [C[0][0], C[1][0], C[2][0], C[3][0], C[4][0], C[5][0], C[6][0]]
    C = np.array(C)
    # Case 1: Consider Real Dynamics
    # val = -C-G+u1
    # val2 = np.matmul(np.linalg.inv(M),val)
    # Case 2: Compensating Real Dynamics
    val2 = np.matmul(np.linalg.inv(M),u1)
    # print(val2)
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

qt_temp = []
qdt_temp = []
error_q_qt = []
error_qd_qt = []
q = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
q_t = [0, 0, 1.57, 0, 0, 0, 0]
qd_t = [0, 0, 0, 0, 0, 0, 0]
Kp = np.array([20, 15, 5, 1, 1, 1, 1])
Kd = np.sqrt(Kp)*0.5
# q1=np.array([1,2,3,4,5,6,7])
# M = M_mtx(q1)
# C = C_mtx(q1, q1)
# G = G_mtx(q1)

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
    # print(q_sol_next)

error_q_qt = np.array(error_q_qt)
error_qd_qt = np.array(error_qd_qt)
q = np.array(q)
qt_temp.insert(0,q_t)
qdt_temp.insert(0,qd_t)
qt_temp = np.array(qt_temp)
qdt_temp = np.array(qdt_temp)

plt.plot(tspan, q[:,0], label = "q1")
plt.plot(tspan, q[:,1], label = "q2")
plt.plot(tspan, q[:,2], label = "q3")
plt.plot(tspan, q[:,3], label = "q4")
plt.grid()
plt.xlabel("Time(sec)")
plt.ylabel("State response")
plt.legend()

fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, figsize=(8, 9))

ax0.plot(tspan, q[:,4], label = "q1")
ax0.plot(tspan, qt_temp[:,4], label = "q1_target")
ax0.set_title('Joint 5')
ax0.legend()

ax1.plot(tspan, q[:,5], label = "q2")
ax1.plot(tspan, qt_temp[:,5], label = "q2_target")
ax1.set_title('Joint 6')
ax1.legend()

ax2.plot(tspan, q[:,6], label = "q3")
ax2.plot(tspan, qt_temp[:,6], label = "q3_target")
ax2.set_title('Joint 7')
ax2.legend()

ax3.plot(tspan, q[:,3], label = "q4")
ax3.plot(tspan, qt_temp[:,3], label = "q4_target")
ax3.set_title('Joint 4')
ax3.legend()

plt.show()









