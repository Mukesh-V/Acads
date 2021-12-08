import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
from scipy import signal

from normal import normal_model
from tmd import tmd_model
from tmd2 import two_tmd_model
from tmd4 import four_tmd_model

from road import road_profile

data1, data2, data3, data4 = normal_model(), tmd_model(), two_tmd_model(), four_tmd_model()
system1, system2, system3, system4 = data1[4], data2[4], data3[4], data4[4]
A1, A2, A3, A4 = data1[0], data2[0], data3[0], data4[0] 
B1, B2, B3, B4 = data1[1], data2[1], data3[1], data4[1]

options = {
    'wheelbase': 2,
    'speed': 20,
    'amplitude': 0.1,
    'type': 1,
    'side': 1
}
road = road_profile(options)

U = []
T = np.arange(0, 10, 0.01)
for i in T:
    U.append(road(i))
U = np.array(U)
U = np.stack(U)

_, k1, l1 = signal.lsim(system1, U, T.tolist())
_, k2, l2 = signal.lsim(system2, U, T.tolist())
_, k3, l3 = signal.lsim(system3, U, T.tolist())
_, k4, l4 = signal.lsim(system4, U, T.tolist())

def dot_calculator(z):
    lf1, lf2, lf3, lf4 = [], [], [] , []
    for i, li1 in enumerate(l1):
        lf1.append(np.matmul(A1, li1)[z] + np.matmul(B1, U[i])[z])
    for i, li2 in enumerate(l2):
        lf2.append(np.matmul(A2, li2)[z] + np.matmul(B2, U[i])[z])
    for i, li3 in enumerate(l3):
        lf3.append(np.matmul(A3, li3)[z] + np.matmul(B3, U[i])[z])
    for i, li4 in enumerate(l4):
        lf4.append(np.matmul(A4, li4)[z] + np.matmul(B4, U[i])[z])
    
    return [lf1, lf2, lf3, lf4]


z_arr = [0, 2, 4]
ylabels = {0:'x', 2: 'theta', 4: 'phi'}
for z in z_arr:
    f, (ax1, ax2) = plt.subplots(1, 2)
    plt.subplots_adjust(wspace=0.283)
    ax1.plot(T, l1[:, z])
    ax1.plot(T, l2[:, z])
    # plt.plot(T, l3[:, z])
    ax1.plot(T, l4[:, z])
    ax1.set_ylabel(ylabels[z])
    ax1.set_xlabel('Time (in seconds)')
    ax1.legend(["Normal", "1-TMD", "5-TMD"], loc ="lower right")

    z += 1
    lf1, lf2, lf3, lf4 = dot_calculator(z)
    ax2.plot(T, lf1)
    ax2.plot(T, lf2)
    # plt.plot(T, lf3)
    ax2.plot(T, lf4)
    ax2.set_ylabel(ylabels[z-1] + ' second derivative wrt time')
    ax2.set_xlabel('Time (in seconds)')
    ax2.legend(["Normal", "1-TMD", "5-TMD"], loc ="lower right")

    plt.show()
    plt.close()