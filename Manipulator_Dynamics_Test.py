
import numpy as np
from get_G import G_mtx
from get_M import M_mtx
from get_C import C_mtx
q = np.array([0,0,0,0,0,0,0])
qd = np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1])

#
# def ff(q,qd):
#     C = C_mtx(q,qd)
#     return C
# a=[2,3,4,5]
# print(a[0:3])
# print(ff(q,qd))
#
print(np.pi)
t1 = G_mtx(q)
t2 = M_mtx(q)
t3 = C_mtx(q,qd)

print(t2)
print(t3)
print(t1)

# print(t1)
# print(np.size(t1))
#
#
# print(t3)
# print(np.size(t3))
# print(type(t3))
