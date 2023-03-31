# from mpl_toolkits import mplot3d
# import numpy as np
# import matplotlib.pyplot as plt


# # Constantes
# R = 4
# n = 2
# m = 1.92
# L_AV = 6.8
# L_AR = 5
# X_M = 3



# def f_AV(x) :
#     try :
#         l = np.log(R) + 1/m * np.log(1 - ((x - X_M)/L_AV)**m)
#         return np.exp(l)
#     except RuntimeWarning:
#         print("hello")

# def f(x) :
#     if (x <= L_AV + X_M and X_M <= x ):
#         return f_AV(x)
#     elif x < X_M and x >= 0 :
#         # return f_M(x,y)
#         return 1
#     else :
#         return 1



# N = 200
# phi = np.linspace(-np.pi, np.pi, N)
# z = np.linspace(X_M,X_M+L_AV,N)
# Z = np.array(N * [z])
# X  = np.zeros(Z.shape)
# Y = np.zeros(Z.shape)
# r = np.zeros(Z.shape)

# for i in range(N) :
#     for j in range(N) :
#         r[i,j] = f(Z[i,j])

# X = r*np.cos(phi)
# Y = r*np.sin(phi)
# Y = Y.T

# #Plot 
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# # plot AV
# ax.plot_surface(X, Y, Z, cmap='binary')

# # ax.contour3D(X, Y_neg, Z_3, 50, cmap='binary')
# # ax.contour3D(X, Y_neg, Z_4, 50, cmap='binary')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z');
# plt.show()
# print("debug")

import matplotlib
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

matplotlib.use('TkAgg',force=True)
print("Switched to:",matplotlib.get_backend())

# Constantes
R = 4
n = 2
m = 1.92
L_AV = 3
L_AR = 5
X_M = 3
N = 500

def pol2cart(theta,r,z):
    return [np.cos(theta)*r,np.sin(theta)*r,z]

def f_av(Z) :
    return R*(1-((Z[0]-X_M)/L_AV)**m)**(1/m)

def f_m(Z) :
    return R * np.ones(Z.shape)

def plotShape(t,r,z):
    [x,y,z] = pol2cart(t,r,z)
    #Plot 
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # plot AV
    ax.plot_surface(x,y,z, cmap='binary')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z');
    plt.show();


theta = np.linspace(0,2*np.pi,N)

z = np.linspace(0,L_AV+X_M,N)
[TH,Z] = np.meshgrid(theta,z)
# r = f_av(Z)
r = np.zeros(Z.shape)
# r[Z<X_M] = f_m(Z[Z<X_M])
r[(Z >= X_M) & (Z <= X_M + L_AV)] = f_av(Z[(Z >= X_M) & (Z <= X_M + L_AV)])


# plotShape(TH_AV,r_av,Z_AV)

# z_m = np.linspace(0,X_M,N)
# [TH_M,Z_M] = np.meshgrid(theta,z_m)
# r_m = f_m(Z_M)

# plotShape(TH_M,r_m,Z_M)

# Z = np.concatenate((Z_M,Z_AV), axis=0)
# TH = np.concatenate((TH_M,TH_AV), axis=0)
# r = np.concatenate((r_m,r_av), axis=0)


[x,y,z] = pol2cart(TH,r,Z)

#Plot 
fig = plt.figure()
ax = plt.axes(projection='3d')
# plot AV
ax.plot_surface(x,y,z, cmap='binary')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');
plt.show();
print("debug")