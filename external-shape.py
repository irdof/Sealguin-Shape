from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt


# Constantes
R = 4
n = 2
m = 1.92
L_AV = 6.8
L_AR = 5
X_M = 3

# functions
def f_AV(x,y) :
    r = np.sqrt((x-X_M)**2+y**2)
    return R * (1-((r)/L_AV)**m)**(1/m)
    
def g_AV(x,y) :
    return -f_AV(x,y)

def f_M(x,y):    
    return R
    
def g_M(x,y):
    return -f_M(x,y)

def f(x,y) :
    if abs(y) <= 2*R :
        if (x <= L_AV + X_M and X_M <= x ):
            return f_AV(x,y)
        elif x < X_M and x >= 0 and abs(y) <= R:
            return f_M(x,y)
        else :
            return np.nan
    else :
        return np.nan
def g(x,y) :
    return -f(x,y)
    

def compute_grid(X,Y,Z,func):
    i = 0
    j = 0
    for coorZip in zip (X,Y) :
        j = 0
        for coor in zip(coorZip[0],coorZip[1]):
            Z[i,j] = func(coor[0],coor[1])
            j = j + 1
        i = i + 1
        
# Compute data
N = 200
x = np.linspace(-L_AR,X_M+L_AV,N)
y= np.linspace(-2*R,2*R,N)

X, Y = np.meshgrid(x, y)
Y_neg = -Y

# AV
Z_1 = np.zeros(X.shape)
Z_2 = np.zeros(X.shape)
Z_3 = np.zeros(X.shape)
Z_4 = np.zeros(X.shape)

compute_grid(X,Y,Z_1,f)
compute_grid(X,Y,Z_2,g)
# compute_grid(X,Y_neg,Z_3,f)
# compute_grid(X,Y_neg,Z_4,g)



#Plot 
fig = plt.figure()
ax = plt.axes(projection='3d')
# plot AV
ax.contour3D(X, Y, Z_1, 50, cmap='binary')
ax.contour3D(X, Y, Z_2, 50, cmap='binary')
# ax.contour3D(X, Y_neg, Z_3, 50, cmap='binary')
# ax.contour3D(X, Y_neg, Z_4, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');
plt.show()
