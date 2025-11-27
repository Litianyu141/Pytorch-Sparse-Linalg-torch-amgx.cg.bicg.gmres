import numpy as np
# import matplotlib
# matplotlib.use("tkagg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
import time
import os

# Use sparse solver for solving pressure_possion equation
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

#define stagger grid shape
nx = 100 # number of cells in x direction
ny =  nx # number of cells in y direction
miu = 0.0025 # dynamic viscosity
lx = 1.0 # length of the domain in x direction
ly = 1.0 # length of the domain in y direction
dx = lx/nx # cell size in x direction
dy = ly/ny  # cell size in y direction
t = 0.0 # absolute start time
nsteps = 1000

# cell centered coordinates
xx = np.linspace(dx/2.0,lx - dx/2.0,nx, endpoint=True) # x-coordinates of the cell centers
yy = np.linspace(dy/2.0,ly - dy/2.0,ny, endpoint=True) # y-coordinates of the cell centers
xcc, ycc = np.meshgrid(xx,yy) 

# x-staggered coordinates
xxs = np.linspace(0,lx,nx+1, endpoint=True)
xu,yu = np.meshgrid(xxs, yy)

# y-staggered coordinates
yys = np.linspace(0,ly,ny+1, endpoint=True)
xv,yv = np.meshgrid(xx, yys)

#Boundary conditions coffecients, all normal velocity components are zero -no inlets whatsoever
Ut = 1.0
Ub = 0.0
Vl = 0.0
Vr = 0.0
Re = Ut*lx/miu #Reynolds number
print('Reynolds Number:', Re) #Reynolds number should be similiar 

#Choose dt based on the linear advection-diffusion constraint,cfl condition
dt = min(0.25*dx*dx/miu, 4.0*miu/Ut/Ut)
print('dt=', dt)

# initialize all velocities - we stagger everything in the negative direction. A scalar cell owns its minus face, only.Then, for example, the u velocity field has a ghost cell at x0 - dx and the plus ghost cell at lx
u = np.ones([ny+2, nx+2]) # include ghost cells

# same thing for the y-velocity component
v = np.zeros([ny+2, nx+2]) # include ghost cells

ut = np.zeros_like(u) # u_tilde
vt = np.zeros_like(u) # v_tilde 

# initialize the pressure
p = np.zeros([nx+2,ny+2]); # include ghost cells

# a bunch of lists for animation purposes
usol=[]
usol.append(u)

vsol=[]
vsol.append(v)

psol = []
psol.append(p)

# build pressure coefficient matrix
Ap = np.zeros([ny,nx])
Ae = 1.0/dx/dx*np.ones([ny,nx])
As = 1.0/dy/dy*np.ones([ny,nx])
An = 1.0/dy/dy*np.ones([ny,nx])
Aw = 1.0/dx/dx*np.ones([ny,nx])
# set left wall coefs
Aw[:,0] = 0.0
# set right wall coefs
Ae[:,-1] = 0.0
# set top wall coefs
An[-1,:] = 0.0
# set bottom wall coefs
As[0,:] = 0.0
Ap = -(Aw + Ae + An + As)

n = nx*ny
d0 = Ap.reshape(n)
# print(d0)
de = Ae.reshape(n)[:-1]
# print(de)
dw = Aw.reshape(n)[1:]
# print(dw)
ds = As.reshape(n)[nx:]
# print(ds)
dn = An.reshape(n)[:-nx]
# print(dn)
A1 = scipy.sparse.diags([d0, de, dw, dn, ds], [0, 1, -1, nx, -nx], format='csr')
#plt.matshow((A1.toarray()))


# start solving, while t < tend:
t0 = time.perf_counter()
momtime = 0.0
solvertime = 0.0


fig1 = plt.figure(figsize=[18,8])

for n in range(0,nsteps):
    #set boundary conditions on u
    # left wall
    u[1:-1,1] = 0.0
    # right wall
    u[1:-1,-1] = 0.0
    # top wall,python indexing for matrix is intutivily from the top left
    u[-1,1:] = 2.0*Ut - u[-2,1:]
    # bottom wall
    u[0,1:] = 2.0*Ub - u[1,1:]

    #set boundary conditions on v
    # left wall
    v[1:,0] = 2.0*Vl - v[1:,1]
    # right wall
    v[1:,-1] = 2.0*Vr - v[1:,-2]
    # bottom wall
    v[1,1:-1] = 0.0
    # top wall
    v[-1,1:-1] = 0.0    
  
    ''' >>> original implemention of the staggered grid >>> '''
    # # do x-momentum first - u is of size (nx + 2) x (ny + 2) - only need to do the interior points
    # tic = time.perf_counter()
    # #for each cell
    # for i in range(2,nx+1):
    #     for j in range(1,ny+1):
    #         ue = 0.5*(u[j,i+1] + u[j,i])
    #         uw = 0.5*(u[j,i]   + u[j,i-1])    
            
    #         un = 0.5*(u[j+1,i] + u[j,i])
    #         us = 0.5*(u[j,i] + u[j-1,i])            
            
    #         vn = 0.5*(v[j+1,i] + v[j+1,i-1])
    #         vs = 0.5*(v[j,i] + v[j,i-1])
            
    #         # convection = - d(uu)/dx - d(vu)/dy
    #         convection = - (ue*ue - uw*uw)/dx - (un*vn - us*vs)/dy
            
    #         # diffusion = d2u/dx2 + d2u/dy2
    #         diffusion = miu*( (u[j,i+1] - 2.0*u[j,i] + u[j,i-1])/dx/dx + (u[j+1,i] - 2.0*u[j,i] + u[j-1,i])/dy/dy )
            
    #         #u_tilde
    #         ut[j,i] = u[j,i] + dt *(convection + diffusion)
    # #for each cell            
    # # do y-momentum - only need to do interior points
    # for i in range(1,nx+1):
    #     for j in range(2,ny+1):
    #         ve = 0.5*(v[j,i+1] + v[j,i])
    #         vw = 0.5*(v[j,i] + v[j,i-1])    
            
    #         ue = 0.5*(u[j,i+1] + u[j-1,i+1])
    #         uw = 0.5*(u[j,i] + u[j-1,i])
            
    #         vn = 0.5*(v[j+1,i] + v[j,i])
    #         vs = 0.5*(v[j,i] + v[j-1,i])            

    #         # convection = d(uv)/dx + d(vv)/dy
    #         convection = - (ue*ve - uw*vw)/dx - (vn*vn - vs*vs)/dy
            
    #         # diffusion = d2u/dx2 + d2u/dy2
    #         diffusion = miu*( (v[j,i+1] - 2.0*v[j,i] + v[j,i-1])/dx/dx + (v[j+1,i] - 2.0*v[j,i] + v[j-1,i])/dy/dy )
            
    #         #v_tilde
    #         vt[j,i] = v[j,i] + dt*(convection + diffusion)  
    ''' <<< original implemention of the staggered grid >>> '''
    
    ''' >>> vectorlized implemention of the staggered grid >>> '''
    # do x-momentum - 向量化实现
    tic = time.perf_counter()

    # 内部点的范围是：ut[1:-1, 2:-1]
    ue = 0.5 * (u[1:-1, 3:] + u[1:-1, 2:-1])
    uw = 0.5 * (u[1:-1, 2:-1] + u[1:-1, 1:-2])

    un = 0.5 * (u[2:, 2:-1] + u[1:-1, 2:-1])
    us = 0.5 * (u[1:-1, 2:-1] + u[0:-2, 2:-1])

    vn = 0.5 * (v[2:, 2:-1] + v[2:, 1:-2])
    vs = 0.5 * (v[1:-1, 2:-1] + v[1:-1, 1:-2])

    # 计算对流项
    convection = - (ue*ue - uw*uw)/dx - (un*vn - us*vs)/dy

    # 计算扩散项
    diffusion = miu * ((u[1:-1, 3:] - 2.0*u[1:-1, 2:-1] + u[1:-1, 1:-2])/dx/dx + 
                    (u[2:, 2:-1] - 2.0*u[1:-1, 2:-1] + u[0:-2, 2:-1])/dy/dy)

    # 更新u_tilde
    ut[1:-1, 2:-1] = u[1:-1, 2:-1] + dt * (convection + diffusion)

    # do y-momentum - 向量化实现
    ve = 0.5 * (v[2:-1, 2:] + v[2:-1, 1:-1])
    vw = 0.5 * (v[2:-1, 1:-1] + v[2:-1, :-2])

    ue = 0.5 * (u[2:-1, 2:] + u[1:-2, 2:])
    uw = 0.5 * (u[2:-1, 1:-1] + u[1:-2, 1:-1])

    vn = 0.5 * (v[3:, 1:-1] + v[2:-1, 1:-1])
    vs = 0.5 * (v[2:-1, 1:-1] + v[1:-2, 1:-1])

    # 计算对流项
    convection = - (ue*ve - uw*vw)/dx - (vn*vn - vs*vs)/dy

    # 计算扩散项
    diffusion = miu * ((v[2:-1, 2:] - 2.0*v[2:-1, 1:-1] + v[2:-1, :-2])/dx/dx + 
                    (v[3:, 1:-1] - 2.0*v[2:-1, 1:-1] + v[1:-2, 1:-1])/dy/dy)

    # 更新v_tilde
    vt[2:-1, 1:-1] = v[2:-1, 1:-1] + dt * (convection + diffusion)
    ''' <<< vectorlized implemention of the staggered grid <<< '''
    # do pressure - p-right-hand-side(prhs) = 1/dt * div(u_tilde)
    # we will only need to fill the interior points. This size is for convenient indexing
    div_ut = np.zeros([ny+2,nx+2]) # initialize divergence of u_tilde
    div_ut[1:-1,1:-1] = (ut[1:-1,2:] - ut[1:-1,1:-1])/dx + (vt[2:,1:-1] - vt[1:-1,1:-1])/dy #calculate divergence of u_tilde
    
    #cacluate pressure_right-hand-side
    prhs = 1.0/dt * div_ut
    toc = time.perf_counter()
    momtime += (toc - tic)
    
    tic=time.perf_counter()
    # solve for pressure
    pt,info = scipy.sparse.linalg.bicg(A1,prhs[1:-1,1:-1].ravel(),atol=1e-10,maxiter=1000) #theta=sc.linalg.solve_triangular(A,d)
    toc=time.perf_counter()
    #record solver time
    solvertime += toc - tic
    p = np.zeros([ny+2,nx+2])
    p[1:-1,1:-1] = pt.reshape([ny,nx])
    
    # push forward u_n+1 and v_n+1 in next time step
    u[1:-1,2:-1] = ut[1:-1,2:-1] - dt * (p[1:-1,2:-1] - p[1:-1,1:-2])/dx
    v[2:-1,1:-1] = vt[2:-1,1:-1] - dt * (p[2:-1,1:-1] - p[1:-2,1:-1])/dy  
    
    #time advance
    t += dt

    # Check mass residual
    divunp1 = np.zeros((ny+2, nx+2))
    divunp1[1:ny+1, 1:nx+1] = (u[1:ny+1, 2:nx+2] - u[1:ny+1, 1:nx+1]) / dx + \
                            (v[2:ny+2, 1:nx+1] - v[1:ny+1, 1:nx+1]) / dy
    residual = np.linalg.norm(divunp1.ravel())

    if n % 100 == 0:
        print('Step:', n)
        print('Time:', t)
        print('dt:', dt)
        print('Momentum time:', momtime)
        print('Solver time:', solvertime)
        print('Pressure solver info:', info)
        print('Mass residual:', residual)
        print('Max velocity:', np.max(np.sqrt(u[1:-1,2:]**2 + v[2:,1:-1]**2)))
        print('Min velocity:', np.min(np.sqrt(u[1:-1,2:]**2 + v[2:,1:-1]**2)))

    
    #printing result in real-time
    if (n <= nsteps and n % 100 == 0):
        
        plt.subplot(1,2,1)
        plt.cla()
        
        # divu = (u[1:-1,2:] - u[1:-1,1:-1])/dx + (v[2:,1:-1] - v[1:-1,1:-1])/dy
        #plt.imshow(divu,origin='upper')
        # print(divu-divunp1[1:-1,1:-1])
        
        ucc = 0.5*(u[1:-1,2:] + u[1:-1,1:-1])
        vcc = 0.5*(v[2:,1:-1] + v[1:-1,1:-1])
        speed = np.sqrt(ucc*ucc + vcc*vcc)
        levels = np.linspace(speed.min(), speed.max(), 256)
        plt.contourf(xx, yy, speed, levels=levels, cmap='RdBu_r')
        # plt.colorbar()
        # plt.contourf(xx,yy,speed,cmap='RdBu_r')
        #plt.colorbar()
        plt.xlabel(r'x',fontsize=18)
        plt.ylabel(r'y',fontsize=18)
        plt.title(r'Velocity Magnitude',fontsize=18)
        ax = plt.gca()
        ax.set_aspect(1)
        plt.pause(0.001)
        
        x = np.linspace(0,lx,nx)
        y = np.linspace(0,ly,ny)
        xx,yy = np.meshgrid(x,y)
        nn = 1
        plt.subplot(1,2,2)
        plt.cla()
        #plt.quiver(xx[::nn,::nn],yy[::nn,::nn],ucc[::nn,::nn],vcc[::nn,::nn])
        plt.xlim([xx[0,0],xx[0,-1]])
        plt.ylim([yy[0,0],yy[-1,0]])
        ax.set_xlim([xx[0,0],xx[0,-1]])
        plt.streamplot(xx,yy,ucc, vcc, color=np.sqrt(ucc*ucc + vcc*vcc),density=1.5,cmap=plt.cm.autumn,linewidth=1.5)
        plt.xlabel(r'x',fontsize=18)
        plt.ylabel(r'y',fontsize=18)
        plt.title(r'Streamline',fontsize=18)
        ax = plt.gca()
        ax.set_aspect(1)

savedir = f"FVM_example/scipy_results/Re{Re:.0f}_nx{nx}_steps{nsteps}_dt{dt:.6f}"
os.makedirs(savedir, exist_ok=True)
plt.savefig(f'{savedir}/Staggered_scipy_steps{nsteps}.png')
