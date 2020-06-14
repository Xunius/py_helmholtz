'''Compute helmholtz decomposition on non-global u- and v- winds.

Implemented following:

> Li, Zhijin and Chao, Yi and McWilliams, James C., 2006: Computation of the
> Streamfunction and Velocity Potential for Limited and Irregular Domains.
> Monthly Weather Review, 3384-3394.


# 1. To solve for streamfunction (psi) and velocity potential (chi) from
u- and v- winds, on a uniform grid (uniform dx and dy everywhere):

u and v are given in even grid (n x m).

streamfunction (psi) and velocity potential (chi) are defined on a dual
grid ((n+1) x (m+1)), where psi and chi are defined on the 4 corners
of u and v.

Define:

    u = u_chi + u_psi
    v = v_chi + v_psi

    u_psi = -dpsi/dy
    v_psi = dpsi/dx
    u_chi = dchi/dx
    v_chi = dchi/dy


Define 2 2x2 kernels:

    k_x = |-0.5 0.5|
          |-0.5 0.5| / dx

    k_y = |-0.5 -0.5|
          |0.5   0.5| / dy

Then u_chi = chi \bigotimes k_x
where \bigotimes is cross-correlation,

Similarly:

    v_chi = chi \bigotimes k_y
    u_psi = psi \bigotimes -k_y
    v_psi = psi \bigotimes k_x

Define cost function J = (uhat - u)**2 + (vhat - v)**2

Gradients of chi and psi:

    dJ/dchi = (uhat - u) du_chi/dchi + (vhat - v) dv_chi/dchi
    dJ/dpsi = (uhat - u) du_psi/dpsi + (vhat - v) dv_psi/dpsi

    (uhat - u) du_chi/dchi = (uhat - u) \bigotimes Rot180(k_x) = (uhat - u) \bigotimes -k_x
    (vhat - v) dv_chi/dchi = (vhat - v) \bigotimes Rot180(k_y) = (vhat - v) \bigotimes -k_y
    (uhat - u) du_psi/dpsi = (uhat - u) \bigotimes k_y
    (vhat - v) dv_psi/dpsi = (vhat - v) \bigotimes Rot180(k_x) = (vhat - v) \bigotimes -k_x

Rot180() is a 180 degree rotation, and for k_x and k_y, it's the same as the
inverting the sign. This Rot180() process is similar as the error back-propagation
process in the convolutional neural network training process.

Add the regularization term:

    J = (uhat - u)**2 + (vhat - v)**2 + lambda(chi**2 + psi**2)


# 2. To solve for streamfunction and velocity potential from u- and v- winds
on irregular grid (e.g. mercator):

Use similar definition of cost function and gradients, except that the computation
of component winds and derivatives are performed on steps for NE, NW, SE, SW
qudarants:

    u_chi =  0.5*((vp[:-1,1:]-vp[:-1,:-1])/dx_n + (vp[1:,1:]-vp[1:,:-1])/dx_s)
    v_chi =  0.5*((vp[1:,:-1]-vp[:-1,:-1])/dy_w + (vp[1:,1:]-vp[:-1,1:])/dy_e)
    u_psi = -0.5*((sf[1:,:-1]-sf[:-1,:-1])/dy_w + (sf[1:,1:]-sf[:-1,1:])/dy_e)
    v_psi =  0.5*((sf[:-1,1:]-sf[:-1,:-1])/dx_n + (sf[1:,1:]-sf[1:,:-1])/dx_s)

    du_chi/dchi, dv_chi/dchi, du_psi/dpsi and dv_psi/dpsi are also compose
    by 4 quadrants, see code for details.


# 3. The `Wind2D` class is designed for computation of netcdf data via the CDAT interface.

CDAT: https://cdat.llnl.gov/index.html


# 4. Example

    # read in some wind data as `u` and `v` via cdms

    u=u(latitude=(5,50),longitude=(100,180))
    v=v(latitude=(5,50),longitude=(100,180))

    # create an wind obj, optimization will use gradient descent
    w1=Wind2D(u,v,'GD')

    # compute streamfunction and velocity potential
    sf1,vp1=w1.getSFVP()

    # get irrotational and non-divergent components
    uchi1,vchi1,upsi1,vpsi1=w1.helmholtz()
    uhat1=uchi1+upsi1
    vhat1=vchi1+vpsi1

    # create an wind obj, optimization will use scipy.optimize (recommended)
    w2=Wind2D(u,v,'optimize')
    sf2,vp2=w2.getSFVP()
    uchi2,vchi2,upsi2,vpsi2=w2.helmholtz()
    uhat2=uchi2+upsi2
    vhat2=vchi2+vpsi2

    # recompute and interpolate to the same grid of u and v
    sf3,vp3=w2.getSFVP(interp=True)


Author: guangzhi XU (xugzhi1987@gmail.com)
Update time: 2020-06-14 11:54:13.
'''




#--------Import modules-------------------------
import cdms2 as cdms
import MV2 as MV
import numpy as np
from scipy import optimize, interpolate
from scipy.signal import fftconvolve
from scipy import integrate


#-----------------------------------------------------------------------
#-                            Utility funcs                            -
#-----------------------------------------------------------------------

#----Get mask for missing data (masked or nan)----
def getMissingMask(slab):
    '''Get a bindary array denoting missing (masked or nan).

    Args:
        slab (ndarray): ndarray possibly contains masked values or nans.

    Returns:
        mask (ndarray): bindary ndarray, 1s for missing, 0s otherwise.
    '''

    nan_mask=np.where(np.isnan(slab),1,0)

    if not hasattr(slab,'mask'):
        mask_mask=np.zeros(slab.shape)
    else:
        if slab.mask.size==1 and slab.mask==False:
            mask_mask=np.zeros(slab.shape)
        else:
            mask_mask=np.where(slab.mask,1,0)

    mask=np.where(mask_mask+nan_mask>0,1,0)

    return mask

#----------Delta_Latitude----------------------------
def dLongitude(var,side='c',R=6371000):
    '''Return a slab of longitudinal increment (meter) delta_x.

    Args:
        var (cdms.TransientVariable): variable from which longitude axis is
            obtained;

    Kwargs:
        side (str): 'n': northern boundary of each latitudinal band;
            's': southern boundary of each latitudinal band;
            'c': central line of latitudinal band;

             -----     'n'
            /-----\     'c'
           /_______\     's'


        R (float): radius of Earth;

    Returns:
        delta_x (ndarray): 2d-array, longitudinal grid lengths.
    '''

    latax=var.getLatitude()
    lonax=var.getLongitude()

    if latax is None:
        raise Exception("<var> has no latitude axis.")
    if lonax is None:
        raise Exception("<var> has no longitude axis.")

    #----------Get axes---------------------
    lonax=var.getLongitude()

    latax_bounds=latax.getBounds()
    lonax_bounds=lonax.getBounds()
    lon_increment=[np.max(ii)-np.min(ii) for ii in lonax_bounds]
    lon_increment=np.array(lon_increment)
    lon_increment=np.pi*lon_increment/180

    delta_x=[]

    for ii in range(len(latax)):

        if side=='n':
            latii=max(latax_bounds[ii])
        elif side=='s':
            latii=min(latax_bounds[ii])
        elif side=='c':
            latii=latax[ii]

        latii=abs(latii)*np.pi/180.
        dx=R*np.cos(latii)*lon_increment

        if np.any(dx<=1e-8):
            dx[np.where(dx<=1e-8)]=1
            print '\n# <dLongitude>: Warning, delta x is 0. Re-assign to 1.'

        delta_x.append(dx)

    #-------Repeat array to get slab---------------
    delta_x=MV.array(delta_x)
    delta_x.setAxisList((latax,lonax))

    return delta_x

#----------Delta_Longitude----------------------------
def dLatitude(var,R=6371000,verbose=True):
    '''Return a slab of latitudinal increment (meter) delta_y.

    Args:
        var (cdms.TransientVariable): variable from which latitude axis is
            obtained;

    Kwargs:
        R (float): radius of Earth;

    Returns:
        delta_y (ndarray): 2d-array, latitudinal grid lengths.
    '''

    latax=var.getLatitude()
    lonax=var.getLongitude()

    if latax is None:
        raise Exception("<var> has no latitude axis.")
    if lonax is None:
        raise Exception("<var> has no longitude axis.")

    #---------Get axes and bounds-------------------
    latax_bounds=latax.getBounds()

    delta_y=[]

    for ii in range(len(latax)):
        d_theta=abs(latax_bounds[ii][0]-latax_bounds[ii][1])*np.pi/180.
        dy=R*d_theta
        delta_y.append(dy)

    #-------Repeat array to get slab---------------
    delta_y=MV.array(delta_y)
    delta_y=MV.reshape(delta_y,(len(latax),1))
    delta_y=MV.repeat(delta_y,len(lonax),axis=1)
    delta_y.setAxisList((latax,lonax))

    return delta_y



#-----------------------------------------------------------------------
#-                    Functions for regular grid                     -
#-----------------------------------------------------------------------

def uRecon_reg(sf,vp,kernel_x,kernel_y):
    '''Reconstruction u from streamfunction and velocity potential'''
    uchi=fftconvolve(vp,-kernel_x,mode='valid')
    upsi=fftconvolve(sf,kernel_y,mode='valid')
    return upsi+uchi

def vRecon_reg(sf,vp,kernel_x,kernel_y):
    '''Reconstruction v from streamfunction and velocity potential'''
    vchi=fftconvolve(vp,-kernel_y,mode='valid')
    vpsi=fftconvolve(sf,-kernel_x,mode='valid')
    return vpsi+vchi

def costFunc_reg(params,u,v,kernel_x,kernel_y,lam):
    '''Compute cost function of optimization, for GD method'''
    sf=params[0]
    vp=params[1]
    uhat=uRecon_reg(sf,vp,kernel_x,kernel_y)
    vhat=vRecon_reg(sf,vp,kernel_x,kernel_y)
    j=(uhat-u)**2+(vhat-v)**2
    j=j.mean()+lam*np.mean(params**2)

    return j,uhat,vhat

def jac_reg(params,u,v,uhat,vhat,kernel_x,kernel_y,lam):
    '''Compute gradients of cost function of optimization, for GD method'''
    du=uhat-u
    dv=vhat-v

    dvp_u=fftconvolve(du,kernel_x,mode='full')
    dvp_v=fftconvolve(dv,kernel_y,mode='full')

    dsf_u=fftconvolve(du,-kernel_y,mode='full')
    dsf_v=fftconvolve(dv,kernel_x,mode='full')

    dsf=dsf_u+dsf_v
    dvp=dvp_u+dvp_v
    #dsf=dsf+lam*params[0]/params[0].size
    #dvp=dvp+lam*params[1]/params[1].size
    dsf=dsf+lam*params[0]
    dvp=dvp+lam*params[1]

    return dsf, dvp

def costFunc2_reg(params,u,v,kernel_x,kernel_y,pad_shape,lam):
    '''Compute cost function of optimization, for optimize method'''
    pp=params.reshape(pad_shape)
    sf=pp[0]
    vp=pp[1]
    uhat=uRecon_reg(sf,vp,kernel_x,kernel_y)
    vhat=vRecon_reg(sf,vp,kernel_x,kernel_y)
    j=(uhat-u)**2+(vhat-v)**2
    j=j.mean()+lam*np.mean(params**2)

    return j

def jac2_reg(params,u,v,kernel_x,kernel_y,pad_shape,lam):
    '''Compute gradients of cost function of optimization, for optimize method'''
    pp=params.reshape(pad_shape)
    sf=pp[0]
    vp=pp[1]
    uhat=uRecon_reg(sf,vp,kernel_x,kernel_y)
    vhat=vRecon_reg(sf,vp,kernel_x,kernel_y)

    du=uhat-u
    dv=vhat-v

    dvp_u=fftconvolve(du,kernel_x,mode='full')
    dvp_v=fftconvolve(dv,kernel_y,mode='full')

    dsf_u=fftconvolve(du,-kernel_y,mode='full')
    dsf_v=fftconvolve(dv,kernel_x,mode='full')

    dsf=dsf_u+dsf_v
    dvp=dvp_u+dvp_v
    #dsf=(dsf_u+dsf_v)/u.size
    #dvp=(dvp_u+dvp_v)/u.size

    re=np.vstack([dsf[None,:,:,],dvp[None,:,:]])
    re=re.reshape(params.shape)
    re=re+lam*params/u.size

    return re

def getSFVP_reg(u,v,dx,dy,method='optimize',lr=0.01,lam=0.001,max_iter=1000,
        threshold=None,interp=False):
    '''Compute streamfunction and velocity potential from u and v on regular grid.

    Args:
        u, v (ndarray): 2D array with shape (nxm), u- and v- velocities,
            need to have same shape and defined on regular grid (uniform dx and dy).
        dx, d> (floats): x- and y- grid size. Unit should be consistent with
            the units of <u> and <v>. E.g. <u>, <v> in m/s, <dx>, <dy> in m.

    Kwargs:
        method (str): 'optimize' for optimization using scipy.
                       'GD' for gradient descent optimization.
        lr (float): learning rate. Only used if <method> == 'GD'.
        lam (float): regularization parameter, <lam> > 0.
        max_iter (int): max number of iterations for <method> == 'GD'.
                Only used if <method> == 'GD'.
        threshold (float or None): If None, optimization runs all iterations
             defined by <max_iter>. If float, allow optimization to
             early stop if all absolute errors are smaller than <threshold>.
             Only used if <method> == 'GD'.
        interp (bool): whether to interpolate streamfunction and velocity
                  potential to the grid of u and v

    Returns:
        sf, vp (ndarray): 2D array with shape ((n+1)x(m+1) if <interp>,
            nxm otherwise), streamfunction and velocity potential.
    '''

    #-------------------Check inputs-------------------
    if np.ndim(u)!=2:
        raise Exception("<u> needs to be 2D.")
    if np.ndim(v)!=2:
        raise Exception("<v> needs to be 2D.")
    if not np.isscalar(dx) and dx>0:
        raise Exception("<dx> needs to be positive scalar")
    if not np.isscalar(dy) and dy>0:
        raise Exception("<dy> needs to be positive scalar")
    if method not in ['optimize', 'GD']:
        raise Exception("<method> is one of ['optimize', 'GD']")
    if not np.isscalar(lr) and lr>0:
        raise Exception("<lr> needs to be positive scalar")
    if not np.isscalar(lam) and lam>0:
        raise Exception("<lam> needs to be positive scalar")
    if not np.isscalar(max_iter) and max_iter>0:
        raise Exception("<max_iter> needs to be positive scalar")
    if threshold is not None:
        if not np.isscalar(threshold) and threshold>0:
            raise Exception("<threshold> needs to be positive scalar")

    #----------------Scale down dx, dy----------------
    scale_y=np.log10(dy)
    scale_x=np.log10(dx)
    scale=10**(0.5*(scale_x+scale_y))
    dx=dx/scale
    dy=dy/scale

    #-------------Get x,y coordinates-------------
    ny,nx=u.shape
    y=np.arange(0,ny*dy,dy)
    x=np.arange(0,nx*dx,dx)
    X,Y=np.meshgrid(x,y)

    #-------------------Get kernels-------------------
    kernel_x=np.array([[-0.5, 0.5],[-0.5, 0.5]])/dx
    kernel_y=np.array([[-0.5, -0.5],[0.5, 0.5]])/dy

    #---------Integrate to get an intial guess---------
    intx=integrate.cumtrapz(v,X,axis=1,initial=0)[0]
    inty=integrate.cumtrapz(u,Y,axis=0,initial=0)
    psi1=intx-inty

    intx=integrate.cumtrapz(v,X,axis=1,initial=0)
    inty=integrate.cumtrapz(u,Y,axis=0,initial=0)[:,0][:,None]
    psi2=intx-inty

    psi=0.5*(psi1+psi2)

    intx2=integrate.cumtrapz(u,X,axis=1,initial=0)[0]
    inty2=integrate.cumtrapz(v,Y,axis=0,initial=0)
    chi1=intx2+inty2

    intx2=integrate.cumtrapz(u,X,axis=1,initial=0)
    inty2=integrate.cumtrapz(v,Y,axis=0,initial=0)[:,0][:,None]
    chi2=intx2+inty2

    chi=0.5*(chi1+chi2)

    #---------------Pad to get dual grid---------------
    psi=np.pad(psi,(1,0),'edge')
    chi=np.pad(chi,(1,0),'edge')
    params=np.vstack([psi[None,:,:], chi[None,:,:]])
    pad_shape=params.shape

    #---------------------Optimize---------------------
    if method=='GD':
        costs=[]
        ii=0
        if threshold is not None:
            check_inter=10
        while True:
            ii+=1
            jii,uhatii,vhatii=costFunc_reg(params,u,v,kernel_x,kernel_y,lam)
            dsf,dvp=jac_reg(params,u,v,uhatii,vhatii,kernel_x,kernel_y,lam)

            old_params=params.copy()

            params[0]=params[0]-lr*dsf
            params[1]=params[1]-lr*dvp

            #--------------Shrink lr if overshoot--------------
            if ii>1 and jii>costs[-1]:
                kk=0
                while jii>costs[-1]:
                    kk+=1
                    params=old_params
                    lr=lr*0.95
                    params[0]=params[0]-lr*dsf
                    params[1]=params[1]-lr*dvp
                    jii,uhatii,vhatii=costFunc_reg(params,u,v,kernel_x,kernel_y,lam)
                    if kk>=50:
                        break

            costs.append(jii)

            #---------------Check if early stop---------------
            if threshold is not None:
                if ii%check_inter==0:
                    duii=uhatii-u
                    dvii=vhatii-v
                    if np.all(abs(duii)<=threshold) and \
                        np.all(abs(dvii)<=threshold):
                        break
            if ii>max_iter:
                break

        costs=np.array(costs)

    elif method=='optimize':

        opt=optimize.minimize(costFunc2_reg,params,
                args=(u,v,kernel_x,kernel_y,pad_shape,lam),
                method='Newton-CG',
                jac=jac2_reg)
        params=opt.x.reshape(pad_shape)

    sf=params[0]
    vp=params[1]

    #-----------------Scale up sf, vp-----------------
    sf=sf*scale
    vp=vp*scale

    #-------------------Interpolate-------------------
    if interp:
        x_dual=np.hstack([x-dx/2.,x[-1]+dx/2.])
        y_dual=np.hstack([y-dy/2.,y[-1]+dy/2.])
        sf_interpf=interpolate.interp2d(x_dual,y_dual,sf,kind='cubic')
        vp_interpf=interpolate.interp2d(x_dual,y_dual,vp,kind='cubic')

        sf=sf_interpf(x,y)
        vp=vp_interpf(x,y)

    return sf, vp

def example_reg():

    y=np.linspace(0,10,40)
    x=np.linspace(0,10,50)
    X,Y=np.meshgrid(x,y)

    u=3*Y**2-3*X**2
    v=6*X*Y

    dx=x[1]-x[0]
    dy=y[1]-y[0]
    #sf,vp=getSFVP_reg(u,v,dx,dy,'GD',lr=1,threshold=1e-4,max_iter=4000)
    sf,vp=getSFVP_reg(u,v,dx,dy,'optimize', lam=0.1)
    #sf,vp=getSFVP_reg(u,v,dx,dy,'optimize',interp=True)

    kernel_x=np.array([[-0.5, 0.5],[-0.5, 0.5]])/dx
    kernel_y=np.array([[-0.5, -0.5],[0.5, 0.5]])/dy

    #uhat=uRecon_reg(sf,vp,kernel_x,kernel_y)
    #vhat=vRecon_reg(sf,vp,kernel_x,kernel_y)
    uhat=uRecon_reg(sf,vp,kernel_x,kernel_y)
    vhat=vRecon_reg(sf,vp,kernel_x,kernel_y)
    #uhat=uRecon_reg(sf,vp,kernel_x,kernel_y)
    #vhat=vRecon_reg(sf,vp,kernel_x,kernel_y)

    return sf,vp,uhat,vhat




#-----------------------------------------------------------------------
#-                    Functions for irregular grid                     -
#-----------------------------------------------------------------------
def getUChi(vp,dx_n,dx_s):
    return 0.5*((vp[:-1,1:]-vp[:-1,:-1])/dx_n + (vp[1:,1:]-vp[1:,:-1])/dx_s)

def getVChi(vp,dy_w,dy_e):
    return 0.5*((vp[1:,:-1]-vp[:-1,:-1])/dy_w + (vp[1:,1:]-vp[:-1,1:])/dy_e)

def getUPsi(sf,dy_w,dy_e):
    return -0.5*((sf[1:,:-1]-sf[:-1,:-1])/dy_w + (sf[1:,1:]-sf[:-1,1:])/dy_e)

def getVPsi(sf,dx_n,dx_s):
    return 0.5*((sf[:-1,1:]-sf[:-1,:-1])/dx_n + (sf[1:,1:]-sf[1:,:-1])/dx_s)

def uRecon(sf,vp,dx_n,dx_s,dy_w,dy_e):
    uchi=getUChi(vp,dx_n,dx_s)
    upsi=getUPsi(sf,dy_w,dy_e)
    return upsi+uchi

def vRecon(sf,vp,dx_n,dx_s,dy_w,dy_e):
    vchi=getVChi(vp,dy_w,dy_e)
    vpsi=getVPsi(sf,dx_n,dx_s)
    return vpsi+vchi

def getIrrotationalComponent(sf,vp,dx_n,dx_s,dy_w,dy_e):
    '''Get irrotational component of u and v from streamfunction and velocity
    potential'''
    uchi=getUChi(vp,dx_n,dx_s)
    vchi=getVChi(vp,dy_w,dy_e)
    return uchi,vchi

def getNondivergentComponent(sf,vp,dx_n,dx_s,dy_w,dy_e):
    '''Get nondivergent component of u and v from streamfunction and velocity
    potential'''
    upsi=getUPsi(sf,dy_w,dy_e)
    vpsi=getVPsi(sf,dx_n,dx_s)
    return upsi,vpsi


def costFunc(params,u,v,dx_n,dx_s,dy_w,dy_e,lam):
    '''Compute cost function of optimization, for GD method'''
    sf=params[0]
    vp=params[1]
    uhat=uRecon(sf,vp,dx_n,dx_s,dy_w,dy_e)
    vhat=vRecon(sf,vp,dx_n,dx_s,dy_w,dy_e)
    j=np.array((uhat-u)**2+(vhat-v)**2).mean()
    j+=lam*np.mean(params**2)

    return j,uhat,vhat

def jac(params,u,v,uhat,vhat,dx_n,dx_s,dy_w,dy_e,lam):
    '''Compute gradient of cost function of optimization, for GD method'''
    du=uhat-u
    dv=vhat-v

    dvp_u=np.zeros(tuple(np.array(u.shape)+1))
    dvp_v=np.zeros(dvp_u.shape)
    dsf_u=np.zeros(dvp_u.shape)
    dsf_v=np.zeros(dvp_u.shape)

    def fillDx(slab,dz):
        slab[:-1,:-1]+=-0.5*dz/dx_n
        slab[1:,:-1]+=-0.5*dz/dx_s
        slab[1:,1:]+=0.5*dz/dx_s
        slab[:-1,1:]+=0.5*dz/dx_n
        return slab

    def fillDy(slab,dz):
        slab[:-1,:-1]+=-0.5*dz/dy_w
        slab[1:,:-1]+=0.5*dz/dy_w
        slab[1:,1:]+=0.5*dz/dy_e
        slab[:-1,1:]+=-0.5*dz/dy_e
        return slab

    dvp_u=fillDx(dvp_u,du)
    dvp_v=fillDy(dvp_v,dv)
    dsf_u=fillDy(dsf_u,-du)
    dsf_v=fillDx(dsf_v,dv)

    dsf=dsf_u+dsf_v
    dvp=dvp_u+dvp_v

    dsf=dsf+lam*params[0]
    dvp=dvp+lam*params[1]

    return dsf, dvp

def costFunc2(params,u,v,dx_n,dx_s,dy_w,dy_e,pad_shape,lam):
    '''Compute cost function of optimization, for optimize method'''
    pp=params.reshape(pad_shape)
    sf=pp[0]
    vp=pp[1]
    uhat=uRecon(sf,vp,dx_n,dx_s,dy_w,dy_e)
    vhat=vRecon(sf,vp,dx_n,dx_s,dy_w,dy_e)
    j=np.array((uhat-u)**2+(vhat-v)**2).mean()
    j+=lam*np.mean(params**2)

    return j


def jac2(params,u,v,dx_n,dx_s,dy_w,dy_e,pad_shape,lam):
    '''Compute gradient of cost function of optimization, for optimize method'''
    pp=params.reshape(pad_shape)
    sf=pp[0]
    vp=pp[1]
    uhat=uRecon(sf,vp,dx_n,dx_s,dy_w,dy_e)
    vhat=vRecon(sf,vp,dx_n,dx_s,dy_w,dy_e)
    du=uhat-u
    dv=vhat-v

    dvp_u=np.zeros(tuple(np.array(u.shape)+1))
    dvp_v=np.zeros(dvp_u.shape)
    dsf_u=np.zeros(dvp_u.shape)
    dsf_v=np.zeros(dvp_u.shape)

    def fillDx(slab,dz):
        slab[:-1,:-1]+=-0.5*dz/dx_n
        slab[1:,:-1]+=-0.5*dz/dx_s
        slab[1:,1:]+=0.5*dz/dx_s
        slab[:-1,1:]+=0.5*dz/dx_n
        return slab

    def fillDy(slab,dz):
        slab[:-1,:-1]+=-0.5*dz/dy_w
        slab[1:,:-1]+=0.5*dz/dy_w
        slab[1:,1:]+=0.5*dz/dy_e
        slab[:-1,1:]+=-0.5*dz/dy_e
        return slab

    dvp_u=fillDx(dvp_u,du)
    dvp_v=fillDy(dvp_v,dv)
    dsf_u=fillDy(dsf_u,-du)
    dsf_v=fillDx(dsf_v,dv)

    dsf=dsf_u+dsf_v
    dvp=dvp_u+dvp_v

    re=np.vstack([dsf[None,:,:,],dvp[None,:,:]])
    re=re.reshape(params.shape)
    re=re+lam*params/u.size

    return re


class Wind2D(object):

    def __init__(self,u,v,method='optimize',lr=1.,lam=0.001,max_iter=3000,
        threshold=0.01,interp=False):
        '''Object of 2d winds

        Args:
            u, v (ndarray): 2D array with shape (nxm), u- and v- velocities,
                need to have same shape and defined on regular grid (uniform dx and dy).
            dx, d> (floats): x- and y- grid size. Unit should be consistent with
                the units of <u> and <v>. E.g. <u>, <v> in m/s, <dx>, <dy> in m.

        Kwargs:
            method (str): 'optimize' for optimization using scipy.
                           'GD' for gradient descent optimization.
            lr (float): learning rate. Only used if <method> == 'GD'.
            lam (float): regularization parameter, <lam> > 0.
            max_iter (int): max number of iterations for <method> == 'GD'.
                    Only used if <method> == 'GD'.
            threshold (float or None): If None, optimization runs all iterations
                 defined by <max_iter>. If float, allow optimization to
                 early stop if all absolute errors are smaller than <threshold>.
                 Only used if <method> == 'GD'.
            interp (bool): whether to interpolate streamfunction and velocity
                      potential to the grid of u and v

        Returns:
            sf, vp (ndarray): 2D array with shape ((n+1)x(m+1) if <interp>,
                nxm otherwise), streamfunction and velocity potential.
        '''

        #-------------------Check inputs-------------------
        if np.ndim(u)!=2:
            raise Exception("<u> needs to be 2D.")
        if np.ndim(v)!=2:
            raise Exception("<v> needs to be 2D.")

        latax=u.getLatitude()
        lonax=u.getLongitude()
        if latax is None:
            raise Exception("<u> has no latitude axis.")
        if lonax is None:
            raise Exception("<u> has no longitude axis.")

        if method not in ['optimize', 'GD']:
            raise Exception("<method> is one of ['optimize', 'GD']")
        if not np.isscalar(lr) and lr>0:
            raise Exception("<lr> needs to be positive scalar")
        if not np.isscalar(lam) and lam>0:
            raise Exception("<lam> needs to be positive scalar")
        if not np.isscalar(max_iter) and max_iter>0:
            raise Exception("<max_iter> needs to be positive scalar")
        if threshold is not None:
            if not np.isscalar(threshold) and threshold>0:
                raise Exception("<threshold> needs to be positive scalar")

        self.u=u(latitude=(-90,90))
        self.v=v(latitude=(-90,90))
        self.method=method
        self.lr=lr
        self.lam=lam
        self.max_iter=max_iter
        self.threshold=threshold
        self.interp=interp

        self.latax=latax
        self.lonax=lonax
        self.axislist=u.getAxisList()
        self.u_data=np.array(u)
        self.v_data=np.array(v)
        self.missing_mask=getMissingMask(u)

        #------------------Get grid sizes------------------
        self.dx_n=np.array(dLongitude(u,'n'))
        self.dx_s=np.array(dLongitude(u,'s'))
        self.dy_w=np.array(dLatitude(u))
        self.dy_e=self.dy_w

        dx_mean=0.5*(self.dx_n+self.dx_s)
        dy_mean=0.5*(self.dy_w+self.dy_e)
        scale_x=np.log10(dx_mean.mean())
        scale_y=np.log10(dy_mean.mean())
        self.scale=10**(0.5*(scale_x+scale_y))

        #-----------------Create dual grid-----------------
        yb=self.latax.getBounds()
        xb=self.lonax.getBounds()
        y_dual=np.unique(yb)
        x_dual=np.unique(xb)

        latax_dual=cdms.createAxis(y_dual)
        latax_dual.designateLatitude()
        latax_dual.id='y'
        latax_dual.units='degree north'

        lonax_dual=cdms.createAxis(x_dual)
        lonax_dual.designateLongitude()
        lonax_dual.id='x'
        lonax_dual.units='degree east'

        self.latax_dual=latax_dual
        self.lonax_dual=lonax_dual
        self.axislist_dual=[latax_dual,lonax_dual]

        self.sf=None
        self.vp=None


    def getSFVP(self,interp=None):
        '''Compute stream function and velocity potential from u and v

        Kwargs:
            interp (bool): whether to interpolate streamfunction and velocity
                      potential to the grid of u and v

        Returns:
            sf, vp (ndarray): 2D array with shape ((n+1)x(m+1) if <interp>,
                nxm otherwise), streamfunction and velocity potential.
        '''

        if interp is None:
            interp=self.interp

        #--------------Get inital guess from an uniform grid-------------
        sf0,vp0=getSFVP_reg(self.u_data,self.v_data,1,1,method='optimize',
                interp=False)
        params=np.vstack([sf0[None,:,:], vp0[None,:,:]])
        del sf0,vp0
        pad_shape=params.shape

        #--------------Scale down grid sizes--------------
        dx_n=self.dx_n/self.scale
        dx_s=self.dx_s/self.scale
        dy_w=self.dy_w/self.scale
        dy_e=self.dy_e/self.scale

        #---------------------Optimize---------------------
        if self.method=='GD':
            lr=self.lr
            if self.threshold is not None:
                check_inter=10
            ii=0
            costs=[]
            while True:
                ii+=1
                jii,uhatii,vhatii=costFunc(params,self.u_data,self.v_data,
                        dx_n,dx_s,dy_w,dy_e,self.lam)
                dsf,dvp=jac(params,self.u_data,self.v_data,uhatii,vhatii,
                        dx_n,dx_s,dy_w,dy_e,self.lam)

                old_params=params.copy()

                params[0]=params[0]-lr*dsf
                params[1]=params[1]-lr*dvp

                #--------------Shrink lr if overshoot--------------
                if ii>1 and jii>costs[-1]:
                    kk=0
                    while jii>costs[-1]:
                        kk+=1
                        params=old_params
                        lr=lr*0.95
                        params[0]=params[0]-lr*dsf
                        params[1]=params[1]-lr*dvp
                        jii,uhatii,vhatii=costFunc(params,self.u_data,self.v_data,
                                dx_n,dx_s,dy_w,dy_e,self.lam)
                        if kk>=50:
                            break

                costs.append(jii)

                if ii%10==1:
                    print '# <getSFVP>: Iteration : %d, cost : %f' %(ii,jii)

                #---------------Check if early stop---------------
                if self.threshold is not None:
                    if ii%check_inter==0:
                        duii=uhatii-self.u_data
                        dvii=vhatii-self.v_data
                        if np.all(abs(duii)<=self.threshold) and \
                            np.all(abs(dvii)<=self.threshold):
                            break
                if ii>self.max_iter:
                    break

            self.costs=np.array(costs)

        elif self.method=='optimize':

            opt=optimize.minimize(costFunc2,params,
                    args=(self.u_data,self.v_data,dx_n,dx_s,dy_w,dy_e,
                        pad_shape,self.lam),
                    method='Newton-CG',
                    jac=jac2)
            params=opt.x.reshape(pad_shape)
            print '\n# <getSFVP>: Cost function after optimization : %f' %opt.fun

        #----------------Scale up sf and vp----------------
        sf=params[0]*self.scale
        vp=params[1]*self.scale

        #-----------------Create variable-----------------
        sf=MV.array(sf)
        sf.setAxisList([self.latax_dual,self.lonax_dual])
        sf.id='sf'
        sf.long_name='streamfunction'
        sf.standard_name=sf.long_name
        sf.title=sf.long_name
        sf.units='m^2/s'

        vp=MV.array(vp)
        vp.setAxisList([self.latax_dual,self.lonax_dual])
        vp.id='vp'
        vp.long_name='velocity potential'
        vp.standard_name=vp.long_name
        vp.title=vp.long_name
        vp.units='m^2/s'

        #---Keep a record of sf and vp to avoid duplicate computes---
        self.sf=sf
        self.vp=vp

        #-------------------Interpolate-------------------
        if interp:
            sf,vp=self.interpolate(sf,vp)

        return sf, vp


    def interpolate(self,sf,vp):
        '''Interpolate to the grid of u and v'''
        x=self.lonax[:]
        y=self.latax[:]
        x_dual=self.lonax_dual[:]
        y_dual=self.latax_dual[:]

        sf_interpf=interpolate.interp2d(x_dual,y_dual,sf,kind='cubic')
        vp_interpf=interpolate.interp2d(x_dual,y_dual,vp,kind='cubic')
        sf=sf_interpf(x,y)
        vp=vp_interpf(x,y)

        sf=MV.array(sf)
        sf.setAxisList([self.latax,self.lonax])
        sf.id='sf'
        sf.long_name='streamfunction'
        sf.standard_name=sf.long_name
        sf.title=sf.long_name
        sf.units='m^2/s'

        vp=MV.array(vp)
        vp.setAxisList([self.latax,self.lonax])
        vp.id='vp'
        vp.long_name='velocity potential'
        vp.standard_name=vp.long_name
        vp.title=vp.long_name
        vp.units='m^2/s'

        return sf,vp


    def getStreamFunc(self,interp=None):
        if interp is None:
            interp=self.interp
        sf,vp=self.getSFVP(interp)
        return sf

    def getVelocityPotential(self,interp=None):
        if interp is None:
            interp=self.interp
        sf,vp=self.getSFVP(interp)
        return vp

    def helmholtz(self):
        '''Perform helmholtz decomposition from u and v winds'''

        if self.sf is None or self.vp is None:
            self.getSFVP(interp=False)

        uchi,vchi=self.getIrrotationalComponent()
        upsi,vpsi=self.getNondivergentComponent()

        return uchi,vchi,upsi,vpsi


    def getIrrotationalComponent(self):
        if self.sf is None or self.vp is None:
            sf,vp=self.getSFVP(False)
        else:
            vp=self.vp

        uchi=getUChi(vp,self.dx_n,self.dx_s)
        vchi=getVChi(vp,self.dy_w,self.dy_e)

        vlist=[uchi,vchi]
        for ii in range(len(vlist)):
            vii=MV.array(vlist[ii])
            vii.mask=self.missing_mask
            vii.setAxisList(self.axislist)
            vlist[ii]=vii
        uchi,vchi=vlist

        uchi.id='uchi'
        uchi.long_name='irrotational u-wind'
        uchi.standard_name=uchi.long_name
        uchi.title=uchi.long_name
        uchi.units='m/s'

        vchi.id='vchi'
        vchi.long_name='irrotational v-wind'
        vchi.standard_name=vchi.long_name
        vchi.title=vchi.long_name
        vchi.units='m/s'
        return uchi,vchi

    def getIC(self):
        return self.getIrrotationalComponent()

    def getNondivergentComponent(self):

        if self.sf is None or self.vp is None:
            sf,vp=self.getSFVP(False)
        else:
            sf=self.sf

        upsi=getUPsi(sf,self.dy_w,self.dy_e)
        vpsi=getVPsi(sf,self.dx_n,self.dx_s)

        vlist=[upsi,vpsi]
        for ii in range(len(vlist)):
            vii=MV.array(vlist[ii])
            vii.mask=self.missing_mask
            vii.setAxisList(self.axislist)
            vlist[ii]=vii
        upsi,vpsi=vlist

        upsi.id='upsi'
        upsi.long_name='non-divergent u-wind'
        upsi.standard_name=upsi.long_name
        upsi.title=upsi.long_name
        upsi.units='m/s'

        vpsi.id='vpsi'
        vpsi.long_name='non-divergent v-wind'
        vpsi.standard_name=vpsi.long_name
        vpsi.title=vpsi.long_name
        vpsi.units='m/s'
        return upsi,vpsi

    def getNC(self):
        return self.getNondivergentComponent()





if __name__=='__main__':

    sf, vp, uhat, vhat=example_reg()

