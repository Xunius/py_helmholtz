# py_helmholtz
Compute streamfunction, velocity potential and helmholtz decomposition from non-global wind data


## 1. To solve for streamfunction (psi) and velocity potential (chi) from u- and v- winds, on a uniform grid (uniform dx and dy everywhere):

u and v are given in even grid (n x m).

streamfunction (psi) and velocity potential (chi) are defined on a dual
grid (`(n+1) x (m+1)`), where psi and chi are defined on the 4 corners
of u and v.

Define:

```
    u = u_chi + u_psi
    v = v_chi + v_psi

    u_psi = -dpsi/dy
    v_psi = dpsi/dx
    u_chi = dchi/dx
    v_chi = dchi/dy
```

Define 2 2x2 kernels:

```
    k_x = |-0.5 0.5|
          |-0.5 0.5| / dx

    k_y = |-0.5 -0.5|
          |0.5   0.5| / dy
```

Then `u_chi = chi \bigotimes k_x`
where `\bigotimes` is cross-correlation,

Similarly:

```
    v_chi = chi \bigotimes k_y
    u_psi = psi \bigotimes -k_y
    v_psi = psi \bigotimes k_x

```

Define cost function `J = (uhat - u)**2 + (vhat - v)**2`

Gradients of chi and psi:

```
    dJ/dchi = (uhat - u) du_chi/dchi + (vhat - v) dv_chi/dchi
    dJ/dpsi = (uhat - u) du_psi/dpsi + (vhat - v) dv_psi/dpsi

    du_chi/dchi = (uhat - u) \bigotimes Rot180(k_x) = (uhat - u) \bigotimes -k_x
    dv_chi/dchi = (vhat - v) \bigotimes Rot180(k_y) = (vhat - v) \bigotimes -k_y
    du_psi/dpsi = (uhat - u) \bigotimes k_x
    dv_psi/dpsi = (vhat - v) \bigotimes Rot180(k_x) = (vhat - v) \bigotimes -k_x
```

Add optional regularization term:

```
    J = (uhat - u)**2 + (vhat - v)**2 + lambda(chi**2 + psi**2)
```

## 2. To solve for streamfunction and velocity potential from u- and v- winds on irregular grid (e.g. mercator):

Use similar definition of cost function and gradients, except that the computation
of component winds and derivatives are performed on steps for NE, NW, SE, SW
qudarants:

```
    u_chi = 0.5*((vp[:-1,1:]-vp[:-1,:-1])/dx_n + (vp[1:,1:]-vp[1:,:-1])/dx_s)
    v_chi = 0.5*((vp[1:,:-1]-vp[:-1,:-1])/dy_w + (vp[1:,1:]-vp[:-1,1:])/dy_e)
    u_psi = -0.5*((sf[1:,:-1]-sf[:-1,:-1])/dy_w + (sf[1:,1:]-sf[:-1,1:])/dy_e)
    v_psi = 0.5*((sf[:-1,1:]-sf[:-1,:-1])/dx_n + (sf[1:,1:]-sf[1:,:-1])/dx_s)
    
```

`du_chi/dchi`, `dv_chi/dchi`, `du_psi/dpsi` and `dv_psi/dpsi` are also compose by 4 quadrants, see code for details.



## 3. Designed for computation of netcdf data loaded via the CDAT interface.

CDAT: https://cdat.llnl.gov/index.html


## 4. Example

```
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

```
