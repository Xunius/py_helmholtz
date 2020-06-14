'''Compare streamfunction and velocity potential computed from windspharm
and this code.

Steps:
    1. get some global u-, v- winds.
    2. compute streamfunction and velocity potential using spherical harmonics.
    3. crop a regional box, and compute sf, vp using this method.
    4. compare the sf, vp in the box from different methods.
'''

from helmholtz import Wind2D
import cdms2 as cdms
from windspharm.cdms import VectorWind
from tools import plot

if __name__=='__main__':

    #----------------Read in global u,v----------------
    abpath_in='/home/guangzhi/datasets/erai/uv_p900_6_2007_Jan.nc'
    print('\n### <helmholtz>: Read in file:\n',abpath_in)
    fin=cdms.open(abpath_in,'r')
    uall=fin('u', time=slice(0,1))
    vall=fin('v', time=slice(0,1))
    fin.close()

    uall=uall(squeeze=1)
    vall=vall(squeeze=1)
    print('uall.shape = ', uall.shape)
    print('vall.shape = ', vall.shape)

    # global wind harmonics
    wf=VectorWind(uall,vall)
    sf0, vp0=wf.sfvp()

    #-----------------Get a region box-----------------
    ubox=uall(latitude=(5,50),longitude=(100,180))
    vbox=vall(latitude=(5,50),longitude=(100,180))
    sf0=sf0(latitude=(5,50),longitude=(100,180))
    vp0=vp0(latitude=(5,50),longitude=(100,180))

    # create an wind obj, optimization will use gradient descent
    w2=Wind2D(ubox,vbox,'optimize')
    sf2,vp2=w2.getSFVP()
    uchi2,vchi2,upsi2,vpsi2=w2.helmholtz()
    uhat2=uchi2+upsi2
    vhat2=vchi2+vpsi2

    #-------------------Plot------------------------
    import matplotlib.pyplot as plt
    figure=plt.figure(figsize=(12,16),dpi=100)

    plot_vars1=[ubox, uhat2]
    plot_vars2=[vbox, vhat2]
    plot_vars3=[sf0, sf2]
    plot_vars4=[vp0, vp2]

    iso1=plot.Isofill(plot_vars1, 14, 1, 1)
    iso2=plot.Isofill(plot_vars2, 14, 1, 1)
    iso3=plot.Isofill(plot_vars3, 14, 1, 1)
    iso4=plot.Isofill(plot_vars4, 14, 1, 1)

    titles1=['U ori', r'$\hat{U}$']
    titles2=['V ori', r'$\hat{V}$']
    titles3=[r'$\psi$ windspharm', r'$\psi$ my code']
    titles4=[r'$\chi$ windspharm', r'$\chi$ my code']

    # plot U
    for ii in range(2):
        axii=figure.add_subplot(4, 2, ii+1)

        plot.plot2(plot_vars1[ii], iso1, axii,
                title=titles1[ii],
                projection='cyl',
                fix_aspect=False,
                legend='local')

    # plot V
    for ii in range(2):
        axii=figure.add_subplot(4, 2, 2*1+ii+1)

        plot.plot2(plot_vars2[ii], iso2, axii,
                title=titles2[ii],
                projection='cyl',
                fix_aspect=False,
                legend='local')

    # plot psi
    for ii in range(2):
        axii=figure.add_subplot(4, 2, 2*2+ii+1)

        plot.plot2(plot_vars3[ii], iso3, axii,
                title=titles3[ii],
                projection='cyl',
                fix_aspect=False,
                legend='local')

    # plot chi
    for ii in range(2):
        axii=figure.add_subplot(4, 2, 2*3+ii+1)

        plot.plot2(plot_vars4[ii], iso4, axii,
                title=titles4[ii],
                projection='cyl',
                fix_aspect=False,
                legend='local')

        #figure.subplots_adjust(wspace=0.075, hspace=0.335)
    figure.show()
    #----------------- Save plot------------
    plot_save_name='compare'
    print('\n# <compare>: Save figure to', plot_save_name)
    figure.savefig(plot_save_name+'.png',dpi=100,bbox_inches='tight')
    figure.savefig(plot_save_name+'.pdf',dpi=100,bbox_inches='tight')


