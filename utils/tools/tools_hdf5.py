# -*- coding: utf-8 -*-
"""
Tools for reading .h5 files.
Author : COMET team at LISN
"""
import numpy as np
import h5py

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.rcParams.update({'font.size': 16})
font = {'family' : 'sans serif', 'weight' : 'normal', 'size'   : 20}
plt.rc('font', **font)

#--------------------------------------------------------------------------
def Read_H5_Spav(datafilename,champ='',keyreading=False,gridreading=False):
            
    #print("datafilename =",datafilename) 
    #print(champ,keyreading,gridreading)

    # open the file
    hf = h5py.File(datafilename, 'r')
    # file content
    if keyreading:
        keys=list(hf.keys())
        print('keyreading : liste des champs', keys )
        # close the file
        hf.close()
        return keys
    # grab each dataset
    elif gridreading:
        hf.keys()
        xx = hf.get('XC')
        yy = hf.get('YC')
        # convert HDF5 dataset object into numpy array
        xx = np.array(xx)
        yy = np.array(yy)
        # close the file
        hf.close()
        return xx,yy
    else:
        hf.keys()
        tt = hf.get('time')
        snap = hf.get(champ)
        # convert HDF5 dataset object into numpy array
        snap = np.array(snap)
        tt= np.array(tt)
        # close the file
        hf.close()
        return tt,snap
#--------------------------------------------------------------------------
def Read_h5_inputdata(datafilename,champ='',Nu='',keyreading=False,gridreading=False):
            
    print("datafilename =",datafilename) 
    print(champ,Nu,keyreading,gridreading)

    # open the file
    hf = h5py.File(datafilename, 'r')
    # file content
    if keyreading:
        keys=list(hf.keys())
        print('keyreading : liste des champs', keys )
        # close the file
        hf.close()
        return keys
    # grab each dataset
    elif gridreading:
        hf.keys()
        tlen = hf.get ('Len_Time')
        xx = hf.get('XC')
        yy = hf.get('YC')
        # convert HDF5 dataset object into numpy array
        tlen = np.array(tlen)
        xx = np.array(xx)
        yy = np.array(yy)
        # close the file
        hf.close()
        return tlen,xx,yy
    else:
        hf.keys()
        tt = hf.get('time')
        snap = hf.get(champ)
        nu = hf.get(Nu)
        # convert HDF5 dataset object into numpy array
        snap = np.array(snap)
        tt= np.array(tt)
        nu = np.array(nu)
        # close the file
        hf.close()
        return tt,snap,nu

#--------------------------------------------------------------------------
def Write_h5_meandata(datafilename,names='',nbc=1,champs='',x='',y=''): #,

    #  initialise the HDF5 file  
    hf = h5py.File(datafilename, 'w')
    # create a dataset
    hf.create_dataset('XC', data=x)
    hf.create_dataset('YC', data=y)
    for i in range (nbc):
        hf.create_dataset(names[i], data=champs[:,:,i])
    
    # close the file
    hf.close()
    return

#--------------------------------------------------------------------------
def Write_h5_snapdata(datafilename='',tt='',names='', champs='',coord=''):

    #  initialise the HDF5 file  
    hf = h5py.File(datafilename, 'w')
    nbc = len(names)
    # create a dataset
    hf.create_dataset('time', data=tt)
    hf.create_dataset('XC', data=coord[0])
    hf.create_dataset('YC', data=coord[1])
    for i in range (nbc):
        hf.create_dataset(names[i], data=champs[:,:,i])
    #hf.create_dataset('L2D2T', data=spav.L2D2T)
    #hf.create_dataset('T', data=spav.T)
    # close the file
    hf.close()
    return