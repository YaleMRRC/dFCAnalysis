import h5py
import time
import os, sys
import glob
from functools import reduce

import numpy as np
import pandas as pd
import pickle

from leida import calc_eigs

import hdbscan
import sklearn.cluster
import scipy.cluster
from scipy import stats,io
from scipy import signal as sg
import sklearn.datasets
from sklearn.decomposition import PCA
import statsmodels.formula.api as sm
from scipy.spatial import ConvexHull
import scipy as sp
import pdb
#### Dynamic Estimates ####

def return_sliding_windows(ipdata,window_size=30):
    '''
    Takes a matrix of size subs x ntimepoints x rois 
    and returns a matrix of sliding window correlations
    of size subs x ntimepoints x rois x rois - window_size
    '''
    nsubs,ntps,nrois=ipdata.shape

    opdata=np.zeros([nsubs,(ntps-window_size),nrois,nrois])

    for sub in range(0,nsubs):
        for nwindow in range(0,(ntps-window_size)):
            opdata[sub,nwindow,:,:]=np.corrcoef(ipdata[sub,nwindow:(nwindow+window_size),:].T)


    return opdata


def cosine_similarity(timeseries):
    """
    Function to calculate similarity between timeseries as a
    function of the angle of the complex representation
    Takes NxM matrix, where M = number of timeseries, and 
    N = number of timepoints
    Returns a matrix of size N x M x M
    """
    n_ts=timeseries.shape[1]
    n_tp=timeseries.shape[0]
    hilt = sg.hilbert(timeseries,axis=0)
    angles = np.angle(hilt)

    pw_diff=np.array([angles[v,:] - a for v in range(0,n_tp) for a in angles[v,:]])
    pw_diff=np.reshape(pw_diff,[n_tp,n_ts,n_ts])

    cos_sim=np.cos(pw_diff)

    return cos_sim

def harmonize_evecs(evec):
    # % Make sure the largest component is negative
    # % This step is important because the same eigenvector can
    # % be returned either as V or its symmetric -V and we need
    # % to make sure it is always the same (so we choose always
    # % the most negative one)
    if np.mean(evec>0)>.5:
        evec=-evec;
    elif np.mean(evec>0)==.5 and np.sum(evec[evec>0])>-np.sum(evec[evec<0]):
        evec=-evec;

    return evec


def calc_eigs(matrices,numevals="All"):
    """
    Takes NxMxM matrix and returns eigenvalues and eigenvectors
    """
    
    if len(matrices.shape) == 3:
        nvols,nrois,_=matrices.shape
    elif len(matrices.shape) == 2:
        #print('2D, Assuming this is an ROIxROI matrix')
        nrois,_=matrices.shape
        nvols=1
    else:
        raise(Exception("Not sure about this matrix shape"))

    evals=np.zeros([nvols,nrois])
    evecs=np.zeros([nvols,nrois,nrois])
    evars=np.zeros([nvols,nrois])

    for volnum in range(0,nvols):
        #print(volnum)

        if len(matrices.shape) == 3:
            eigs=sp.linalg.eigh(matrices[volnum,:,:])
        else:
            eigs=sp.linalg.eigh(matrices)


        tevals=eigs[0]
        tevecs=eigs[1]

        tevecs=np.array([harmonize_evecs(tevecs[:,i]) for i in range(0,tevecs.shape[1])]).T

        evsort=np.argsort(tevals)
        tevals=tevals[evsort[::-1]]
        evals[volnum,:]=tevals
        evecs[volnum,:,:]=tevecs[:,evsort[-1::]]
        evars[volnum,:]=np.array([tevals[i]/np.sum(tevals,axis=None) for i in range(0,tevals.shape[0])])



        #evecs=np.array([evecs[i,:,evsort[i,:]] for i in range(0,len(evsort))])
        #evars=np.array([evals[i,:]/np.sum(evals[i,:],axis=None) for i in range(0,evals.shape[0])])


    opdict={}

    if numevals == 'All':
        opdict['EigVals']=evals
        opdict['EigVecs']=evecs
        opdict['EigVars']=evars

    else:
        opdict['EigVals']=evals[:,0:numevals]
        opdict['EigVecs']=evecs[:,:,0:numevals]
        opdict['EigVars']=evars[:,0:numevals]

    return opdict


def tsfilt(timeseries):
    """
    Demean and detrend input timeseries of one subject
    accepts array of size NxM, N = timepoints, M = timeseries
    """
    ts_detrend=sg.detrend(timeseries,axis=0)
    ts_demean=ts_detrend-ts_detrend.mean(axis=0)
    
    return ts_demean


def dotnorm(v1,v2):
    return  np.dot(v1,v2)/np.linalg.norm(v1)/np.linalg.norm(v2)   

def indv_leida_mats(onesubdata,numeigs=1):
    """
    """

    filtered_data=tsfilt(onesubdata)

    cos_sim_data=cosine_similarity(filtered_data)

    opdict=calc_eigs(cos_sim_data,numevals=numeigs)

    evecs=opdict['EigVecs']
    tp,ts,numevec=evecs.shape
    fcd_list=[]
    for fcdi in range(0,numevec):
        evec=np.squeeze(evecs[:,:,fcdi])
        dns=np.array([dotnorm(e1, e2) for e1 in evec for e2 in evec])
        fcd_list.append(np.reshape(dns,[tp,tp]))


    opdict['FCD']=fcd_list

    return opdict

def return_leading_evecs(onesubdata,numeigs=1):

    filtered_data=tsfilt(onesubdata)

    cos_sim_data=cosine_similarity(filtered_data)

    opdict=calc_eigs(cos_sim_data,numevals=numeigs)

    evecs=np.squeeze(opdict['EigVecs'])

    return evecs

def meanphase_dump(args):

    ts_parcel,opdir,windowtps,window_anchor,subid,tplist=args

    wa_options=['start','middle','end']
    if window_anchor not in wa_options:
        raise Exception('Window anchor must be specified as one of: ', wa_options)
    elif window_anchor == 'middle' and not windowtps % 2:
        raise Exception('You specified an even window length, but to anchor the window at the middle timepoint, unsure how to proceed')



    ntps=ts_parcel.shape[0]
    phasecon=cosine_similarity(ts_parcel)
    fpaths=[]


    if not tplist:
        beginshift_dct={
        'start':0,
        'middle':np.ceil(windowtps/2).astype(int)-1,
        'end':windowtps-1}

        endshift_dct={
        'start':windowtps-1,
        'middle':np.ceil(windowtps/2).astype(int)-1,
        'end':0}


        beginshift=beginshift_dct[window_anchor]
        endshift=endshift_dct[window_anchor]
        tplist=range(0+beginshift,ntps-endshift)


    finalopfname='indvphasecon_tp_'+str(tplist[-1]).zfill(3)+'_'+window_anchor+'_numtps_'+str(windowtps).zfill(3)+'_sub_'+subid+'.pkl'
    finalopfpath=os.path.join(opdir,finalopfname)
    if os.path.isfile(finalopfpath):
        print('Final data already exists: ',finalopfpath)

    else:
        for tp in tplist:

            opfname='indvphasecon_tp_'+str(tp).zfill(3)+'_'+window_anchor+'_numtps_'+str(windowtps).zfill(3)+'_sub_'+subid+'.pkl'
            opfpath=os.path.join(opdir,opfname)
            fpaths.append(opfpath)

            if window_anchor == 'start':
                mean_window_phasecon=np.mean(phasecon[tp:(tp+windowtps),:,:],axis=0)
            elif window_anchor == 'middle':
                winstart=tp-round((windowtps-.5)/2)
                winend=tp+round((windowtps-.5)/2)+1
                mean_window_phasecon=np.mean(phasecon[winstart:winend,:,:],axis=0)
            elif window_anchor == 'end':
                mean_window_phasecon=np.mean(phasecon[(tp-windowtps+1):(tp+1),:,:],axis=0)

            tp_phasecon=np.expand_dims(mean_window_phasecon,0)
            pickle.dump(tp_phasecon,open(opfpath,'wb'))
        print('Final data written to: ',opfpath)

    return fpaths


def meanConfoundDump(ipname,windowtps,window_anchor):

    
    confoundDf=pd.read_csv(ipname,index_col=0)


    wa_options=['start','middle','end']
    if window_anchor not in wa_options:
        raise Exception('Window anchor must be specified as one of: ', wa_options)
    elif window_anchor == 'middle' and not windowtps % 2:
        raise Exception('You specified an even window length, but to anchor the window at the middle timepoint, unsure how to proceed')


    confoundArr=confoundDf.values
    colnames=confoundDf.columns
    ntps=confoundDf.values.shape[0] #???    

    beginshift_dct={
    'start':0,
    'middle':np.ceil(windowtps/2).astype(int)-1,
    'end':windowtps-1}

    endshift_dct={
    'start':windowtps-1,
    'middle':np.ceil(windowtps/2).astype(int)-1,
    'end':0}


    beginshift=beginshift_dct[window_anchor]
    endshift=endshift_dct[window_anchor]
    tplist=range(0+beginshift,ntps-endshift)

    newArr=np.zeros([len(tplist),len(colnames)])
    
    for i,tp in enumerate(tplist):

        if window_anchor == 'start':
            newArr[i,:]=np.mean(confoundArr[tp:(tp+windowtps),:],axis=0)
        elif window_anchor == 'middle':
            winstart=tp-round((windowtps-.5)/2)
            winend=tp+round((windowtps-.5)/2)+1
            
            newArr[i,:]=np.mean(confoundArr[winstart:winend,:],axis=0)
        elif window_anchor == 'end':
            newArr[i,:]=np.mean(confoundArr[(tp-windowtps+1):(tp+1),:],axis=0)


    opdf=pd.DataFrame(newArr,columns=colnames,index=tplist)

    return opdf 


def gather_meanphase(args):
    ippaths,oppath=args
    numfs=len(ippaths)


    if os.path.isfile(oppath):
        print('Already exists:', oppath)
        #for ipf in ippaths:
        #    os.remove(ipf)
    
    else:
        if numfs == 1:
            fpath=ippaths[0]
            av_pc=pickle.load(open(fpath,'rb'))
        #    os.remove(fpath)


        else:
            gather_mats=[pickle.load(open(ipf,'rb')) for ipf in ippaths]
            av_pc=np.stack(gather_mats).squeeze()
            #for ipf in ippaths:
            #    os.remove(ipf)

        pickle.dump(av_pc,open(oppath,'wb'))
  
        print('Wrote to:', oppath)

    return oppath

def calcLeadingEig_dump(args):
    ippath,oppath=args
    
    av_pc=pickle.load(open(ippath,'rb'))
    os.remove(ippath)

    opdict=calc_eigs(av_pc,numevals=1)

    evecs=np.squeeze(opdict['EigVecs'])
     
    pickle.dump(evecs,open(oppath,'wb'))

    print('Wrote evecs to:', oppath)

    return oppath

