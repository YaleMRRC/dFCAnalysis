import h5py
import time
import os, sys
import glob
from functools import reduce
import pickle
import argparse
import pdb
import warnings

import numpy as np
import pandas as pd
import random

import copy

import matplotlib
if (os.name == 'posix' and "DISPLAY" in os.environ) or (os.name == 'nt'):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import pyplot as plt
    import seaborn as sns
    import matplotlib.animation as animation

elif os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('agg')
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import pyplot as plt
    import seaborn as sns
    import matplotlib.animation as animation


from scipy import stats,io
import statsmodels.formula.api as sm
import scipy as sp


import dfc
import cpm


from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool


import yaml



def createDFC(ts_parcel,avtps,window_anchor,tps_with_RT,insta_pc_dir,subject_list):
    # Figuring out start and end timepoints based on imaging data and 
    # prediction targets available
    beginshift_dct={
    'start':0,
    'middle':np.ceil(avtps/2).astype(int)-1,
    'end':avtps-1}

    endshift_dct={
    'start':avtps-1,
    'middle':np.ceil(avtps/2).astype(int)-1,
    'end':0}

    beginshift=beginshift_dct[window_anchor]
    endshift=endshift_dct[window_anchor]

    subs  = list(ts_parcel.keys())
    ntps = ts_parcel[subs[0]].shape[0]

    # Put in catch so code doesnt try to include tps not in imaging data
    tps_to_run_image=set(range(0+beginshift,ntps-endshift))
    tps_to_run=sorted(list(set.intersection(set(tps_with_RT),tps_to_run_image)))

    # Determine output names of phase connectivity data
    opnames_gather_meanphase=[os.path.join(insta_pc_dir,'pc_tp_'+str(tp).zfill(3)+'_av'+str(avtps).zfill(3)+'_'+window_anchor+'.pkl') for tp in tps_to_run]

    # Figure out if phase connectivity also exists
    ogm_mask=np.array([os.path.isfile(ogm) for ogm in opnames_gather_meanphase])


    # Run for any phase connectivity that doesnt exist
    if not all(ogm_mask):
        print('Calculating instantaneous PC \n Starting threading avtps: ',avtps)
        # Figure out which timepoints already have data and exclude from calculation
        tps_to_run_arr=np.array(tps_to_run)
        tps_to_dump=list(tps_to_run_arr[~ogm_mask])


        # Input array for multithreaded calculation of phase connectivity 
        thread_ips_pcmats=[(ts_parcel[subject_list[j]],insta_pc_dir,avtps,window_anchor,str(j).zfill(3),tps_to_dump) for j in range(0,len(subject_list))] # formerly nsubs

        # Dump out each windowed calculation for each subject
        with ThreadPool(15) as p:
            x=p.map(dfc.meanphase_dump,thread_ips_pcmats)
        p.join()

        # Create array of all those filenames
        aggfiles=np.stack(x).T

        # Filter out timepoints we want to keep
        tps_in_aggfiles=list(map(lambda x: x.split('_')[x.split('_').index('tp')+1],aggfiles[:,0]))
        tps_in_aggfiles=np.array(tps_in_aggfiles).astype(int)
        include_arr=np.isin(tps_in_aggfiles,tps_to_run)
        aggfiles_to_delete=aggfiles[~include_arr,:]
        aggfiles_to_include=aggfiles[include_arr,:]

        # Delete the ones we dont want
        for aggdel in aggfiles_to_delete.flatten():
             os.remove(aggdel)

        # Input the timepoints we want to keep
        ogm_newlist=list(np.array(opnames_gather_meanphase)[~ogm_mask])
        ipfiles=list(zip(aggfiles_to_include,ogm_newlist))
 
        print("Gathering instantaneous PC into timelocked matrix across subs")
        # Create one  matrix per timepoint with one window from all subs in it
        with ThreadPool(15) as p:
            cpmfiles=p.map(dfc.gather_meanphase,ipfiles)

        p.join()

        # These are the input files to CPM
        cpmfiles=opnames_gather_meanphase


        # Delete the ones we dont want
        for aggdel in aggfiles_to_include.flatten():
             os.remove(aggdel)

    else:
        cpmfiles=opnames_gather_meanphase
        print("input matrices already exist")

    return cpmfiles,tps_to_run



def run_apply_model(model,modeltp,thresh,subs_to_run,opnametag):
    pass




if __name__ == '__main__':



    ymlFile = sys.argv[1]

    with open(ymlFile,'r') as fload: 
        optionDct = yaml.safe_load(fload) 
    


    modeldct = optionDct['modeldct']




    subject_list=optionDct['subject_list_path']
    ipfile=optionDct['ipfile']
    prediction_target=optionDct['prediction_target']


    avtps_list=optionDct['avtps_list']
    window_anchor=optionDct['window_anchor']



    workdir = optionDct['workdir']



    subject_list=sorted(list(np.load(subject_list,allow_pickle=True)))


    subs_to_run = optionDct['subs_to_run']
    niters = optionDct['niters']


    subListInds = optionDct['subject_list_indices']

    subject_list=subject_list[subListInds[0]:subListInds[1]]

    threshList = list(map(lambda x : round(x,2),optionDct['threshList']))



    # Make sure working directory exists
    if not os.path.isdir(workdir):
        raise Exception('Working directory specified does not exist')

    # Declare directories to put FC and results of CPM
    insta_pc_dir=os.path.join(workdir,'insta_pc')
    resdir=os.path.join(workdir,'results')

    # If they dont exist make them
    for fpath in [insta_pc_dir,resdir]:
        if not os.path.isdir(fpath):
            os.makedirs(fpath)


    print("Gathering subject IDs")
    # Load data using function to extract from matlab files, should get rid
    print("Loading data")

    ts_parcel=np.load(ipfile,allow_pickle=True).item()
    subs = ts_parcel.keys()

    
    
    # Make sure all subs from sublist are in the timeseries object
    if not all([s in subs for s in subject_list]):
        missing=[s for s in subject_list if s not in subs]
        raise Exception('The following subjects do not have imaging data: '+','.join(missing))



    # Read in target variable(s) and pull out subjects
    prediction_target=pd.read_csv(prediction_target,index_col=0)
    pt_subs=prediction_target.columns

    # Check if all subjects in subject list are in the prediction target array
    if not all([s in pt_subs for s in subject_list]):
        missing=[s for s in pt_subs if s not in subs]
        raise Exception('The following subjects do not have prediction targets: '+','.join(missing))


    # Reduce prediction_target to pertinent subjects
    prediction_target=prediction_target[subject_list]

    # Pick out timepoints with no nan values
    tps_with_RT=prediction_target.dropna(axis=0,how = 'all').index.values.astype(int)

    tps_with_enough_subs = prediction_target[np.sum(~np.isnan(prediction_target),axis=1) > (subs_to_run+(subs_to_run*0.2))].index.values.astype(int)

    tps_with_RT=tps_with_RT[np.in1d(tps_with_RT,tps_with_enough_subs)]

    ### Attempting to have varying subjectlist by timepoint based on those with prediction
    ### target set to zero or NaN
    # Pick timepoints with no response
    mask=(prediction_target != 0) & ~prediction_target.isna()
    
    # Apply mask to subject list to create array of varying sublists
    sublistarr=np.array(subject_list)
    pred_target_subsbytp=mask.apply(lambda x : sublistarr[x],axis=1).values
    #    Keep subs that are excluded by timepoint
    subbytp_exclude=mask.apply(lambda x : sublistarr[~x],axis=1).values
    
    # Turn prediction target into array
    prediction_target=prediction_target.values


    for modelTag in modeldct:
        subModelDct = modeldct[modelTag]

        model_results_file=os.path.join(subModelDct['ipfile'])

        tps = subModelDct['tps']
        modelFitTag = subModelDct['modelFit']
        edgesTag = subModelDct['featureEdges']

        resDict = {}



        ### Reading in data, subids and timepoints ###


        modelObj=np.load(model_results_file,allow_pickle=True)
        runGather = []
        for tp in tps:

            tpStr = str(tp)

            modelTp = [mO for mO in modelObj if len(mO) > 0 and mO['tp'] == tp][0]
            modelFit = np.mean(np.concatenate([r for r in modelTp[modelFitTag]]),axis=0)
            modelEdges =  np.mean(np.concatenate([r for r in modelTp[edgesTag]]),axis=0)


            ### Iterating over window lengths to calculate dFC ###
            ### and run CPM                                    ###    

            for avtps in avtps_list:

                print('Generating dfc for window length: ', avtps)
                avtpsStr = str(avtps)
                cpmfiles,tps_to_run=createDFC(ts_parcel,avtps,window_anchor,tps_with_RT,insta_pc_dir,subject_list)


                for i,fnum in enumerate(tps_to_run):
                    fnumStr = str(fnum)



                    for thresh in threshList:
                        threshStr=str(thresh).zfill(1).replace('.','_')

                        result_name = f"model-{modelTag}_ModelTp-{tpStr}_TestTp-{fnumStr}_windowLen-{avtpsStr}_Thresh-{threshStr}"
                        predscore_agg=np.zeros(niters)

                        for itr in range(0,niters):

                            ipDict = {}
                            ipDict['ipmats'] = cpmfiles[i]
                            ipDict['pmats'] = prediction_target[fnum,:]
                            ipDict['modelEdges'] = modelEdges > thresh
                            ipDict['modelFit'] = modelFit
                            ipDict['mask'] = mask.values[fnum,:]
                            ipDict['readfile'] = True
                            ipDict['subsToRun'] = subs_to_run
                            ipDict['randomize'] = True
                            ipDict['tag'] = f"model-{modelTag}_ModelTp-{tpStr}_TestTp-{fnumStr}_windowLen-{avtpsStr}_Thresh-{threshStr}_Iter-{itr}"
                            runGather.append(ipDict)


        print("Running CPM")
        with Pool(8) as p:
            Rval_dict=p.map(cpm.apply_cpm_para,runGather)

        # Write result
        opfile_name = f"model-{modelTag}.npy"
        oppath=os.path.join(resdir,opfile_name)
        print("Saving results: ",oppath)
        np.save(oppath,Rval_dict)
