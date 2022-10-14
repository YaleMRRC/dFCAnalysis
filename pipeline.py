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


import hdbscan
import sklearn.cluster
import scipy.cluster
from scipy import stats,io
import sklearn.datasets
from sklearn.decomposition import PCA
import statsmodels.formula.api as sm
from scipy.spatial import ConvexHull
import scipy as sp


import dfc
import utils
import cpm


from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool

import yaml


def produceDynFiles():
    pass



if __name__ == '__main__':



    ##$  Config Start  $##
    

    parser=argparse.ArgumentParser(description='Run split half CV CPM on instantaneous and windowed connectivity matrices')
    parser.add_argument('timeseries',help="Path to parcellated timeseries to be used to calculate dynamic connectivity")
    parser.add_argument('prediction_target',help="Path to csv with target predictions, with rows as timepoints and columns as subjects")
    parser.add_argument('subject_list',help="Path to csv/numpy file")
    parser.add_argument('nsubs',type=int,help='Number of subs in group pool')
    parser.add_argument('nsubs_to_run',type=int,help='Number of subs to randomly sample in each CPM run')
    parser.add_argument('workdir')
    parser.add_argument('window_lengths',help='For window lengths of 1,2 and 3, pass like so: "1,2,3" ')
    parser.add_argument('connectivity_type',help='static or dynamic')
    parser.add_argument('--window_anchor',type=str,help='Must be start middle or end')
    parser.add_argument('result_name',type=str,help='thing to append to end of resultfile')
    parser.add_argument('--trim_data',action='store_true',help='Remove time points from end of timeseries')
    parser.add_argument('--keepvols',type=int,help='Specify number of vols to keep')
    parser.add_argument('--shuffle',action='store_true')
    parser.add_argument('--calc_evec',action='store_true')

    parser.add_argument('--temporalmask',help='Path to mask; Specify which timepoints to include in analysis, only for static right now')

    parser.add_argument('--corrtype',help='"pearsonr" or "partial"; if partial please specify confound',type=str,default = 'pearsonr')
    parser.add_argument('--confound',help='path to confound csv')
    parser.add_argument('--cpmconfig',help='path to yaml file containting parameters for CPM run')


    
    args=parser.parse_args()

    ipfile=args.timeseries
    prediction_target=args.prediction_target
    subject_list=args.subject_list
    nsubs=args.nsubs
    workdir=args.workdir
    subs_to_run=args.nsubs_to_run
    trim_data=args.trim_data
    conn_type=args.connectivity_type
    shuff=args.shuffle
    calc_evec=args.calc_evec
    avtps_list=list(map(int,args.window_lengths.split(',')))
    window_anchor=args.window_anchor
    result_name=args.result_name

    corrtype=args.corrtype
    confoundfile=args.confound

    maskfile=args.temporalmask

    cpmConfigPath = args.cpmconfig


    with open(cpmConfigPath,'r') as fload: 
        cpmDict = yaml.safe_load(fload)

    cpmDict['subs_to_run'] = subs_to_run
    cpmDict['sublist'] = subject_list
    cpmDict['corrtype'] = corrtype



    ## paralell processing vals

    cpmParallelRuns = 15
    dFCCalcParallelRuns = 15
    
    ##$  Config End  $##





    ### Establishing paths and options ###

    # Whether or not to trim the timeseries
    if (trim_data) and not (type(args.keepvols) == int):
        raise Exception('If you want trimvols, must also set number of vols to keep')
    else:
        keepvols=args.keepvols

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


    ### Reading in data, subids and timepoints ###

    print("Gathering subject IDs")



    subject_list=list(map(str,np.load(subject_list, allow_pickle=True)))


    # Use only first x subs, again should get rid of this
    subject_list=subject_list[:nsubs]

    

    # Load data using function to extract from matlab files, should get rid
    print("Loading data")
    ts_parcel, subs = utils.load_timeseries('',ipfile,'','')

    

    ntpsTimeseries = ts_parcel[subject_list[0]].shape[0]

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




    if conn_type == 'dynamic':

        # Pick out timepoints with no nan values
        tps_with_RT=prediction_target.dropna(axis=0,how = 'all').index.values.astype(int)

        tps_with_enough_subs = prediction_target[np.sum(~np.isnan(prediction_target),axis=1) >= subs_to_run].index.values.astype(int)

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


    if corrtype == 'partial':

        # Read in target variable(s) and pull out subjects
        confoundDf=pd.read_csv(confoundfile,index_col=0)
        confound_subs=confoundDf.columns

        # Check if all subjects in subject list are in the prediction target array
        if not all([s in confound_subs for s in subject_list]):
            missing=[s for s in pt_subs if s not in subs]
            raise Exception('The following subjects do not have confound values: '+','.join(missing))

        # Reduce prediction_target to pertinent subjects
        confoundDf=confoundDf[subject_list]

        # Turn prediction target into array
        confoundArr=confoundDf.values


    
    ### Enacting options if specified ###

    # Trim timeseries if requested, maybe should get rid
    if trim_data:
        print("Trimming timeseries selected, trimming to ",keepvols," volumes")
        ts_parcel={k:ts_parcel[k][:keepvols,:] for k in ts_parcel}

    # Shuffle timeseries if requested   
    if shuff:

        print("Shuffling timeseries selected")
        # Created randomly shuffled temporal indices
        randinds=[np.random.permutation(ntpsTimeseries) for i in range(0,len(ts_parcel))]
        # Apply
        ts_parcel={k:ts_parcel[k][randinds[i],:] for i,k in enumerate(ts_parcel)}

        # Save random indices and subject list in results directory
        randdict={}
        randdict['Randinds']=randinds
        randdict['keys']=list(ts_parcel.keys())
        randdict['filteredsubs']=subject_list
        savepath=os.path.join(resdir,'randinds_subs.npy')
        np.save(savepath,randdict)





    
    ### Iterating over window lengths to calculate dFC ###
    ### and run CPM                                    ###    

    print('Running analysis')


    if conn_type == 'dynamic':
        for avtps in avtps_list:

            print(avtps)

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



            # Put in catch so code doesnt try to include tps not in imaging data
            tps_to_run_image=set(range(0+beginshift,ntpsTimeseries-endshift))
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



            # Pair down prediction target
            #pmats=pmats[:nsubs]
            #prediction_target=prediction_target[:,:nsubs]



            # Possibility to calculate leading eigenvector of connectivity matrices
            if calc_evec:
                raise Exception('Not sure things are indexing adequately')
                evec_ipfiles=[(cpmfiles[fnum],cpmfiles[fnum].replace('.pkl','_evecs.pkl')) for fnum in tps_to_run]
                with Pool(15) as p:
                    evecfiles=p.map(dfc.calcLeadingEig_dump,evec_ipfiles)

                cpm_ipfiles=[(evecfiles[fnum],prediction_target[fnum,:],fnum,True,subs_to_run) for fnum in tps_to_run]
                cpm_files=[cpmfiles[fnum].replace('.pkl','_evecs.pkl') for fnum in tps_to_run]

            else:
                if corrtype == 'partial':

                    #cpm_ipfiles=[(cpmfiles[i],prediction_target[fnum,:],fnum,True,subs_to_run,mask.values[fnum,:],subject_list,corrtype,confoundArr[fnum,:]) for i,fnum in enumerate(tps_to_run)]

                    cpm_ipfiles = []

                    for i,fnum in enumerate(tps_to_run):
                        subDct = copy.deepcopy(cpmDict)
                        tpDct = { 'ipmats' : cpmfiles[i],
                                  'pmats' : prediction_target[fnum,:],
                                  'tp' : fnum,
                                  'tpmask' : mask.values[fnum,:],
                                  'confound' : confoundArr[fnum,:]}

                        subDct.update(tpDct)

                        cpm_ipfiles.append(subDct)

                    


                else:
                                        
                    #cpm_ipfiles=[(cpmfiles[i],prediction_target[fnum,:],fnum,True,subs_to_run,mask.values[fnum,:],subject_list,corrtype,None) for i,fnum in enumerate(tps_to_run)]

                    cpm_ipfiles = []

                    for i,fnum in enumerate(tps_to_run):
                        subDct = copy.deepcopy(cpmDict)
                        tpDct = { 'ipmats' : cpmfiles[i],
                                  'pmats' : prediction_target[fnum,:],
                                  'tp' : fnum,
                                  'tpmask' : mask.values[fnum,:],
                                  'confound' : None}

                        subDct.update(tpDct)

                        cpm_ipfiles.append(subDct)




            # Run CPM at each timepoint
            print("Running CPM")
            with Pool(15) as p:
                Rval_dict=p.map(cpm.run_cpm,cpm_ipfiles)

            #for cf in cpm_files:
            #    os.remove(cf)

            # Write result

            opfile_name='dCPM_tps_'+str(avtps).zfill(3)+'_'+window_anchor+'_results_'+result_name+'.npy'
            oppath=os.path.join(resdir,opfile_name)
            print("Saving results: ",oppath)
            np.save(oppath,Rval_dict)

    

    if conn_type == 'static':

        if type(maskfile) == str and os.path.isfile(maskfile):
            print("Masking timeseries")
            mask=pd.read_csv(maskfile,index_col=0)
            mask=mask[subject_list]
            ts_parcel={k:ts_parcel[k][mask[k].values,:] for k in subject_list}

            if corrtype == 'partial':
                 confoundArr=np.vstack([np.mean(confoundArr[mask.values[:,i],i]) for i in range(0,len(subject_list)) ]).T
        
        elif type(maskfile) == str and not os.path.isfile(maskfile):
            raise Exception('Temporal mask does not exist')
        
        elif not maskfile:
            pass
        else:
            raise Exception('Not sure what is going on with the mask variable')


        # Determine output names of phase connectivity data
        opnames_gather_meanphase=[os.path.join(insta_pc_dir,'pc_static.pkl')]

        # Figure out if phase connectivity also exists
        ogm_mask=np.array([os.path.isfile(ogm) for ogm in opnames_gather_meanphase])

        # Run for any phase connectivity that doesnt exist
        if not all(ogm_mask):
            # Input array for multithreaded calculation of phase connectivity 
            thread_ips_pcmats=[(ts_parcel[subject_list[j]],insta_pc_dir,ts_parcel[subject_list[j]].shape[0],'start',str(j).zfill(3),[]) for j in range(0,len(subject_list))] # formerly nsubs

            # Dump out each windowed calculation for each subject
            with ThreadPool(15) as p:
                x=p.map(dfc.meanphase_dump,thread_ips_pcmats)
            p.join()

            # Create array of all those filenames
            aggfiles=np.stack(x).T

            # Input the timepoints we want to keep
            ipfiles=list(zip(aggfiles,opnames_gather_meanphase))
     
            print("Gathering instantaneous PC into timelocked matrix across subs")
            # Create one  matrix per timepoint with one window from all subs in it
            with ThreadPool(15) as p:
                cpmfiles=p.map(dfc.gather_meanphase,ipfiles)

            p.join()

            # These are the input files to CPM
            cpmfiles=opnames_gather_meanphase
        else:
            cpmfiles=opnames_gather_meanphase
        

        if corrtype == 'pearsonr':
            cpm_ipfiles=[(cpmfiles[0],prediction_target[0],0,True,subs_to_run,False,subject_list,corrtype,None)]
        else:
            cpm_ipfiles=[(cpmfiles[0],prediction_target[0],0,True,subs_to_run,False,subject_list,corrtype,confoundArr[0])]




        # Run CPM at each timepoint
        print("Running CPM")
        with ThreadPool(10) as p:
            Rval_dict=p.map(cpm.run_cpm,cpm_ipfiles)


        # Write result
        opfile_name='dCPM_static_results_'+result_name+'.npy'
        oppath=os.path.join(resdir,opfile_name)
        print("Saving results: ",oppath)
        np.save(oppath,Rval_dict)



