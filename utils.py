import h5py
import time
import os, sys
import glob
from functools import reduce

import numpy as np
import pandas as pd

import hdbscan
import sklearn.cluster
import scipy.cluster
from scipy import stats,io
import sklearn.datasets
from sklearn.decomposition import PCA
import statsmodels.formula.api as sm
from scipy.spatial import ConvexHull

## Utils

def otherstuff():
    clus=run_hdbscan(rest_wm_agg_wevs[:1607,:])
    k_means = sklearn.cluster.KMeans(10)
    clus=k_means.fit(rest_wm_agg_wevs)
    np.save('59subs_kmeans10.npy',clus)
    
    rest_wm_agg_wevs_res=np.reshape(rest_wm_agg_wevs,[59,1607,268])
    hdbscan_indv=[run_hdbscan(rest_wm_agg_wevs_res[i,:,:]) for i in range(0,59)]
    np.save('59subs_hdbscan_indv.npy',hdbscan_indv)
    dbscan_indv=[[sklearn.cluster.dbscan(rest_wm_agg_wevs_res[j,:,:],eps=i) for i in [0.1,0.5,1,2,3,4,5]] for j in range(0,59)]
    x=word_lbls*59
    xtrue=[True if x1 == 'rest_le' else False for x1 in x]
    subsdf=pd.DataFrame(subs,columns=['Subject'])
    pmat_df=pd.read_csv('./pmatfilter.csv')
    subsdf.Subject = subsdf.Subject.astype('int')
    subs59pmat=pd.merge(subsdf,pmat_df,on='Subject')
    plt.hist(subs59pmat.PMAT24_A_CR)
    plt.show()
    
    restkmeanslabels=clus.labels_[xtrue]
    restkmeanslabels_res=np.reshape(restkmeanslabels,[59,1200])
    kmeans_rest_switches=np.diff(restkmeanslabels_res) != 0
    krestswitch_bysub=np.sum(kmeans_rest_switches,axis=1)
    np.corrcoef(subs59pmat.PMAT24_A_CR,krestswitch_bysub)
    xload=np.load('hdbscan59subsle.npy')
    hdbscan_clus=xload.item()
    rest_hdb_labels=hdbscan_clus.labels_[xtrue]
    rest_hdb_labels_res=np.reshape(rest_hdb_labels,[59,1200])
    hdblabels_df=pd.DataFrame(rest_hdb_labels_res.T)
    hdb_ffill=hdblabels_df.apply(lambda x : x.replace(to_replace=-1,method='ffill'),axis=0)
    hdbswitches=np.sum(hdb_ffill.diff() != 0)
    np.corrcoef(subs59pmat.PMAT24_A_CR,hdbswitches)


    
    state_mem_vec=np.concatenate([np.sum(restkmeanslabels_res == i,axis=1) for i in range(0,10)])
    state_list=np.repeat(['state_'+str(i).zfill(2) for i in range(0,10)],59)
    sublist=['sub'+str(i).zfill(2) for i in range(1,60)]*10
    statememdf=pd.DataFrame(np.stack([state_mem_vec,state_list,sublist]).T,columns=['NumVols','State','Subject']) 
    sns.swarmplot(x='State',y='NumVols',data=statememdf)



def get_data(i=0):
    x,y = np.random.normal(loc=i,scale=3,size=(2, 260))
    return x,y



time_samples = [1000, 2000, 5000, 10000, 25000, 50000, 75000, 100000, 250000, 500000, 750000,
               1000000, 2500000, 5000000, 10000000, 50000000, 100000000, 500000000, 1000000000]

def get_timing_series(data, quadratic=True):
    '''
    Taken from: https://hdbscan.readthedocs.io/en/latest/performance_and_scalability.html
    '''
    if quadratic:
        data['x_squared'] = data.x**2
        model = sm.ols('y ~ x + x_squared', data=data).fit()
        predictions = [model.params.dot([1.0, i, i**2]) for i in time_samples]
        return pd.Series(predictions, index=pd.Index(time_samples))
    else: # assume n log(n)
        data['xlogx'] = data.x * np.log(data.x)
        model = sm.ols('y ~ x + xlogx', data=data).fit()
        predictions = [model.params.dot([1.0, i, i*np.log(i)]) for i in time_samples]
        return pd.Series(predictions, index=pd.Index(time_samples))


def benchmark_algorithm(dataset_sizes, cluster_function, function_args, function_kwds,
                        dataset_dimension=10, dataset_n_clusters=10, max_time=45, sample_size=2):
    '''
    Taken from: https://hdbscan.readthedocs.io/en/latest/performance_and_scalability.html
    '''
    # Initialize the result with NaNs so that any unfilled entries
    # will be considered NULL when we convert to a pandas dataframe at the end
    result = np.nan * np.ones((len(dataset_sizes), sample_size))
    for index, size in enumerate(dataset_sizes):
        for s in range(sample_size):
            print("Running for dataset size: ",size,"Features",dataset_dimension)
            # Use sklearns make_blobs to generate a random dataset with specified size
            # dimension and number of clusters
            data, labels = sklearn.datasets.make_blobs(n_samples=size,
                                                       n_features=dataset_dimension,
                                                       centers=dataset_n_clusters)

            # Start the clustering with a timer
            start_time = time.time()
            cluster_function(data, *function_args, **function_kwds)
            time_taken = time.time() - start_time

            # If we are taking more than max_time then abort -- we don't
            # want to spend excessive time on slow algorithms
            if time_taken > max_time:
                result[index, s] = time_taken
                return pd.DataFrame(np.vstack([dataset_sizes.repeat(sample_size),
                                               result.flatten()]).T, columns=['x','y'])
            else:
                result[index, s] = time_taken

    # Return the result as a dataframe for easier handling with seaborn afterwards
    return pd.DataFrame(np.vstack([dataset_sizes.repeat(sample_size),
                                   result.flatten()]).T, columns=['x','y'])


def loadmatv73(fname):
    f_load = h5py.File(fname, 'r')
    opdict={}
    for k,v in f_load.items():
        opdict[k]=v

    return opdict

def loadmatv73_tree(fname):
    '''
    Help from:
    https://codereview.stackexchange.com/questions/38038/recursively-convert-a-list-of-lists-into-a-dict-of-dicts
    '''
    f_load = h5py.File(fname, 'r')

    
    def recurs_dict(thing):
        opdict={}


        for k,v in thing.items():
            print(k)
            if 'items' in dir(thing[k]):
                opdict[k]=recurs_dict(thing[k])
            else:
                opdict[k]=thing[k].value
        

        return opdict

    bdct=recurs_dict(f_load)

    return bdct


def load_timeseries(ippath,savepath,tier1,tier2):
    
    if not os.path.isfile(savepath):
        ts_parcel=loadmatv73_tree(ippath)
        ts_parcel=ts_parcel[tier1][tier2]
        np.save(savepath,ts_parcel)
    else:
        ts_parcel=np.load(savepath,allow_pickle=True).item()
    

    #subs=[k.replace('sub','') for k in ts_parcel.keys()]
    subs=[k for k in ts_parcel.keys()]


    return ts_parcel,subs





def compare_staticcon_phasecon(restsubs,wmsubs,eventsubs,ts_parcel_wm,ts_parcel_rest):

    # Event subs
    # Find possible working memory spreadsheets
    fpaths=glob.glob(f'../HCP-WM-LR-EPrime/*/*_3T_WM_run*_TAB_filtered.csv')
    eventsubs=[f.split('/')[2] for f in fpaths]
    
    # Filter subs based on whats common to both modalities
    subs_combo=list(sorted(set(restsubs).intersection(set(wmsubs)).intersection(set(eventsubs))))


    wm_phasecon={k:np.mean(dfc.cosine_similarity(ts_parcel_wm['sub'+k]),axis=0) for k in subs_combo}
    wm_phasecon=np.stack(wm_phasecon)



    rest_phasecon={k:np.mean(dfc.cosine_similarity(ts_parcel_rest[k])) for k in ts_parcel_rest.keys()}
    rest_phasecon=np.stack(rest_phasecon)


    rest_sc=np.transpose(rest_sc,[2,0,1])
    
    ts_parcel_wm={k:ts_parcel_wm[k] for k in subs_combo[:50]}
    ts_parcel_rest={k:ts_parcel_rest[k] for k in subs_combo[:50]}



    # Iterate over subjects
    for nsub in range(0,1):
        ts_wm=ts_parcel_wm[subs_combo[nsub]]
        wm_static_corr=np.corrcoef(ts_wm.T).flatten()
        wm_mean_phase_corr=np.mean(dfc.cosine_similarity(ts_wm),axis=0).flatten()
        wm_var_phase_corr=np.var(dfc.cosine_similarity(ts_wm),axis=0).flatten()
        wm_cv_phase_corr=wm_var_phase_corr/wm_mean_phase_corr

        corrs=np.corrcoef([wm_static_corr,wm_mean_phase_corr,wm_var_phase_corr,wm_cv_phase_corr])
        
        print('WM:',corrs)



        ts_rest=ts_parcel_rest[subs_combo[nsub]]
        rest_static_corr=np.corrcoef(ts_rest.T).flatten()
        rest_mean_phase_corr=np.mean(dfc.cosine_similarity(ts_rest),axis=0).flatten()
        rest_var_phase_corr=np.var(dfc.cosine_similarity(ts_rest),axis=0).flatten()
        rest_cv_phase_corr=rest_var_phase_corr/rest_mean_phase_corr

        corrs=np.corrcoef([rest_static_corr,rest_mean_phase_corr,rest_var_phase_corr,rest_cv_phase_corr])
        
        print('Rest:',corrs)



def gather_unfilt_ts(ip_globstr):
    fs=glob.glob(ip_globstr)
    fs=list(sorted(fs))
    sublist=list(sorted(['sub'+f.split('/')[7] for f in fs]))

    opdict={}
    for i,fp in enumerate(fs):
        print(fp)
        subname=sublist[i]
        tsdf=pd.read_csv(fp,sep='\t',index_col=0)
        tsdf=tsdf.dropna(axis=1)
        opdict[subname]=tsdf.values
    np.save('./wm_ts_unfiltered.npy',opdict)



def produce_pheno():
    
    ## Accuracy and Response Time
    accdf=pd.read_csv('/data15/mri_group/dave_data/dcpm/AccuracyByTP.csv',index_col=0)
    RTdf=pd.read_csv('/data15/mri_group/dave_data/dcpm/ResponseTimeByTP.csv',index_col=0)
    RTdf[RTdf == 0] = 2000
    RTdf[accdf == 0] = 2000
    RTdf.columns=list(map(lambda x: x.replace('sub',''),RTdf.columns))
    RTdf.to_csv('/data15/mri_group/dave_data/dcpm/RT_inputtoCPM.csv')

    pmat_old=pd.read_csv('/home/dmo39/pmat.csv')
    newdata=np.tile(np.expand_dims(pmat_old.PMAT24_A_CR.values,axis=1),405)
    pmat_new=pd.DataFrame(newdata.T,columns=pmat_old.Subject)
    pmat_new.to_csv('/data15/mri_group/dave_data/dcpm/PMATs_inputtoCPM.csv')


    #pmat_all=pd.read_csv('/home/dmo39/pmat.csv')
    #pmat_filter=pmat_all[pmat_all.Subject.isin(subs_combo_pmat)]
    #pmats=pmat_filter.PMAT24_A_CR.values.astype(int)


