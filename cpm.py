
'''

General functions for running CPM

Author: Dave O'Connor

'''


import numpy as np 
import scipy as sp
import pandas as pd
#from matplotlib import pyplot as plt
#import seaborn as sns
import glob
from scipy import stats,io,special
import random
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},reload_support=True)
import corr_multi
import pickle
import pdb


def RvalstoPvals(Rvals,df):
    '''
    Convert an array of R values to p values
    Rvals: Array of R values
    df: degrees of freedom 
    '''
    tvals=(Rvals*np.sqrt(df))/np.sqrt(1-Rvals**2)
    pvals=stats.t.sf(np.abs(tvals),df)*2

    return pvals



def train_cpm(ipmats, pheno, pthresh = 0.01, corrtype = 'pearsonr', confound = False):

    '''
    Accepts input matrices (N features x Nsubs) and pheno data
    Returns model

    ipmat: features
    pheno: target variables
    pthresh: the p threshold to use in feature selection
    corrtype: the type of correlation method to use in feature selection
    can be pearsons or partial
    confound: if corrtype is partial, confounds must be provided
    '''

    num_pheno=len(pheno)
    

    if corrtype == 'pearsonr':
        df=num_pheno-2
        Rvals=corr_multi.corr_multi_cy(pheno,ipmats.T)
        pvals = RvalstoPvals(Rvals,df) 
        posedges=(Rvals > 0) & (pvals < pthresh)
        posedges=posedges.astype(int)
        negedges=(Rvals < 0) & (pvals < pthresh)
        negedges=negedges.astype(int)
        pe=ipmats[posedges.flatten().astype(bool),:]
        ne=ipmats[negedges.flatten().astype(bool),:]
        pe=pe.sum(axis=0)/2
        ne=ne.sum(axis=0)/2

    elif corrtype == 'partial':
        if type(confound) != np.ndarray:
             raise Exception('if corrtype is partial confounds must be specified')

        df=num_pheno-3
       
        y=pheno
        y=(y-np.mean(y))/np.std(y)

        z=confound
        z=(z-np.mean(z))/np.std(z)

        

        ipmatMean=np.vstack(np.mean(ipmats,axis=1))
        ipmatStdv=np.vstack(np.std(ipmats,axis=1))


        

        ipmatNorm=(ipmats-ipmatMean)/ipmatStdv

        Rvals=corr_multi.partial_corr(ipmatNorm.T,y,z)
        pvals = RvalstoPvals(Rvals,df)

       
        
        posedges=(Rvals > 0) & (pvals < pthresh)
        posedges=posedges.astype(int)
        negedges=(Rvals < 0) & (pvals < pthresh)
        negedges=negedges.astype(int)
        pe=ipmats[posedges.flatten().astype(bool),:]
        ne=ipmats[negedges.flatten().astype(bool),:]
        pe=pe.sum(axis=0)/2
        ne=ne.sum(axis=0)/2

    else:
        raise Exception('corrtype must be "pearsonr" or "partial"')


    if np.sum(pe) != 0:
        fit_pos=np.polyfit(pe,pheno,1)
    else:
        fit_pos=[]

    if np.sum(ne) != 0:
        fit_neg=np.polyfit(ne,pheno,1)
    else:
        fit_neg=[]


    return fit_pos,fit_neg,posedges,negedges,Rvals



def kfold_cpm(ipmats, pheno, k = 2, corrtype = 'pearsonr', confound = False, pthresh = 0.01):
    '''
    Run k fold cross validation of CPM

    
    ipmat: feature matrix (N features x n subs)
    pheno: target variables
    k: number of folds
    corrtype: the type of correlation method to use in feature selection
    can be pearsons or partial
    confound: if corrtype is partial, confounds must be provided

    '''
    nrois, numsubs = ipmats.shape

    randinds=np.arange(0,numsubs)
    random.shuffle(randinds)

    samplesize=int(np.floor(float(numsubs)/k))


    behav_pred_pos=np.zeros([k,samplesize])
    behav_pred_neg=np.zeros([k,samplesize])
    behav_actual=np.zeros([k,samplesize])
    rvals_gather=np.zeros([k,nrois]).astype(np.float16)

    nedges=ipmats.shape[0]

    posedge_gather=[]
    negedge_gather=[]
    pf_gather=[]
    nf_gather=[]

    for fold in range(0,k):
        print("Running fold:",fold+1)
        si=fold*samplesize
        fi=(fold+1)*samplesize


        if fold != k-1:
            testinds=randinds[si:fi]
        else:
            testinds=randinds[si:]

        traininds=randinds[~np.isin(randinds,testinds)]


        trainmats=ipmats[:,traininds]
        trainpheno=pheno[traininds]
 
        testmats=ipmats[:,testinds]
        testpheno=pheno[testinds]

        if corrtype == 'partial':
            trainconf=confound[traininds]
            testconf=confound[testinds]
        else:
            trainconf=confound
            testconf=confound

        behav_actual[fold,:]=testpheno


        pos_fit,neg_fit,posedges,negedges,Rvals=train_cpm(trainmats,trainpheno,corrtype = corrtype,confound = trainconf, pthresh = pthresh)

        pe=np.sum(testmats[posedges.flatten().astype(bool),:], axis=0)/2
        ne=np.sum(testmats[negedges.flatten().astype(bool),:], axis=0)/2


        posedge_gather.append(posedges.astype(bool))
        negedge_gather.append(negedges.astype(bool))
        pf_gather.append(pos_fit)
        nf_gather.append(neg_fit)

        rvals_gather[fold,:] = Rvals.astype(np.float16)

        if len(pos_fit) > 0:
            behav_pred_pos[fold,:]=pos_fit[0]*pe + pos_fit[1]
        else:
            behav_pred_pos[fold,:]='nan'

        if len(neg_fit) > 0:
            behav_pred_neg[fold,:]=neg_fit[0]*ne + neg_fit[1]
        else:
            behav_pred_neg[fold,:]='nan'

    #posedge_gather=posedge_gather/k
    #negedge_gather=negedge_gather/k


    return behav_pred_pos,behav_pred_neg,behav_actual,posedge_gather,negedge_gather,pf_gather, nf_gather, rvals_gather



def run_validate(ipmats, pheno, cvtype = 'splithalf', corrtype = 'pearsonr', confound = False, pthresh = 0.01, niters = 1):
    '''
    Wrapper for k folds CV, makes it easier to just run
    split half, five fold, ten fold, LOO or integer less than num subs

    
    ipmat: features (N features x n subs)
    pheno: target variables
    cvtype: 
    corrtype: the type of correlation method to use in feature selection
    can be pearsons or partial
    confound: if corrtype is partial, confounds must be provided

    '''



    nrois,numsubs=ipmats.shape
    #ipmats=np.reshape(ipmats,[-1,numsubs])

    cvstr_dct={
    'LOO' : numsubs,
    'splithalf' : 2,
    '5k' : 5,
    '10k' : 10}


    if type(cvtype) == str:
        if cvtype not in cvstr_dct.keys():
            raise Exception('cvtype must be LOO, 5k, 10k, or splithalf (case sensitive)')
        else:
            knum=cvstr_dct[cvtype]
    elif type(cvtype) == int:
        knum=cvtype

    else:
        raise Exception('cvtype must be an int, representing number of folds, or a string descibing CV type')




    #Rvals=np.zeros((niters,1))

    pe_gather=[]
    ne_gather=[]
    bp_gather=[]
    bn_gather=[]
    ba_gather=[]
    pf_gather=[]
    nf_gather=[]
    pe_gather_save=[]
    featRvalGather=[]


    for i in range(0,niters):
        print("Running iteration:",i+1,' out of ', niters, 'iterations')

        bp,bn,ba,pe,ne,pf,nf,rvalArr = kfold_cpm(ipmats, pheno, k = knum, corrtype = corrtype, confound = confound, pthresh = pthresh)

        bp_res=np.reshape(bp,numsubs)
        bn_res=np.reshape(bn,numsubs)
        ba_res=np.reshape(ba,numsubs)

        #Rpos=stats.pearsonr(bp_res,ba_res)[0]
        #Rneg=stats.pearsonr(bn_res,ba_res)[0]



        #Rvals[i]=Rp
        pe_gather.append(pe)
        ne_gather.append(ne)
        bp_gather.append(bp)
        bn_gather.append(bn)
        ba_gather.append(ba)
        pf_gather.append(pf)
        nf_gather.append(nf)
        featRvalGather.append(rvalArr)


    #pe_gather = pe_gather/niters
    #ne_gather = ne_gather/niters
    bp_gather=np.stack(bp_gather)
    bn_gather=np.stack(bn_gather)
    ba_gather=np.stack(ba_gather)


    return pe_gather,ne_gather,bp_gather,bn_gather,ba_gather,pf_gather,nf_gather,featRvalGather


def resampleCPM(ipmats, pheno, resampleSize, corrtype = 'pearsonr', confound = False, replacement = False, iters = 1, pthresh = 0.01):
    '''
    Fit CPM on subsample/upsample/bootstrapped version of data

    
    ipmat: feature matrix (N features x n subs)
    pheno: target variables

    corrtype: the type of correlation method to use in feature selection
    can be pearsons or partial
    confound: if corrtype is partial, confounds must be provided

    '''

    nedges, numsubs = ipmats.shape
    sampleInds=np.arange(0,numsubs)
    

    posedge_gather=[]#np.zeros(nedges)
    negedge_gather=[]#np.zeros(nedges)
    pf_gather=[]
    nf_gather=[]
    randindsGather = []
    behav_actual=np.zeros([iters,resampleSize])
    rvals_gather = np.zeros([iters,nedges])

    for iter1 in range(0,iters):
        print("Running iteration:",iter1+1,' out of ', iters, 'iterations')

        resampleInds = np.random.choice(sampleInds,size=resampleSize,replace=replacement) 
        trainmats=ipmats[:,resampleInds]
        trainpheno=pheno[resampleInds]
 

        if corrtype == 'partial':
            trainconf=confound[resampleInds]
        else:
            trainconf=confound

        behav_actual[iter1,:]=trainpheno

        pos_fit,neg_fit,posedges,negedges,Rvals = train_cpm(trainmats, trainpheno, corrtype = corrtype, confound = trainconf, pthresh = pthresh)

        posedge_gather.append(posedges.astype(bool))#+posedges.flatten()
        negedge_gather.append(negedges.astype(bool))#+negedges.flatten()
        pf_gather.append(pos_fit)
        nf_gather.append(neg_fit)
        randindsGather.append(resampleInds)

        rvals_gather[iter1,:] = Rvals.astype(np.float16)


    return behav_actual,posedge_gather,negedge_gather,pf_gather, nf_gather, rvals_gather,randindsGather





def run_cpm(argDict = None):

    '''
    Interface for multiprocessing run
    Can iteratively run cpm with resampling/CV many times and aggregates outputs
    Accepts input dictionary with all parameters
    '''

    #niters=10
    #ipmats,pmats,tp,readfile,subs_to_run,tpmask,sublist,corrtype,confound=args


    #if argDict is not None and type(argDict) == dict:
    #    niters = argDict['niters']
    #    ipmats = argDict['ipmats']
    #    pmats = argDict['pmats']
    #    tp = argDict['tp']
    #    readfile = argDict['readfile']
    #    subs_to_run = argDict['subs_to_run']
    #    tpmask = argDict['tpmask']
    #    sublist = argDict['sublist']
    #    corrtype = argDict['corrtype']
    #    confound = argDict['confound']
    #    method = argDict['method']
    #    pthresh = argDict['pthresh']

     


    # Default args for function
    #argNamesDefaultVals = {'ipmat':None,
    #                        'pheno':None,
    #                        'corrtype':'pearsonr',
    #                        'confound':False,
    #                        'pthresh':0.01,
    #                        'replacement' : False,
    #                        'resampleSize' : None,
    #                        'niters' : None,
    #                        'tp' : 0,
    #                        'method' : None,
    #                        'sublist' : None,
    #                        'tpmask' : False,
    #                        'readfile' : False}

    # Check if vars have class definition
    #argNameDict={}
    #for argName,defaultVal in argNamesDefaultVals.items():
    #    argNameDict[argName] = {}

    #    try:
    #        exec('argNameDict["'+argName+'"]["val"] = '+argName)
    #    except NameError:
    #        exec('argNameDict["'+argName+'"]["val"] = None')
    #    argNameDict[argName]['default'] = defaultVal

    #varDict = self.checkVars(argNameDict)
    varDict = argDict

    niters = varDict['niters']
    ipmats = varDict['ipmats']
    pmats = varDict['pmats']
    tp = varDict['tp']
    readfile = varDict['readfile']
    subs_to_run = varDict['subs_to_run']
    tpmask = varDict['tpmask']
    sublist = varDict['sublist']
    corrtype = varDict['corrtype']
    confound = varDict['confound']
    method = varDict['method']
    pthresh = varDict['pthresh']

    if method == 'crossVal' and 'cvType' not in argDict.keys():
        raise Exception('If you want to run cross validation cvType must exist in input dictionary and be set to one of LOO, 5k, 10k, or splithalf (case sensitive)')
    elif  method == 'crossVal' and 'cvType' in argDict.keys():
        cvType = argDict['cvType']



    print('(Applies to dynamics analysis) timepoint: ',tp)
    

    if readfile == True:
        ipmats=pickle.load(open(ipmats,'rb'))

    ### For now we have reshaping, but should enforce a features x datapoint structure
    if len(ipmats.shape) == 3:
        ipmats=np.transpose(ipmats,[1,2,0])
        if ipmats.shape[0] != ipmats.shape[1]:
            raise Exception('This is a 3D array but the dimensions typically designated ROIs are not equal')
        nrois=ipmats.shape[0]**2
        numsubs=ipmats.shape[2]
        ipmats=ipmats.reshape(nrois,numsubs)
        
    elif len(ipmats.shape) == 2:
        ipmats=np.transpose(ipmats,[1,0])
        nrois=ipmats.shape[0]
        numsubs=ipmats.shape[1]
    else:
        raise Exception('Input matrix should be 2 or 3 Dimensional (Nsubs x Nfeatures) or (Nsubs x Nrois x Nrois)')
        

    ## Take lower triangle, should do this in a better way

    triangMask = np.triu(np.ones([268,268]),k=1).reshape(-1).astype(bool)
    ipmats = ipmats[triangMask,:]



    if type(tpmask) == np.ndarray and tpmask.dtype == bool:

        ipmats=ipmats[:,tpmask]
        pmats=pmats[tpmask]
        if corrtype == 'partial':
            condfound=confound[tpmask]

        if ipmats.shape[1] < subs_to_run:
            Warning('Not enough subs, returning empty dict')
            return {}
        if pmats.shape[0] < subs_to_run:
            raise Exception('Not enough dependent variables')

        ipmats=ipmats[:,:subs_to_run]
        pmats=pmats[:subs_to_run]
        if corrtype == 'partial':
            confound=confound[:subs_to_run]

        numsubs=subs_to_run

    elif type(tpmask) == bool and tpmask == False:

        pass
    else:
        raise Exception('Datatype of mask not recognized, must be a boolean ndarray or boolean of value "False"')
        



    #ipmats=np.arctanh(ipmats)
    #ipmats[ipmats == np.inf] = np.arctanh(0.999999)

    if method == 'crossVal':
        #if 'nfolds' in argDict.keys():
        #    k = argDict['nfolds']
        #else:
        #    raise Exception('if method is crossVal, must specify number of folds')

        Rvals=np.zeros((niters,1))
        randinds=np.arange(0,numsubs)


        #pe_gather=np.zeros(nrois)
        #ne_gather=np.zeros(nrois)
        #bp_gather=[]
        #bn_gather=[]
        #ba_gather=[]
        #randinds_gather=[]
        #pf_gather=[]
        #nf_gather=[]
        #pe_gather_save=[]
        #featRvalGather=[]


        #for i in range(0,niters):
        #    print('iter: ',i)



        #    random.shuffle(randinds)
        #    randinds_torun=randinds[:subs_to_run]
            #randinds_to_run=randinds  

        #    ipmats_rand=ipmats[:,randinds_torun]
        #    pmats_rand=pmats[randinds_torun]

            # run_validate(ipmats, pheno, cvtype = 'splithalf', corrtype = 'pearsonr', confound = False, pthresh = 0.01, niters = 1)
        pe_gather,ne_gather,bp_gather,bn_gather,ba_gather,pf_gather,nf_gather,featureRvalGather= run_validate(ipmats = ipmats,pheno = pmats,cvtype = cvType,corrtype = corrtype,confound = confound,pthresh = pthresh, niters = niters)

        #    if i < 5:
        #        pe_gather_save.append(pe)
        #    pe_gather=pe_gather+pe
        #    ne_gather=ne_gather+ne
        #    bp_gather.append(bp)
        #    bn_gather.append(bn)
        #    ba_gather.append(ba)
        #    randinds_gather.append(randinds_torun)
        #    pf_gather.append(pf)
        #    nf_gather.append(nf)
        #    featRvalGather.append(featureRval)

        #pe_gather=pe_gather
        #ne_gather=ne_gather
        #bp_gather=np.stack(bp_gather)
        #bn_gather=np.stack(bn_gather)
        #ba_gather=np.stack(ba_gather)
        #randinds_gather=np.stack(randinds_gather)

        opdict={}
        opdict['tp']=tp
        #opdict['rvals']=Rvals
        opdict['posedges']=pe_gather
        #opdict['posedgesIndv']=pe_gather_save
        opdict['negedges']=ne_gather
        opdict['posbehav']=bp_gather
        opdict['negbehav']=bn_gather
        opdict['actbehav']=ba_gather
        #opdict['randinds']=randinds_gather
        opdict['posfits']=pf_gather
        opdict['negfits']=nf_gather
        opdict['sublist']=sublist
        opdict['triangmask']=triangMask
        
        #opdict['featureRvals'] = featureRvalGather
        opdict[method] = method




    elif method == 'resample':
        if 'replacement' in argDict.keys():
            resampleReplace = argDict['replacement']
        else:
            raise Exception('if method is resample, must specify "replacement" as True or False') 
        if 'resampleSize' in argDict.keys():
            resampleReplace = argDict['resampleSize']
        else:
            raise Exception('if method is resample, must specify "resampleSize"')

        
        
        behav_actual,posedge_gather,negedge_gather,pf_gather, nf_gather, rvals_gather,randindsGather = resampleCPM(ipmats = ipmat,pheno = pmat,corrtype = corrtype,confound = confound,replacement = replacement,resampleSize = resampleSize,iters = iters)


        opdict={}
        opdict['tp']=tp
        opdict['posedges']=posedge_gather
        opdict['negedges']=negedge_gather
        opdict['actbehav']=behav_actual
        opdict['randinds']=randindsGather
        opdict['posfits']=pf_gather
        opdict['negfits']=nf_gather
        opdict['sublist']=sublist
        opdict['triangmask']=triangMask
        opdict['featureRvals'] = featRvalGather
        opdict[method] = method



    else:
        raise Exception('method must be "crossVal" or "resample"')





    if type(tpmask) == np.ndarray and tpmask.dtype == bool:
        opdict['tpmask']=tpmask


    return opdict


def apply_cpm(ipmats,pmats,edges,model,tpmask,readfile,subs_to_run,perf = 'corr', randomize = False):

    '''
    Accepts input matrices, edges and model
    Returns predicted behavior
    '''    

    if readfile == True:
        ipmats=pickle.load(open(ipmats,'rb'))
        if len(ipmats.shape) == 3:
            ipmats=np.transpose(ipmats,[1,2,0])
            if ipmats.shape[0] != ipmats.shape[1]:
                raise Exception('This is a 3D array but the dimensions typically designated ROIs are not equal')
            nrois=ipmats.shape[0]**2
            numsubs=ipmats.shape[2]
            ipmats=ipmats.reshape(nrois,numsubs)
            
        elif len(ipmats.shape) == 2:
            ipmats=np.transpose(ipmats,[1,0])
            nrois=ipmats.shape[0]
            numsubs=ipmats.shape[1]
        else:
            raise Exception('Input matrix should be 2 or 3 Dimensional (Nsubs x Nfeatures) or (Nsubs x Nrois x Nrois)')

    else:
        if len(ipmats.shape) == 3:
            ipmats=np.transpose(ipmats,[1,2,0])
            if ipmats.shape[0] != ipmats.shape[1]:
                raise Exception('This is a 3D array but the dimensions typically designated ROIs are not equal')
            nrois=ipmats.shape[0]**2
            numsubs=ipmats.shape[2]
            ipmats=ipmats.reshape(nrois,numsubs)
            
        elif len(ipmats.shape) == 2:
            ipmats=np.transpose(ipmats,[1,0])
            nrois=ipmats.shape[0]
            numsubs=ipmats.shape[1]
        else:
            raise Exception('Input matrix should be 2 or 3 Dimensional (Nsubs x Nfeatures) or (Nsubs x Nrois x Nrois)')





    if randomize:
        randinds=np.arange(0,numsubs)
        random.shuffle(randinds)
        ipmats=ipmats[:,randinds]
        pmats=pmats[randinds]
        tpmask = tpmask[randinds]


        #while (~np.isnan(pmats[:subs_to_run])).sum() < subs_to_run:

        #    random.shuffle(randinds)
        #    ipmats=ipmats[:,randinds]
        #    pmats=pmats[randinds]
        #    tpmask = tpmask[randinds]

     

    if type(tpmask) == np.ndarray and tpmask.dtype == bool:

        ipmats=ipmats[:,tpmask]
        pmats=pmats[tpmask]

        if ipmats.shape[1] < subs_to_run:
            raise Exception('Not enough subs')
        if pmats.shape[0] < subs_to_run:
            raise Exception('Not enough dependent variables')

        ipmats=ipmats[:,:subs_to_run]
        pmats=pmats[:subs_to_run]

        numsubs=subs_to_run

    elif type(tpmask) == bool and tpmask == False:
        pass
    else:
        raise Exception('Datatype of mask not recognized, must be a boolean ndarray or boolean of value "False"')
        



    
    edgesum=np.sum(ipmats[edges.flatten().astype(bool),:], axis=0)/2   
    behav_pred=model[0]*edgesum + model[1]



    predscore=calcPerf(behav_pred,pmats,method=perf)#np.corrcoef(behav_pred,pmats)[0,1]


    return predscore




def apply_cpm_para(argDict=None):

    '''
    Accepts input matrices, edges and model
    Returns predicted behavior
    '''    

    #ipmats,pmats,edges,model,tpmask,readfile,subs_to_run,perf = 'corr', randomize = False
    perf = 'corr'
    ipmats = argDict['ipmats']
    pmats = argDict['pmats']
    edges = argDict['modelEdges']
    model = argDict['modelFit']
    tpmask = argDict['mask']
    readfile = argDict['readfile']
    subs_to_run = argDict['subsToRun']
    randomize = argDict['randomize']

    if 'tag' in argDict.keys():
        print(argDict['tag'])

    if readfile == True:
        ipmats=pickle.load(open(ipmats,'rb'))
        if len(ipmats.shape) == 3:
            ipmats=np.transpose(ipmats,[1,2,0])
            if ipmats.shape[0] != ipmats.shape[1]:
                raise Exception('This is a 3D array but the dimensions typically designated ROIs are not equal')
            nrois=ipmats.shape[0]**2
            numsubs=ipmats.shape[2]
            ipmats=ipmats.reshape(nrois,numsubs)
            
        elif len(ipmats.shape) == 2:
            ipmats=np.transpose(ipmats,[1,0])
            nrois=ipmats.shape[0]
            numsubs=ipmats.shape[1]
        else:
            raise Exception('Input matrix should be 2 or 3 Dimensional (Nsubs x Nfeatures) or (Nsubs x Nrois x Nrois)')

    else:
        if len(ipmats.shape) == 3:
            ipmats=np.transpose(ipmats,[1,2,0])
            if ipmats.shape[0] != ipmats.shape[1]:
                raise Exception('This is a 3D array but the dimensions typically designated ROIs are not equal')
            nrois=ipmats.shape[0]**2
            numsubs=ipmats.shape[2]
            ipmats=ipmats.reshape(nrois,numsubs)
            
        elif len(ipmats.shape) == 2:
            ipmats=np.transpose(ipmats,[1,0])
            nrois=ipmats.shape[0]
            numsubs=ipmats.shape[1]
        else:
            raise Exception('Input matrix should be 2 or 3 Dimensional (Nsubs x Nfeatures) or (Nsubs x Nrois x Nrois)')


    ## Take lower triangle, should do this in a better way

    triangMask = np.triu(np.ones([268,268]),k=1).reshape(-1).astype(bool)
    ipmats = ipmats[triangMask,:]


    if randomize:
        randinds=np.arange(0,numsubs)
        random.shuffle(randinds)
        ipmats=ipmats[:,randinds]
        pmats=pmats[randinds]
        tpmask = tpmask[randinds]


        #while (~np.isnan(pmats[:subs_to_run])).sum() < subs_to_run:

        #    random.shuffle(randinds)
        #    ipmats=ipmats[:,randinds]
        #    pmats=pmats[randinds]
        #    tpmask = tpmask[randinds]

     

    if type(tpmask) == np.ndarray and tpmask.dtype == bool:

        ipmats=ipmats[:,tpmask]
        pmats=pmats[tpmask]

        if ipmats.shape[1] < subs_to_run:
            raise Exception('Not enough subs')
        if pmats.shape[0] < subs_to_run:
            raise Exception('Not enough dependent variables')

        ipmats=ipmats[:,:subs_to_run]
        pmats=pmats[:subs_to_run]

        numsubs=subs_to_run

    elif type(tpmask) == bool and tpmask == False:
        pass
    else:
        raise Exception('Datatype of mask not recognized, must be a boolean ndarray or boolean of value "False"')
        



    
    edgesum=np.sum(ipmats[edges.flatten().astype(bool),:], axis=0)/2   
    behav_pred=model[0]*edgesum + model[1]



    predscore=calcPerf(behav_pred,pmats,method=perf)#np.corrcoef(behav_pred,pmats)[0,1]

    if 'tag' in argDict.keys():
        return {argDict['tag']:predscore}
    else:
        return predscore


def calcPerf(pred,actu,method = 'corr'):

    if method == 'corr':
        perf = np.corrcoef(pred,actu)[0,1]
    elif method == 'expVar1':
        perf = np.corrcoef(pred,actu)[0,1]**2
    elif method == 'expVar2':
        resid = actu-pred
        perf = 1 - (resid.var()/actu.var())
    elif method == 'MSE':
        perf = np.mean((pred-actu)**2)
    elif method == 'MAE':
        perf = np.mean(np.abs(pred-actu))

    return perf


"""
class CPM:

    def __init__(self,argDict = {}):

        for varName, val in argDict.items():
            exec('self.'+varName+' = val')

        #self.niters = argDict['niters']
        #self.ipmats = argDict['ipmats']
        #self.pmats = argDict['pmats']
        #self.tp = argDict['tp']
        #self.readfile = argDict['readfile']
        #self.subs_to_run = argDict['subs_to_run']
        #self.tpmask = argDict['tpmask']
        #self.sublist = argDict['sublist']
        #self.corrtype = argDict['corrtype']
        #self.confound = argDict['confound']
        #self.method = argDict['method']


    def checkVars(self,argNamesDefaultVals):
        '''
        Compare Class attributes to a dictionary of default
        values, and fill in those that dont exist or set to
        "None"
        '''

        opDict={}
        for argName,subDict in argNamesDefaultVals.items():
            
            if subDict['val'] is None:
                try:
                    exec('opDict["'+argName+'"] = self.'+argName)
                except AttributeError:
                    opDict[argName] = subDict['default']
                    
                if opDict[argName] is None:
                    raise Exception(argName+' was not specified in function and is not a class attribute')
            else:
                opDict[argName] = argNamesDefaultVals[argName]['val']

        return opDict


    def train_cpm_run(self,ipmats=None,pheno=None,pthresh=None, corrtype = None, confound = None):

        '''
        Class wrapper for function "train_cpm"

        Accepts input matrices (N features x Nsubs) and pheno data
        Returns model

        ipmat: features
        pheno: target variables
        pthresh: the p threshold to use in feature selection
        corrtype: the type of correlation method to use in feature selection
        can be pearsons or partial
        confound: if corrtype is partial, confounds must be provided
        '''

        # Default args for function
        argNamesDefaultVals = {'ipmats':None,
                                'pheno':None,
                                'pthresh':0.01,
                                'corrtype':'pearsonsr',
                                'confound':False}

        # Check if vars have class definition
        argNameDict={}
        for argName,defaultVal in argNamesDefaultVals.items():
            argNameDict[argName] = {}

            exec('argNameDict["'+argName+'"]["val"] = '+argName)
            argNameDict[argName]['default'] = defaultVal

        varDict = self.checkVars(argNameDict)


        ipmats = varDict['ipmats']
        pheno = varDict['pheno']
        pthresh = varDict['pthresh']
        corrtype = varDict['corrtype']
        confound = varDict['confound']


        
        fit_pos,fit_neg,posedges,negedges,Rvals = train_cpm(ipmats, pheno, pthresh = pthresh, corrtype = corrtype, confound = confound)



        return fit_pos,fit_neg,posedges,negedges,Rvals


    def kfold_cpm_run(self, ipmats = None,pheno = None,numsubs = None,k = None,corrtype = None,confound = None,pthresh=None):
        '''
        Class wrapper for function "kfold_cpm"

        
        ipmat: feature matrix (N features x n subs)
        pheno: target variables
        numsubs: number of participants/datapoints
        k: number of folds
        corrtype: the type of correlation method to use in feature selection
        can be pearsons or partial
        confound: if corrtype is partial, confounds must be provided

        '''

        # Default args for function
        argNamesDefaultVals = {'ipmats':None,
                                'pheno':None,
                                'numsubs':None,
                                'k':None,
                                'corrtype':'pearsonr',
                                'confound':False,
                                'pthresh':0.01}

        # Check if vars have class definition
        argNameDict={}
        for argName,defaultVal in argNamesDefaultVals.items():
            argNameDict[argName] = {}

            exec('argNameDict["'+argName+'"]["val"] = '+argName)
            argNameDict[argName]['default'] = defaultVal

        varDict = self.checkVars(argNameDict)


        ipmats = varDict['ipmats']
        pheno = varDict['pheno']
        numsubs = varDict['numsubs']
        k = varDict['k']
        corrtype = varDict['corrtype']
        confound = varDict['confound']
        pthresh = varDict['pthresh']


        behav_pred_pos,behav_pred_neg,behav_actual,posedge_gather,negedge_gather,pf_gather, nf_gather, rvals_gather = kfold_cpm(ipmats, pheno, k = k, corrtype = corrtype, confound = confound, pthresh = pthresh)


        return behav_pred_pos,behav_pred_neg,behav_actual,posedge_gather,negedge_gather,pf_gather, nf_gather, rvals_gather


    def run_validate_run(self, ipmats = None, pheno = None, cvtype = None, corrtype = None, confound = None, pthresh = None):
        '''
        Class wrapper for the function "run_validate"
        split half, five fold, ten fold or LOO

        
        ipmat: features (N features x n subs)
        pheno: target variables
        cvtype: 
        corrtype: the type of correlation method to use in feature selection
        can be pearsons or partial
        confound: if corrtype is partial, confounds must be provided

        '''

        # Default args for function
        argNamesDefaultVals = {'ipmats':None,
                                'pheno':None,
                                'cvtype':'split-half',
                                'corrtype':'pearsonr',
                                'confound':False,
                                'pthresh':0.01}

        # Check if vars have class definition
        argNameDict={}
        for argName,defaultVal in argNamesDefaultVals.items():
            argNameDict[argName] = {}

            exec('argNameDict["'+argName+'"]["val"] = '+argName)
            argNameDict[argName]['default'] = defaultVal

        varDict = self.checkVars(argNameDict)


        ipmats = varDict['ipmats']
        pheno = varDict['pheno']
        cvtype = varDict['cvtype']
        corrtype = varDict['corrtype']
        confound = varDict['confound']
        pthresh = varDict['pthresh']


        Rpos,Rneg,pe,ne,bp_res,bn_res,ba_res,pf,nf,rvalArr = run_validate(ipmats, pheno, cvtype = cvtype, corrtype = corrtype, confound = confound, pthresh = pthresh)



        return Rpos,Rneg,pe,ne,bp_res,bn_res,ba_res,pf,nf,rvalArr


    def resampleCPM_run(self, ipmats = None, pheno = None, corrtype = None, confound = None, replacement = None, resampleSize = None, iters = None, pthresh=None):
        '''
        Fit CPM on subsample/upsample/bootstrapped version of data

        
        ipmat: feature matrix (N features x n subs)
        pheno: target variables

        corrtype: the type of correlation method to use in feature selection
        can be pearsons or partial
        confound: if corrtype is partial, confounds must be provided

        '''

        # Default args for function
        argNamesDefaultVals = {'ipmats':None,
                                'pheno':None,
                                'corrtype':'pearsonr',
                                'confound':False,
                                'pthresh':0.01,
                                'replacement' : False,
                                'resampleSize' : None,
                                'iters' : None}

        # Check if vars have class definition
        argNameDict={}
        for argName,defaultVal in argNamesDefaultVals.items():
            argNameDict[argName] = {}

            exec('argNameDict["'+argName+'"]["val"] = '+argName)
            argNameDict[argName]['default'] = defaultVal

        varDict = self.checkVars(argNameDict)


        ipmats = varDict['ipmats']
        pheno = varDict['pheno']

        corrtype = varDict['corrtype']
        confound = varDict['confound']
        pthresh = varDict['pthresh']
        replacement = varDict['replacement']
        resampleSize = varDict['resampleSize']
        iters = varDict['iters']

        behav_actual, posedge_gather, negedge_gather, pf_gather, nf_gather, rvals_gather, randindsGather = resampleCPM(ipmats, pheno, resampleSize, corrtype = corrtype, confound = confound, replacement = replacement, iters = iters, pthresh = pthresh)

        return behav_actual, posedge_gather, negedge_gather, pf_gather, nf_gather, rvals_gather, randindsGather




    def run_cpm(self, argDict = None):

        '''
        Interface for multiprocessing run
        Can iteratively run cpm with resampling/CV many times and aggregates outputs
        Accepts input dictionary with all parameters
        '''


        if argDict is not None and type(argDict) == dict:
            niters = argDict['niters']
            ipmats = argDict['ipmats']
            pmats = argDict['pmats']
            tp = argDict['tp']
            readfile = argDict['readfile']
            subs_to_run = argDict['subs_to_run']
            tpmask = argDict['tpmask']
            sublist = argDict['sublist']
            corrtype = argDict['corrtype']
            confound = argDict['confound']
            method = argDict['method']
            pthresh = argDict['pthresh']


        # Default args for function
        argNamesDefaultVals = {'ipmat':None,
                                'pheno':None,
                                'corrtype':'pearsonr',
                                'confound':False,
                                'pthresh':0.01,
                                'replacement' : False,
                                'resampleSize' : None,
                                'niters' : None,
                                'tp' : 0,
                                'method' : None,
                                'sublist' : None,
                                'tpmask' : False,
                                'readfile' : False}

        # Check if vars have class definition
        argNameDict={}
        for argName,defaultVal in argNamesDefaultVals.items():
            argNameDict[argName] = {}

            try:
                exec('argNameDict["'+argName+'"]["val"] = '+argName)
            except NameError:
                exec('argNameDict["'+argName+'"]["val"] = Nonw')
            argNameDict[argName]['default'] = defaultVal

        varDict = self.checkVars(argNameDict)


        niters = varDict['niters']
        ipmats = varDict['ipmats']
        pmats = varDict['pmats']
        tp = varDict['tp']
        readfile = varDict['readfile']
        subs_to_run = varDict['subs_to_run']
        tpmask = varDict['tpmask']
        sublist = varDict['sublist']
        corrtype = varDict['corrtype']
        confound = varDict['confound']
        method = varDict['method']
        pthresh = varDict['pthresh']



        print('(Applies to dynamics analysis) timepoint: ',tp)
        

        if readfile == True:
            ipmats=pickle.load(open(ipmats,'rb'))

        ### For now we have reshaping, but should enforce a features x datapoint structure
        if len(ipmats.shape) == 3:
            ipmats=np.transpose(ipmats,[1,2,0])
            if ipmats.shape[0] != ipmats.shape[1]:
                raise Exception('This is a 3D array but the dimensions typically designated ROIs are not equal')
            nrois=ipmats.shape[0]**2
            numsubs=ipmats.shape[2]
            ipmats=ipmats.reshape(nrois,numsubs)
            
        elif len(ipmats.shape) == 2:
            ipmats=np.transpose(ipmats,[1,0])
            nrois=ipmats.shape[0]
            numsubs=ipmats.shape[1]
        else:
            raise Exception('Input matrix should be 2 or 3 Dimensional (Nsubs x Nfeatures) or (Nsubs x Nrois x Nrois)')
            

        if type(tpmask) == np.ndarray and tpmask.dtype == bool:

            ipmats=ipmats[:,tpmask]
            pmats=pmats[tpmask]
            if corrtype == 'partial':
                condfound=confound[tpmask]

            if ipmats.shape[1] < subs_to_run:
                raise Exception('Not enough subs')
            if pmats.shape[0] < subs_to_run:
                raise Exception('Not enough dependent variables')

            ipmats=ipmats[:,:subs_to_run]
            pmats=pmats[:subs_to_run]
            if corrtype == 'partial':
                confound=confound[:subs_to_run]

            numsubs=subs_to_run

        elif type(tpmask) == bool and tpmask == False:

            pass
        else:
            raise Exception('Datatype of mask not recognized, must be a boolean ndarray or boolean of value "False"')
            



        #ipmats=np.arctanh(ipmats)
        #ipmats[ipmats == np.inf] = np.arctanh(0.999999)

        if method == 'crossVal':
            if 'nfolds' in argDict.keys():
                k = argDict['nfolds']
            else:
                raise Exception('if method is crossVal, must specify number of folds')
            Rvals=np.zeros((niters,1))
            randinds=np.arange(0,numsubs)


            pe_gather=np.zeros(nrois)
            ne_gather=np.zeros(nrois)
            bp_gather=[]
            bn_gather=[]
            ba_gather=[]
            randinds_gather=[]
            pf_gather=[]
            nf_gather=[]
            pe_gather_save=[]
            featRvalGather=[]


            for i in range(0,niters):
                print('iter: ',i)



                random.shuffle(randinds)
                randinds_torun=randinds[:subs_to_run]
                #randinds_to_run=randinds  

                ipmats_rand=ipmats[:,randinds_torun]
                pmats_rand=pmats[randinds_torun]


                Rp,Rn,pe,ne,bp,bn,ba,pf,nf,featureRval=self.run_validate(ipmats = ipmats_rand,pheno = pmats_rand,cvtype = 'splithalf',corrtype = corrtype,confound = confound,pthresh = pthresh)
                Rvals[i]=Rp
                if i < 5:
                    pe_gather_save.append(pe)
                pe_gather=pe_gather+pe
                ne_gather=ne_gather+ne
                bp_gather.append(bp)
                bn_gather.append(bn)
                ba_gather.append(ba)
                randinds_gather.append(randinds_torun)
                pf_gather.append(pf)
                nf_gather.append(nf)
                featRvalGather.append(featureRval)

            pe_gather=pe_gather
            ne_gather=ne_gather
            bp_gather=np.stack(bp_gather)
            bn_gather=np.stack(bn_gather)
            ba_gather=np.stack(ba_gather)
            randinds_gather=np.stack(randinds_gather)

            opdict={}
            opdict['tp']=tp
            opdict['rvals']=Rvals
            opdict['posedges']=pe_gather
            opdict['posedgesIndv']=pe_gather_save
            opdict['negedges']=ne_gather
            opdict['posbehav']=bp_gather
            opdict['negbehav']=bn_gather
            opdict['actbehav']=ba_gather
            opdict['randinds']=randinds_gather
            opdict['posfits']=pf_gather
            opdict['negfits']=nf_gather
            opdict['sublist']=sublist
            opdict['featureRvals'] = featRvalGather
            opdict[method] = method




        elif method == 'resample':
            if 'replacement' in argDict.keys():
                resampleReplace = argDict['replacement']
            else:
                raise Exception('if method is resample, must specify "replacement" as True or False') 
            if 'resampleSize' in argDict.keys():
                resampleReplace = argDict['resampleSize']
            else:
                raise Exception('if method is resample, must specify "resampleSize"')

            
            
            behav_actual,posedge_gather,negedge_gather,pf_gather, nf_gather, rvals_gather,randindsGather = self.resampleCPM(ipmats = ipmat,pheno = pmat,corrtype = corrtype,confound = confound,replacement = replacement,resampleSize = resampleSize,iters = iters)


            opdict={}
            opdict['tp']=tp
            opdict['posedges']=posedge_gather
            opdict['negedges']=negedge_gather
            opdict['actbehav']=behav_actual
            opdict['randinds']=randindsGather
            opdict['posfits']=pf_gather
            opdict['negfits']=nf_gather
            opdict['sublist']=sublist
            opdict['featureRvals'] = featRvalGather
            opdict[method] = method



        else:
            raise Exception('method must be "crossVal" or "resample"')





        if type(tpmask) == np.ndarray and tpmask.dtype == bool:
            opdict['tpmask']=tpmask


        return opdict


    def apply_cpm(self,ipmats,pmats,edges,model,tpmask,readfile,subs_to_run):

        '''
        Accepts input matrices, edges and model
        Returns predicted behavior
        '''    

        if readfile == True:
            ipmats=pickle.load(open(ipmats,'rb'))
            if len(ipmats.shape) == 3:
                ipmats=np.transpose(ipmats,[1,2,0])
                if ipmats.shape[0] != ipmats.shape[1]:
                    raise Exception('This is a 3D array but the dimensions typically designated ROIs are not equal')
                nrois=ipmats.shape[0]**2
                numsubs=ipmats.shape[2]
                ipmats=ipmats.reshape(nrois,numsubs)
                
            elif len(ipmats.shape) == 2:
                ipmats=np.transpose(ipmats,[1,0])
                nrois=ipmats.shape[0]
                numsubs=ipmats.shape[1]
            else:
                raise Exception('Input matrix should be 2 or 3 Dimensional (Nsubs x Nfeatures) or (Nsubs x Nrois x Nrois)')

        else:
            if len(ipmats.shape) == 3:
                ipmats=np.transpose(ipmats,[1,2,0])
                if ipmats.shape[0] != ipmats.shape[1]:
                    raise Exception('This is a 3D array but the dimensions typically designated ROIs are not equal')
                nrois=ipmats.shape[0]**2
                numsubs=ipmats.shape[2]
                ipmats=ipmats.reshape(nrois,numsubs)
                
            elif len(ipmats.shape) == 2:
                ipmats=np.transpose(ipmats,[1,0])
                nrois=ipmats.shape[0]
                numsubs=ipmats.shape[1]
            else:
                raise Exception('Input matrix should be 2 or 3 Dimensional (Nsubs x Nfeatures) or (Nsubs x Nrois x Nrois)')

                

        if type(tpmask) == np.ndarray and tpmask.dtype == bool:

            ipmats=ipmats[:,tpmask]
            pmats=pmats[tpmask]

            if ipmats.shape[1] < subs_to_run:
                raise Exception('Not enough subs')
            if pmats.shape[0] < subs_to_run:
                raise Exception('Not enough dependent variables')

            ipmats=ipmats[:,:subs_to_run]
            pmats=pmats[:subs_to_run]

            numsubs=subs_to_run

        elif type(tpmask) == bool and tpmask == False:
            pass
        else:
            raise Exception('Datatype of mask not recognized, must be a boolean ndarray or boolean of value "False"')
            


        
        edgesum=np.sum(ipmats[edges.flatten().astype(bool),:], axis=0)/2   
        behav_pred=model[0]*edgesum + model[1]



        predscore=np.corrcoef(behav_pred,pmats)[0,1]


        return predscore

    def bootstrapApplyCPM(self,ipmats,pmats,edges,model,tpmask,readfile,subs_to_run,niters):
        '''
        Accepts input matrices, edges and model
        Bootstraps smaller samples from input matrices
        Returns array of predicted behavior
        '''

        ipmats=pickle.load(open(ipmats,'rb'))

        numsubs=ipmats.shape[0]

        Rvals=np.zeros((niters,1))
        randinds=np.arange(0,numsubs)

        for i in range(0,niters):
            print('iter: ',i)

            random.shuffle(randinds)
            randinds_torun=randinds[:subs_to_run]

            ipmats_rand=ipmats[randinds_torun,:,:]
            pmats_rand=pmats[randinds_torun]

            Rvals[i,:]=apply_cpm(ipmats_rand,pmats_rand,edges,model,False,False,False)

        return Rvals



###### Old stuff ###################################


def read_mats(iplist):

    '''
    Read in data from a list of tsv files
    '''

    x=[pd.read_csv(m,sep='\t',header=None) for m in iplist]
    x=[df.dropna(axis=1).values for df in x]
    ipmats=np.stack(x,axis=2)

    return ipmats


def sample_500(ipmats,pheno,cvtype):

    numsubs=ipmats.shape[2]

    randinds=np.arange(0,numsubs)
    random.shuffle(randinds)

    randinds500=randinds[:500]

    ipmats_rand=ipmats[:,:,randinds500]
    pheno_rand=pheno[randinds500]

    opdict={}



    Rpos_loo,Rneg_loo=run_validate(ipmats = ipmats_rand,pheno = pheno_rand,cvtype = 'LOO')
    
    Rpos_2k,Rneg_2k=run_validate(ipmats = ipmats_rand,pheno = pheno_rand,cvtype = 'splithalf')

    Rpos_5k,Rneg_5k=run_validate(ipmats = ipmats_rand,pheno = pheno_rand,cvtype = '5k')

    Rpos_10k,Rneg_10k=run_validate(ipmats = ipmats_rand,pheno = pheno_rand,cvtype = '10k')

    opdict['LOO_Rpos'] = Rpos_loo
    opdict['LOO_Rneg'] = Rneg_loo
    opdict['2k_Rpos'] = Rpos_2k
    opdict['2k_Rneg'] = Rneg_2k
    opdict['5k_Rpos'] = Rpos_5k
    opdict['5k_Rneg'] = Rneg_5k
    opdict['10k_Rpos'] = Rpos_10k
    opdict['10k_Rneg'] = Rneg_10k
    opdict['Sample_Indices']=randinds500

    return opdict



def testcorr():
    ipdata=io.loadmat('../../Fingerprinting/ipmats.mat')
    ipmats=ipdata['ipmats']
    pmatvals=ipdata['pmatvals'][0]
    ipmats_res=np.reshape(ipmats,[-1,843])
    pmats_rep=np.repeat(np.vstack(pmatvals),71824,axis=1)
    cc=corr2_coeff(ipmats_res,pmats_rep.T)

    return cc




def corr2_coeff(A,B):
	# from: https://stackoverflow.com/questions/30143417/computing-the-correlation-coefficient-between-two-multi-dimensional-arrays/30143754#30143754
    # https://stackoverflow.com/questions/45403071/optimized-computation-of-pairwise-correlations-in-python?noredirect=1&lq=1
    # Rowwise mean of input arrays & subtract from input arrays themeselves

    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1);
    ssB = (B_mB**2).sum(1);

    # Finally get corr coeff
    return np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))
"""
