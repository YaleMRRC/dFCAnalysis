# File: corr_multi.pyx


cdef extern from "math.h":
    double sqrt(double m)
    
cdef int numel
cdef int numel_m1
cdef int numcompare
cdef int n
cdef int xsize
cdef int ymean 
cdef int zmean
cdef int num_yz
cdef int den_yz
cdef int den_xz
cdef int b1_yz
cdef int b0_yz
cdef int xmean
cdef int num_xz
cdef int b1_xz
cdef int b0_xz


from numpy cimport ndarray
cimport numpy as np
cimport cython
import bottleneck as bn


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True) 
@cython.profile(True)




#def corr_multi_cy( ndarray[float32_t, ndim=1]  arr, ndarray[float32_t, ndim=2]  mat):
def corr_multi_cy(arr,mat):

    matsize=mat.shape


    if len(matsize) == 2:
        numcompare = len(mat.T)

        numel = int(len(arr))
        numel_m1 = int(numel-1)

        coll=ndarray(numcompare)

        arrdemean=arr-bn.nanmean(arr)
        arrss=sqrt(bn.ss(arrdemean))
        arrstd=bn.nanstd(arr)


        for n in range(0,numcompare):
            submat=mat[:,n]
            bdemean=submat-bn.nanmean(submat)
            
            bss=sqrt(bn.ss(bdemean))
            
            cross_mul=bn.nansum(arrdemean*bdemean)
        
            if bss == 0:
                r=0
            else:
                r=cross_mul/(arrss*bss)


            coll[n]=(r)

    elif len(matsize) == 1:
        numcompare = 1
        
        numel = int(len(arr))
        numel_m1 = int(numel-1)

        arrdemean=arr-bn.nanmean(arr)
        arrss=sqrt(bn.ss(arrdemean))
        arrstd=bn.nanstd(arr)

        bdemean=mat-bn.nanmean(mat)
        
        bss=sqrt(bn.ss(bdemean))
        
        cross_mul=bn.nansum(arrdemean*bdemean)
    
        if bss*arrss == 0:
            r=0
        else:
            r=cross_mul/(arrss*bss)

        coll=r

    return coll

def partial_corr(x,y,z):

    # Z is independent variable with shared influence on x and y

    numel = len(y)

    xsize=x.shape

    ymean = bn.nanmean(y)
    zmean = bn.nanmean(z)
    ydemean = y - ymean
    zdemean = z - zmean

    num_yz = bn.nansum(zdemean * ydemean)
    den_yz = bn.ss(zdemean)
    den_xz = den_yz

    b1_yz = num_yz/den_yz
    b0_yz = ymean - (b1_yz * zmean)

    ypred = b0_yz + b1_yz * z
    yres = y - ypred

    if len(xsize) == 2:

        numcompare = len(x.T)
        coll=ndarray(numcompare)

        for n in range(0,numcompare):
            #print(n)

            subx=x[:,n]


            xmean=bn.nanmean(subx)
            xdemean=subx-xmean

            
            num_xz = bn.nansum(zdemean * xdemean)
            
            b1_xz = num_xz/den_xz
            b0_xz = xmean - (b1_xz * zmean)

            xpred = b0_xz + b1_xz * z
            xres = subx - xpred
           
            coll[n]=corr_multi_cy(xres,yres)

    else:
        numcompare = 1
        
        xmean=bn.nanmean(x)
        xdemean=x-xmean

        
        num_xz = bn.nansum(zdemean * xdemean)
        
        b1_xz = num_xz/den_xz
        b0_xz = xmean - (b1_xz * zmean)

        xpred = b0_xz + b1_xz * z
        xres = x - xpred
       
        coll=corr_multi_cy(xres,yres)
        

    return coll


