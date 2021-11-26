'''
# Adaptive-sampling.py
# 2021 M. Matsuo

## Adaptive sampling
## Author: Mitsuaki Matsuo (Keio University)

## Authors provide no guarantees for this code.  Use as-is and for academic research use only; no commercial use allowed without permission. For citations, please use the reference below:
# Ref: M. Matsuo, T. Nakamura, M. Morimoto, K. Fukami, K. Fukagata,
#     "Supervised convolutional network for three-dimensional fluid data reconstruction from sectional flow fields with adaptive super-resolution assistance,"
#     in Review, arXiv preprint, arXiv:2103.09020, 2021
#
# The code is written for educational clarity and not for speed.
# -- version 1: Nov 25, 2021
'''

# To run this code, listed modules are required
import numpy as np
from skimage.measure import block_reduce

# Parameters
datasetSerial =1000 # the number of snapshots
comp = 3            # the number of components
alpha = 8           # the initial coarseness before adaptive sampling
alpha2 = 2          # downsampling factor of 1st threshold
alpha3 = 4          # downsampling factor of 2nd threshold
Thru = 0.1          # the 1st threshold of u component (if the standard deviation (SD) is Thru2 < SD < Thru, the flow field is pooled to 1/(alpha2^2) )
Thru2 = 0.08        # the 2nd threshold of u component (if the standard deviation (SD) is SD < Thru2, the flow field is pooled to 1/(alpha3^2) )
Thrv = 0.08         # the 1st threshold of v component
Thrv2 = 0.06        # the 2nd threshold of v component
Thrw = 0.07         # the 1st threshold of w component
Thrw2 = 0.05        # the 2nd threshold of w component

# Configurations of three-dimensional flow variables as follows:
'''
uvw3D_field: time=1000, nx =256, ny=128, nz=160, components=3
'''

# Calculate the standard deviation of the velocity within each cross-section
uvw_std = np.zeros((len(datasetSerial),int(nx/alpha),int(ny/alpha),nz,comp))
x_st = 0
x_ed = alpha
y_st = 0
y_ed = alpha

for xx in range(int(nx/alpha)):
    y_st = 0
    y_ed = alpha
    for yy in range(int(ny/alpha)):
        uvw_std[:,xx,yy,:,:] = np.std(uvw3D_field[:,x_st:x_ed,y_st:y_ed,:,:],axis=(1,2))
        y_st = y_st+alpha
        y_ed = y_st+alpha
    x_st = x_st+alpha
    x_ed = x_st+alpha
    
# Prepare the snapshot which is pooled to 1/alpha before adaptive sampling
low_output = np.zeros((len(datasetSerial),int(nx/alpha),int(ny/alpha),nz,comp))
for zz in tqdm(range(nz)):
    for m in range(comp):
        for n in range(len(datasetSerial)):
            low_output[n,:,:,zz,m] = block_reduce(uvw3D_field[n,:,:,zz,m],block_size=(r_late,r_late), func=np.mean)
            
# ========================================== Adaptive sampling ========================================== #
# === U component === #
au = np.zeros((len(datasetSerial),int(nx/alpha),int(ny/alpha),nz),dtype='float32) # Represents the geometry information
for tt in range(len(datasetSerial)):
    for zz in range(nz):
        for xx in range(int(nx/alpha-(alpha3-1))):
            for yy in range(int(ny/alpha-(alpha3-1))):
                beta2u_0 = uvw_std[tt,xx,yy,zz,0]
                beta2u_1 = uvw_std[tt,xx+1,yy,zz,0]
                beta2u_2 = uvw_std[tt,xx+2,yy,zz,0]
                beta2u_3 = uvw_std[tt,xx+3,yy,zz,0]
                beta2u_4 = uvw_std[tt,xx,yy+1,zz,0]
                beta2u_5 = uvw_std[tt,xx,yy+2,zz,0]
                beta2u_6 = uvw_std[tt,xx,yy+3,zz,0]
                beta2u_7 = uvw_std[tt,xx+1,yy+1,zz,0]
                beta2u_8 = uvw_std[tt,xx+1,yy+2,zz,0]
                beta2u_9 = uvw_std[tt,xx+1,yy+3,zz,0]
                beta2u_10= uvw_std[tt,xx+2,yy+1,zz,0]
                beta2u_11= uvw_std[tt,xx+2,yy+2,zz,0]
                beta2u_12= uvw_std[tt,xx+2,yy+3,zz,0]
                beta2u_13= uvw_std[tt,xx+3,yy+1,zz,0]
                beta2u_14= uvw_std[tt,xx+3,yy+2,zz,0]
                beta2u_15= uvw_std[tt,xx+3,yy+3,zz,0]
              
                if au[tt,xx,yy,zz]==0 and au[tt,xx+1,yy,zz]==0 and au[tt,xx+2,yy,zz]==0 and au[tt,xx+3,yy,zz]==0\
                and au[tt,xx,yy+1,zz]==0 and au[tt,xx+1,yy+1,zz]==0 and au[tt,xx+2,yy+1,zz]==0 and au[tt,xx+3,yy+1,zz]==0\
                and au[tt,xx,yy+2,zz]==0 and au[tt,xx+1,yy+2,zz]==0 and au[tt,xx+2,yy+2,zz]==0 and au[tt,xx+3,yy+2,zz]==0\
                and au[tt,xx,yy+3,zz]==0 and au[tt,xx+1,yy+3,zz]==0 and au[tt,xx+2,yy+3,zz]==0 and au[tt,xx+3,yy+3,zz]==0:
                    if beta2u_0   < Thru2 and beta2u_1  < Thru2 and beta2u_2 < Thru2 and beta2u_3  < Thru2 and beta2u_4 < Thru2\
                    and beta2u_5  < Thru2 and beta2u_6  < Thru2 and beta2u_7 < Thru2 and beta2u_8  < Thru2 and beta2u_9 < Thru2\
                    and beta2u_10 < Thru2 and beta2u_11 < Thru2 and beta2u_12< Thru2 and beta2u_13 < Thru2 and beta2u_14< Thru2\
                    and beta2u_15 < Thru2:
                        u_matrix = np.zeros((len(datasetSerial),alpha3,alpha3,z_nz))
                        low_u2 = np.zeros((len(datasetSerial),1,1,z_nz))
                        u_matrix[tt,0,0,zz] = low_output[tt,xx,yy,zz,0]
                        u_matrix[tt,1,0,zz] = low_output[tt,xx+1,yy,zz,0]
                        u_matrix[tt,2,0,zz] = low_output[tt,xx+2,yy,zz,0]
                        u_matrix[tt,3,0,zz] = low_output[tt,xx+3,yy,zz,0]
                        u_matrix[tt,0,1,zz] = low_output[tt,xx,yy+1,zz,0]
                        u_matrix[tt,0,2,zz] = low_output[tt,xx,yy+2,zz,0]
                        u_matrix[tt,0,3,zz] = low_output[tt,xx,yy+3,zz,0]
                        u_matrix[tt,1,1,zz] = low_output[tt,xx+1,yy+1,zz,0]
                        u_matrix[tt,1,2,zz] = low_output[tt,xx+1,yy+2,zz,0]
                        u_matrix[tt,1,3,zz] = low_output[tt,xx+1,yy+3,zz,0]
                        u_matrix[tt,2,1,zz] = low_output[tt,xx+2,yy+1,zz,0]
                        u_matrix[tt,2,2,zz] = low_output[tt,xx+2,yy+2,zz,0]
                        u_matrix[tt,2,3,zz] = low_output[tt,xx+2,yy+3,zz,0]
                        u_matrix[tt,3,1,zz] = low_output[tt,xx+3,yy+1,zz,0]
                        u_matrix[tt,3,2,zz] = low_output[tt,xx+3,yy+2,zz,0]
                        u_matrix[tt,3,3,zz] = low_output[tt,xx+3,yy+3,zz,0]
                        low_u2[tt,0,0,zz] = block_reduce(u_matrix[tt,:,:,zz],block_size=(alpha3,alpha3), func=np.mean)
                        low_output[tt,xx:xx+alpha3,yy:yy+alpha3,zz,0] = low_u2[tt,0,0,zz]
                        au[tt,xx,yy,zz]=2
                        au[tt,xx,yy+1,zz]=2
                        au[tt,xx,yy+2,zz]=2
                        au[tt,xx,yy+3,zz]=2
                        au[tt,xx+1,yy,zz]=2
                        au[tt,xx+1,yy+1,zz]=2
                        au[tt,xx+1,yy+2,zz]=2
                        au[tt,xx+1,yy+3,zz]=2
                        au[tt,xx+2,yy,zz]=2
                        au[tt,xx+2,yy+1,zz]=2
                        au[tt,xx+2,yy+2,zz]=2
                        au[tt,xx+2,yy+3,zz]=2
                        au[tt,xx+3,yy,zz]=2
                        au[tt,xx+3,yy+1,zz]=2
                        au[tt,xx+3,yy+2,zz]=2
                        au[tt,xx+3,yy+3,zz]=2

        for xx in range(int(nx/alpha-(alpha2-1))):
            for yy in range(int(nx/alpha-(alpha2-1))):                
                betau = uvw_std[tt,xx,yy,zz,0]
                betau1 = uvw_std[tt,xx+1,yy,zz,0]
                betau2 = uvw_std[tt,xx,yy+1,zz,0]
                betau3 = uvw_std[tt,xx+1,yy+1,zz,0]
                if au[tt,xx,yy,zz]==0 and au[tt,xx+1,yy,zz]==0 and au[tt,xx+1,yy+1,zz]==0 and au[tt,xx,yy+1,zz]==0:
                    if betau < Thru and betau1 < Thru and betau2 < Thru and betau3 < Thru:
                        u_matrix = np.zeros((len(datasetSerial),alpha2,alpha2,z_nz))
                        low_u = np.zeros((len(datasetSerial),1,1,z_nz)) 
                        u_matrix[tt,0,0,zz] = low_output[tt,xx,yy,zz,0]
                        u_matrix[tt,1,0,zz] = low_output[tt,xx+1,yy,zz,0]
                        u_matrix[tt,0,1,zz] = low_output[tt,xx,yy+1,zz,0]
                        u_matrix[tt,1,1,zz] = low_output[tt,xx+1,yy+1,zz,0]
                        low_u[tt,0,0,zz] = block_reduce(u_matrix[tt,:,:,zz],block_size=(alpha2,alpha2), func=np.mean)
                        low_output[tt,xx:xx+alpha2,yy:yy+alpha2,zz,0] = low_u[tt,0,0,zz]                                
                        au[tt,xx,yy,zz]=1
                        au[tt,xx+1,yy,zz]=1
                        au[tt,xx,yy+1,zz]=1
                        au[tt,xx+1,yy+1,zz]=1

# === V component === #
av = np.zeros((len(datasetSerial),int(nx/alpha),int(ny/alpha),nz))
for tt in tqdm(range(len(datasetSerial))):
    for zz in range(nz):
        for xx in range(int(nx/alpha-(alpha3-1))):
            for yy in range(int(ny/alpha-(alpha3-1))):
                beta2v_0 = uvw_std[tt,xx,yy,zz,1]
                beta2v_1 = uvw_std[tt,xx+1,yy,zz,1]
                beta2v_2 = uvw_std[tt,xx+2,yy,zz,1]
                beta2v_3 = uvw_std[tt,xx+3,yy,zz,1]
                beta2v_4 = uvw_std[tt,xx,yy+1,zz,1]
                beta2v_5 = uvw_std[tt,xx,yy+2,zz,1]
                beta2v_6 = uvw_std[tt,xx,yy+3,zz,1]
                beta2v_7 = uvw_std[tt,xx+1,yy+1,zz,1]
                beta2v_8 = uvw_std[tt,xx+1,yy+2,zz,1]
                beta2v_9 = uvw_std[tt,xx+1,yy+3,zz,1]
                beta2v_10= uvw_std[tt,xx+2,yy+1,zz,1]
                beta2v_11= uvw_std[tt,xx+2,yy+2,zz,1]
                beta2v_12= uvw_std[tt,xx+2,yy+3,zz,1]
                beta2v_13= uvw_std[tt,xx+3,yy+1,zz,1]
                beta2v_14= uvw_std[tt,xx+3,yy+2,zz,1]
                beta2v_15= uvw_std[tt,xx+3,yy+3,zz,1]


                if av[tt,xx,yy,zz]==0 and av[tt,xx+1,yy,zz]==0 and av[tt,xx+2,yy,zz]==0 and av[tt,xx+3,yy,zz]==0\
                and av[tt,xx,yy+1,zz]==0 and av[tt,xx+1,yy+1,zz]==0 and av[tt,xx+2,yy+1,zz]==0 and av[tt,xx+3,yy+1,zz]==0\
                and av[tt,xx,yy+2,zz]==0 and av[tt,xx+1,yy+2,zz]==0 and av[tt,xx+2,yy+2,zz]==0 and av[tt,xx+3,yy+2,zz]==0\
                and av[tt,xx,yy+3,zz]==0 and av[tt,xx+1,yy+3,zz]==0 and av[tt,xx+2,yy+3,zz]==0 and av[tt,xx+3,yy+3,zz]==0:
                    if beta2v_0   < Thrv2 and beta2v_1  < Thrv2 and beta2v_2 < Thrv2 and beta2v_3  < Thrv2 and beta2v_4 < Thrv2\
                    and beta2v_5  < Thrv2 and beta2v_6  < Thrv2 and beta2v_7 < Thrv2 and beta2v_8  < Thrv2 and beta2v_9 < Thrv2\
                    and beta2v_10 < Thrv2 and beta2v_11 < Thrv2 and beta2v_12< Thrv2 and beta2v_13 < Thrv2 and beta2v_14< Thrv2\
                    and beta2v_15 < Thrv2:

                        v_matrix = np.zeros((len(datasetSerial),alpha3,alpha3,nz))
                        low_v2 = np.zeros((len(datasetSerial),1,1,nz))
                        v_matrix[tt,0,0,zz] = low_output[tt,xx,yy,zz,1]
                        v_matrix[tt,1,0,zz] = low_output[tt,xx+1,yy,zz,1]
                        v_matrix[tt,2,0,zz] = low_output[tt,xx+2,yy,zz,1]
                        v_matrix[tt,3,0,zz] = low_output[tt,xx+3,yy,zz,1]
                        v_matrix[tt,0,1,zz] = low_output[tt,xx,yy+1,zz,1]
                        v_matrix[tt,0,2,zz] = low_output[tt,xx,yy+2,zz,1]
                        v_matrix[tt,0,3,zz] = low_output[tt,xx,yy+3,zz,1]
                        v_matrix[tt,1,1,zz] = low_output[tt,xx+1,yy+1,zz,1]
                        v_matrix[tt,1,2,zz] = low_output[tt,xx+1,yy+2,zz,1]
                        v_matrix[tt,1,3,zz] = low_output[tt,xx+1,yy+3,zz,1]
                        v_matrix[tt,2,1,zz] = low_output[tt,xx+2,yy+1,zz,1]
                        v_matrix[tt,2,2,zz] = low_output[tt,xx+2,yy+2,zz,1]
                        v_matrix[tt,2,3,zz] = low_output[tt,xx+2,yy+3,zz,1]
                        v_matrix[tt,3,1,zz] = low_output[tt,xx+3,yy+1,zz,1]
                        v_matrix[tt,3,2,zz] = low_output[tt,xx+3,yy+2,zz,1]
                        v_matrix[tt,3,3,zz] = low_output[tt,xx+3,yy+3,zz,1]
                        low_v2[tt,0,0,zz] = block_reduce(v_matrix[tt,:,:,zz],block_size=(alpha3,alpha3), func=np.mean)
                        low_output[tt,xx:xx+alpha3,yy:yy+alpha3,zz,1] = low_v2[tt,0,0,zz]
                        av[tt,xx,yy,zz]=2
                        av[tt,xx,yy+1,zz]=2
                        av[tt,xx,yy+2,zz]=2
                        av[tt,xx,yy+3,zz]=2
                        av[tt,xx+1,yy,zz]=2
                        av[tt,xx+1,yy+1,zz]=2
                        av[tt,xx+1,yy+2,zz]=2
                        av[tt,xx+1,yy+3,zz]=2
                        av[tt,xx+2,yy,zz]=2
                        av[tt,xx+2,yy+1,zz]=2
                        av[tt,xx+2,yy+2,zz]=2
                        av[tt,xx+2,yy+3,zz]=2
                        av[tt,xx+3,yy,zz]=2
                        av[tt,xx+3,yy+1,zz]=2
                        av[tt,xx+3,yy+2,zz]=2
                        av[tt,xx+3,yy+3,zz]=2
              
        for xx in range(int(nx/alpha-(alpha2-1))):
            for yy in range(int(ny/alpha-(alpha2-1))):                
                betav = uvw_std[tt,xx,yy,zz,1]
                betav1 = uvw_std[tt,xx+1,yy,zz,1]
                betav2 = uvw_std[tt,xx,yy+1,zz,1]
                betav3 = uvw_std[tt,xx+1,yy+1,zz,1]
                if av[tt,xx,yy,zz]==0 and av[tt,xx+1,yy,zz]==0 and av[tt,xx+1,yy+1,zz]==0 and av[tt,xx,yy+1,zz]==0:
                    if betav < Thrv and betav1 < Thrv and betav2 < Thrv and betav3 < Thrv:
                        v_matrix = np.zeros((len(datasetSerial),alpha2,alpha2,nz))
                        low_v = np.zeros((len(datasetSerial),1,1,nz)) 
                        v_matrix[tt,0,0,zz] = low_output[tt,xx,yy,zz,1]
                        v_matrix[tt,1,0,zz] = low_output[tt,xx+1,yy,zz,1]
                        v_matrix[tt,0,1,zz] = low_output[tt,xx,yy+1,zz,1]
                        v_matrix[tt,1,1,zz] = low_output[tt,xx+1,yy+1,zz,1]
                        low_v[tt,0,0,zz] = block_reduce(v_matrix[tt,:,:,zz],block_size=(alpha2,alpha2), func=np.mean)
                        low_output[tt,xx:xx+alpha2,yy:yy+alpha2,zz,1] = low_v[tt,0,0,zz]  
                        av[tt,xx,yy,zz]=1
                        av[tt,xx+1,yy,zz]=1
                        av[tt,xx,yy+1,zz]=1
                        av[tt,xx+1,yy+1,zz]=1              
              
# === w component === #
aw = np.zeros((len(datasetSerial),int(nx/alpha),int(ny/alpha),nz))
for tt in tqdm(range(len(datasetSerial))):
    for zz in range(nz):
        for xx in range(int(nx/alpha-3)):
            for yy in range(int(ny/alpha-3)):
                beta2w_0 = uvw_std[tt,xx,yy,zz,2]
                beta2w_1 = uvw_std[tt,xx+1,yy,zz,2]
                beta2w_2 = uvw_std[tt,xx+2,yy,zz,2]
                beta2w_3 = uvw_std[tt,xx+3,yy,zz,2]
                beta2w_4 = uvw_std[tt,xx,yy+1,zz,2]
                beta2w_5 = uvw_std[tt,xx,yy+2,zz,2]
                beta2w_6 = uvw_std[tt,xx,yy+3,zz,2]
                beta2w_7 = uvw_std[tt,xx+1,yy+1,zz,2]
                beta2w_8 = uvw_std[tt,xx+1,yy+2,zz,2]
                beta2w_9 = uvw_std[tt,xx+1,yy+3,zz,2]
                beta2w_10= uvw_std[tt,xx+2,yy+1,zz,2]
                beta2w_11= uvw_std[tt,xx+2,yy+2,zz,2]
                beta2w_12= uvw_std[tt,xx+2,yy+3,zz,2]
                beta2w_13= uvw_std[tt,xx+3,yy+1,zz,2]
                beta2w_14= uvw_std[tt,xx+3,yy+2,zz,2]
                beta2w_15= uvw_std[tt,xx+3,yy+3,zz,2]


                if aw[tt,xx,yy,zz]==0 and aw[tt,xx+1,yy,zz]==0 and aw[tt,xx+2,yy,zz]==0 and aw[tt,xx+3,yy,zz]==0\
                and aw[tt,xx,yy+1,zz]==0 and aw[tt,xx+1,yy+1,zz]==0 and aw[tt,xx+2,yy+1,zz]==0 and aw[tt,xx+3,yy+1,zz]==0\
                and aw[tt,xx,yy+2,zz]==0 and aw[tt,xx+1,yy+2,zz]==0 and aw[tt,xx+2,yy+2,zz]==0 and aw[tt,xx+3,yy+2,zz]==0\
                and aw[tt,xx,yy+3,zz]==0 and aw[tt,xx+1,yy+3,zz]==0 and aw[tt,xx+2,yy+3,zz]==0 and aw[tt,xx+3,yy+3,zz]==0:
                    if beta2w_0   < Thrw2 and beta2w_1  < Thrw2 and beta2w_2 < Thrw2 and beta2w_3  < Thrw2 and beta2w_4 < Thrw2\
                    and beta2w_5  < Thrw2 and beta2w_6  < Thrw2 and beta2w_7 < Thrw2 and beta2w_8  < Thrw2 and beta2w_9 < Thrw2\
                    and beta2w_10 < Thrw2 and beta2w_11 < Thrw2 and beta2w_12< Thrw2 and beta2w_13 < Thrw2 and beta2w_14< Thrw2\
                    and beta2w_15 < Thrw2:

                        w_matrix = np.zeros((len(datasetSerial),alpha3,alpha3,nz))
                        low_w2 = np.zeros((len(datasetSerial),1,1,nz))
                        w_matrix[tt,0,0,zz] = low_output[tt,xx,yy,zz,2]
                        w_matrix[tt,1,0,zz] = low_output[tt,xx+1,yy,zz,2]
                        w_matrix[tt,2,0,zz] = low_output[tt,xx+2,yy,zz,2]
                        w_matrix[tt,3,0,zz] = low_output[tt,xx+3,yy,zz,2]
                        w_matrix[tt,0,1,zz] = low_output[tt,xx,yy+1,zz,2]
                        w_matrix[tt,0,2,zz] = low_output[tt,xx,yy+2,zz,2]
                        w_matrix[tt,0,3,zz] = low_output[tt,xx,yy+3,zz,2]
                        w_matrix[tt,1,1,zz] = low_output[tt,xx+1,yy+1,zz,2]
                        w_matrix[tt,1,2,zz] = low_output[tt,xx+1,yy+2,zz,2]
                        w_matrix[tt,1,3,zz] = low_output[tt,xx+1,yy+3,zz,2]
                        w_matrix[tt,2,1,zz] = low_output[tt,xx+2,yy+1,zz,2]
                        w_matrix[tt,2,2,zz] = low_output[tt,xx+2,yy+2,zz,2]
                        w_matrix[tt,2,3,zz] = low_output[tt,xx+2,yy+3,zz,2]
                        w_matrix[tt,3,1,zz] = low_output[tt,xx+3,yy+1,zz,2]
                        w_matrix[tt,3,2,zz] = low_output[tt,xx+3,yy+2,zz,2]
                        w_matrix[tt,3,3,zz] = low_output[tt,xx+3,yy+3,zz,2]
                        low_w2[tt,0,0,zz] = block_reduce(w_matrix[tt,:,:,zz],block_size=(alpha3,alpha3), func=np.mean)
                        low_output[tt,xx:xx+alpha3,yy:yy+alpha3,zz,2] = low_w2[tt,0,0,zz]
                        aw[tt,xx,yy,zz]=2
                        aw[tt,xx,yy+1,zz]=2
                        aw[tt,xx,yy+2,zz]=2
                        aw[tt,xx,yy+3,zz]=2
                        aw[tt,xx+1,yy,zz]=2
                        aw[tt,xx+1,yy+1,zz]=2
                        aw[tt,xx+1,yy+2,zz]=2
                        aw[tt,xx+1,yy+3,zz]=2
                        aw[tt,xx+2,yy,zz]=2
                        aw[tt,xx+2,yy+1,zz]=2
                        aw[tt,xx+2,yy+2,zz]=2
                        aw[tt,xx+2,yy+3,zz]=2
                        aw[tt,xx+3,yy,zz]=2
                        aw[tt,xx+3,yy+1,zz]=2
                        aw[tt,xx+3,yy+2,zz]=2
                        aw[tt,xx+3,yy+3,zz]=2

        for xx in range(int(nx/alpha-1)):
            for yy in range(int(ny/alpha-1)):                
                betaw = uvw_std[tt,xx,yy,zz,1]
                betaw1 = uvw_std[tt,xx+1,yy,zz,1]
                betaw2 = uvw_std[tt,xx,yy+1,zz,1]
                betaw3 = uvw_std[tt,xx+1,yy+1,zz,1]
                if aw[tt,xx,yy,zz]==0 and aw[tt,xx+1,yy,zz]==0 and aw[tt,xx+1,yy+1,zz]==0 and aw[tt,xx,yy+1,zz]==0:
                    if betaw < Thrw and betaw1 < Thrw and betaw2 < Thrw and betaw3 < Thrw:
                        w_matrix = np.zeros((len(datasetSerial),alpha2,alpha2,nz))
                        low_w = np.zeros((len(datasetSerial),1,1,nz)) 
                        w_matrix[tt,0,0,zz] = low_output[tt,xx,yy,zz,2]
                        w_matrix[tt,1,0,zz] = low_output[tt,xx+1,yy,zz,2]
                        w_matrix[tt,0,1,zz] = low_output[tt,xx,yy+1,zz,2]
                        w_matrix[tt,1,1,zz] = low_output[tt,xx+1,yy+1,zz,2]
                        low_w[tt,0,0,zz] = block_reduce(w_matrix[tt,:,:,zz],block_size=(alpha2,alpha2), func=np.mean)
                        low_output[tt,xx:xx+alpha2,yy:yy+alpha2,zz,2] = low_w[tt,0,0,zz]
                        aw[tt,xx,yy,zz]=1
                        aw[tt,xx+1,yy,zz]=1
                        aw[tt,xx,yy+1,zz]=1
                        aw[tt,xx+1,yy+1,zz]=1
              
# Remake the adaptive sampled-field to the original domain size
Adaptive_field = np.zeros((len(datasetSerial),nx,ny,nz,comp))
for tt in range(num_ts):
    for cc in range(comp):
        for zz in range(nz):
            st_x = 0
            end_x = alpha
            for xx in range (int(nx/alpha)):
                st_y = 0
                end_y = alpha
                for yy in range (int(ny/alpha)):
                    Adaptive_field[tt,st_x:end_x,st_y:end_y,zz,cc] = low_output[tt,xx,yy,zz,cc]
                    st_y =st_y+alpha
                    end_y = end_y+alpha
                st_x = st_x+alpha
                end_x = end_x+alpha
