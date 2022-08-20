import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

# converting the input image into numpy array
photo = Image.open("38.jpg")
arr=np.asarray(photo,dtype=np.double)

# A = U W V_t ( singular value decomposition )
# A*A_t = U W v_t V W_t U_t = U W^2 U_t ( since V is orthonormal )
# W^-1 * U_t * A = V_t
# from second equation we find U,W using eigen value decomposition of A*A_t and from third , we find V_t

arr_transpose=arr.transpose()
singval,singvec1=np.linalg.eig(arr@arr_transpose)
n=np.size(singval)
singval=np.diag(singval)
for i in range(0,n):
    if(singval[i][i]<0):
        singval[i][i]=0;
    singval[i][i]=math.sqrt(singval[i][i])
singvec2=(np.linalg.inv(singval))@(singvec1.transpose())@(arr)

# storing the indices of singular values in descending order
ind=np.argsort(np.diag(singval),axis=0)
ind=ind[::-1][:n]

# storing the values of k to experiment on
k_values=[1,2,3,5,10,15,30,50,60,100,130,150,200,230,250]
fig,ax=plt.subplots(15,2,figsize=(30,30))
fig_idx=0

for k in k_values :

    # new_val is the matrix with k largest singular values 
    new_val = np.zeros([n,n],dtype=np.double)
    for i in range (0,k):
        new_val[ind[i],:]=singval[ind[i],:]

    # new_arr is the array corresponding to the k largest singular values 
    new_arr=singvec1@new_val@singvec2
    new_arr = np.abs(new_arr)

    # plotting the reconstructed image along with its error image
    ax[fig_idx][0].imshow(new_arr,cmap="gray",vmin=0,vmax=255)
    ax[fig_idx][0].axis('off')
    ax[fig_idx][0].set_title("k = "+str(k))
    ax[fig_idx][1].imshow(arr-new_arr,cmap="gray",vmin=0,vmax=255)
    ax[fig_idx][1].set_title("error image")
    ax[fig_idx][1].axis('off')

    fig_idx=fig_idx+1

    # printing frobenius norm for the report
    frobenious_norm=np.linalg.norm(arr-new_arr,'fro')
    print("k = " + str(k) +" , norm = " + str(frobenious_norm) )

fig.tight_layout()
fig.savefig("svd.jpg",dpi=96)




