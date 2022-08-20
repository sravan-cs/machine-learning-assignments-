import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# converting the input image into numpy array
photo=Image.open('38.jpg')
arr=np.asarray(photo)

# finding the eigen vectors and the eigen values using linalg library
eigval,eigvec=np.linalg.eig(arr)
eigvecinv=np.linalg.inv(eigvec)
eigval=np.diag(eigval)
eigvalmod = np.abs(eigval)

# storing the indices of elemnets in descending order 
ind=np.argsort(np.diag(eigvalmod),axis=0)
n=np.size(ind)
ind=ind[::-1][:n]

# storing the values of k to experiment on
k_values=[1,2,3,5,10,15,30,50,60,100,130,150,200,230,250]
fig,ax=plt.subplots(15,2,figsize=(30,30))
fig_idx=0

for k in k_values :

    # new_val is the matrix with k largest eigen values 
    new_val = np.zeros([n,n],dtype=complex)
    for i in range (0,k):
        new_val[ind[i],:]=eigval[ind[i],:]

    # new_arr is the array corresponding to the k largest eigen values 
    new_arr=eigvec@new_val@eigvecinv
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
    frobenious_norm=np.linalg.norm(abs(arr)-new_arr,'fro')
    print("k = " + str(k) +" , norm = " + str(frobenious_norm) )


fig.tight_layout()
fig.savefig("evd.jpg",dpi=96)



