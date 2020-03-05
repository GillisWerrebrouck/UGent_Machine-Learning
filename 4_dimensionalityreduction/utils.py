from sklearn.datasets import make_blobs
import numpy as np
import seaborn as sns

def generate_anisotropicgaussian(no_samples,no_features):
    #generate isotropic gaussian data
    X_orig, labels = make_blobs(n_samples=no_samples,center_box=(2,2),n_features=no_features,centers=1,cluster_std=0.3,shuffle=True)
    #now apply a linear transformation on this data
    theta = np.radians(70)
    t = np.tan(theta)
    shear_x = np.array(((1, t), (0, 1))).T
    X_rotated = X_orig.dot(shear_x)
    return X_rotated


def plot2DScatter(X):
    ax = sns.scatterplot(x=X[:,0],y=X[:,1],legend=False);
    ax.set(xlim=(5, 10));
    ax.set(ylim=(1,3));
    ax.set(xlabel='$x_1$');
    ax.set(ylabel='$x_2$');
    
    
#fit and plot ellipsoid
from skimage.measure import EllipseModel
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt


def plotscatterwithellipse(data,xmin,xmax,ymin,ymax):
    ell = EllipseModel();
    ell.estimate(data);
    xc, yc, a,b , theta = ell.params
    ell_patch = Ellipse((xc, yc), 2*a, 2*b, theta*180/np.pi, edgecolor='red', facecolor='none')
    sns.set_style("whitegrid")    
    ax = sns.scatterplot(x=data[:,0],y=data[:,1],legend=False);
    ax.set(xlim=(xmin,xmax));
    ax.set(ylim=(ymin,ymax));
    ax.set(xlabel='$x_1$');
    ax.set(ylabel='$x_2$');
    ax.add_patch(ell_patch)
    ax.scatter(xc,yc,color='red',s=20)
    #ax.scatter(xc-a*np.sin(theta),yc+a*np.sin(theta),color='red',s=20)
    #ax.scatter(xc+b*np.sin(theta),yc-b*np.cos(theta),color='green',s=20)
    #a and b scaled a little bit for better visuals
    scalefactor=1.2
    delta_endarrow1 = (scalefactor*b*np.sin(theta),scalefactor*(-b)*np.cos(theta))
    delta_endarrow2 = (scalefactor*(-a)*np.sin(theta),scalefactor*a*np.sin(theta))    
    ax.arrow(xc,yc,delta_endarrow1[0],delta_endarrow1[1],width=0.02,color='red')
    ax.arrow(xc,yc,delta_endarrow2[0],delta_endarrow2[1],width=0.02,color='green')
    
    style1 = dict(size=20,color='red')
    style2 = dict(size=20,color='green')
    ax.text(xc+delta_endarrow1[0]+0.2,yc+delta_endarrow1[1],"$z_1$",ha='left',**style1)
    ax.text(xc+delta_endarrow2[0],yc+delta_endarrow2[1]+0.2,"$z_2$",ha='left',**style2)
    plt.show()
    

def plotProjection(X):
    #zero-mean the data and rotate axes to line up with directions of highest variance
    X_zeromean = X-X.mean(axis=0)
    from numpy import linalg as LA
    w,v = LA.eig(np.cov(X_zeromean,rowvar=False,bias=True).T)
    Z = X_zeromean.dot(v)
    ax = sns.scatterplot(Z[:,0],Z[:,1])
    ax.set(xlim=(-3,+3));
    ax.set(ylim=(-2,2));
    ax.set(xlabel='$z_1$');
    ax.set(ylabel='$z_2$');