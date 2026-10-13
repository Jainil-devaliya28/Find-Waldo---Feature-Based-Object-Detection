# %%
# Imports
# ================================================================
# Don't update these imports.
import os
import cv2
import getpass
import numpy as np
import matplotlib.pyplot as plt
from task1 import *
from task2 import *

# %%
# Helper Functions
# ================================================================
# These are helper functions that you can use in your implementation.
# Don't update these functions.

os.makedirs("plots", exist_ok=True)


TEMPLATE_KEYPOINTS = 160 #TODO: Make your own Choice
SCENE_KEYPOINTS = 1200    #TODO: Make your own chouce


def sample_bilinear(image: np.ndarray, ys: np.ndarray,
                    xs: np.ndarray) -> np.ndarray:
    """Read the image at fractional coordinates, clamping outside the border."""
    h, w = image.shape
    ys = np.clip(ys, 0, h - 1)
    xs = np.clip(xs, 0, w - 1)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    fy, fx = ys - y0, xs - x0
    return ((1 - fy) * (1 - fx) * image[y0, x0]
            + (1 - fy) * fx * image[y0, x1]
            + fy * (1 - fx) * image[y1, x0]
            + fy * fx * image[y1, x1])


def orientation_histogram(magnitude: np.ndarray, orientation: np.ndarray,
                          n_bins: int, weights: np.ndarray = None) -> np.ndarray:
    """Histogram of gradient directions over [0, 360), weighted by magnitude.

    Each sample goes entirely into one bin (hard assignment). Bin b covers
    [b * 360 / n_bins, (b + 1) * 360 / n_bins).
    """
    mag = np.asarray(magnitude).ravel()
    ori = np.asarray(orientation).ravel()
    w = mag if weights is None else mag * np.asarray(weights).ravel()
    index = np.clip(np.floor(ori / (360.0 / n_bins)).astype(int), 0, n_bins - 1)
    return np.bincount(index, weights=w, minlength=n_bins).astype(np.float64)


# ================================================================
# %%
# Task 3
# ======
#

def harris_response(image: np.ndarray, sigma_d: float = 1.0,
                    sigma_i: float = 2.0, k: float = 0.04) -> np.ndarray:
    """Harris cornerness: how much the image changes in EVERY direction at once.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image of shape (H, W)
    sigma_d : float
        Blur applied before differentiating
    sigma_i : float
        Blur applied to the gradient products
    k : float
        Harris constant, usually 0.04 to 0.06

    Returns
    -------
    np.ndarray
        Cornerness map of shape (H, W)
    """
    g=gaussian_blur(np.asarray(image,float),sigma_d); gy,gx=np.gradient(g)
    jxx=gaussian_blur(gx*gx,sigma_i); jyy=gaussian_blur(gy*gy,sigma_i); jxy=gaussian_blur(gx*gy,sigma_i)
    tr=jxx+jyy; return jxx*jyy-jxy*jxy-k*tr*tr
    # return np.zeros_like(image)  # comment this line and write your code for the function


def detect_corners(image: np.ndarray, threshold_rel: float = 0.02,
                   min_distance: int = 3, max_corners: int = 120,
                   border: int = 4) -> np.ndarray:
    """Turn the Harris map into a list of corner positions.


    Parameters
    ----------
    image : np.ndarray
        Grayscale image of shape (H, W)
    threshold_rel : float
        Fraction of the maximum response a corner must reach
    min_distance : int
        Minimum separation between corners
    max_corners : int
        Keep at most this many
    border : int
        Width of the ignored frame

    Returns
    -------
    np.ndarray
        Shape (N, 2), columns (y, x), strongest first. (0, 2) if none.
    """
    r=np.maximum(np.asarray(image,float),0); mx=r.max() if r.size else 0
    if mx<=0: return np.zeros((0,2),int)
    y,x=np.nonzero(r>=threshold_rel*mx); v=r[y,x]; ok=(y>=border)&(y<r.shape[0]-border)&(x>=border)&(x<r.shape[1]-border); y,x,v=y[ok],x[ok],v[ok]
    keep=[]
    for i in np.argsort(-v):
        yy,xx=int(y[i]),int(x[i])
        if all(max(abs(yy-a),abs(xx-b))>=min_distance for a,b in keep):
            keep.append((yy,xx))
            if len(keep)>=max_corners: break
    return np.asarray(keep,int).reshape(-1,2)
    # return np.zeros((0, 2))  # comment this line and write your code for the function


def dog_keypoints(image: np.ndarray, sigma0: float = 0.8, n_scales: int = 8,
                  k: float = 1.25, threshold: float = 0.005,
                  border: int = 4, max_keypoints: int = 800) -> np.ndarray:
    """Find blob-like keypoints, and the SIZE each one lives at.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image of shape (H, W)
    sigma0, n_scales, k : float, int, float
        The scale ladder
    threshold : float
        Minimum absolute response
    border : int
        Ignored frame width
    max_keypoints : int
        Keep at most this many

    Returns
    -------
    np.ndarray
        Shape (N, 3), columns (y, x, sigma), strongest first. (0, 3) if none.
    """
    g=np.asarray(image,float); sig=sigma0*(k**np.arange(n_scales)); gs=[gaussian_blur(g,float(s)) for s in sig]; dogs=[gs[i+1]-gs[i] for i in range(n_scales-1)]; H,W=g.shape; cand=[]
    for z in range(1,len(dogs)-1):
        cur=dogs[z]; core=cur[border:H-border,border:W-border]; mask=np.abs(core)>=threshold; mx=np.ones(core.shape,bool); mn=np.ones(core.shape,bool)
        for dz in (-1,0,1):
          for dy in (-1,0,1):
            for dx in (-1,0,1):
              if dz==0 and dy==0 and dx==0: continue
              nb=dogs[z+dz][border+dy:H-border+dy,border+dx:W-border+dx]; mx &= core>nb; mn &= core<nb
        yy,xx=np.nonzero(mask&(mx|mn))
        for y,x in zip(yy+border,xx+border): cand.append((int(y),int(x),float(sig[z+1]),float(abs(cur[y,x]))))
    cand.sort(key=lambda q:q[3],reverse=True); keep=[]
    for q in cand:
        y,x=q[:2]
        if all(max(abs(y-a),abs(x-b))>=3 for a,b,_,_ in keep):
            keep.append(q)
            if len(keep)>=max_keypoints: break
    return np.asarray([(y,x,s) for y,x,s,_ in keep],float).reshape(-1,3)

    # return np.zeros((0, 3))  # comment this line and write your code for the function


def dominant_orientation(image: np.ndarray, y: float, x: float,
                         sigma: float = 1.6, n_bins: int = 36) -> float:
    """The direction a keypoint's neighbourhood mostly points in.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image of shape (H, W)
    y, x : float
        Keypoint position
    sigma : float
        Scale the keypoint was found at
    n_bins : int
        Number of orientation bins

    Returns
    -------
    float
        Angle in degrees in [0, 360), or 0.0 if the window has no gradient
    """
    g=np.asarray(image,float); r=max(1,int(np.ceil(3*sigma))); yy,xx=np.mgrid[int(y)-r:int(y)+r+1,int(x)-r:int(x)+r+1]
    yyf,xxf=yy.ravel().astype(float),xx.ravel().astype(float)
    gx=.5*(sample_bilinear(g,yyf,xxf+1)-sample_bilinear(g,yyf,xxf-1)); gy=.5*(sample_bilinear(g,yyf+1,xxf)-sample_bilinear(g,yyf-1,xxf))
    mag=np.hypot(gx,gy); ori=(np.degrees(np.arctan2(gy,gx))+360)%360; w=np.exp(-((yyf-y)**2+(xxf-x)**2)/(2*sigma*sigma)); h=orientation_histogram(mag,ori,n_bins,w)
    return 0.0 if h.max()<=1e-15 else float((np.argmax(h)+.5)*360/n_bins%360)

    # return 0.0  # comment this line and write your code for the function


def describe_keypoints(image: np.ndarray, keypoints: np.ndarray,
                       orientations: np.ndarray = None) -> np.ndarray:
    """Describe each keypoint by a grid of gradient-orientation histograms.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image of shape (H, W)
    keypoints : np.ndarray
        Shape (N, 3), columns (y, x, sigma)
    orientations : np.ndarray, optional
        Shape (N,) in degrees; None means upright descriptors

    Returns
    -------
    np.ndarray
        Shape (N, 128), one row per keypoint
    """
    g=np.asarray(image,float); kp=np.asarray(keypoints,float).reshape(-1,3); n=len(kp)
    if n==0: return np.zeros((0,128))
    angs=np.zeros(n) if orientations is None else np.asarray(orientations,float)
    gy, gx = np.gradient(g)
    sigma = np.maximum(kp[:, 2], 0.5)
    theta = np.deg2rad(angs)
    ct, st = np.cos(theta), np.sin(theta)    
    q=(np.arange(-8,8,dtype=float)+.5); gx0=q[None,:,None]*sigma[:,None,None]; gx0=np.broadcast_to(gx0,(n,16,16)); gy0=np.broadcast_to(gx0.transpose(0,2,1),(n,16,16))
    dx=ct[:,None,None]*gx0-st[:,None,None]*gy0; dy=st[:,None,None]*gx0+ct[:,None,None]*gy0; ys=kp[:,0,None,None]+dy; xs=kp[:,1,None,None]+dx
    h,w=g.shape; ys=np.clip(ys,0,h-1); xs=np.clip(xs,0,w-1); y0=np.floor(ys).astype(int); x0=np.floor(xs).astype(int); y1=np.minimum(y0+1,h-1); x1=np.minimum(x0+1,w-1); fy=ys-y0; fx=xs-x0
    def bil(arr):
        return ((1-fy)*(1-fx)*arr[y0,x0]+(1-fy)*fx*arr[y0,x1]+fy*(1-fx)*arr[y1,x0]+fy*fx*arr[y1,x1])
    sx,sy=bil(gx),bil(gy); m=np.hypot(sx,sy); a=(np.degrees(np.arctan2(sy,sx))-angs[:,None,None])%360
    wt=np.exp(-(gx0*gx0+gy0*gy0)/(2*(8*sigma[:,None,None])**2)); bins=np.floor(a/45).astype(int).clip(0,7); cells=(np.arange(16)[:,None]//4)*4+(np.arange(16)[None,:]//4)
    D=np.zeros((n,128)); np.add.at(D,(np.arange(n)[:,None,None],cells[None,:,:]*8+bins),m*wt)
    z=np.linalg.norm(D,axis=1,keepdims=True); D=np.divide(D,z,out=np.zeros_like(D),where=z>1e-12); D=np.minimum(D,.2); z=np.linalg.norm(D,axis=1,keepdims=True); D=np.divide(D,z,out=np.zeros_like(D),where=z>1e-12); return D

    # return np.zeros((len(keypoints), 128))  # comment this line and write your code for the function


if __name__ == "__main__":

    template = load_image_as_grayscale("imgs/waldo_template.png")
    scene = load_image_as_grayscale("imgs/scene_easy.png")

    # #############################################
    # Comment these lines and write your code here
    # response = np.zeros_like(template)
    # corners = np.zeros((0, 2))
    # keypoints = np.zeros((0, 3))
    # angles = np.zeros(0)
    # descriptors = np.zeros((0, 128))
    response = harris_response(template)
    corners = detect_corners(response)
    keypoints = dog_keypoints(template, max_keypoints=TEMPLATE_KEYPOINTS)
    angles = np.array([dominant_orientation(template,y,x,s) for y,x,s in keypoints])
    descriptors = describe_keypoints(template,keypoints,angles)
    # #############################################

    print("Harris corners on the template :", len(corners))
    print("DoG keypoints on the template  :", len(keypoints))
    print("sigmas found                   :",
          np.round(np.unique(np.round(keypoints[:, 2], 2)), 2))
    norms = np.linalg.norm(descriptors, axis=1) if len(descriptors) else np.zeros(1)
    print("descriptors                    :", descriptors.shape,
          "row norms in [{:.3f}, {:.3f}]".format(norms.min(), norms.max()))

    scene_keypoints = dog_keypoints(scene, max_keypoints=SCENE_KEYPOINTS)
    print("DoG keypoints on scene_easy    :", len(scene_keypoints))

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(template, cmap="gray")
    axes[0].set_title("Template")
    axes[1].imshow(response, cmap="gray")
    axes[1].set_title("Harris response R")
    axes[2].imshow(template, cmap="gray")
    if len(corners):
        axes[2].scatter(corners[:, 1], corners[:, 0], s=18,
                        facecolors="none", edgecolors="lime")
    axes[2].set_title("Harris corners")
    axes[3].imshow(template, cmap="gray")
    if len(keypoints):
        axes[3].scatter(keypoints[:, 1], keypoints[:, 0], s=keypoints[:, 2] * 12,
                        facecolors="none", edgecolors="red")
    axes[3].set_title("DoG keypoints (size = sigma)")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("plots/task3_results.png", metadata={"Author": getpass.getuser()})
    print("\nwrote plots/task3_results.png")
