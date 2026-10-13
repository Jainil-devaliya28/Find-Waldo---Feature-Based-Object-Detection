# %%
# Imports
# ================================================================
# Don't update these imports.
import os
import cv2
import getpass
import numpy as np
import matplotlib.pyplot as plt

# %%
# Helper Functions
# ================================================================
# These are helper functions that you can use in your implementation.
# Don't update these functions.

os.makedirs("plots", exist_ok=True)


def load_image_as_rgb(image_path: str) -> np.ndarray:
    """Load an image as RGB floats in [0, 1], shape (H, W, 3).

    This is the normal way to load an image in this assignment. Waldo is
    defined by his RED and WHITE stripes, and throwing that away costs you
    real information -- one of the questions asks you to measure how much.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV loads BGR
    return image.astype(np.float64) / 255.0




def load_image_as_grayscale(image_path: str) -> np.ndarray:
    """Load straight to luminance, shape (H, W). Used by Task 3."""
    return to_grayscale(load_image_as_rgb(image_path))


def add_gaussian_noise(image: np.ndarray, sigma: float = 0.06,
                       seed: int = 0) -> np.ndarray:
    """Add zero-mean Gaussian noise to every channel, then clip to [0, 1]."""
    rng = np.random.default_rng(seed)
    return np.clip(image + rng.normal(0.0, sigma, image.shape), 0.0, 1.0)


def add_salt_pepper_noise(image: np.ndarray, amount: float = 0.04,
                          seed: int = 0) -> np.ndarray:
    """Knock a fraction of PIXELS to pure black or pure white.

    Note it is whole pixels, not individual channel samples: a dead sensor
    pixel loses all three channels at once. Corrupting channels separately
    would give coloured confetti, which is not what this noise model is.
    """
    rng = np.random.default_rng(seed)
    out = image.copy()
    hit = rng.random(image.shape[:2]) < amount
    values = rng.integers(0, 2, int(hit.sum())).astype(np.float64)
    if image.ndim == 3:
        out[hit] = values[:, None]          # same value in all channels
    else:
        out[hit] = values
    return out




# ================================================================
# %%
# Task 1
# ======


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur, done as two 1-D passes instead of one 2-D convolution.

    Provided because Tasks 3 and 4 call it hundreds of times and need it to be
    fast. A 2-D Gaussian is separable, so blurring rows and then columns costs
    O(H*W*k) instead of O(H*W*k^2) -- worth understanding, and it is one of
    the report questions.

    Works on (H, W) and on (H, W, 3): a linear filter acts on each colour
    channel independently, so only the two spatial axes are padded.
    """
    #TODO: Write your Code for Gaussian Blur
    image = np.asarray(image, dtype=np.float64)
    if sigma <= 0: return image.copy()
    radius = max(1, int(np.ceil(3.0*sigma)))
    x = np.arange(-radius, radius+1, dtype=np.float64)
    k = np.exp(-(x*x)/(2.0*sigma*sigma)); k /= k.sum()
    image = convolve2d(convolve2d(image,k.reshape(1,-1)),k.reshape(-1,1))
    return image

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Collapse an RGB image to one luminance channel, shape (H, W).

    The Rec.601 weights below are not arbitrary: the eye is far more
    sensitive to green than to blue, so a plain (R+G+B)/3 makes blues look
    lighter than they are.

    Gradient-based code (Sobel, Harris, SIFT-like descriptors in Task 3)
    needs a SCALAR field to differentiate -- "the direction of steepest
    increase" has no meaning for a 3-vector without first choosing how to
    combine the channels. So Task 3 works on this, while Tasks 1 and 2 work
    in full colour. A 2-D array is passed through unchanged.
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim == 2:
        return image

    #TODO: Write the code for Grayscale
    image = 0.299*image[...,0]+0.587*image[...,1]+0.114*image[...,2]
    return image

def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve an image with a kernel, keeping the same output size.

    Pad the image by half the kernel on each side (reflect the edge pixels),
    flip the kernel on both axes, then slide it over every pixel.

    COLOUR: the input may be (H, W) or (H, W, 3). A linear filter treats the
    channels independently, so the same kernel is applied to each one and
    nothing mixes between them. You do not need a separate code path for
    this -- pad only the two SPATIAL axes and the arithmetic broadcasts.
    Getting that padding wrong (padding the channel axis as well) is the
    single most common bug here.

    Parameters
    ----------
    image : np.ndarray
        Image of shape (H, W) or (H, W, 3), values in [0, 1]
    kernel : np.ndarray
        Kernel of shape (kh, kw), both odd

    Returns
    -------
    np.ndarray
        Filtered image, same shape as the input
    """
    image=np.asarray(image,dtype=np.float64); kernel=np.asarray(kernel,dtype=np.float64)
    if kernel.ndim!=2 or any(s%2==0 for s in kernel.shape): raise ValueError('kernel must have odd dimensions')
    py,px=kernel.shape[0]//2,kernel.shape[1]//2
    pads=((py,py),(px,px)) if image.ndim==2 else ((py,py),(px,px),(0,0))
    padded=np.pad(image,pads,mode='reflect'); k=kernel[::-1,::-1]
    out=np.zeros_like(image,dtype=np.float64)
    for y in range(kernel.shape[0]):
        for x in range(kernel.shape[1]):
            out += k[y,x]*padded[y:y+image.shape[0],x:x+image.shape[1]]
    return out
    # return np.zeros_like(image)  # comment this line and write your code for the function


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Build a (size, size) Gaussian kernel that sums to 1.

    Sample exp(-(x^2 + y^2) / (2 * sigma^2)) on a grid centred on zero, then
    divide by the sum of the samples (not by the analytic constant).

    Parameters
    ----------
    size : int
        Side length, must be odd
    sigma : float
        Standard deviation in pixels

    Returns
    -------
    np.ndarray
        Kernel of shape (size, size), summing to 1

    """
    if size<=0 or size%2==0 or sigma<=0: raise ValueError('size must be positive odd')
    r=size//2; yy,xx=np.mgrid[-r:r+1,-r:r+1].astype(np.float64)
    k=np.exp(-(xx*xx+yy*yy)/(2*sigma*sigma)); return k/k.sum()
    # return np.zeros((size, size))  # comment this line and write your code for the function


def median_filter(image: np.ndarray, size: int) -> np.ndarray:
    """Replace every pixel by the median of its (size, size) neighbourhood.

    This one cannot be written as a convolution -- explain why in your answers.

    COLOUR: take the median of each channel separately. Be aware that this
    is a choice, not the only option: the output colour at a pixel may be one
    that appears nowhere in the input window, because the red median and the
    blue median can come from different neighbours. A "vector median" that
    returns an actual input pixel avoids that. You are asked about this.

    Parameters
    ----------
    image : np.ndarray
        Image of shape (H, W) or (H, W, 3)
    size : int
        Side length of the window, must be odd

    Returns
    -------
    np.ndarray
        Filtered image, same shape as the input
    """
    image=np.asarray(image,dtype=np.float64)
    if size<=0 or size%2==0: raise ValueError('size must be positive odd')
    r=size//2; pads=((r,r),(r,r)) if image.ndim==2 else ((r,r),(r,r),(0,0))
    padded=np.pad(image,pads,mode='reflect')
    w=np.lib.stride_tricks.sliding_window_view(padded,(size,size),axis=(0,1))
    return np.median(w,axis=(2,3)) if image.ndim==2 else np.median(w,axis=(3,4))

    # return np.zeros_like(image)  # comment this line and write your code for the function


def sobel_edges(image: np.ndarray) -> tuple:
    """Image gradients from the Sobel kernels, and their polar form.


    Parameters
    ----------
    image : np.ndarray
        Image of shape (H, W) or (H, W, 3)

    Returns
    -------
    tuple
        gx (H, W), gy (H, W), magnitude (H, W), orientation (H, W) in degrees
        wrapped into [0, 360)
    """
    g=to_grayscale(image)
    kx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]],float)/8
    ky=np.array([[-1,-2,-1],[0,0,0],[1,2,1]],float)/8
    gx,gy=convolve2d(g,kx),convolve2d(g,ky); mag=np.hypot(gx,gy)
    ori=(np.degrees(np.arctan2(gy,gx))+360)%360
    return gx,gy,mag,ori
    # # return (np.zeros(image.shape[:2]), np.zeros(image.shape[:2]),
    #         np.zeros(image.shape[:2]), np.zeros(image.shape[:2]))  # comment this line and write your code for the function


def psnr(image: np.ndarray, reference: np.ndarray) -> float:
    """Peak signal-to-noise ratio, in decibels. Higher is better.

        MSE  = mean((image - reference) ** 2)
        PSNR = 10 * log10(1.0 / MSE)          because our images are in [0, 1]


    Parameters
    ----------
    image : np.ndarray
        The image being judged, shape (H, W) or (H, W, 3)
    reference : np.ndarray
        The clean image to compare against, same shape

    Returns
    -------
    float
        PSNR in dB, or float("inf") when the two images are identical
    """
    mse=float(np.mean((np.asarray(image)-np.asarray(reference))**2))
    return float('inf') if mse==0 else float(10*np.log10(1.0/mse))

    # return 0.0  # comment this line and write your code for the function


def ssim(image: np.ndarray, reference: np.ndarray) -> tuple:
    """Structural similarity: compare local statistics, not just pixel error.

    Inside a Gaussian window of sigma 1.5, with local means mu, variances var
    and covariance cov, and C1 = 0.01 ** 2, C2 = 0.03 ** 2:

        SSIM = ((2*mu_a*mu_b + C1) * (2*cov + C2))
               / ((mu_a**2 + mu_b**2 + C1) * (var_a + var_b + C2))

    Every one of those local statistics is a Gaussian-weighted average, so the
    provided gaussian_blur computes all five of them.


    Parameters
    ----------
    image : np.ndarray
        The image being judged, shape (H, W) or (H, W, 3)
    reference : np.ndarray
        The clean image to compare against, same shape

    Returns
    -------
    tuple
        mean_ssim (float, 1.0 means identical) and ssim_map, the same shape
        as the input. The map is worth plotting: it shows you WHERE a filter
        did damage.
    """
    a=np.asarray(image,float); b=np.asarray(reference,float); C1=.01**2; C2=.03**2
    ma,mb=gaussian_blur(a,1.5),gaussian_blur(b,1.5)
    va=np.maximum(gaussian_blur(a*a,1.5)-ma*ma,0); vb=np.maximum(gaussian_blur(b*b,1.5)-mb*mb,0)
    cov=gaussian_blur(a*b,1.5)-ma*mb
    m=((2*ma*mb+C1)*(2*cov+C2))/((ma*ma+mb*mb+C1)*(va+vb+C2))
    return float(np.mean(m)),m
    # return 0.0, np.zeros_like(image)  # comment this line and write your code for the function


if __name__ == "__main__":

    clean = load_image_as_rgb("imgs/scene_easy.png")
    noisy = add_gaussian_noise(clean, sigma=0.06)
    salted = add_salt_pepper_noise(clean, amount=0.04)

    # Denoise each kind of noise with each kind of filter, then score them.
    # #############################################
    # Comment these lines and write your code here
    # box_on_gaussian = np.zeros_like(noisy)
    # gauss_on_gaussian = np.zeros_like(noisy)
    # median_on_salt = np.zeros_like(salted)
    # gauss_on_salt = np.zeros_like(salted)
    box_on_gaussian = convolve2d(noisy, np.ones((9,9))/81.0)
    gauss_on_gaussian = gaussian_blur(noisy, 2.0)
    median_on_salt = median_filter(salted, 5)
    gauss_on_salt = gaussian_blur(salted, 1.5)
    # #############################################

    results = [("box 9x9 on gaussian", noisy, box_on_gaussian),
               ("gaussian s=2 on gaussian", noisy, gauss_on_gaussian),
               ("median 5x5 on salt+pepper", salted, median_on_salt),
               ("gaussian s=1.5 on salt+pepper", salted, gauss_on_salt)]

    print("Colour images, {} channels".format(clean.shape[2]))
    print("{:<32}{:>10}{:>10}{:>10}{:>10}".format(
        "filter", "PSNR in", "PSNR out", "SSIM in", "SSIM out"))
    for label, before, after in results:
        print("{:<32}{:>10.2f}{:>10.2f}{:>10.4f}{:>10.4f}".format(
            label, psnr(before, clean), psnr(after, clean),
            ssim(before, clean)[0], ssim(after, clean)[0]))

    _gx, _gy, magnitude, _orientation = sobel_edges(clean)
    _mean_ssim, ssim_map = ssim(median_on_salt, clean)

    # 2x3 plots: noisy inputs and their best filter, then the SSIM map
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    panels = [(clean, "Clean", None), (noisy, "Gaussian noise", None),
              (gauss_on_gaussian, "Gaussian filtered", None),
              (salted, "Salt & pepper", None),
              (median_on_salt, "Median filtered", None),
              (ssim_map.mean(axis=2), "SSIM map (dark = damaged)", "gray")]
    for ax, (img, title, cmap) in zip(axes.ravel(), panels):
        ax.imshow(np.clip(img, 0.0, 1.0), cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("plots/task1_results.png", metadata={"Author": getpass.getuser()})
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(clean)
    axes[0].set_title("Input (RGB)")
    axes[1].imshow(magnitude, cmap="gray")
    axes[1].set_title("Sobel gradient magnitude (on luminance)")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("plots/task1_edges.png", metadata={"Author": getpass.getuser()})
    plt.close()
    print("\nwrote plots/task1_results.png and plots/task1_edges.png")
