# Subjective Answers — Task 1

## 1.7 Predict First, Then Measure

### Predictions Before Running the Experiment

- PSNR: I predict that applying a 9 x 9 box filter to the Gaussian-noisy image will make PSNR worse because the large kernel may blur important image details along with reducing noise.
- SSIM: I predict that the box filter will improve SSIM because smoothing should suppress noise and make the image structurally more similar to the clean image.

### Measured Results

| Metric | Before Filtering | After Filtering |
|---|---:|---:|
| PSNR (dB) | 24.53 | 21.55 |
| SSIM | 0.4241 | 0.7225 |

### Comparison with Predictions

- PSNR prediction: Correct. PSNR decreased from 24.53 dB to 21.55 dB. Although filtering reduces noise, the large kernel also blurs edges and removes fine details. In this experiment, the resulting pixel-wise reconstruction error increased.
- SSIM prediction: Correct. SSIM increased from 0.4241 to 0.7225. Noise suppression makes the overall structure more similar to the clean image, even though some fine details are lost.

Thus, the experiment demonstrates that filtering can improve structural similarity while reducing PSNR.

## 1.8 The Two Metrics Disagree

PSNR and SSIM measure different aspects of image quality. PSNR measures pixel-wise reconstruction error, whereas SSIM measures structural similarity based on luminance, contrast, and structure.

For the Gaussian-noisy image, applying the 9 x 9 box filter reduced PSNR from 24.53 dB to 21.55 dB, while SSIM increased from 0.4241 to 0.7225.

This happens because filtering suppresses noise and improves the overall structural appearance of the image. However, the large kernel also blurs edges and fine details, changing pixel values and increasing the squared reconstruction error.

For template matching, I would focus on preserving distinctive edges, shapes, and local patterns rather than relying on PSNR alone. SSIM provides useful structural information, but neither metric guarantees better template matching. The final performance depends on whether the features required for matching remain distinguishable.

## 1.9 What Colour Costs You

The original experiment uses colour images with three channels. Converting the clean image to grayscale makes each filter operate on a single intensity channel instead of processing the RGB channels separately.

The results may change because grayscale conversion combines red, green, and blue information into a single intensity value. Colour differences between objects may disappear, and the PSNR and SSIM values may change because the metrics are now calculated on grayscale images.

Colour is not useless, however. In the Waldo detection task, the red-and-white striped clothing provides a useful visual cue. Objects with similar grayscale intensities may have different colours, allowing colour-based template matching to distinguish them.

Therefore, grayscale simplifies image processing and feature detection, while colour can provide additional information for distinguishing objects during template matching.

## 1.10 Padding

When `convolve2d` uses zero padding, pixels outside the image boundary are treated as zero. Near the borders, the filter averages real image pixels with artificial zero-valued pixels.

With a large kernel, this effect becomes more noticeable because a larger part of the filter window may extend outside the image. This can cause artificial darkening near the boundaries and introduce intensity transitions that do not exist in the original image.

Reflect padding is safer for template matching because it extends the image by reflecting pixel values across the boundaries. This generally reduces artificial dark regions and better preserves local intensity patterns near the edges.

Therefore, reflect padding is preferable to zero padding when filtering images for template matching, because artificial boundary artefacts may interfere with the similarity between a template and a scene.