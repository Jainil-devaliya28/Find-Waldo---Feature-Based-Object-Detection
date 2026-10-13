
# Subjective Answers — Task 2

## 2.7 Why NCC and Not Sum-of-Squared-Differences?

Normalized Cross-Correlation (NCC) is less sensitive to changes in image brightness and contrast than Sum of Squared Differences (SSD).

Let the template be **T** and an image patch be **I**. NCC is defined as:

NCC(I, T) = Σᵢ [(Iᵢ − Ī)(Tᵢ − T̄)] / [√Σᵢ(Iᵢ − Ī)² × √Σᵢ(Tᵢ − T̄)²]

where **Ī** and **T̄** are the mean intensities of the image patch and template, respectively.

Suppose the image patch undergoes a positive contrast change and a brightness shift:

I′ = aI + b, where a > 0

Its mean becomes:

Ī′ = aĪ + b

Therefore:

I′ − Ī′ = a(I − Ī)

Substituting into NCC gives:

NCC(I′, T) = [a × Σᵢ(Iᵢ − Ī)(Tᵢ − T̄)] / [a × √Σᵢ(Iᵢ − Ī)² × √Σᵢ(Tᵢ − T̄)²]

Since **a > 0**, the factor **a** cancels between the numerator and denominator. Hence:

**NCC(aI + b, T) = NCC(I, T)**

Thus, NCC remains unchanged under a positive linear contrast change and an additive brightness shift, provided the image patch has nonzero variance.

SSD is defined as:

SSD(I, T) = Σᵢ (Iᵢ − Tᵢ)²

After the transformation:

SSD(aI + b, T) = Σᵢ (aIᵢ + b − Tᵢ)²

In general, this is not equal to the original SSD. Therefore, SSD can change when brightness or contrast changes, whereas NCC is invariant to these changes under the stated conditions.

This makes NCC useful for template matching when the same object appears under different brightness or contrast conditions. However, NCC is not invariant to arbitrary colour changes, rotations, occlusion, or geometric distortions.

## 2.8 Grayscale Against Colour, Measured

The original experiment performs multiscale template matching using colour images. For the grayscale experiment, both the scene and the template are converted to grayscale before detection.

Grayscale conversion combines the red, green, and blue channels into a single intensity channel. This reduces the amount of channel information used by the matching algorithm and may change the NCC scores and detected locations.

### Colour Results

The colour experiment produced the following results:

| Scene | Best NCC Score | Best Scale | IoU |
|---|---:|---:|---:|
| scene_easy | 0.889 | 1.00 | 1.000 |
| scene_medium | 0.863 | 1.35 | 1.000 |
| scene_rotated | 0.829 | 1.15 | 0.000 |
| scene_hard | 0.809 | 1.50 | 1.000 |

### Grayscale Results

The grayscale experiment produced the following results:

| Scene | Best NCC Score | Best Scale | IoU |
|---|---:|---:|---:|
| scene_easy | 0.862 | 1.00 | 1.000 |
| scene_medium | 0.844 | 1.35 | 1.000 |
| scene_rotated | 0.797 | 1.15 | 0.000 |
| scene_hard | 0.786 | 1.50 | 1.000 |

### Comparison and Explanation

The winning scale remains unchanged for all four scenes. The NCC score decreases after grayscale conversion in every scene.

| Scene | Colour Score | Grayscale Score | Change |
|---|---:|---:|---:|
| scene_easy | 0.889 | 0.862 | -0.027 |
| scene_medium | 0.863 | 0.844 | -0.019 |
| scene_rotated | 0.829 | 0.797 | -0.032 |
| scene_hard | 0.809 | 0.786 | -0.023 |

The results show that the colour representation gives higher maximum NCC scores in this experiment. However, the winning scales and IoU values remain the same for all four scenes.

Colour can provide useful information for detecting Waldo because his red-and-white striped clothing is distinctive. Grayscale conversion removes explicit colour differences, making objects with similar intensity patterns harder to distinguish. At the same time, grayscale images are simpler to process because they contain only one channel.

The results demonstrate that grayscale conversion changes the correlation scores, but it does not change the detected scale or IoU in these experiments. The zero IoU for `scene_rotated` indicates that the detector does not correctly localize that target, even though it obtains a relatively high NCC score.

Therefore, colour information can be useful for template matching, especially when the target has distinctive colours. The measured results do not show an improvement in localization accuracy from colour for these particular scenes, since the IoU values are unchanged.

## 2.9 The Mask You Were Given and Never Used

The file `imgs/waldo_mask.png` identifies the pixels belonging to Waldo within the template rectangle. I implemented masked NCC so that the correlation is calculated using only the pixels selected by the mask.

At each scale, the template is resized, and the mask is resized using nearest-neighbour interpolation so that its dimensions match the resized template.

### Results on scene_hard

| Method | Best NCC Score | Best Scale |
|---|---:|---:|
| Unmasked NCC | 0.8091 | 1.50 |
| Masked NCC | 0.9282 | 1.50 |

The masked NCC score increases from 0.8091 to 0.9282, an absolute increase of 0.1191. Both methods select the same scale of 1.50.

### Explanation

Unmasked NCC uses all pixels within the rectangular template, including background pixels surrounding Waldo. These background pixels can reduce the usefulness of the comparison because they may differ between the template and the scene.

Masked NCC focuses on the pixels selected by the mask, reducing the contribution of the surrounding background. This can make the score more representative of the similarity between Waldo's visible appearance in the template and the corresponding scene region.

This is particularly relevant to `scene_hard`, where an occluder covers part of Waldo. Background and occlusion can make matching more difficult when all template pixels are included.

In this experiment, masked NCC produces a higher maximum score than unmasked NCC. The increase in score supports the usefulness of the mask for this scene. However, a higher NCC score alone does not prove that the detected bounding box is more accurate; localization accuracy must also be evaluated using the ground-truth bounding box and IoU.

## 2.10 Scores Are Not Comparable Across Scales

To examine how NCC scores vary across scales, I evaluated the score map on `scene_hard` for each scale. I recorded the maximum NCC score, the standard deviation of the score map, and the number of pixels in the resized template.

### Results

| Scale | Template Pixels | Maximum NCC | Standard Deviation |
|---:|---:|---:|---:|
| 0.60 | 578 | 0.6275 | 0.0995 |
| 0.72 | 800 | 0.6760 | 0.0999 |
| 0.85 | 1152 | 0.6242 | 0.0998 |
| 1.00 | 1568 | 0.7042 | 0.0928 |
| 1.15 | 2048 | 0.6987 | 0.0968 |
| 1.35 | 2888 | 0.4502 | 0.0921 |
| 1.50 | 3528 | 0.8091 | 0.0892 |
| 1.70 | 4560 | 0.3450 | 0.0846 |

### Analysis

The maximum NCC score varies substantially across scales. The highest score is 0.8091 at scale 1.50, whereas the lowest is 0.3450 at scale 1.70. The template pixel count increases from 578 pixels at scale 0.60 to 4560 pixels at scale 1.70.

The standard deviation generally decreases as the template becomes larger, although it does not decrease strictly at every scale. This indicates that the spread of NCC scores changes with scale.

A smaller template contains fewer pixels and may match background patterns by chance. A larger template includes more image information, but it can also become less similar to the target if its size does not match the target's actual size. The number of candidate locations also changes with scale, affecting the chance of observing a high maximum score.

Consequently, the maximum NCC values at different scales should not automatically be interpreted as equally reliable measurements.

One possible approach is to divide each score map by its own standard deviation. This measures score variation relative to the spread of that map. However, standard-deviation normalization alone does not guarantee fair comparison across scales because it does not account for differences in the mean, distribution shape, or number of candidate locations.

The measured table shows that scale 1.50 produces the highest raw maximum NCC score of 0.8091. Whether standard-deviation normalization improves scale selection must be established by calculating the normalized values and comparing the resulting detections against the ground truth. The current measurements alone are insufficient to conclude that normalization would improve detection.
