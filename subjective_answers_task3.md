# Subjective Answers — Task 3

## 3.6 Corner or Blob?

A Harris corner is represented by (y, x), which specifies its position in the image. A Difference of Gaussian (DoG) keypoint is represented by (y, x, σ), where σ is the characteristic scale at which the keypoint is detected.

The third number, σ, comes from the scale-space representation used by the DoG detector. The image is smoothed at different scales, and the difference between Gaussian-blurred images at successive scales is used to identify keypoints. The scale associated with a detected keypoint indicates the approximate size of the image structure that produced the response.

In my experiment, the template produced 59 Harris corners and 25 DoG keypoints. The detected DoG keypoint scales ranged from approximately 1.25 to 2.44.

Harris corners alone would not be sufficient for the scale-aware matching approach used in Task 4 because they do not inherently provide scale information. When the template appears at a different size in the scene, its corresponding corners may be detected at different locations and densities. DoG keypoints provide scale estimates that can help determine the relative size of the template in the scene.

Therefore, Harris corners are useful for detecting distinctive local corners, whereas DoG keypoints provide both position and scale information, which is valuable for matching objects that appear at different sizes.

## 3.7 The Ladder Is a Trade

The DoG detector uses three parameters: σ₀, n_scales, and k.

- σ₀ is the initial Gaussian smoothing scale.
- n_scales controls the number of scales examined.
- k determines the multiplicative increase in scale between successive levels.

Increasing the number of scales or changing the initial scale changes which image structures can be detected. Small scales are more suitable for fine details, while larger scales can detect broader structures.

The following table must be completed using the actual output from three different parameter settings.

| Experiment | σ₀ | n_scales | k | Keypoint count | Minimum σ | Maximum σ |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.6 | 10 | 1.2 | To measure | To measure | To measure |
| 2 | 0.8 | 8 | 1.3 | To measure | To measure | To measure |
| 3 | 1.0 | 12 | 1.4 | To measure | To measure | To measure |

At one end of the scale ladder, very small scales may be sensitive to fine details, noise, and pixel-level variations. At the other end, very large scales may smooth away small structures and fail to detect them reliably. The chosen parameters therefore involve a trade-off between detecting fine details and detecting larger structures.

The final conclusion should be based on the measured keypoint counts and scale ranges for the three experiments.

## 3.8 More Keypoints, Fewer Matches

In Task 4, the template is enlarged before keypoint detection using the upsample parameter. Increasing this parameter makes the template larger in pixel dimensions, but it does not necessarily increase the number of matches that survive the ratio test.

The experiment must be performed with upsample set to 1.5, 2.0, 3.0, and 4.0.

| Upsample | Template keypoints | Surviving matches |
|---:|---:|---:|
| 1.5 | To measure | To measure |
| 2.0 | To measure | To measure |
| 3.0 | To measure | To measure |
| 4.0 | To measure | To measure |

The number of surviving matches depends not only on the number of detected keypoints but also on the distinctiveness of their descriptors.

Waldo's red-and-white striped shirt contains repeated, visually similar patterns. Enlarging the template can produce additional keypoints on these repeated stripes. Their descriptors may become similar to one another, making it difficult to identify a unique corresponding descriptor.

The ratio test compares the distance to the nearest descriptor match with the distance to the second-nearest match. A match is retained only when the nearest match is sufficiently better than the second-nearest one. In this implementation, the condition is controlled by the ratio threshold.

When repeated patterns produce two similarly good candidate matches, the ratio test rejects the match. Consequently, increasing the template size may increase the number of detected keypoints without increasing the number of accepted matches. The actual trend must be established from the four experimental measurements.

## 3.9 Why Does This Task Throw the Colour Away?

Task 3 converts the image to grayscale before computing gradients and detecting keypoints. Grayscale represents each pixel using a single intensity value instead of separate red, green, and blue values.

For a grayscale image I(y, x), the horizontal and vertical gradients are calculated as:

Gₓ = ∂I/∂x

Gᵧ = ∂I/∂y

The gradient orientation is then calculated as:

θ = atan2(Gᵧ, Gₓ)

This produces a single, well-defined gradient orientation at each pixel.

For a red-to-green edge, the red channel may decrease while the green channel increases. The blue channel may remain almost unchanged. Therefore, the gradients in the individual colour channels can point in different directions or have different magnitudes.

If gradient orientation were calculated directly from a three-channel image without defining how the channel gradients should be combined, there would not be one unambiguous orientation. A method would have to specify how the three channel gradients are combined.

Converting the image to grayscale first produces one intensity gradient and therefore one orientation. This simplifies orientation estimation and makes it possible to construct consistent local descriptors for matching.

However, grayscale conversion also discards some colour information. Two regions with different colours but similar luminance may become difficult to distinguish. Thus, grayscale simplifies feature detection, while colour may provide additional information for distinguishing visually similar objects.