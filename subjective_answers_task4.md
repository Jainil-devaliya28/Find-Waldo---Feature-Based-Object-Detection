# Subjective Answers – Task 4

## 4.6 Comparison of Consensus Estimation and Median

The plain median method estimates the object centre by taking the median of all predicted centre coordinates. However, incorrect feature matches can generate outliers, which affect the estimated centre and scale.

In my experiment, the plain median method produced an IoU of **0.000 for all four scenes**: `scene_easy`, `scene_medium`, `scene_rotated`, and `scene_hard`.

This indicates that the median-based estimate did not overlap with the ground-truth bounding box in these cases. The predicted centres may have been affected by incorrect feature matches and outliers.

A consensus-based method attempts to identify a group of predictions that agree with one another. In principle, this can reduce the influence of outliers. However, the results obtained in my experiment show that the current implementation does not produce accurate bounding boxes for all scenes.

## 4.7 Comparison of Task 2 and Task 4

Task 2 uses feature matching to identify corresponding keypoints between the template and the scene. However, it does not explicitly estimate the rotation of the object.

Task 4 extends this approach by using the scale and orientation information associated with keypoints to estimate the object's centre, scale, and rotation. The orientation difference between corresponding keypoints is used to account for rotation, while the scale ratio helps estimate changes in object size.

The initial Task 4 results were:

| Scene | Matches | Votes | IoU | Agreement |
|---|---:|---:|---:|---:|
| scene_easy | 111 | 10 | 0.930 | 0.930 |
| scene_medium | 133 | 12 | 0.000 | 0.000 |
| scene_rotated | 103 | 8 | 0.000 | 0.000 |
| scene_hard | 55 | 3 | 0.000 | 0.000 |

The results show that the method performed well for `scene_easy`, obtaining an IoU of 0.930. However, it failed to produce an overlapping bounding box for the other three scenes.

Although Task 4 incorporates scale and orientation information, its performance depends on the accuracy of feature matching and centre estimation. Incorrect correspondences can produce inaccurate centre predictions and reduce localization accuracy.

Therefore, Task 4 provides additional information for handling scale and rotation, but the experimental results show that this alone does not guarantee accurate localization in every scene.

## 4.8 Effect of Mutual Nearest Neighbour (MNN) Matching

To reduce incorrect feature correspondences, I modified the matching procedure to use Mutual Nearest Neighbour (MNN) matching along with the ratio test.

The ratio test checks whether the best match is sufficiently better than the second-best match. MNN additionally requires the match to be reciprocal: the scene descriptor must select the template descriptor as its nearest neighbour, and the template descriptor must select the same scene descriptor as its nearest neighbour.

The results before and after applying MNN are shown below.

| Scene | Matches Before | Votes Before | Matches After MNN | Votes After MNN | IoU After MNN |
|---|---:|---:|---:|---:|---:|
| scene_easy | 111 | 10 | 20 | 3 | 0.000 |
| scene_medium | 133 | 12 | 20 | 4 | 0.000 |
| scene_rotated | 103 | 8 | 16 | 6 | 0.000 |
| scene_hard | 55 | 3 | 10 | 2 | 0.000 |

MNN substantially reduced the number of feature matches in every scene. For example, the number of matches in `scene_easy` decreased from 111 to 20, while the number in `scene_rotated` decreased from 103 to 16.

However, the IoU remained 0.000 for all four scenes after applying MNN. This means that reducing the number of correspondences did not improve the final bounding-box localization in this experiment.

One possible explanation is that, although MNN removes some ambiguous matches, the remaining matches may still produce inaccurate centre predictions. The number of votes may also be insufficient for the current consensus requirements.

Thus, MNN changes the set of feature correspondences, but the results indicate that further improvement in centre estimation and consensus selection would be necessary to achieve accurate localization.

## 4.9 Comparison of IoU and Agreement

Intersection over Union (IoU) measures the overlap between the predicted bounding box and the ground-truth bounding box. It is calculated as:

IoU = Area of Intersection / Area of Union

A higher IoU indicates better localization. An IoU of 0.000 means that the predicted and ground-truth bounding boxes have no overlapping area.

Agreement measures how closely the estimated result agrees with the ground-truth result according to the evaluation procedure used in the assignment.

The initial Task 4 results were:

| Scene | IoU | Agreement |
|---|---:|---:|
| scene_easy | 0.930 | 0.930 |
| scene_medium | 0.000 | 0.000 |
| scene_rotated | 0.000 | 0.000 |
| scene_hard | 0.000 | 0.000 |

The `scene_easy` result has an IoU and agreement of 0.930, indicating good localization. In contrast, the remaining scenes have zero IoU and agreement, indicating that the method did not correctly localize the object in those cases.

After applying MNN, the IoU was 0.000 for all four scenes. Therefore, the modified matching procedure did not improve the final localization results.

Overall, the experiment demonstrates that reliable feature correspondences are important for object localization. However, improving feature matching alone may not be sufficient; the centre predictions, consensus selection, and final bounding-box estimation must also be accurate.