# Hand Pose Estimation for Drone Control

## Project Purpose

The goal of this project is to evaluate the performance and applicability of different hand pose estimation frameworks under varying recording conditions. Based on this evaluation, the most suitable model is selected to control a drone in a simulated environment using hand gesture recognition. The project covers the full pipeline, from data acquisition and labeling to model training, evaluation, and potential integration with a drone simulation.

## Used Gestures and Their Function

The selected gestures represent intuitive commands for drone control. Each gesture is mapped to a specific drone action:

- 👍 **Thumbs up**: Take off / ascend
- ✋ **Open palm**: Stop / hover
- 👉 **Point into direction**: Fly in the pointed direction
- 👋 **Wave left / Wave right**: Yaw left or right
- 🤏 **Pinch gesture**: Take a photo using the front camera
- ✨ **Circle motion (clockwise / counterclockwise)**: Rotate / orbit around the current position
- 👎 **Thumbs down**: Land / descend
- ✊ **Closed fist**: Hold position and ignore further input

A total of 10 gestures were defined, each with an official code to ensure consistency during dataset creation and processing.

## Methodology Followed

The project methodology was divided into several stages. First, each participant collected and labeled their own video dataset following a standardized naming convention and predefined recording conditions. Next, multiple gesture recognition frameworks were trained using this data.

Each framework was evaluated using quantitative metrics, mainly confusion matrices, in order to analyze classification accuracy, class confusion, and overall feasibility. Based on these results, the most suitable framework was selected for future integration with the drone simulation.

## Number of Samples, Recording Conditions, and Data Storage

For each of the 10 defined gestures, 20 short videos were recorded, each with a duration of approximately 10 seconds, resulting in a total of 200 samples.

The recordings were performed under 20 different conditions, including variations in lighting, background, camera angle, resolution, distance, and users. Examples include natural light, backlight, messy background, diagonal 45° angle, low light, and left-hand recordings.

Each video was stored in a gesture-specific subdirectory and named using the standardized format:

```
[gesture]_[condition].mp4
```

All videos were additionally registered in an annotation file with six columns (path, pose_id, pose, condition, index, split). The dataset was split into training, validation, and test sets (train, val, test), ensuring reproducibility and compatibility with the training and evaluation scripts.

## Used Models

Three different hand pose estimation frameworks were initially planned for this project:

### MediaPipe Hands (Python)
Used for hand detection and feature extraction, followed by a gesture classification model trained on the collected dataset.

### YOLOv8 (Ultralytics)
Initially implemented as a hand detector, with an additional classification layer for gesture recognition.

### OpenPose
Intended for extracting hand keypoints and training a gesture classifier based on these keypoints.

## Selected Model

After experimentation and evaluation, **MediaPipe** was chosen as the main framework for this project. It provided the best balance between training speed, implementation simplicity, and gesture classification accuracy. Additionally, MediaPipe required significantly fewer computational resources compared to YOLOv8 and OpenPose, making it more suitable for real-time applications and drone simulation integration.

To improve gesture discrimination between visually similar gestures the MediaPipe-based pipeline incorporated several complementary techniques. First, data augmentation was applied to the temporal landmark sequences to increase robustness against variations in execution. This included the addition of Gaussian noise, temporal warping, temporal shifts, random scaling, and small 2D rotations of the hand landmarks. These augmentations helped the model generalize better to different users, speeds, and recording conditions.

Second, a rich feature extraction strategy was used instead of relying solely on raw landmark coordinates. In addition to basic statistical features, the model extracted angular features that describe circular motion patterns, such as total accumulated rotation, angular velocity, direction consistency, and direction changes. These features were particularly important for distinguishing clockwise and counterclockwise circle gestures.

Furthermore, relative distance features between hand landmarks were computed by measuring pairwise Euclidean distances across joints. These distance-based features capture hand shape and finger configuration independently of absolute hand position, which improved the separation between static gestures such as closed fist, open palm, and pinch.

All extracted features were normalized using a standard scaler and classified using a Random Forest classifier, which proved effective for handling the high-dimensional feature space and heterogeneous feature types derived from temporal hand landmark data. Overall, this combination of data augmentation, motion-aware features, and distance-based hand representation significantly improved classification performance and reduced confusion between similar gestures.

## Camera

*(To be completed)*

## Simulation

*(To be completed)*

## Drone–Model Connection

*(To be completed)*

## Troubleshooting

During the development of the project, several relevant issues were identified:

- At an early stage of the project, the MediaPipe model showed difficulties in correctly distinguishing between certain gestures and in identifying the direction of motion in dynamic gestures (e.g., circular movements and pointing directions). To address this issue, additional feature engineering techniques were introduced. In particular, angular features were used to capture rotational patterns and motion direction, while relative distance features between hand landmarks were implemented to better represent hand shape and finger configuration. These features significantly improved the model's ability to differentiate between similar gestures and motion directions.

- A frequent confusion between the circular rotation gestures (clockwise and counterclockwise) and the closed fist gesture was observed. To address this, the circular gesture was modified to include three raised fingers, making it more distinguishable and significantly reducing misclassification.

- The YOLOv8-based model required long training times and high computational resources. Due to these constraints, it was discarded midway through the project. Similarly, OpenPose proved to be too complex to install and configure on the available machines and was therefore also discarded. As a result, the project focused exclusively on MediaPipe, which showed faster training and better overall performance.