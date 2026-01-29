# Hand Pose Estimation for Drone Control

# 1. Project Purpose

The goal of this project is to evaluate the performance and applicability of different hand pose estimation frameworks under varying recording conditions. Based on this evaluation, the most suitable model is selected to control a drone in a simulated environment using hand gesture recognition. The project covers the full pipeline, from data acquisition and labeling to model training, evaluation, and potential integration with a drone simulation.

# 2. Used Gestures and Their Function

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

# 3. Methodology

## 3.1 Development Phases

The project was structured into **well-defined development phases**, allowing a progressive and controlled evolution from conceptual design to a functional gesture-based drone control system. Each phase addressed a specific objective while feeding results into the next stage.

---

### Phase 1 – Research & Framework Selection

The first phase focused on understanding the problem domain and exploring existing approaches for vision-based hand gesture recognition. Different computer vision frameworks were reviewed and compared at a high level, considering factors such as:

- Robustness
- Flexibility
- Real-time performance
- Ease of integration with external systems

**Outcome:** Selection of multiple candidate frameworks to be experimentally evaluated later, rather than committing early to a single solution.

---

### Phase 2 – Dataset Design and Collection

In the second phase, a custom dataset was designed to reflect realistic and challenging conditions for hand gesture recognition. A standardized data collection protocol was defined to ensure consistency across recordings, including:

- Gesture definitions
- Lighting conditions
- Camera angles
- Background variations
- User diversity

The dataset was collaboratively collected by multiple participants, following strict naming conventions and quality checklists. This phase ensured that the data foundation of the project was **diverse, reproducible, and suitable for fair model comparison**.

---

### Phase 3 – Feature Engineering & Model Training

During this phase, the collected dataset was prepared for training across different frameworks. Data preprocessing pipelines were established, and multiple gesture recognition models were trained independently.

Rather than focusing on a single architecture, this phase emphasized **experimentation and iteration**, allowing the team to explore different representations and learning strategies while maintaining a consistent evaluation methodology.

---

### Phase 4 – Evaluation & Troubleshooting

Once initial models were trained, a systematic evaluation phase was conducted. Performance metrics such as **classification accuracy** and **confusion matrices** were used to analyze strengths and weaknesses of each approach.

Based on these results, models that did not meet the project requirements were refined or discarded. This phase was highly iterative, with insights from evaluation directly influencing model selection and design decisions.

---

### Phase 5 – Simulation & Drone Control Integration

The final phase focused on integrating the selected gesture recognition model into a drone simulation environment. Real-time gesture detection was connected to drone control commands, enabling **hands-free interaction**.

This phase validated the **end-to-end feasibility** of the system, demonstrating how gesture recognition outputs could be reliably translated into meaningful drone behaviors within a simulated environment.

---

## 3.2 Timeline and Task Management (Notion)

The entire project was planned and managed using **[Notion](https://www.notion.so/Cronograma-IP5-Hand-Pose-Drone-Project-2b866f34b39180e88f9debf834b714c0)** as the primary project management tool. Notion was used to organize tasks, track progress, and coordinate work across team members throughout all development phases.

A structured timeline was created, dividing the project into **weekly blocks** aligned with the main development phases. Each week contained clearly defined tasks and subtasks, ensuring transparency and accountability.

### Task Organization and Workflow

The project workflow followed a task-based structure that included:

- **Centralized task backlog** containing all planned activities
- **Clearly defined task states:** To Do, In Progress, and Done
- **Subtasks** for complex activities such as dataset recording, model training, and integration
- **Ownership assignment** to ensure responsibility for each deliverable

This structure allowed the team to monitor progress in real time and quickly identify bottlenecks.

### Milestones and Iterations

Each development phase was associated with concrete milestones, such as:

- Completing the dataset
- Finishing model training
- Achieving successful gesture-based drone control in simulation

Notion also supported **iterative development**: when certain approaches proved ineffective (e.g., specific models or frameworks), tasks were revisited, adjusted, or replaced. This was particularly relevant during the experimentation with multiple frameworks, where some options (such as YOLO or OpenPose) required reevaluation or were partially discarded based on performance and feasibility.

### Methodological Benefits

Using Notion provided a clear overview of the project's evolution, supported iterative decision-making, and ensured that technical experimentation remained aligned with the overall project goals. The tool enabled a balance between **structured planning and flexibility**, which was essential for an experimental, research-driven project.

---

# 4. Development Environments

This section describes the **hardware, software, and execution environments** used throughout the development of the project. Defining these environments is essential to ensure reproducibility, transparency, and a clear understanding of the constraints under which the system was developed and evaluated.

---

## 4.1 Hardware Environment

### Computing Device

All stages of the project were primarily developed and executed on a personal laptop belonging to one of the team members. This device was used as the main reference platform for data processing, model training, evaluation, and simulation.

**Hardware Specifications:**

| Component    | Specification                               |
| ------------ | ------------------------------------------- |
| Manufacturer | HP                                          |
| Processor    | AMD Ryzen 5 3500G                           |
| Memory       | 16 GB RAM                                   |
| Graphics     | Integrated GPU (no dedicated graphics card) |

This configuration reflects a **standard consumer-grade device**, aligning with the project's objective of building a system that can run on accessible, non-specialized hardware.

---

### Webcam

For real-time gesture capture and part of the dataset collection, the **integrated laptop webcam** was used. The native webcam resolution was **720p**, which was also representative of real-world usage scenarios.

For dataset diversity and robustness testing, additional videos were recorded at higher resolutions, ranging from **720p up to 4K**, using external devices such as smartphones.

---

### Physical Recording Conditions

Data collection took place in a wide range of physical environments to improve generalization:

- Private indoor spaces (homes)
- Public locations
- Controlled environments using virtual backgrounds

**Lighting conditions** varied intentionally and included:

- Natural light
- Artificial light
- Mixed lighting
- Low-light scenarios
- Challenging conditions (backlighting and flickering light)

This variability was a **deliberate design choice** to simulate real-world operating conditions.

---

## 4.2 Software Environment

### Operating System

All development, training, evaluation, and simulation tasks were conducted on **Ubuntu 22.04 LTS**, which served as the primary operating system for the project. This choice ensured compatibility with the drone simulation framework and related SDKs.

---

### Programming Language

**Python version:** `3.10`

Python was used as the main programming language for dataset processing, model training, evaluation, and integration with the drone control system.

---

### Main Libraries and Frameworks

The project relied on a set of widely adopted computer vision and machine learning libraries:

| Library          | Purpose                                                  |
| ---------------- | -------------------------------------------------------- |
| **MediaPipe**    | Hand and gesture-related representations                 |
| **OpenCV**       | Video processing and frame-level operations              |
| **NumPy**        | Numerical computations and data manipulation             |
| **scikit-learn** | Classical machine learning models and evaluation metrics |

Additionally, the **Olympe SDK** was used to interface gesture recognition outputs with the drone simulation environment.

---

### Development Tools

The following tools were used during development:

- **Visual Studio Code** – Primary code editor
- **Terminal** – Script execution and environment management
- **GitLab** – Version control and collaborative development
- **ChatGPT** – Supporting tool for ideation, debugging, and documentation assistance

---

## 4.3 Execution Environments

### Data Collection Environment

Gesture data was collected using **personal laptops and smartphones**, without any specialized capture hardware. Recordings were performed locally, using built-in cameras and standard recording applications, ensuring **accessibility and ease of replication**.

---

### Training Environment

All model training was executed **locally** on the described laptop hardware. No cloud-based services or external GPU resources were used. This constraint reinforced the project's focus on **lightweight models** and feasible deployment on consumer-grade devices.

---

### Evaluation Environment

Model evaluation, including the generation of **confusion matrices** and **performance metrics**, was carried out in the same local environment used for training. Evaluation scripts were implemented in Python and executed directly on the Ubuntu system.

---

### Simulation Environment

Drone simulation and control integration were performed locally on **Ubuntu 22.04 LTS** using **Parrot Sphinx**, a professional drone simulation framework.

The simulated drone model was the **Parrot Anafi AI**, which was controlled programmatically through Python using the **Olympe SDK**.

This environment allowed the validation of gesture-based control logic in a **safe and fully simulated setting**, without requiring physical drone hardware.

---

# 5. Dataset Design and Acquisition

This section describes the **design principles, data acquisition process, and structural organization** of the gesture dataset used in the project. The dataset was specifically designed to support fair model comparison, reproducibility, and robustness under real-world conditions.

---

## 5.1 Gesture Definition and Coding

The set of gestures used in this project was **predefined by the course instructor**, ensuring consistency across teams and alignment with the academic objectives of the assignment. A total of **10 distinct hand gestures** were considered.

Each gesture was explicitly mapped to a **single drone command**, following a one-to-one correspondence between gesture and action. This clear semantic mapping simplified interpretation and reduced ambiguity during both training and real-time control.

All gestures were performed using **one hand only**. To increase variability and robustness, gestures could be executed using either the **right or left hand**, which was explicitly accounted for in the dataset through the inclusion of left-hand recordings.

### Gesture Coding

To ensure consistency across scripts and frameworks, each gesture was assigned:

- A **human-readable label** (e.g., `thumbsUp`, `openPalm`)
- A **numeric identifier** (`pose_id`), used internally for training and evaluation

---

## 5.2 Recording Protocol

Each sample in the dataset consists of a **short video** with a maximum duration of approximately **10 seconds**. Videos were recorded following a consistent protocol to minimize uncontrolled variation.

Within each video, the target gesture was **repeated multiple times at different speeds**, allowing the dataset to capture natural variations in execution while maintaining a single gesture per sample. This approach increased temporal diversity without mixing multiple gestures in the same recording.

Videos were recorded in both **horizontal and vertical orientations**, reflecting realistic usage scenarios. Orientation differences were handled during preprocessing to ensure compatibility with the training pipelines.

---

## 5.3 Recording Conditions

To improve generalization and evaluate robustness, recordings were intentionally performed under a **wide range of conditions**. Each video was labeled with one dominant recording condition, even though minor secondary variations could naturally occur.

The dataset includes **20 predefined conditions**, covering:

### Lighting Variations

- Natural light
- Backlight
- Low light
- Flickering light

### Background Complexity

- Plain backgrounds
- Messy backgrounds
- Moving backgrounds
- Virtual backgrounds

### Camera Perspective and Framing

- Front angle
- Top angle
- Diagonal 45°
- Close-up
- Wide shot

### Video Quality and Resolution

- 4K
- 720p
- Low-quality/compressed

### User Variability

- Different individuals performing the gestures

These conditions were designed both **intentionally**, to cover edge cases, and **exploratorily**, based on realistic recording scenarios.

The `multiUser` condition specifically represents recordings performed by different individuals, increasing **inter-subject variability** and reducing overfitting to a single performer.

---

## 5.4 Dataset Structure and Storage

The dataset was organized using a **clear and reproducible directory structure**. Videos were stored in gesture-specific subdirectories, for example:

```
CollectedData/
 ├── thumbsUp/
 ├── openPalm/
 ├── point/
 └── ...
```

### Naming Convention

Each video file followed a **standardized naming convention**:

```
[gesture]_[condition].mp4
```

This naming scheme enabled automatic identification of both the gesture and the recording condition by processing scripts.

### Annotation File

In addition to the directory structure, all samples were registered in a **centralized annotation file** (`dataset.csv`), which served as the single source of truth for the dataset. Each row in this file contained exactly **six fields**:

```
path, pose_id, pose, condition, index, split
```

This design ensured compatibility across different frameworks and simplified dataset loading, filtering, and evaluation.

All dataset files and annotations were **version-controlled using GitLab**, ensuring traceability and collaborative consistency.

---

## 5.5 Train / Validation / Test Split

The dataset was split into **training, validation, and test sets** following a deterministic and reproducible strategy.

### Split Distribution

For each gesture:

| Set        | Number of Videos |
| ---------- | ---------------- |
| Training   | 14               |
| Validation | 3                |
| Test       | 3                |

The split was performed **by index**, rather than randomly, ensuring that the same samples were consistently assigned to each subset across experiments.

> ⚠️ **Important:** The split was defined **before model training**, preventing data leakage and enabling fair comparison between different models and frameworks.

---

# 6. Methodology

This section describes the methodological approach used to process gesture videos, extract features, and perform gesture classification. The methodology follows a **structured pipeline** that transforms raw video input into a discrete gesture identifier, suitable for both offline evaluation and real-time drone control.

---

## 6.1 Overall Pipeline

The overall processing pipeline follows a **sequential and modular structure**:

```
Video → Landmarks → Features → Classification → Gesture ID
```

1. **Video Processing:** Gesture videos are processed frame by frame
2. **Landmark Extraction:** For each frame, hand landmarks are extracted using a vision-based hand tracking model
3. **Feature Conversion:** Landmarks are converted into numerical feature representations
4. **Sequence Aggregation:** Features are aggregated into fixed-length sequences
5. **Classification:** A classifier maps each sequence to a single **Gesture ID**

Gesture recognition is performed at the **video level**, meaning that each video produces exactly one predicted gesture. The output of the model is directly the predicted gesture class, **without additional thresholding or post-processing**.

This pipeline was designed to be **simple, interpretable, and compatible with real-time extensions**.

---

## 6.2 Feature Extraction Strategy

Feature extraction is based **exclusively on hand landmarks**, without using full-body or arm information. This choice reduces computational complexity and focuses the model on the most relevant information for gesture recognition.

### Landmark-Based Features

For each detected hand, **21 hand landmarks** are extracted per frame. Each landmark is represented by its **3D coordinates** (x, y, z), as provided by the hand tracking model.

The resulting per-frame feature vector has a dimensionality of:

```
21 landmarks × 3 coordinates = 63 features per frame
```

These coordinates are used directly, forming the core representation of hand pose geometry.

---

### Temporal Representation

Landmarks are extracted **frame by frame** across each video. To ensure consistent input dimensions across samples, each video is represented as a **fixed-length sequence of 48 frames**.

- **If a video contains more than 48 valid frames:** Frames are uniformly sampled across the sequence
- **If fewer frames are available:** The sequence is zero-padded

This approach preserves temporal structure while maintaining compatibility with standard machine learning models.

---

### Distance, Angular, and Motion Features

**No explicit** distance-based, angular, or motion-specific features were computed. Instead, temporal information such as gesture speed and motion patterns is **implicitly encoded** in the sequence of landmark coordinates over time.

This design choice simplifies the feature pipeline while still allowing the classifier to learn dynamic gesture patterns from the temporal evolution of landmarks.

---

## 6.3 Data Augmentation

**No explicit data augmentation techniques** were applied during training.

Specifically:

- ❌ No artificial noise was added to landmarks or features
- ❌ No temporal warping (speed-up or slow-down) was performed
- ❌ No rotations or scaling were applied to either videos or landmark coordinates

Instead, dataset variability was achieved **naturally through the data collection process itself**. Variations in:

- Lighting
- Background
- Camera angle
- Resolution
- Execution speed
- Hand orientation
- Users

...provided sufficient diversity to support model generalization without synthetic augmentation.

---

## 6.4 Implementation Summary

In summary, the methodology prioritizes:

✓ **Simplicity and reproducibility**  
✓ **Landmark-based representations**  
✓ **Fixed-length temporal sequences**  
✓ **Minimal preprocessing assumptions**

This approach allows for **transparent analysis, fair model comparison**, and straightforward integration with downstream systems such as drone control.

---

# 7. Evaluated Models

This section describes the models and frameworks that were evaluated during the project, focusing on **technical feasibility, computational requirements, and integration complexity** rather than quantitative performance.

---

## 7.1 MediaPipe Hands

**MediaPipe Hands** was used as the primary framework for hand landmark extraction. The project relied on the **MediaPipe Tasks API**, using a pretrained hand landmark model without architectural modifications.

### Key Features

The framework provides **21 hand landmarks** per detected hand, each represented by **three-dimensional coordinates** (x, y, z). Only **one hand** was considered per frame, as the gesture set was explicitly designed to be single-handed. Landmark extraction was performed frame by frame, and frames where no hand was detected were intentionally discarded.

**Handedness information** (left/right classification) was **not used**. Instead, the system was designed to be **hand-agnostic**, allowing gestures to be performed with either hand without affecting classification.

### Selection Rationale

MediaPipe was selected for its:

- **Lightweight nature**
- **Stable real-time performance on CPU-only systems**
- **Seamless integration with Python-based pipelines**

---

## 7.2 YOLOv8 (Discarded)

YOLOv8 was initially explored as an alternative approach, aiming to perform both **hand detection and gesture classification** within a single deep learning pipeline.

### Experimentation

Partial training experiments were conducted using a subset of the dataset. However, when attempting to train the model on the complete dataset, the available hardware resources proved **insufficient**. Training on CPU resulted in:

- High computational cost
- Excessive training time
- Impractical iterative experimentation

### Reasons for Discarding

Additionally, the YOLOv8-based pipeline introduced **significantly more complexity** compared to the MediaPipe-based approach. This complexity affected:

- Preprocessing
- Annotation formats
- Training configuration
- Downstream integration

> ❌ Due to its **high computational demands** and **pipeline complexity**, YOLOv8 was deemed unsuitable for the project's real-time constraints and was therefore discarded.

---

## 7.3 OpenPose (Discarded)

OpenPose was considered as a potential framework for extracting hand keypoints. However, it was **never executed in practice**.

### Reasons for Discarding

The decision to discard OpenPose was based on:

- ❌ **Complex installation process**
- ❌ **Strong dependency on GPU acceleration**
- ❌ **Overly heavy architecture** relative to the project's scope

Given the project's focus on real-time performance and deployment on consumer-grade hardware, OpenPose was considered **overkill** for single-hand gesture recognition and was excluded early in the development process.

---

# 8. Model Selection Rationale

The final selection of **MediaPipe Hands** was driven by a qualitative comparison of the evaluated frameworks, emphasizing **practical constraints** rather than raw accuracy.

## Decision Factors

The most influential factors in the decision were:

| Factor                                 | Description                                                                                        |
| -------------------------------------- | -------------------------------------------------------------------------------------------------- |
| **Simplicity of the pipeline**         | MediaPipe offered a clean separation between landmark extraction and classification                |
| **Low computational requirements**     | The framework performed reliably on CPU-only hardware                                              |
| **Robustness to recording conditions** | MediaPipe handled variations in lighting, background, and camera angle effectively                 |
| **Ease of integration**                | The landmark-based representation integrated naturally with the drone simulation and control logic |

### Project Priorities

The primary objective of the system was **real-time drone control using hand gestures**, rather than offline gesture recognition. Consequently, **lightweight inference** and **low latency** were prioritized over more complex deep learning architectures.

> ✅ The decision to adopt MediaPipe was made **before drone integration**, ensuring that the selected model aligned with real-time constraints from the outset.

---

# 9. Model Training and Evaluation

## 9.1 Classifier Selection

After landmark extraction, gesture classification was performed using a **Random Forest classifier** implemented with **scikit-learn**.

### Why Random Forest?

Random Forest was selected due to:

- ✓ Its ability to handle **high-dimensional input data**
- ✓ **Robust performance** without extensive hyperparameter tuning
- ✓ **Fast training and inference on CPU**
- ✓ **Interpretability and stability** for small to medium-sized datasets

**Default hyperparameters** were used, as the primary goal was to evaluate the effectiveness of the feature representation rather than perform extensive model optimization.

---

## 9.2 Evaluation Metrics

Model performance was evaluated using **standard classification metrics**:

- **Overall accuracy**
- **Confusion matrix**

These metrics provided both a **global performance overview** and **class-level insight** into misclassifications.

> ⚠️ Evaluation was performed on the **test set**, which was defined **prior to training** to avoid data leakage and ensure fair assessment.

---

## 9.3 Observed Confusions

The most frequent confusions occurred between gestures with **similar spatial or temporal characteristics**, including:

- 🔄 **Clockwise vs. counterclockwise** circular motions
- ✋ **Open palm vs. closed fist**
- 👍 **Thumbs up vs. thumbs down**

### Root Causes

These confusions are primarily attributed to:

1. **Visual and geometric similarity** between gestures
2. **Limitations of landmark-based representations** in capturing subtle orientation differences
3. **Variability in gesture execution** across users

---

# 10. Troubleshooting

During the development of the project, several relevant issues were identified:

- At an early stage of the project, the MediaPipe model showed difficulties in correctly distinguishing between certain gestures and in identifying the direction of motion in dynamic gestures (e.g., circular movements and pointing directions). To address this issue, additional feature engineering techniques were introduced. In particular, angular features were used to capture rotational patterns and motion direction, while relative distance features between hand landmarks were implemented to better represent hand shape and finger configuration. These features significantly improved the model's ability to differentiate between similar gestures and motion directions.

- A frequent confusion between the circular rotation gestures (clockwise and counterclockwise) and the closed fist gesture was observed. To address this, the circular gesture was modified to include three raised fingers, making it more distinguishable and significantly reducing misclassification.

- The YOLOv8-based model required long training times and high computational resources. Due to these constraints, it was discarded midway through the project. Similarly, OpenPose proved to be too complex to install and configure on the available machines and was therefore also discarded. As a result, the project focused exclusively on MediaPipe, which showed faster training and better overall performance.

# 11. Camera Setup

This section describes the **camera configuration and operational constraints** used during real-time gesture recognition. The setup was designed to be **simple, reproducible, and compatible with real-time drone control** using consumer-grade hardware.

---

## 11.1 Camera Position

The system uses a **single monocular RGB camera**, accessed through OpenCV using a configurable camera index. In practice, this corresponds to the **built-in laptop webcam**.

The camera is positioned **in front of the user**, facing the upper body and hands. Frames are **horizontally flipped** to provide a mirrored view, improving user interaction and making gesture execution more intuitive.

No physical mounting constraints were imposed; instead, the setup assumes a **stable camera position** typical of a laptop placed on a desk.

---

## 11.2 Field of View

The effective field of view corresponds to the **default field of view of the laptop webcam**. No optical zoom or camera calibration was applied.

Gesture recognition relies on **MediaPipe hand detection**, which dynamically adapts to the visible hand region. As a result, the system does **not require precise framing**, as long as:

- ✓ One hand is clearly visible
- ✓ The hand remains within the camera's field of view for a sufficient number of frames

Only **one hand** is processed at any time (`num_hands = 1`), consistent with the gesture design.

---

## 11.3 Distance and Angle Constraints

The system is designed to operate under **flexible distance and angle conditions**, without strict geometric constraints.

### Distance

- ✓ The hand must be **close enough** for reliable landmark detection
- ⚠️ Excessive distance may reduce landmark precision
- ⚠️ Extremely close positioning may cause partial occlusion

**Temporal buffering** is used to accumulate a fixed-length sequence of frames before classification. Gesture prediction is only performed once the sequence buffer is full, ensuring **temporal consistency**.

### Angle

- ✓ Gestures can be performed at **slight horizontal or vertical angles**
- ✓ The system is tolerant to **moderate hand rotation**
- ⚠️ Extreme out-of-plane rotations may reduce recognition accuracy

For **pointing gestures**, direction is inferred using the **relative position between the wrist and the index finger tip**, allowing directional commands such as:

- Forward
- Backward
- Left
- Right

---

## 11.4 Limitations

Despite its flexibility, the camera setup presents several limitations:

| Limitation                   | Description                                                                                                         |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Single-camera dependency** | Depth perception relies entirely on inferred landmark depth (z-coordinate), not true stereo vision                  |
| **Lighting sensitivity**     | Very poor lighting conditions may affect hand detection reliability                                                 |
| **Occlusions**               | Partial occlusion of fingers or the wrist can lead to missing landmarks and discarded frames                        |
| **One-hand constraint**      | The system does not support bimanual gestures                                                                       |
| **Buffer latency**           | Gesture recognition occurs only after accumulating a full temporal window, introducing a small but consistent delay |

> ℹ️ These limitations are **inherent to monocular, vision-based gesture recognition systems** and were considered acceptable within the scope of real-time drone control in a simulated environment.

---

# 12. Simulation Environment

This section describes the **drone simulation environment** used to validate the gesture-based control system, as well as the set of commands supported and their semantic mapping.

---

## 12.1 Drone Simulator Description

Drone control was validated using a **professional drone simulation environment** provided by **Parrot**, executed locally on **Ubuntu 22.04**. The simulator emulates a real drone and exposes a control interface compatible with the **Olympe SDK**, allowing programmatic control via Python.

The simulated drone behaves **identically to a real platform** in terms of:

- Command structure
- State transitions
- Motion dynamics

This approach enabled **safe experimentation** without requiring physical hardware, while maintaining realism in flight behavior.

Communication with the simulated drone is established through a **network interface** (`DRONE_IP`), and all commands are issued **asynchronously** using Olympe's Python API.

---

## 12.2 Supported Commands

The simulation environment supports a **discrete set of flight commands** corresponding to high-level drone behaviors. These commands are internally translated into low-level motion primitives (PCMD, TakeOff, Landing) and executed continuously via a control loop.

### Command List

The supported actions include:

- ✈️ **Takeoff**
- 🛬 **Landing**
- 🔄 **Hover** (hold position)
- ➡️ **Linear motion:** forward, backward, left, right
- 🔃 **Yaw rotation:** left and right
- 🌀 **Orbital motion:** clockwise and counterclockwise circular flight
- 🚨 **Emergency stop / input lock**

> ℹ️ Altitude changes during takeoff and landing are handled **incrementally** to ensure smooth transitions and prevent abrupt movements.

---

## 12.3 Mapping Gestures to Actions

Each recognized gesture is mapped to a **unique internal command identifier**, which is then translated into a drone state. The mapping follows a **one-to-one correspondence**, ensuring deterministic behavior.

### Gesture-to-Action Examples

| Gesture              | Action                                        |
| -------------------- | --------------------------------------------- |
| 👍 Thumbs up         | Takeoff / ascend                              |
| 👎 Thumbs down       | Descend / land                                |
| 👉 Pointing gestures | Directional movement                          |
| 👋 Wave left / right | Yaw rotation                                  |
| 🔄 Circular gestures | Orbital motion (clockwise / counterclockwise) |
| ✋ Open palm         | Hover / stop                                  |
| ✊ Closed fist       | Lock input                                    |

### Directional Inference

**Directional pointing gestures** are further refined by computing the **relative vector between the wrist and the index finger tip**, allowing the system to infer:

- Forward
- Backward
- Left
- Right

All recognized gestures are placed into a **thread-safe queue** and processed sequentially by the drone control loop.

---

# 13. Drone–Model Integration

This section describes how gesture recognition outputs are integrated with the drone control logic, including **latency considerations** and **safety mechanisms**.

---

## 13.1 Gesture-to-Command Pipeline

The integration pipeline follows a **modular and asynchronous design**:

```
Camera Input → Gesture Recognition → Gesture Queue → State Machine → Drone Commands
```

### Pipeline Flow

1. **Gesture recognition** runs continuously using the camera feed
2. Once a valid gesture is detected with sufficient confidence, its corresponding identifier is **pushed into a shared queue**
3. A dedicated **drone loop** reads gesture identifiers from the queue and passes them to a **gesture handler**
4. The gesture handler updates the drone's **finite state machine (FSM)**
5. The FSM ensures that only **valid transitions** occur and that commands are executed consistently

Low-level motion commands are sent at a **fixed rate** through a control loop, decoupling gesture recognition from drone actuation.

---

## 13.2 Latency Considerations

Latency in the system arises primarily from **temporal gesture buffering and classification**:

- ⏱️ Gesture recognition requires a **full temporal window** (fixed-length sequence) before classification
- ⚡ Classification and state updates are **lightweight** and executed locally on CPU
- 🔄 Drone commands are issued at **regular intervals** (`CONTROL_LOOP_DT`)

Although this introduces a **small delay** between gesture execution and drone response, the latency is **predictable and stable**, which is critical for safe control.

> ✅ The system favors **robustness and consistency** over immediate responsiveness.

---

## 13.3 Safety Constraints (Stop / Hold Gesture)

Several **safety mechanisms** are implemented to prevent unintended or dangerous behavior:

### Safety Features

| Mechanism                         | Description                                                                      |
| --------------------------------- | -------------------------------------------------------------------------------- |
| ✋ **Open palm gesture**          | Immediately stops motion and forces the drone into a hover state                 |
| ✊ **Closed fist gesture**        | Locks further input, ignoring all subsequent gestures until explicitly released  |
| 🔒 **State-based constraints**    | Certain actions (e.g., landing) are only allowed when the drone is airborne      |
| 📏 **Controlled altitude limits** | Ascent and descent are bounded by predefined minimum and maximum altitude values |

> 🛡️ These constraints ensure that gesture-based control remains **predictable** and that the drone can always be brought to a **safe state** using a simple and intuitive gesture.

---

# 16. Conclusion

This project demonstrated the **feasibility of controlling a drone in a simulated environment** using vision-based hand gesture recognition on **consumer-grade hardware**. By evaluating multiple hand pose estimation frameworks and systematically comparing their practical constraints, the project was able to identify a solution that balances **accuracy, computational efficiency, and real-time performance**.

---

## Key Learnings

### Pipeline Simplicity in Real-Time Systems

One of the main learnings of this project is the **importance of pipeline simplicity** in real-time systems. Although more complex deep learning approaches such as **YOLOv8** and **OpenPose** offer powerful capabilities, their computational cost and integration complexity made them unsuitable under the given hardware constraints. In contrast, a **landmark-based approach using MediaPipe** provided a reliable and efficient foundation for gesture recognition.

### Critical Role of Dataset Design

Another important insight was the **critical role of dataset design**. Carefully controlling:

- Gesture definitions
- Recording protocols
- Environmental variability

...proved to be **as impactful as model selection itself**. The inclusion of diverse lighting conditions, backgrounds, camera angles, and users significantly improved robustness **without relying on synthetic data augmentation**.

### Value of Explicit System Constraints

Additionally, the project highlighted the **value of explicit system constraints**, such as:

- Fixed temporal windows
- Deterministic gesture-to-command mappings
- State-based control logic

These design decisions contributed to **predictable behavior**, which is essential for safe drone operation.

---

## Viability of the Approach

The results confirm that **real-time gesture-based drone control is viable** using:

- ✓ A monocular camera
- ✓ Classical machine learning techniques
- ✓ CPU-only hardware

The selected **MediaPipe-based pipeline** met the project's real-time requirements while remaining:

- Interpretable
- Reproducible
- Easy to deploy

The integration with a **professional drone simulator** further validated the end-to-end system, showing that gesture recognition outputs can be **reliably translated into meaningful drone actions** through a well-defined control architecture.

---

## Project Impact

### Human–Computer Interaction in Robotics

This project demonstrates how **human–computer interaction principles** can be effectively applied to robotics and autonomous systems. By enabling **intuitive, hands-free control** through gestures, the system provides a foundation for more natural interaction paradigms in drone operation.

### Transferability to Other Domains

Beyond drone control, the **methodology and system architecture** developed in this project are transferable to other domains, such as:

- 🤖 Robotic manipulation
- 🏠 Smart environments
- ♿ Assistive technologies

The emphasis on **modularity, safety constraints, and real-time feasibility** makes the approach particularly relevant for applications where **reliability and user trust** are critical.

---

## Final Remarks

In conclusion, the project successfully bridges **computer vision, machine learning, and real-time control**, delivering a functional and extensible gesture-based drone control system that balances **technical rigor with practical applicability**.

> ✅ The system proves that sophisticated human-drone interaction can be achieved with accessible technology, opening pathways for future research and real-world deployment.
