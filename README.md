# Integrated CPS Project "IP5 - Performance of pose estimation frameworks"

You signed in to the project "IP5 - Performance of pose estimation frameworks". I will supervise and grade your work.

Therefore i have set up a gitlab project (https://git.unileoben.ac.at/p2575021/integrated-cps-project-ip5-performance-of-pose-estimation-frameworks) and i will add you as a collaborator as soon as you send me your student number (p*)

## Contact 

It is the best if we share all information and questions about the project as issue in the gitlab project (https://git.unileoben.ac.at/p2575021/integrated-cps-project-ip5-performance-of-pose-estimation-frameworks/-/issues). We will use it as a kind of forum, so everybody has the same information.

If you want to discuss something or meet me in person please write me an email 2 oer 3 days before because I am not everyday in my office in the university.

My email is `guenther.hutter@unileoben.ac.at` and i am usually located in the "Haus der Digitalisierung", Office in room 209. My response time is usually between 8 and 48 hours.


## Project context

The goal of the project is to evaluate the performance and applicability of different hand pose estimation frameworks under varying conditions and then to use this information to control a (simulated) drone.

### Hand pose estimation
In this context, varying conditions means all the real-world factors that could influence how well a hand-pose estimation model performs — especially those that are not part of the gesture itself, but part of the environment, camera, or user. For example:

Typical varying conditions you would deliberately test

* Lighting
  * well-lit vs. dim light
  * strong shadows / backlight / flickering neon / sunlight

* Background complexity
  * plain wall vs. messy lab vs. moving background

* Camera distance
  * full upper body in frame vs. only hand close-up

* Camera angle & perspective
  * frontal vs. from above vs. 45° diagonal

* Camera quality
  * high-resolution DSLR vs. cheap laptop webcam vs. smartphone front cam

* Video quality / compression / framerate
  * 4K vs. 720p vs. heavily compressed WhatsApp-style blur

* User variation
  * different hand sizes, skin tones, accessories (rings, sleeves, gloves)
  * left vs. right hand

* Motion speed
 * slow controlled movement vs. fast “realistic” motion


### Drone simulation

A professional drone simulation framework is the one from Parrot (https://developer.parrot.com/docs/sphinx/index.html) - It allows you to fly a virtual drone in a virtual environment which can be set up under `Ubuntu 22.04`.

You can fly around an Anafi AI drone (https://www.parrot.com/en/drones/anafi-ai) by controlling it via an application (QDroneControl), or by controlling it via python code by utilizing the Olympe SDK (https://developer.parrot.com/docs/olympe/index.html) 

A good ressource for developer questions is the parrot forum (https://forum.developer.parrot.com/c/anafi-ai/42)


## Deliverables

First of all - log in into gitlab and join the project. There everyone of you should find a directory for each subtask which is described here:


### Collected Data (30 Points)

Everyone of you will need to collect and label his own datasets. Therefore you will train the following poses under varying conditions:

* 👍 **Thumbs up** - Take off / ascend
* ✋ **Open palm** - stop / hover	Immediately hold position / freeze
* 👉 **Point into direction** - Fly in the pointed direction
* 👋 **Wave left/right** - Yaw to left / right
* 🤏 **Pinch gesture** - Make a photo of the front camera
* ✨ **Circle motion with hand** - Rotate / orbit around current position
* 👎 **Thumbs down** - Land / descend
* ✊ **Closed fist** - Hold and ignore further input until the 

Record at least ___20 short videos for each pose___ (5–10 seconds each) using any available device (smartphone, laptop webcam, meta quest 3, etc.). Each video should clearly show exactly one hand gesture or hand pose, and should be stored in the corresponding subdirectory.

Hint: You can also use our lab to create these conditions (bright vs. dark, different backgrounds, ...) 


### Trained Models (30 Points)

After you have finished the data aquisition, the group needs to select **three different hand pose frameworks** and **train them on the generated data** to recognize the 10 different gestures. 

Here are some suggestions - if you find better ones you are free to use yours:

* MediaPipe Hands (Python) → install and run in 3 minutes
* OpenPose Python API (if GPU available) → heavy but very powerful
* YOLOv8 (Ultralytics) Hand Detection + add classification layer → nice mix of detection + ML training

Compare the frameworks by computing and plotting confusion matrices per framework AND cross-framework comparison.

Therefore craft small python examples that allow you to get these metrics by just calling it. You can use any frameworks and helpers (like AI coding assistants) for that - as long as you understand the code and results that are produced there.

Finally **write a short sumamry** in the `/TrainedModels/README.md` folder that summarizes your results and findings across all models. This should include all different confusion matrices from all frameworks, as well as the overall comparison.


### Drone Simulation (30 Points)

The goal of this subtask is to set up the drone simulator and use the best performing framework to control the virtual drone to perform the actions that correspond with the detected poses.
To make it more realistic you can choose to use a webcam as input device  instead of the pre recorded videos to control the drone.

Be aware and plan some time for this because this can be tricky to set up and the hardware requirements / needed software skills are high - so we will most likely need do coordinate on this specific topic.


## Grading

The final grade will be the sum of the project reults and the final presentation.

* **80 points** for the project results (code, effort, team management, sumamry, ...). The individual effort will be measured by the git commit history and the commited content. 
* **20 points** for the final presentation (20 min per group + 5 min. questions).
* **up to 20 bonus points** for extraordinary efforts



| Cumulative Points | Final Grade |
|-------------------|-------------|
| 0 – 49.9          | 5           |
| 50 – 65.9         | 4           |
| 66 – 79.9         | 3           |
| 80 – 91.9         | 2           |
| 92 – 100          | 1           |

