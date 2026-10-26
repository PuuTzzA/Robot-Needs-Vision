Task Specification: Robot Butler Course Project
1. Objective
Build an autonomous mobile robot system that can perceive objects in a venue, understand written task instructions, navigate safely to relevant locations, and report accurate results.

The project has two stages:

Stereo vision: Build and evaluate a stereo-camera depth-estimation system.
Intelligent robot: Integrate vision with a mobile robot to read tasks, navigate, inspect objects, measure them, and answer questions.
2. System and operating environment
Robot: TurtleBot 4 Lite, with a movable base and 2D LiDAR.
Development platform: Python and ROS 2.
Computing: A separate Linux PC is required; Ubuntu 20.04 LTS is recommended.
Camera: The robot’s built-in camera may be used. Integrating the team’s custom stereo camera is eligible for bonus credit.
The robot must avoid obstacles and operate autonomously wherever possible.
3. Stage 1 — Stereo vision
Objective
Build a stereo depth-estimation system using the provided pair of homogeneous cameras. The project materials specify AR0330 sensors, 70° field-of-view lenses, and USB connections.

Required work
Select a camera baseline and construct a suitable camera mount.
Calibrate the stereo camera and obtain its calibration parameters.
Rectify image pairs.
Compute stereo correspondence/disparity and produce a depth map.
Test the system on objects approximately 20–50 cm from the camera.
Report depth-estimation performance, including accuracy, speed, and compactness.
A typical pipeline is: capture stereo images → rectify → match corresponding pixels → compute disparity → convert disparity to depth.

Assessment
Stage 1 is 40% of the project grade:

Camera preparation and calibration: 25%
Stereo-matching performance: 15%, divided into:
Accuracy: 5%
Speed: 5%
Compactness: 5%
The detailed depth-map criteria award up to 5% each for:

Map completeness: 0 defects / fewer than 5 / fewer than 10 defects → 5% / 3% / 1%
Output time: within 1 s / 3 s / 5 s → 5% / 3% / 1%
Depth error: less than 3 cm / 5 cm / 10 cm → 5% / 3% / 1%
The materials do not define “defect” or specify how these thresholds are treated at their boundaries; confirm the evaluation procedure with the teaching staff.

4. Stage 2 — Intelligent robot challenge
Objective
The robot must collect task instructions from the designated task center, interpret them, plan and execute an appropriate route, and report the requested results.

Task types
The challenge contains five tasks in total:

Level 1 — Exhibition inspection (“What is there?”): Go to a specified point and identify the object there.
Level 2 — Lost-item search (“Where is it?”): Find the point where a named object is located.
Level 3 — Inventory (“How many match the criteria?”): Count objects satisfying a requested condition, such as color.
For each task, the robot is expected to report the object’s type and height using depth-camera information. Responses should also address the task’s requested location, identity, or count.

Operating requirements
Visually read and interpret the task instructions; do not assume the instructions are supplied directly to the software.
Navigate to the locations relevant to the tasks and use vision to inspect objects.
Use depth information to measure object height.
Avoid obstacles and collisions.
Complete the tasks within the stated 10-minute execution limit.
Decide autonomously which tasks to perform and in what order.
Return to the starting point after completing a level to receive the next level’s tasks. The scene and corresponding map change between levels.
Scene and task-location rules
The supplementary material describes scenes with six objects placed at six predefined locations, with two locations each classified as Easy, Medium, and Difficult. It further specifies:

Level 1 targets: one randomly selected Easy location and one randomly selected Medium location.
Level 2 targets: one randomly selected Easy location and one randomly selected Difficult location.
Objects are selected from the available object categories. The robot should treat task locations as fixed, mapped points and should not assume that objects can appear elsewhere.

Assessment
Stage 2 is 60% of the project grade:

Detection, measurement, and caption accuracy: 45%
Two Level 1 tasks: 7 points each
Two Level 2 tasks: 9 points each
One Level 3 task: 13 points
Object-seeking speed: 10%, ranked by overall completion time
Collision avoidance: 5%
Bonuses:

10% for using the team’s custom stereo camera in Stage 2 instead of the robot’s built-in camera.
10% for fully autonomous operation. Two points are deducted for each manual intervention.
5. Suggested system outputs
For each completed task, record at least:

The interpreted instruction
The target location, if one is specified or found
The detected object category
The estimated object height
The Level 3 count or other requested answer
Confidence or an uncertainty indicator when the result is ambiguous
Completion time and any navigation or perception failures
