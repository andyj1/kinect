# Azure Kinect Low Cost Motion Capture System - Progress
### Problems
- the depth information provided by the Azure Kinect is far less accurate than that from LIDAR systems
- with bundle adjustment and constraints imposed by a multi-device system, arriving at the correct (depth) positional information with the nonlinear optimization (as used in stero calibration in opencv) still causes distortion and re-projection errors, possibly introducing too many degrees of freedom
> - closed issue: [GitHub Discussion](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/issues/937)
- parallax: is this actually parallax, or algorithm estimation/2-D re-projection error? 
- 
### Notes
- Kinect (v2) does not retrieve IR and RGB streams simultaneously
- the intrinsic parameters as pre-calibrated in the Azure Kinect SDK are, by default, normalized with respect to the image width (for fx) and height (for fy)
- Kinect returns processed depth image, which is not aligned with the original infrared image
- 

### Possible solutions
- use computed (not normalized) intrinsic parameters by Zhang's method
- instead of assuming zero distortion coefficients, assume some constants
> - [3D with Kinect](https://www.researchgate.net/publication/305108995_3D_with_Kinect) Paper: "non-negligibly increased the overall accuracy of 3D measurement"
- use RGB and IR collectively to compute the relative positions in 3-D for higher accuracy
- test with other fields of view for depth camera
- analyze performance by error metrics, such as residual/projection error

### Speed optimization
- compute transformation parameters at the first
> - currently in place: calculating R and T for each subordinate device on the first frame only


### Accurately Calibrate Kinect Sensor Using Indoor Control Field
- setting control  





# General Goals
- take existing algorithms and Kinects and Vicon and compare across their 3-D positions
- hardware -- maybe some electronic components for frame sampling for Kinect / Vicon fusion
- software -- modularized functionalities that can be applied in other systems easily; starting with Kinect SDK (current version)

# TODO
- combine 1,2,3 systems into single file
- smoothen
- test the speed improvement

smoothing:
(1) - introduce another structure to hold the confidence interval
	 - keep the higher confident joint from previous frame if the current joint is not as confident
(2) - average over past 2-3 frames to reduce spikes
	 - 