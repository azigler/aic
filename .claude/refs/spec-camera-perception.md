# Spec: Camera-Based Port Detection Pipeline

## 1. Overview

Replace blind descent with vision-guided insertion. Use the 3 wrist-mounted
cameras (1152x1024, 20Hz) to detect the target port's position, then servo
the gripper toward it for insertion.

### 1.1 Why This Is Needed
- Board position and port offsets are randomized each trial
- Hardcoded offsets don't generalize (exp-008: 79.9, worse than blind 93.4)
- The cameras exist specifically for this purpose
- CheatCode proves insertion works if you know the port position

### 1.2 Scope
- Detect SFP ports and SC ports from camera images
- Estimate port 3D position relative to TCP
- Guide approach and insertion using visual feedback
- NOT in scope: full learned policy (ACT/diffusion) -- that's a separate branch

## 2. Camera Setup

### Physical Configuration
- 3 cameras (left, center, right) on wrist mount
- Resolution: 1152 x 1024 px, 20 FPS
- Center camera: pointing roughly at the workspace below the gripper
- Left/Right: angled ±30° from center for stereo-like coverage
- Horizontal FOV: ~50°
- Near clip: 7cm

### Camera Intrinsics (from CameraInfo)
Available via `obs.center_camera_info.k` (3x3 intrinsic matrix):
- fx, fy: focal lengths in pixels
- cx, cy: principal point

### Frame Conventions
- Camera optical frame: X=right, Y=down, Z=forward (OpenCV convention)
- ROS transform from camera_link to optical frame handles this

## 3. Port Detection Approach

### 3.1 Start Simple: Color/Shape Filtering

SFP ports and SC ports have distinctive visual features:
- SFP ports: rectangular metallic openings on NIC cards
- SC ports: circular/rectangular fiber optic connectors
- Both are mounted on a colored task board

**Pipeline:**
```python
image = obs.center_image  # 1152x1024 RGB
# 1. Convert to grayscale or HSV
# 2. Apply region of interest (lower half of image = board area)
# 3. Edge detection (Canny) or color threshold
# 4. Find contours matching port shape
# 5. Compute centroid in pixel coordinates
# 6. Convert to 3D using camera intrinsics + known Z distance
```

### 3.2 Template Matching (if simple filtering fails)

```python
# Pre-captured templates of SFP and SC ports
template = cv2.imread("sfp_port_template.png")
result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
_, _, _, max_loc = cv2.minMaxLoc(result)
port_pixel = max_loc  # (px, py)
```

### 3.3 Depth Estimation

With 3 cameras, we can estimate depth:
- **Option A:** Use known Z height of the board (~0.13m for SFP, ~0.01m for SC)
  and project pixel coordinates to 3D using camera intrinsics
- **Option B:** Stereo matching between left and right cameras for true depth
- **Option C:** Use TCP Z position as proxy (we know how far down we've gone)

**Option A is simplest** and should work since the board is at a known height range.

### 3.4 Pixel to 3D Conversion

```python
# Given port at pixel (u, v) in center camera, at known depth Z_cam:
X_cam = (u - cx) * Z_cam / fx
Y_cam = (v - cy) * Z_cam / fy
# Transform from camera frame to TCP frame using known camera mount offset
# Then from TCP frame to base_link frame using controller_state.tcp_pose
```

## 4. Visual Servoing Loop

### 4.1 Architecture

```
while not inserted:
    obs = get_observation()
    image = obs.center_image

    # Detect port in image
    port_pixel = detect_port(image, task.plug_type)

    # Convert to 3D offset from TCP
    port_offset = pixel_to_3d(port_pixel, camera_info, tcp_pose)

    # Compute target TCP pose (current + offset toward port)
    target = compute_target(tcp_pose, port_offset, phase)

    # Command robot
    move_robot(target)
```

### 4.2 Phases

1. **Approach (far):** Large steps toward port center. Camera detects port.
2. **Approach (close):** Smaller steps, refined detection. Port fills more of image.
3. **Align:** Center port in camera view. Very small adjustments.
4. **Insert:** Switch to compliant control, push forward along insertion axis.
   May lose visual contact as port goes behind gripper.

## 5. Implementation Plan

### 5.1 Dependencies
- OpenCV: already available via pixi (or install via `pixi add opencv`)
- NumPy: already available
- No neural network needed for Phase 1

### 5.2 Files to Create/Modify
- `aic_example_policies/aic_example_policies/ros/VisionPolicy.py` -- main policy
- `aic_example_policies/aic_example_policies/perception.py` -- port detection module
- May need port templates (captured from sim)

### 5.3 Testing
- Save camera images during policy execution for debugging
- Overlay detection results on saved images
- Compare detected port position vs ground truth (when available)

## 6. Fallback Strategy

If camera detection is unreliable:
- Fall back to DirectApproach (blind descent) for that trial
- Use camera only for coarse XY alignment, then blind insertion
- Hybrid: camera for approach, force feedback for final insertion

## 7. Expected Scores

| Approach | SFP Trials | SC Trial | Total |
|----------|-----------|----------|-------|
| Current (blind) | 40-50 each | 1 | 81-101 |
| Camera coarse alignment | 50-60 each | 30-50 | 130-170 |
| Camera + insertion | 60-75 each | 50-75 | 170-225 |
| Camera + full insertion | 80-95 each | 70-90 | 230-280 |

## 8. Open Questions

- What do the ports actually look like in camera images? (Need to capture)
- Is OpenCV available in the pixi env? (Need to check)
- How accurate is pixel-to-3D conversion at 5-15cm range?
- Does the port remain visible as the gripper approaches? (May get occluded)
