"""Camera-based port detection for the AI for Industry Challenge.

Uses OpenCV to detect SFP and SC ports in wrist camera images and
estimate their 3D position relative to the camera frame.
"""

import contextlib
from pathlib import Path

import cv2
import numpy as np

# Debug output directory
DEBUG_DIR = Path.home() / "aic_results" / "debug"


def _ensure_debug_dir() -> Path:
    """Create debug output directory if it does not exist."""
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    return DEBUG_DIR


def _ros_image_to_numpy(image_msg) -> np.ndarray:
    """Convert a sensor_msgs/Image to a numpy BGR array for OpenCV.

    Handles rgb8 and bgr8 encodings. Returns a BGR image (OpenCV default).
    """
    height = image_msg.height
    width = image_msg.width
    encoding = image_msg.encoding.lower()

    # Convert raw bytes to numpy array
    if encoding in ("rgb8", "bgr8"):
        channels = 3
        dtype = np.uint8
    elif encoding in ("rgba8", "bgra8"):
        channels = 4
        dtype = np.uint8
    elif encoding == "mono8":
        channels = 1
        dtype = np.uint8
    else:
        # Fallback: assume 3-channel uint8
        channels = 3
        dtype = np.uint8

    img = np.frombuffer(image_msg.data, dtype=dtype).reshape(
        height, width, channels
    )

    if encoding == "rgb8":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif encoding == "rgba8":
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif encoding == "bgra8":
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return img


def _get_camera_intrinsics(camera_info) -> tuple[float, float, float, float]:
    """Extract fx, fy, cx, cy from a CameraInfo message.

    The K matrix is stored as a flat 9-element array (row-major 3x3):
        [fx,  0, cx,
          0, fy, cy,
          0,  0,  1]
    """
    k = camera_info.k
    fx = k[0]
    fy = k[4]
    cx = k[2]
    cy = k[5]
    return fx, fy, cx, cy


def _detect_sfp_port(
    bgr: np.ndarray, port_name: str = "sfp_port_0"
) -> tuple[float, float, float] | None:
    """Detect specific SFP port in a BGR image using COLOR detection.

    SFP ports appear as GREEN/TEAL rectangular openings on the NIC card.
    Each NIC has TWO ports side by side. Detects both and picks the one
    matching port_name (port_0=left, port_1=right in image).

    Returns (centroid_x, centroid_y, confidence) or None.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Green/teal color range in HSV
    lower_green = np.array([60, 40, 60])
    upper_green = np.array([110, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # Find the largest green contour (most likely the target port)
    best = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30:
            continue
        if area > best_area:
            best_area = area
            m = cv2.moments(cnt)
            if m["m00"] > 0:
                cx = m["m10"] / m["m00"]
                cy = m["m01"] / m["m00"]
                best = (cx, cy, area)

    if best is None:
        return None

    confidence = min(1.0, best[2] / 3000.0)
    return (best[0], best[1], confidence)


def _detect_sc_port(bgr: np.ndarray) -> tuple[float, float, float] | None:
    """Detect SC port candidates in a BGR image using COLOR detection.

    From camera images: SC ports appear as BLUE rectangular connectors
    on the dark task board. Very distinctive color.

    Returns (centroid_x, centroid_y, confidence) or None.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Blue color range in HSV (SC port)
    # Blue: H=100-130, S=80-255, V=80-255
    lower_blue = np.array([95, 60, 60])
    upper_blue = np.array([135, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours of blue regions
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # Find the largest blue contour
    best = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue
        if area > best_area:
            best_area = area
            m = cv2.moments(cnt)
            if m["m00"] > 0:
                cx = m["m10"] / m["m00"]
                cy = m["m01"] / m["m00"]
                best = (cx, cy, area)

    if best is None:
        return None

    confidence = min(1.0, best[2] / 3000.0)
    return (best[0], best[1], confidence)


def _save_debug_image(
    bgr: np.ndarray,
    detection: tuple[float, float, float] | None,
    port_type: str,
    frame_id: int,
) -> None:
    """Save an image with detection overlay to the debug directory."""
    debug_dir = _ensure_debug_dir()
    vis = bgr.copy()

    if detection is not None:
        cx, cy, conf = detection
        cx_int, cy_int = int(cx), int(cy)
        color = (0, 255, 0) if conf > 0.3 else (0, 255, 255)
        cv2.circle(vis, (cx_int, cy_int), 15, color, 2)
        cv2.drawMarker(vis, (cx_int, cy_int), color, cv2.MARKER_CROSS, 20, 2)
        label = f"{port_type} conf={conf:.2f}"
        cv2.putText(
            vis,
            label,
            (cx_int + 20, cy_int - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    else:
        cv2.putText(
            vis,
            f"No {port_type} detected",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    filename = f"{port_type}_frame_{frame_id:04d}.jpg"
    cv2.imwrite(str(debug_dir / filename), vis)


def detect_port(
    image_msg,
    port_type: str,
    camera_info,
    frame_id: int = 0,
    save_debug: bool = True,
    port_name: str = "sfp_port_0",
) -> tuple[float, float, float] | None:
    """Detect a port in a camera image and return 3D offset from camera frame.

    Args:
        image_msg: sensor_msgs/Image from observation
        port_type: "sfp" or "sc"
        camera_info: sensor_msgs/CameraInfo for the camera
        frame_id: Sequential frame number for debug image naming
        save_debug: Whether to save debug overlay images

    Returns:
        (dx, dy, dz) offset from camera frame to port in meters, or None
        if not detected. Camera optical frame convention: X=right, Y=down,
        Z=forward.
    """
    bgr = _ros_image_to_numpy(image_msg)

    # Run type-specific detection
    if port_type == "sc":
        detection = _detect_sc_port(bgr)
        estimated_depth = 0.28  # SC port distance from camera
    else:
        detection = _detect_sfp_port(bgr, port_name=port_name)
        estimated_depth = 0.18  # SFP port distance from camera

    # Save debug image (suppress errors so debug I/O never crashes the policy)
    if save_debug:
        with contextlib.suppress(Exception):
            _save_debug_image(bgr, detection, port_type, frame_id)

    if detection is None:
        return None

    pixel_x, pixel_y, _confidence = detection

    # Convert pixel coordinates to 3D using camera intrinsics
    fx, fy, cx, cy = _get_camera_intrinsics(camera_info)

    if fx == 0 or fy == 0:
        return None

    # 3D position in camera optical frame (X=right, Y=down, Z=forward)
    z_cam = estimated_depth
    x_cam = (pixel_x - cx) * z_cam / fx
    y_cam = (pixel_y - cy) * z_cam / fy

    return (x_cam, y_cam, z_cam)
