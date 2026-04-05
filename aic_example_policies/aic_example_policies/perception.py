"""Camera-based port detection for the AI for Industry Challenge.

Uses OpenCV to detect SFP and SC ports in wrist camera images and
estimate their 3D position relative to the camera frame.
"""

import contextlib
import math
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


def _detect_sfp_port(bgr: np.ndarray) -> tuple[float, float, float] | None:
    """Detect SFP port candidates in a BGR image.

    SFP ports are rectangular metallic openings on NIC cards. They appear
    as dark rectangular regions with strong edges. We look in the lower
    portion of the image where the task board is expected.

    Returns (centroid_x, centroid_y, confidence) or None.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Focus on the lower 70% of the image (board area during approach)
    roi_top = int(h * 0.2)
    roi = gray[roi_top:, :]

    # Enhance contrast via CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(roi)

    # Edge detection
    edges = cv2.Canny(enhanced, 50, 150)

    # Dilate to close gaps in edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best = None
    best_score = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # SFP port should be a moderate-sized rectangle in the image
        # At ~18cm distance in a 1152x1024 image, the port is roughly 40-200px wide
        if area < 200 or area > 50000:
            continue

        # Fit a rotated rectangle
        rect = cv2.minAreaRect(cnt)
        rect_w, rect_h = rect[1]
        if rect_w == 0 or rect_h == 0:
            continue

        # SFP ports are roughly 2:1 to 4:1 aspect ratio
        aspect = max(rect_w, rect_h) / min(rect_w, rect_h)
        if aspect < 1.3 or aspect > 6.0:
            continue

        # Rectangularity: how much of the bounding rect is filled
        rect_area = rect_w * rect_h
        rectangularity = area / rect_area if rect_area > 0 else 0
        if rectangularity < 0.4:
            continue

        # Score: prefer larger, more rectangular contours in center of image
        cx_cnt = rect[0][0]
        cy_cnt = rect[0][1] + roi_top  # Adjust back to full image coords
        center_dist = abs(cx_cnt - w / 2) / w + abs(cy_cnt - h / 2) / h
        score = area * rectangularity * (1.0 - 0.3 * center_dist)

        if score > best_score:
            best_score = score
            best = (cx_cnt, cy_cnt, score)

    if best is None:
        return None

    # Normalize confidence to 0-1 range (rough scaling)
    confidence = min(1.0, best[2] / 10000.0)
    return (best[0], best[1], confidence)


def _detect_sc_port(bgr: np.ndarray) -> tuple[float, float, float] | None:
    """Detect SC port candidates in a BGR image.

    SC ports are circular/cylindrical fiber optic connectors on the patch
    panel. We look for circular features or small rectangular openings.

    Returns (centroid_x, centroid_y, confidence) or None.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Focus on the lower 70% of the image
    roi_top = int(h * 0.2)
    roi = gray[roi_top:, :]

    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(roi)

    # Try circle detection first (Hough circles)
    blurred = cv2.GaussianBlur(enhanced, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=100,
        param2=40,
        minRadius=8,
        maxRadius=80,
    )

    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        # Pick the circle closest to center of image
        best_circle = None
        best_dist = float("inf")
        for cx_c, cy_c, r in circles:
            cy_full = cy_c + roi_top
            dist = math.sqrt((cx_c - w / 2) ** 2 + (cy_full - h / 2) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_circle = (float(cx_c), float(cy_full), float(r))

        if best_circle is not None:
            confidence = min(
                1.0, best_circle[2] / 40.0
            )  # Larger radius = more confident
            return (best_circle[0], best_circle[1], confidence)

    # Fallback: use edge/contour approach similar to SFP but with different params
    edges = cv2.Canny(enhanced, 60, 160)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best = None
    best_score = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100 or area > 30000:
            continue

        # SC connectors are closer to square/circular than SFP
        rect = cv2.minAreaRect(cnt)
        rect_w, rect_h = rect[1]
        if rect_w == 0 or rect_h == 0:
            continue

        aspect = max(rect_w, rect_h) / min(rect_w, rect_h)
        if aspect > 3.0:
            continue

        # Circularity
        perimeter = cv2.arcLength(cnt, True)
        circularity = (
            4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        )

        cx_cnt = rect[0][0]
        cy_cnt = rect[0][1] + roi_top
        center_dist = abs(cx_cnt - w / 2) / w + abs(cy_cnt - h / 2) / h
        score = area * (0.5 + circularity) * (1.0 - 0.3 * center_dist)

        if score > best_score:
            best_score = score
            best = (cx_cnt, cy_cnt, score)

    if best is None:
        return None

    confidence = min(1.0, best[2] / 8000.0)
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
        estimated_depth = 0.28  # SC port is ~28cm from camera during approach
    else:
        detection = _detect_sfp_port(bgr)
        estimated_depth = 0.18  # SFP port is ~18cm from camera during approach

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
