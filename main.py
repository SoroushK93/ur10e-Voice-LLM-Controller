import openai
import json
import os
import time
import numpy as np
import cv2
import pyrealsense2 as rs
import cv2.aruco as aruco
from URBasic import robotModel, urScriptExt
import threading
import speech_recognition as sr
import pyaudio
import tkinter as tk
from tkinter import ttk
from SupportCodes import GripperFunctions
import math

# =================================================================================
# PART 1: CONFIGURATION & INITIALIZATION
# =================================================================================

# --- OpenAI Configuration ---
try:
    # It's recommended to set the API key as an environment variable
    if "OPENAI_API_KEY" not in os.environ:
        # Replace with your key if you must hardcode, but environment variable is safer
        os.environ[
            "OPENAI_API_KEY"] = ""

    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "YOUR_API_KEY_HERE":
        print("ERROR: OPENAI_API_KEY is not set.")
        print("Please set the OPENAI_API_KEY environment variable or replace the placeholder in the script.")
        exit()

    client = openai.OpenAI()
    LLM_MODEL = "gpt-4o"
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    exit()

# --- Robot Configuration ---
ROBOT_IP = '192.168.1.10'
RTDE_CONF_FILE = 'URBasic/rtdeConfigurationDefault.xml'
TCP_ORIENTATION = np.array([0.0, 3.14, 0.0], dtype=np.float32)
DROP_OFF_LOCATIONS = {
    "right": [0.3605, 0.0602, 0.0002, 0.0, 3.14, 0.0],
    "left": [.55634, -.18028, .00187, 0.0, 3.14, 0.0],
    "center": [.55784, .0602, .00121, 0.0, 3.14, 0.0]
}

# --- Safety Configuration ---
SAFE_ZONE_X = [-0.60, 0.575]  # Min and Max X in meters
SAFE_ZONE_Y = [-0.60, 0.60]  # Min and Max Y in meters
SAFE_ZONE_Z = [0.002, 1.0]  # Min and Max Z in meters

# --- RealSense & ArUco Configuration ---
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
ARUCO_PARAMS = aruco.DetectorParameters()
SHRINK_RATIO = 0.82
SELECTED_AOI_IDS = [0, 2, 4, 6]
minimum_size = 200
SHAPE_RECOGNITION_FRAMES = 7  # Number of frames to average for shape detection
CLASSIFICATION_REFRESH_RATE = SHAPE_RECOGNITION_FRAMES/30  # Frames/FPS

# --- Computer Vision & Calibration Data ---
ROBOT_POINTS_CALIB = np.array([
    [.55784, .0602, .00121], [.55634, -.18028, .00187], [.55844, -.41785, .0028],
    [.36157, -.41903, .00116], [.16667, -.42055, .001]
], dtype=np.float32)  # Position of ArUco markers in robot's space

# Color definitions for object detection in HSV
COLOR_RANGES = {
    "white": [(np.array([0, 0, 125]), np.array([180, 33, 255]))],
    "red": [(np.array([125, 100, 0]), np.array([180, 255, 255]))],
    "green": [(np.array([35, 10, 100]), np.array([85, 230, 255]))]
}
# BGR colors for drawing bounding boxes
COLOR_BGR = {
    "white": (200, 200, 200),
    "red": (0, 0, 255),
    "green": (0, 255, 0)
}

home_joints = np.array(
    [np.radians(24.12), np.radians(-80.22), np.radians(-130.89), np.radians(-58.88), np.radians(90.12),
     np.radians(114.33)])  # Joint angles of Robot's home position above marker 0

# --- Global Variables ---
camera_to_robot_transform = None
keep_running = True
aoi_polygon_pts = None
aoi_centroid_robot_coords = None
OBJECT_CACHE = []
STACK_CACHE = {}
HELD_OBJECT = None
DISPLAY_OBJECTS = []
display_lock = threading.Lock()


# =================================================================================
# PART 2: CORE HELPER FUNCTIONS (Calibration and Vision)
# =================================================================================

def detect_shape(contour):
    """
    Identifies the shape of a given contour and classifies it as 'cylinder' or 'block'.
    """
    shape = "unidentified"
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04*peri, True)
    num_vertices = len(approx)

    if num_vertices == 3:
        shape = "block"  # Triangles are treated as blocks
    elif num_vertices == 4:
        shape = "block"
    elif num_vertices > 4:
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = float(area)/hull_area
            if solidity > 0.885:
                shape = "cylinder"
            else:
                shape = "block"
    return shape


def get_mode(data_list):
    """Helper function to return the most common element in a list."""
    if not data_list:
        return None
    return max(set(data_list), key=data_list.count)


def compute_kabsch_transform(A, B):
    """Computes the optimal rotation R and translation t for transforming points A to B."""
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    A_centered = A - centroid_A
    B_centered = B - centroid_B
    H = A_centered.T@B_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T@U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T@U.T
    t = centroid_B - R@centroid_A
    return R, t


def setup_calibration(pipeline, robot):
    """Uses ArUco markers to compute the camera-to-robot transformation."""
    global camera_to_robot_transform, keep_running
    move_home(robot)
    GripperFunctions.activate_gripper(robot)
    print("Attempting to establish camera-to-robot transform...")
    print("Please ensure ArUco markers 0, 1, 2, 3, and 4 are visible to the camera.")
    align = rs.align(rs.stream.color)
    while camera_to_robot_transform is None and keep_running:
        try:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
        except RuntimeError as e:
            print(f"Could not get frames, check camera connection: {e}")
            time.sleep(1)
            continue
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        color_image = np.asanyarray(color_frame.get_data())
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray_image, ARUCO_DICT, parameters=ARUCO_PARAMS)
        if ids is not None and len(ids) >= 5:
            camera_points, robot_points_subset = [], []
            intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            for i in range(5):
                if i in ids:
                    idx = np.where(ids.flatten() == i)[0][0]
                    center_pixel = np.mean(corners[idx][0], axis=0).astype(int)
                    depth = depth_frame.get_distance(center_pixel[0], center_pixel[1])
                    if depth > 0:
                        camera_points.append(rs.rs2_deproject_pixel_to_point(intrinsics, center_pixel, depth))
                        robot_points_subset.append(ROBOT_POINTS_CALIB[i])
            if len(camera_points) >= 4:
                R, t = compute_kabsch_transform(np.array(camera_points), np.array(robot_points_subset))
                camera_to_robot_transform = (R, t)
                print("\n✅ Camera-to-Robot Transform Established Successfully!\n")
                time.sleep(1)
                break
        cv2.imshow("Calibration View", color_image)
        if cv2.waitKey(1) & 0xFF == 27:
            print("Calibration cancelled by user.")
            keep_running = False


def find_multiple_objects(pipeline):
    """
    Finds ALL objects, stabilizing shape detection by averaging over multiple frames.
    This function is for the ROBOT's logic, not the display.
    """
    global aoi_polygon_pts
    if camera_to_robot_transform is None or aoi_polygon_pts is None:
        return []

    align = rs.align(rs.stream.color)
    R, t = camera_to_robot_transform

    potential_objects = {}
    grid_size = 0.05

    for _ in range(SHAPE_RECOGNITION_FRAMES):
        if not keep_running: break
        try:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
        except Exception as e:
            print(f"Warning: Skipping a frame due to error: {e}")
            continue

        color_image = np.asanyarray(color_frame.get_data())
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        roi_mask = np.zeros(color_image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(roi_mask, [aoi_polygon_pts], 255)
        hsv_masked = cv2.bitwise_and(hsv_image, hsv_image, mask=roi_mask)
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        for color_name, hsv_ranges in COLOR_RANGES.items():
            color_mask = cv2.inRange(hsv_masked, hsv_ranges[0][0], hsv_ranges[0][1])
            for i in range(1, len(hsv_ranges)):
                color_mask |= cv2.inRange(hsv_masked, hsv_ranges[i][0], hsv_ranges[i][1])

            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > minimum_size:
                    M = cv2.moments(cnt)
                    if M["m00"] == 0: continue
                    cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                    depth = depth_frame.get_distance(cx, cy)
                    if depth > 0:
                        camera_point = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth)
                        robot_pos = R@np.array(camera_point) + t
                        grid_key = (int(robot_pos[0]/grid_size), int(robot_pos[1]/grid_size), color_name)
                        shape = detect_shape(cnt)
                        if grid_key not in potential_objects:
                            potential_objects[grid_key] = {
                                "positions": [], "areas": [], "center_pixels": [], "shapes": [], "color": color_name
                            }
                        potential_objects[grid_key]["positions"].append(robot_pos)
                        potential_objects[grid_key]["areas"].append(area)
                        potential_objects[grid_key]["center_pixels"].append((cx, cy))
                        potential_objects[grid_key]["shapes"].append(shape)
        time.sleep(0.05)

    stable_objects = []
    for group in potential_objects.values():
        if len(group["shapes"]) > SHAPE_RECOGNITION_FRAMES/2:
            stable_shape = get_mode(group["shapes"])
            avg_position = np.mean(group["positions"], axis=0)
            avg_area = np.mean(group["areas"])
            avg_center_pixel = tuple(np.mean(group["center_pixels"], axis=0).astype(int))
            stable_objects.append({
                "position": avg_position, "area": avg_area, "center_pixel": avg_center_pixel,
                "shape": stable_shape, "color": group["color"]
            })
    return stable_objects


def periodic_detection_thread(pipeline):
    """
    Runs in the background to periodically update the list of detected objects.
    """
    global DISPLAY_OBJECTS, keep_running
    print("Starting periodic object detection thread...")
    while keep_running:
        if aoi_polygon_pts is not None and camera_to_robot_transform is not None:
            detected_objects = find_multiple_objects(pipeline)
            with display_lock:
                DISPLAY_OBJECTS = detected_objects
        time.sleep(CLASSIFICATION_REFRESH_RATE)
    print("Periodic object detection thread stopped.")


def live_vision_thread(pipeline):
    """
    Continuously fetches camera frames and annotates them with STABLE object data.
    """
    global keep_running, camera_to_robot_transform, aoi_polygon_pts, aoi_centroid_robot_coords, DISPLAY_OBJECTS
    print("Live vision feed started. Press ESC in the video window to quit.")
    print("Waiting for AOI markers to be detected...")
    align = rs.align(rs.stream.color)

    while keep_running:
        try:
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame: continue
            color_image = np.asanyarray(color_frame.get_data())
        except RuntimeError:
            print("Timeout waiting for frames. Check camera connection.")
            continue

        if aoi_polygon_pts is None:
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco.detectMarkers(gray_image, ARUCO_DICT, parameters=ARUCO_PARAMS)
            if ids is not None:
                id_to_center = {marker_id: corner[0].mean(axis=0).astype(int)
                                for corner, marker_id in zip(corners, ids.flatten())
                                if marker_id in SELECTED_AOI_IDS}
                if all(id in id_to_center for id in SELECTED_AOI_IDS):
                    sorted_points = np.array([id_to_center[i] for i in sorted(SELECTED_AOI_IDS)], dtype=np.float32)
                    centroid = np.mean(sorted_points, axis=0)
                    shrunk_pts = centroid + SHRINK_RATIO*(sorted_points - centroid)
                    aoi_polygon_pts = np.array(shrunk_pts, dtype=np.int32)
                    print("✅ Area of Interest has been locked.")
                    if camera_to_robot_transform is not None:
                        try:
                            intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
                            R, t = camera_to_robot_transform
                            aoi_centroid_pixel = np.mean(aoi_polygon_pts, axis=0).astype(int)
                            depth = depth_frame.get_distance(aoi_centroid_pixel[0], aoi_centroid_pixel[1])
                            if depth > 0:
                                camera_point = rs.rs2_deproject_pixel_to_point(intrinsics, aoi_centroid_pixel, depth)
                                aoi_centroid_robot_coords = R@np.array(camera_point) + t
                                print(f"✅ AOI centroid in robot coords: {np.round(aoi_centroid_robot_coords, 3)}")
                            else:
                                print("Warning: Could not get depth for AOI centroid.")
                        except Exception as e:
                            print(f"Error calculating AOI centroid in robot coordinates: {e}")

        display_image = color_image.copy()
        if aoi_polygon_pts is not None:
            cv2.polylines(display_image, [aoi_polygon_pts], isClosed=True, color=(255, 0, 0), thickness=2)
            roi_mask = np.zeros(color_image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(roi_mask, [aoi_polygon_pts], 255)

            with display_lock:
                stable_objects = list(DISPLAY_OBJECTS)

            hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            hsv_masked = cv2.bitwise_and(hsv_image, hsv_image, mask=roi_mask)

            all_contours = []
            for color_name, hsv_ranges in COLOR_RANGES.items():
                color_mask = cv2.inRange(hsv_masked, hsv_ranges[0][0], hsv_ranges[0][1])
                for i in range(1, len(hsv_ranges)):
                    color_mask |= cv2.inRange(hsv_masked, hsv_ranges[i][0], hsv_ranges[i][1])
                contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if cv2.contourArea(cnt) > minimum_size:
                        all_contours.append(cnt)

            for cnt in all_contours:
                M = cv2.moments(cnt)
                if M["m00"] == 0: continue
                cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])

                closest_stable_obj = None
                min_dist = float('inf')
                for stable_obj in stable_objects:
                    dist = math.hypot(cx - stable_obj['center_pixel'][0], cy - stable_obj['center_pixel'][1])
                    if dist < min_dist:
                        min_dist = dist
                        closest_stable_obj = stable_obj

                if closest_stable_obj and min_dist < 35:
                    stable_shape = closest_stable_obj['shape']
                    stable_color = closest_stable_obj['color']
                    box_color = COLOR_BGR.get(stable_color, (0, 0, 0))

                    if stable_shape == 'cylinder':
                        (x, y), radius = cv2.minEnclosingCircle(cnt)
                        center = (int(x), int(y))
                        radius = int(radius)
                        cv2.circle(display_image, center, radius, box_color, 2)
                        label_text = f"{stable_color} {stable_shape}"
                        cv2.putText(display_image, label_text, (center[0], center[1] + radius + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    else:  # block
                        x, y, w, h = cv2.boundingRect(cnt)
                        cv2.rectangle(display_image, (x, y), (x + w, y + h), box_color, 2)
                        label_text = f"{stable_color} {stable_shape}"
                        cv2.putText(display_image, label_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 255, 255), 2)

        cv2.imshow("Live Robot View", display_image)
        if cv2.waitKey(1) & 0xFF == 27:
            keep_running = False


# =================================================================================
# PART 3: ROBOT ACTION FUNCTIONS (with Safety Checks, Hover & Caching)
# =================================================================================

def is_pose_safe(pose):
    """Checks if a target pose is within the defined SAFE_ZONE."""
    x, y, z = pose[0], pose[1], pose[2]
    if not (SAFE_ZONE_X[0] <= x <= SAFE_ZONE_X[1]):
        return False, f"X-coordinate {x:.3f} is outside the safe zone {SAFE_ZONE_X}"
    if not (SAFE_ZONE_Y[0] <= y <= SAFE_ZONE_Y[1]):
        return False, f"Y-coordinate {y:.3f} is outside the safe zone {SAFE_ZONE_Y}"
    if not (SAFE_ZONE_Z[0] <= z <= SAFE_ZONE_Z[1]):
        return False, f"Z-coordinate {z:.3f} is outside the safe zone {SAFE_ZONE_Z}"
    return True, "Pose is within safe limits."


def safe_move_to_pose(robot, pose, speed=0.15, acceleration=0.25):
    """Checks pose safety before moving the robot."""
    is_safe, message = is_pose_safe(pose)
    if is_safe:
        robot.movel(pose=pose, a=acceleration, v=speed)
        return f"Move executed to {np.round(pose[:3], 3)}."
    else:
        error_message = f"Error: Safety violation. {message}. Move cancelled."
        print(error_message)
        return error_message


def move_home(robot):
    """Moves the robot to its home position."""
    print("Moving robot to home position to ensure clear view...")
    robot.movej(q=home_joints, a=0.5, v=0.5);
    time.sleep(1)
    return "Move home command executed."


def scan_and_cache_objects(pipeline, robot):
    """Scans the scene, caches all objects, clears stack cache, and returns a summary."""
    global OBJECT_CACHE, STACK_CACHE
    print("Scanning environment and clearing all caches...")
    move_home(robot)
    OBJECT_CACHE = find_multiple_objects(pipeline)
    STACK_CACHE = {}

    if not OBJECT_CACHE:
        summary = "Scan complete. No objects were found."
    else:
        object_summary = {}
        for obj in OBJECT_CACHE:
            color = obj.get("color", "unknown_color")
            shape = obj.get("shape", "unidentified")
            if color not in object_summary:
                object_summary[color] = {}
            object_summary[color][shape] = object_summary[color].get(shape, 0) + 1

        summary_parts = []
        for color, shapes in object_summary.items():
            shape_details = ", ".join([f"{count} {shape}{'s' if count > 1 else ''}" for shape, count in shapes.items()])
            summary_parts.append(f"{color}: {shape_details}")

        summary = f"Scan complete. Cached {len(OBJECT_CACHE)} total objects. Breakdown - {' | '.join(summary_parts)}."
    print(f"✅ {summary}")
    return summary


def get_objects(pipeline, use_cache: bool = False):
    """Helper function to get objects from cache or a live scan."""
    global OBJECT_CACHE
    if use_cache:
        print("Using cached object locations.")
        if not OBJECT_CACHE:
            print("Warning: Object cache is empty. Performing a live scan instead.")
            return find_multiple_objects(pipeline)
        return OBJECT_CACHE
    else:
        print("Performing a live scan without using or modifying the cache.")
        return find_multiple_objects(pipeline)


def move_linear_pose(robot, x, y, z, rx=0.0, ry=3.14, rz=0.0, speed=0.15, acceleration=0.25):
    """Moves the robot's tool in a straight line to a target Cartesian pose."""
    target_pose = [x, y, z, rx, ry, rz]
    return safe_move_to_pose(robot, pose=target_pose, speed=speed, acceleration=acceleration)


def move_relative_pose(robot, x=0.0, y=0.0, z=0.0, speed=0.15, acceleration=0.25):
    """Moves the robot's tool by a specified offset from its current position."""
    current_pose = robot.get_actual_tcp_pose()
    if current_pose is None: return "Could not get current robot pose."
    target_pose = [current_pose[0] + x, current_pose[1] + y, current_pose[2] + z,
                   current_pose[3], current_pose[4], current_pose[5]]
    return safe_move_to_pose(robot, pose=target_pose, speed=speed, acceleration=acceleration)


def _get_pick_pose(object_pos, z_offset):
    """Helper to calculate approach and pick poses."""
    approach_pose = np.concatenate((object_pos, TCP_ORIENTATION))
    approach_pose[2] += z_offset

    if object_pos[2] > 0.5:
        pick_offset = -0.04
    elif object_pos[2] > 0.04:
        pick_offset = -0.03
    elif object_pos[2] > 0.03:
        pick_offset = -0.02
    else:
        pick_offset = -0.01

    pick_pose = np.concatenate((object_pos, TCP_ORIENTATION))
    pick_pose[2] += pick_offset

    return approach_pose, pick_pose, pick_offset


def _execute_pick_sequence(robot, approach_pose, pick_pose, selected_object, pick_offset, use_cache):
    """Helper to execute the physical pick motion and update caches."""
    global HELD_OBJECT, OBJECT_CACHE
    open_gripper(robot)
    if "Error" in (result := safe_move_to_pose(robot, pose=approach_pose, speed=0.25, acceleration=0.25)): return result
    if "Error" in (result := safe_move_to_pose(robot, pose=pick_pose, speed=0.1, acceleration=0.1)): return result
    close_gripper(robot)
    # time.sleep(0.5)
    if "Error" in (result := safe_move_to_pose(robot, pose=approach_pose, speed=0.1, acceleration=0.1)): return result

    HELD_OBJECT = {"position": selected_object['position'], "pick_offset": pick_offset, "data": selected_object}

    if use_cache:
        obj_to_remove_id = selected_object['center_pixel']
        OBJECT_CACHE = [obj for obj in OBJECT_CACHE if obj['center_pixel'] != obj_to_remove_id]
        print(f"Removed picked object from cache. {len(OBJECT_CACHE)} objects remaining.")

    return f"Successfully picked up the {selected_object.get('color', '')} {selected_object.get('shape', '')}."


def _filter_and_select_object(objects, color=None, shape=None, descriptor=None, rank=None, direction=None, target=None):
    """
    Centralized function to filter and select a single object based on a priority of criteria.
    """
    filtered_list = list(objects)
    if not filtered_list:
        return None, "Error: The list of objects to filter is empty."

    # --- Attribute Filtering (Color and Shape) ---
    if color:
        filtered_list = [obj for obj in filtered_list if obj.get("color") == color.lower()]
        if not filtered_list:
            return None, f"Error: No objects with color '{color}' found."

    if shape and shape.lower() != 'any':
        user_shape = shape.lower()
        search_shape = None
        if user_shape in ['circle', 'cylinder']:
            search_shape = 'cylinder'
        elif user_shape in ['square', 'rectangle', 'block', 'box', 'cube']:
            search_shape = 'block'

        if search_shape:
            filtered_list = [obj for obj in filtered_list if obj.get("shape") == search_shape]
        else:
            filtered_list = []  # Unrecognized shape yields no results

        if not filtered_list:
            return None, f"Error: No objects matching shape '{shape}' found."

    # --- Sorting and Selection (in order of priority) ---

    # Priority 1: Size-based descriptor ('smallest', 'biggest') with optional rank.
    # This handles commands like "the smallest" or "the second biggest".
    if descriptor:
        is_biggest = descriptor.lower() in ['biggest', 'larger', 'largest']
        filtered_list.sort(key=lambda o: o['area'], reverse=is_biggest)
        rank_to_use = rank if rank is not None else 1
        try:
            return filtered_list[rank_to_use - 1], "OK"
        except IndexError:
            return None, f"Error: Invalid rank {rank_to_use}. Only {len(filtered_list)} objects match the size criteria."

    # Priority 2: Position-based direction ('left_to_right') with optional rank.
    # This handles "the second from the left".
    if direction:
        is_reverse = (direction == 'right_to_left')
        filtered_list.sort(key=lambda o: o['center_pixel'][0], reverse=is_reverse)
        rank_to_use = rank if rank is not None else 1
        try:
            return filtered_list[rank_to_use - 1], "OK"
        except IndexError:
            return None, f"Error: Invalid rank {rank_to_use}. Only {len(filtered_list)} objects match the direction criteria."

    # Priority 3: General spatial target ('left', 'center', 'right').
    # This handles simple commands like "the one on the left".
    if target:
        filtered_list.sort(key=lambda o: o['center_pixel'][0])  # Sort left-to-right by x-pixel
        try:
            if target.lower() == 'left':
                return filtered_list[0], "OK"
            elif target.lower() == 'right':
                return filtered_list[-1], "OK"
            elif target.lower() == 'center':
                return filtered_list[len(filtered_list)//2], "OK"
        except IndexError:
            return None, "Error: No objects found for the specified spatial target."

    # Fallback: If no specific sorting criteria were given, return the first available object.
    return (filtered_list[0], "OK") if filtered_list else (None, "Error: No objects match the specified criteria.")


def pick_object(robot, pipeline, color=None, shape=None, descriptor=None, rank=None, direction=None,
                target=None, z_offset=0.1, hover_only=False, use_cache=False):
    """Universal function to pick or hover over an object based on criteria."""
    objects = get_objects(pipeline, use_cache)
    if not objects: return "No objects found to pick."

    selected_object, msg = _filter_and_select_object(objects, color, shape, descriptor, rank, direction, target)
    if not selected_object: return msg

    object_pos = selected_object['position']
    approach_pose, pick_pose, pick_offset = _get_pick_pose(object_pos, z_offset)

    if hover_only:
        result = safe_move_to_pose(robot, pose=approach_pose, speed=0.25, acceleration=0.25)
        return f"Hovered over the {selected_object.get('color', '')} {selected_object.get('shape', '')}." if "Error" not in str(
            result) else result
    else:
        return _execute_pick_sequence(robot, approach_pose, pick_pose, selected_object, pick_offset, use_cache)


def place_object(robot, location_name: str = "center"):
    """Places the held object on the TABLE at a named location. Does not stack."""
    global HELD_OBJECT
    if HELD_OBJECT is None: return "Error: Not holding anything to place."
    if location_name not in DROP_OFF_LOCATIONS: return f"Error: Unknown drop-off location '{location_name}'."

    z_adjustment = HELD_OBJECT["position"][2] + HELD_OBJECT["pick_offset"]
    target_pose = np.array(DROP_OFF_LOCATIONS[location_name])
    target_pose[2] += z_adjustment
    approach_pose = np.copy(target_pose);
    approach_pose[2] += 0.1

    if "Error" in (r := safe_move_to_pose(robot, pose=approach_pose, speed=0.25, acceleration=0.25)): return r
    if "Error" in (r := safe_move_to_pose(robot, pose=target_pose, speed=0.1, acceleration=0.1)): return r
    open_gripper(robot)
    HELD_OBJECT = None
    time.sleep(0.5)
    if "Error" in (r := safe_move_to_pose(robot, pose=approach_pose, speed=0.1, acceleration=0.1)): return r
    return f"Successfully placed object at {location_name}."


def stack_object(robot, pipeline, location_name=None, target_color=None, target_shape=None, use_cache=True):
    """Places the held object on top of another object or a stack."""
    global HELD_OBJECT, STACK_CACHE, OBJECT_CACHE
    if HELD_OBJECT is None:
        return "Error: Cannot stack object because the robot is not holding anything."

    base_pos_xy = None
    top_of_stack_z = 0
    stack_key = None

    if location_name:
        if location_name not in DROP_OFF_LOCATIONS: return f"Error: Unknown drop-off location '{location_name}'."
        base_pose_3d = DROP_OFF_LOCATIONS[location_name]
        base_pos_xy = base_pose_3d[:2]
        top_of_stack_z = base_pose_3d[2]
        stack_key = location_name
    elif target_color or target_shape:
        objects_to_search = OBJECT_CACHE if use_cache else find_multiple_objects(pipeline)
        if not objects_to_search: return "Error: Could not find any objects to stack on."
        target_object, msg = _filter_and_select_object(objects_to_search, color=target_color, shape=target_shape)
        if not target_object: return f"Error: Could not find target to stack on. {msg}"
        base_pos_xy = target_object['position'][:2]
        top_of_stack_z = target_object['position'][2]
        stack_key = f"stack_at_{target_object['center_pixel'][0]}_{target_object['center_pixel'][1]}"
    else:
        location_name = "center"
        base_pose_3d = DROP_OFF_LOCATIONS[location_name]
        base_pos_xy = base_pose_3d[:2]
        top_of_stack_z = base_pose_3d[2]
        stack_key = location_name

    if stack_key in STACK_CACHE and STACK_CACHE[stack_key]:
        top_of_stack_z = STACK_CACHE[stack_key][-1]['position'][2]

    held_object_height = HELD_OBJECT['position'][2]
    pick_offset = HELD_OBJECT['pick_offset']
    tcp_target_z = top_of_stack_z + held_object_height + pick_offset
    target_pose = np.array([base_pos_xy[0], base_pos_xy[1], tcp_target_z, *TCP_ORIENTATION])

    approach_pose = np.copy(target_pose);
    approach_pose[2] += 0.1
    if "Error" in (r := safe_move_to_pose(robot, pose=approach_pose, speed=0.25, acceleration=0.25)): return r
    if "Error" in (r := safe_move_to_pose(robot, pose=target_pose, speed=0.1, acceleration=0.1)): return r

    placed_object_info = HELD_OBJECT.copy()
    open_gripper(robot)
    HELD_OBJECT = None
    # time.sleep(0.5)

    new_object_top_z = top_of_stack_z + held_object_height
    placed_object_info['data']['position'] = np.array([base_pos_xy[0], base_pos_xy[1], new_object_top_z])
    if stack_key not in STACK_CACHE: STACK_CACHE[stack_key] = []
    STACK_CACHE[stack_key].append(placed_object_info['data'])

    if "Error" in (r := safe_move_to_pose(robot, pose=approach_pose, speed=0.1, acceleration=0.1)): return r
    return f"Successfully stacked object on {stack_key}."


def place_in_circle(robot, pipeline, index: int, total_objects: int, diameter: float = 0.2,
                    center_object_color: str = None, center_object_shape: str = None, use_cache: bool = True):
    """
    Places the currently held object into a designated spot in a circle.
    This should be called after a successful pick_object().
    """
    global HELD_OBJECT, OBJECT_CACHE, aoi_centroid_robot_coords

    if HELD_OBJECT is None:
        return "Error: Not holding anything to place in a circle."

    # --- Determine the Center Point ---
    center_point_xy = None
    if center_object_color or center_object_shape:
        if not use_cache:
            return "Error: Finding a center object requires using the cache. Please scan first."

        # We need to find the center object without modifying the main cache
        # The _filter_and_select_object function is perfect for this.
        center_object, msg = _filter_and_select_object(OBJECT_CACHE, color=center_object_color,
                                                       shape=center_object_shape)

        if not center_object:
            return f"Error: Could not find the specified center object to arrange around. {msg}"

        center_point_xy = center_object['position'][:2]
        print(f"Using object at {np.round(center_point_xy, 2)} as the circle center.")
    else:
        if aoi_centroid_robot_coords is None:
            return "Error: No center object specified and the AOI centroid has not been calculated."
        center_point_xy = aoi_centroid_robot_coords[:2]
        print(f"Using AOI centroid at {np.round(center_point_xy, 2)} as the circle center.")

    # --- Calculate Placement Pose ---
    if total_objects <= 0:
        return "Error: Total number of objects must be positive."
    if index >= total_objects:
        return f"Error: Index {index} is out of bounds for a circle of {total_objects} objects."

    radius = diameter/2.0
    angle_step = (2*math.pi)/total_objects
    angle = index*angle_step

    # Calculate target x, y
    x = center_point_xy[0] + radius*math.cos(angle)
    y = center_point_xy[1] + radius*math.sin(angle)

    # Use the held object's height information for correct placement
    z_adjustment = HELD_OBJECT["position"][2] + HELD_OBJECT["pick_offset"]
    place_pose = np.array([x, y, 0.002 + z_adjustment, *TCP_ORIENTATION])
    approach_pose = np.copy(place_pose)
    approach_pose[2] += 0.1  # Approach from 10cm above

    # --- Execute Placement ---
    if "Error" in (r := safe_move_to_pose(robot, pose=approach_pose, speed=0.25, acceleration=0.25)): return r
    if "Error" in (r := safe_move_to_pose(robot, pose=place_pose, speed=0.1, acceleration=0.1)): return r
    open_gripper(robot)
    HELD_OBJECT = None  # Object is no longer held
    time.sleep(0.5)
    if "Error" in (r := safe_move_to_pose(robot, pose=approach_pose, speed=0.1, acceleration=0.1)): return r

    return f"Successfully placed object {index + 1}/{total_objects} in the circle."


def place_in_line(robot, index: int, total_objects: int, line_x=0.55, line_y_start=-0.3, line_y_end=0.06):
    """
    Places the currently held object into a designated spot in a line.
    This should be called after a successful pick_object().
    """
    global HELD_OBJECT

    if HELD_OBJECT is None:
        return "Error: Not holding anything to place in a line."
    if not isinstance(index, int) or not isinstance(total_objects, int) or total_objects <= 0:
        return f"Error: Invalid arguments. Index and total_objects must be integers, and total_objects > 0."
    if index >= total_objects:
        return f"Error: Index {index} is out of bounds for a line of {total_objects} objects."

    # Calculate the target position for this specific object in the line
    if total_objects == 1:
        # If there's only one object, place it in the middle of the range
        target_y = (line_y_start + line_y_end)/2
    else:
        # Otherwise, distribute objects evenly along the line
        target_y = np.linspace(line_y_start, line_y_end, total_objects)[index]

    target_pos = [line_x, target_y, 0.002]  # Assuming a small default z for the table

    # Use the held object's height information for correct placement
    z_adjustment = HELD_OBJECT["position"][2] + HELD_OBJECT["pick_offset"]
    place_pose = np.array([target_pos[0], target_pos[1], target_pos[2] + z_adjustment, *TCP_ORIENTATION])
    approach_pose = np.copy(place_pose)
    approach_pose[2] += 0.1  # Approach from 10cm above

    # Execute the placement sequence
    if "Error" in (r := safe_move_to_pose(robot, pose=approach_pose, speed=0.25, acceleration=0.25)): return r
    if "Error" in (r := safe_move_to_pose(robot, pose=place_pose, speed=0.1, acceleration=0.1)): return r
    open_gripper(robot)
    HELD_OBJECT = None  # Object is no longer held
    time.sleep(0.5)
    if "Error" in (r := safe_move_to_pose(robot, pose=approach_pose, speed=0.1, acceleration=0.1)): return r

    return f"Successfully placed object {index + 1}/{total_objects} in the line at {np.round(target_pos, 2)}."


def get_object_count(pipeline, color=None, shape=None):
    """Counts objects based on optional color and shape criteria."""
    objects = find_multiple_objects(pipeline)

    # Use the same filtering logic as the main selection function
    filtered_list, _ = _filter_and_select_object(objects, color=color, shape=shape)

    # _filter_and_select_object is designed to return one object, we need to adapt to count all
    # Re-implementing the filter logic here for clarity and correctness
    filtered_list = list(objects)
    if color:
        filtered_list = [obj for obj in filtered_list if obj.get("color") == color.lower()]

    if shape and shape.lower() != 'any':
        search_shape = None
        user_shape = shape.lower()
        if user_shape in ['circle', 'cylinder']:
            search_shape = 'cylinder'
        elif user_shape in ['square', 'rectangle', 'block', 'box', 'cube']:
            search_shape = 'block'

        if search_shape:
            filtered_list = [obj for obj in filtered_list if obj.get("shape") == search_shape]
        else:
            filtered_list = []

    return f"Found {len(filtered_list)} objects matching the criteria."


def open_gripper(robot):
    """Opens the gripper."""
    print("Opening gripper...")
    GripperFunctions.open_gripper(robot)
    return "Gripper opened."


def close_gripper(robot):
    """Closes the gripper."""
    print("Closing gripper...")
    GripperFunctions.close_gripper(robot)
    return "Gripper closed."


def get_current_pose(robot):
    """Gets the current Cartesian pose of the robot's tool."""
    return f"Current pose: {robot.get_actual_tcp_pose()}"


# =================================================================================
# PART 4: LLM INTERACTION & GUI
# =================================================================================

SYSTEM_PROMPT = """
You are a Universal Robots UR10e controller. Your goal is to translate user commands into a sequence of function calls. You MUST be precise and thorough.

**Primary Directive: JSON Formatting**
- Your **only** output must be a single, valid JSON object.
- This object must have a single key: `"commands"`.
- The value of `"commands"` must be a list of JSON objects.
- Each object in the list represents one function call and **must** have two keys:
    1. `"function_name"`: The name of the function to call (e.g., `"pick_object"`).
    2. `"args"`: A JSON object containing the arguments for that function (e.g., `{"color": "red", "use_cache": true}`).
- If a task is complete or no action is needed, return a JSON object with an empty list: `{"commands": []}`.

**Core Concepts:**
- **Colors:** 'red', 'green', 'white'.
- **Shapes:** 'cylinder' and 'block'.
- **Interpreting Commands:** You may need to interpret vague terms. 'Dark colored' could mean 'red'. 'Light colored' could mean 'white'. 'Blocks' can mean 'cubes' or 'squares'. Use context to make the best choice.

**The Golden Rule: Scan First, Then Plan**
- For ANY task involving more than one object (sorting, stacking, arranging, lining up), you MUST use a two-step process:
    1. **First Turn:** Call `scan_and_cache_objects()` and nothing else.
    2. **Second Turn:** After receiving the scan results, generate the COMPLETE plan for the entire task. All functions in this plan MUST use `use_cache=True`.

**Critical Planning Rule: Count Everything!**
- After a `scan_and_cache_objects` call, you will receive a summary like `Scan complete. Cached 6 total objects. Breakdown - white: 1 block | red: 3 cylinders, 1 block | green: 1 block.`
- Your **ABSOLUTE HIGHEST PRIORITY** is to create a plan that handles **EVERY SINGLE OBJECT** that matches the user's request, based on this summary.
- **Example of a Failure:**
    - **User:** "Separate the red objects."
    - **Scan Result:** `...red: 3 cylinders, 1 block...` (This means there are 4 red objects).
    - **INCORRECT Plan:** A plan that only calls `pick_object` three times.
    - **CORRECT Plan:** A plan that calls `pick_object` and `place_object` **four times**, once for each red object.
- You must meticulously count the objects from the function result (e.g., `3 cylinders + 1 block = 4 objects`) and ensure your command list is complete. Failure to account for all relevant objects is a critical error.

**Function Workflows**

- **STACKING:** To place an object on top of another, use `stack_object()`. To place an object on the table, use `place_object()`.
    - **Workflow:**
        1. `scan_and_cache_objects()`
        2. `pick_object(descriptor='biggest', use_cache=true)`
        3. `stack_object(location_name='center')`
        4. `pick_object(descriptor='biggest', use_cache=true)`
        5. `stack_object(location_name='center')`
        6. ...and so on for all objects.

- **LINING UP:** To line up objects, repeatedly call `pick_object` then `place_in_line`.
    - **Workflow (Line up 3 blocks by size):**
        1. `scan_and_cache_objects()`
        2. `pick_object(shape='block', descriptor='smallest', use_cache=true)`
        3. `place_in_line(index=0, total_objects=3)`
        4. `pick_object(shape='block', descriptor='smallest', use_cache=true)`
        5. `place_in_line(index=1, total_objects=3)`
        6. `pick_object(shape='block', descriptor='smallest', use_cache=true)`
        7. `place_in_line(index=2, total_objects=3)`

- **ARRANGING IN A CIRCLE:** To arrange objects, repeatedly call `pick_object` then `place_in_circle`.
    - **Workflow (Arrange 2 blocks in the center of the table):**
        1. `scan_and_cache_objects()`
        2. `pick_object(shape='block', use_cache=true)`
        3. `place_in_circle(index=0, total_objects=2)`
        4. `pick_object(shape='block', use_cache=true)`
        5. `place_in_circle(index=1, total_objects=2)`


**AVAILABLE FUNCTIONS:**

- **`pick_object(color: str, shape: str, descriptor: str, rank: int, direction: str, target: str, hover_only: bool, use_cache: bool)`**:
    - Primary function to select an object. 
    - `shape`: 'cylinder', 'block'
    - `descriptor`: 'biggest', 'smallest'.
    - `rank`: Number (e.g., 1 for first). Used with `direction` ('left_to_right', 'right_to_left') or `descriptor`. (If picking multiple objects from a side, always use rank 1)
    - `direction`: Use for positional ranking, e.g., 'second from the left'.
    - `target`: General location ('left', 'center', 'right').
    - `use_cache`: **MUST be `true` for any step after a scan.**

- **`stack_object(location_name: str, target_color: str, target_shape: str, use_cache: bool)`**:
    - Places the HELD object on top of something.
    - **MUST use `use_cache=True` if targeting a specific object.**
    - Use `location_name` ('center', 'left', 'right') to start or add to a stack at a drop-off point.
    - Use `target_color` and/or `target_shape` to stack on a specific object.
    - If no args, defaults to `location_name='center'`.

- **`place_object(location_name: str = 'center')`**:
    - Places the held object directly on the TABLE at a named location. Does NOT stack.

- **`place_in_line(index: int, total_objects: int, line_x: float, line_y_start: float, line_y_end: float)`**:
    - Places the HELD object into a specific position in a line. **MUST be called after `pick_object`.**
    - `index`: The zero-based position in the line for the current object (e.g., 0 for the first, 1 for the second).
    - `total_objects`: The total number of objects that will be in the complete line.
    - `line_x`, `line_y_start`, `line_y_end`: Define the line's coordinates. Can usually be omitted to use defaults.

- **`place_in_circle(index: int, total_objects: int, diameter: float, center_object_color: str, center_object_shape: str, use_cache: bool)`**:
    - Places the HELD object into a specific position in a circle. **MUST be called after `pick_object`.**
    - `index`: The zero-based position in the circle for the current object.
    - `total_objects`: The total number of objects that will be in the complete circle.
    - `diameter`: The diameter of the circle in meters. Default is 0.2. ALlowed range of 0.15 to 0.25. Specify if user gives size or something like tight, wide, small, etc.
    - `center_object_color`, `center_object_shape`: Specify an object in the cache to use as the center. You MUST omit unless user specifies or wants to cluster objects.
    - `use_cache`: **MUST be `true` if specifying a center object.**

- **`scan_and_cache_objects()`**:
    - Scans the environment. **REQUIRED first step for any multi-object task.** Clears all caches.

- **`get_object_count(color: str, shape: str)`**: Counts objects.
- **`move_home()`**: Moves to home position.
- **`move_linear_pose(x, y, z, rx, ry, rz)`**: Moves to an absolute pose (meters, radians).
- **`move_relative_pose(x, y, z)`**: Moves by an offset (meters).
- **`open_gripper()` / `close_gripper()`**: Controls the gripper.
"""


def get_llm_response(conversation_history):
    """Sends the user command to OpenAI and gets a JSON response."""
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    messages.extend(conversation_history)
    try:
        response = client.chat.completions.create(model=LLM_MODEL, messages=messages,
                                                  response_format={"type": "json_object"}, temperature=0.0)
        return response.choices[0].message.content
    except Exception as e:
        return f'{{"error": "Error communicating with OpenAI: {e}"}}'


class RecorderApp:
    """A persistent GUI control panel for the robot."""

    def __init__(self, root, recognizer, command_dispatcher):
        self.root = root
        self.recognizer = recognizer
        self.command_dispatcher = command_dispatcher
        self.is_recording = False
        self.task_in_progress = False
        self._setup_gui()

        self.start_time = 0
        self.transcription_end_time = 0
        self.execution_end_time = 0
        self.total_llm_time = 0

    def _setup_gui(self):
        self.root.title("Robot Voice Controller")
        self.root.geometry("450x250")
        self.root.attributes('-topmost', True)
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Helvetica", 12), padding=10)
        self.status_label = ttk.Label(self.root, text="Press and hold the button to record.",
                                      font=("Helvetica", 10), wraplength=430)
        self.status_label.pack(pady=10)
        self.record_button = ttk.Button(self.root, text="Hold to Record")
        self.record_button.pack(pady=20, ipadx=20, ipady=10)
        self.exit_button = ttk.Button(self.root, text="Exit Application", command=self.exit_app)
        self.exit_button.pack(pady=10)
        self.record_button.bind("<ButtonPress-1>", self.start_recording)
        self.record_button.bind("<ButtonRelease-1>", self.stop_recording)

    def start_recording(self, event):
        if self.task_in_progress:
            self.status_label.config(text="Cannot record, a task is already in progress.")
            return
        if self.is_recording: return
        self.status_label.config(text="🔴 Recording...")
        self.record_button.config(text="Recording...")
        self.is_recording = True
        threading.Thread(target=self._record_and_process_audio, daemon=True).start()

    def stop_recording(self, event):
        if not self.is_recording: return
        self.is_recording = False
        self.task_in_progress = True
        self.status_label.config(text="Transcribing...")
        self.record_button.config(text="Hold to Record")
        self.start_time = time.time()

    def _record_and_process_audio(self):
        """Handles the entire audio lifecycle in one thread."""
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
        frames = []
        while self.is_recording:
            try:
                frames.append(stream.read(1024))
            except IOError:
                pass
        stream.stop_stream()
        stream.close()
        p.terminate()

        if not frames:
            self.status_label.config(text="No audio recorded. Try again.")
            self.task_in_progress = False
            return

        audio_data = sr.AudioData(b''.join(frames), 16000, p.get_sample_size(pyaudio.paInt16))
        try:
            text = self.recognizer.recognize_openai(audio_data, model="whisper-1")
            self.transcription_end_time = time.time()
            self.status_label.config(text=f"Transcribed: {text}")
            if any(word in text.lower().strip() for word in ['exit', 'stop', 'quit']):
                self.exit_app()
                return
            self._execute_task(text)
        except (sr.RequestError, sr.UnknownValueError) as e:
            self.status_label.config(text=f"Could not process audio: {e}")
            self.task_in_progress = False

    def _report_timings(self):
        """Calculates and prints the timings for each stage."""
        transcription_time = self.transcription_end_time - self.start_time
        llm_time = self.total_llm_time
        total_time = self.execution_end_time - self.start_time
        execution_time = total_time - transcription_time - llm_time

        print("\n--- Task Timing Report ---")
        print(f"Transcription Time: {transcription_time:.2f} seconds")
        print(f"LLM Processing Time (Total): {llm_time:.2f} seconds")
        print(f"Task Execution Time: {execution_time:.2f} seconds")
        print(f"Total Time: {total_time:.2f} seconds")
        print("--------------------------\n")

    def _execute_task(self, user_input):
        """Contains the planning and execution loop for a given user command."""
        self.status_label.config(text=f"Executing: '{user_input}'")
        print(f"\n> User Command: '{user_input}'")
        conversation_history = [{"role": "user", "content": user_input}]
        self.total_llm_time = 0

        while True:
            if not keep_running: break

            llm_start_time = time.time()
            llm_json_str = get_llm_response(conversation_history)
            llm_end_time = time.time()
            self.total_llm_time += (llm_end_time - llm_start_time)

            print(f"LLM Plan: {llm_json_str}")
            try:
                command_data = json.loads(llm_json_str)
            except json.JSONDecodeError:
                self.status_label.config(text="Error: Received invalid plan from AI.")
                break

            commands_to_execute = command_data.get("commands", [])
            if not commands_to_execute:
                self.execution_end_time = time.time()
                self._report_timings()
                self.status_label.config(text="✅ Task Complete! Ready for new command.")
                break

            conversation_history.append({"role": "assistant", "content": llm_json_str})

            all_results = []
            error_occurred = False
            for command in commands_to_execute:
                function_name = command.get("function_name")
                args = {k: v for k, v in command.get("args", {}).items() if v is not None}
                self.status_label.config(text=f"Running: {function_name}({args})")
                result = self.command_dispatcher.get(function_name,
                                                     lambda **kwargs: f"Error: Unknown function '{function_name}'")(
                    **args)
                print(f"Execution Result: {result}")
                all_results.append(str(result))
                if "Error" in str(result):
                    error_occurred = True
                    break

            context_for_llm = f"Function results: [{', '.join(all_results)}]. Based on these results, what is the next complete sequence of commands? If the task is finished, return an empty list."
            conversation_history.append({"role": "user", "content": context_for_llm})

            if error_occurred:
                self.status_label.config(text=f"Error during execution. Ready for new command.")
                break

        self.task_in_progress = False

    def exit_app(self):
        """Gracefully shuts down the application."""
        global keep_running
        if keep_running:
            print("Exit button clicked. Shutting down.")
            keep_running = False
            self.root.destroy()

    def run(self):
        """Starts the GUI event loop."""
        self.root.mainloop()


# =================================================================================
# Main Execution Block
# =================================================================================
def main():
    global keep_running
    robot = None
    pipeline = None
    vision_thread = None
    detection_thread = None

    try:
        print("Initializing Robot...")
        model = robotModel.RobotModel()
        robot = urScriptExt.UrScriptExt(host=ROBOT_IP, robotModel=model, conf_filename=RTDE_CONF_FILE)

        print("Initializing RealSense Camera...")
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, 30)
        pipeline.start(config)
        time.sleep(2)

        setup_calibration(pipeline, robot)
        if not keep_running:
            raise SystemExit("Shutdown during calibration.")

        vision_thread = threading.Thread(target=live_vision_thread, args=(pipeline,), daemon=True)
        detection_thread = threading.Thread(target=periodic_detection_thread, args=(pipeline,), daemon=True)
        vision_thread.start()
        detection_thread.start()

        COMMAND_DISPATCHER = {
            "scan_and_cache_objects": lambda **kwargs: scan_and_cache_objects(pipeline, robot),
            "get_object_count": lambda **kwargs: get_object_count(pipeline, **kwargs),
            "pick_object": lambda **kwargs: pick_object(robot, pipeline, **kwargs),
            "place_object": lambda **kwargs: place_object(robot, **kwargs),
            "stack_object": lambda **kwargs: stack_object(robot, pipeline, **kwargs),
            "place_in_circle": lambda **kwargs: place_in_circle(robot, pipeline, **kwargs),
            "place_in_line": lambda **kwargs: place_in_line(robot, **kwargs),
            "move_home": lambda **kwargs: move_home(robot),
            "move_linear_pose": lambda **kwargs: move_linear_pose(robot, **kwargs),
            "move_relative_pose": lambda **kwargs: move_relative_pose(robot, **kwargs),
            "get_current_pose": lambda **kwargs: get_current_pose(robot),
            "open_gripper": lambda **kwargs: open_gripper(robot),
            "close_gripper": lambda **kwargs: close_gripper(robot),
        }

        recognizer = sr.Recognizer()
        print("\nSetup complete. Launching command GUI...")

        root = tk.Tk()
        app = RecorderApp(root, recognizer, COMMAND_DISPATCHER)
        app.run()

    except SystemExit as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred in main: {e}")
    finally:
        print("\nShutting down all systems...")
        keep_running = False
        if detection_thread and detection_thread.is_alive():
            detection_thread.join(timeout=2)
        if vision_thread and vision_thread.is_alive():
            vision_thread.join(timeout=2)
        if robot:
            robot.close()
        if pipeline:
            pipeline.stop()
        cv2.destroyAllWindows()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
