import numpy as np
from numpy.typing import ArrayLike

def compute_racing_line(centerline: ArrayLike, left_boundary: ArrayLike, right_boundary: ArrayLike) -> ArrayLike:
    n = len(centerline)
    racing_line = np.copy(centerline)

    for i in range(n):
        prev_idx = (i - 3) % n
        next_idx = (i + 3) % n

        p_prev = centerline[prev_idx]
        p_curr = centerline[i]
        p_next = centerline[next_idx]

        v1 = p_curr - p_prev
        v2 = p_next - p_curr

        cross = v1[0] * v2[1] - v1[1] * v2[0]

        angle1 = np.arctan2(v1[1], v1[0])
        angle2 = np.arctan2(v2[1], v2[0])
        angle_diff = angle2 - angle1
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

        dist = np.linalg.norm(v1) + np.linalg.norm(v2)
        curvature = abs(angle_diff) / max(dist, 0.1)

        if curvature > 0.015:
            if cross > 0:
                inside_boundary = right_boundary[i]
            else:
                inside_boundary = left_boundary[i]

            shift_factor = min(0.25, curvature * 3.0)
            racing_line[i] = centerline[i] + shift_factor * (inside_boundary - centerline[i])

    return racing_line

def find_closest_point(position: ArrayLike, path: ArrayLike) -> int:
    distances = np.linalg.norm(path - position, axis=1)
    return np.argmin(distances)

def find_lookahead_point(position: ArrayLike, heading: float, path: ArrayLike, lookahead_distance: float) -> tuple:
    closest_idx = find_closest_point(position, path)

    cumulative_dist = 0.0
    for i in range(len(path)):
        idx = (closest_idx + i) % len(path)
        next_idx = (closest_idx + i + 1) % len(path)

        segment_dist = np.linalg.norm(path[next_idx] - path[idx])
        cumulative_dist += segment_dist

        if cumulative_dist >= lookahead_distance:
            return path[next_idx], next_idx

    return path[(closest_idx + len(path)//4) % len(path)], (closest_idx + len(path)//4) % len(path)

def compute_path_curvature(path: ArrayLike, idx: int, window: int = 5) -> float:
    n = len(path)

    idx_prev = (idx - window) % n
    idx_next = (idx + window) % n

    p1 = path[idx_prev]
    p2 = path[idx]
    p3 = path[idx_next]

    v1 = p2 - p1
    v2 = p3 - p2

    angle1 = np.arctan2(v1[1], v1[0])
    angle2 = np.arctan2(v2[1], v2[0])

    angle_diff = angle2 - angle1
    angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

    dist = np.linalg.norm(v1) + np.linalg.norm(v2)

    if dist > 0.1:
        curvature = abs(angle_diff) / dist
    else:
        curvature = 0.0

    return curvature

def pure_pursuit_steering(position: ArrayLike, heading: float, path: ArrayLike, wheelbase: float, velocity: float) -> float:
    min_lookahead = 8.0
    max_lookahead = 20.0
    lookahead_gain = 0.20

    lookahead_distance = np.clip(min_lookahead + lookahead_gain * abs(velocity), min_lookahead, max_lookahead)

    lookahead_point, _ = find_lookahead_point(position, heading, path, lookahead_distance)

    dx = lookahead_point[0] - position[0]
    dy = lookahead_point[1] - position[1]

    dx_veh = np.cos(heading) * dx + np.sin(heading) * dy
    dy_veh = -np.sin(heading) * dx + np.cos(heading) * dy

    alpha = np.arctan2(dy_veh, dx_veh)
    ld = np.sqrt(dx_veh**2 + dy_veh**2)

    if ld > 0.1:
        delta_desired = np.arctan2(2.0 * wheelbase * np.sin(alpha), ld)
    else:
        delta_desired = 0.0

    return delta_desired

def compute_desired_velocity(path: ArrayLike, current_idx: int, max_velocity: float, max_acceleration: float, current_velocity: float = 50.0) -> float:
    if not hasattr(compute_desired_velocity, 'prev_desired_velocity'):
        compute_desired_velocity.prev_desired_velocity = None

    max_lat_accel_factor = 0.15
    min_velocity = 8.0
    desired_velocity_smoothing_alpha = 0.5

    lookahead_points = 60
    max_curvature = 0.0
    max_curvature_distance = 0.0
    cumulative_distance = 0.0

    curvature_signs = []
    curvature_values = []
    distances = []

    for i in range(lookahead_points):
        idx = (current_idx + i) % len(path)
        next_idx = (current_idx + i + 1) % len(path)

        curvature = compute_path_curvature(path, idx, window=3)

        if i > 0 and i < lookahead_points - 1:
            prev_idx = (current_idx + i - 1) % len(path)
            v1 = path[idx] - path[prev_idx]
            v2 = path[next_idx] - path[idx]
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            signed_curvature = curvature if cross > 0 else -curvature
        else:
            signed_curvature = curvature

        curvature_signs.append(np.sign(signed_curvature) if abs(signed_curvature) > 0.01 else 0)
        curvature_values.append(curvature)
        distances.append(cumulative_distance)

        if curvature > max_curvature:
            max_curvature = curvature
            max_curvature_distance = cumulative_distance

        segment_dist = np.linalg.norm(path[next_idx] - path[idx])
        cumulative_distance += segment_dist

    is_chicane = False
    chicane_severity = 0.0

    window_size = 8
    for i in range(len(curvature_signs) - window_size):
        window_signs = curvature_signs[i:i+window_size]
        window_curves = curvature_values[i:i+window_size]

        direction_changes = 0
        for j in range(len(window_signs) - 1):
            if abs(window_curves[j]) > 0.02 and abs(window_curves[j+1]) > 0.02:
                if window_signs[j] != 0 and window_signs[j+1] != 0 and window_signs[j] != window_signs[j+1]:
                    direction_changes += 1

        if direction_changes >= 1:
            is_chicane = True
            chicane_severity = np.mean([c for c in window_curves if c > 0.02])
            break

    hairpin_threshold = 0.10
    is_hairpin = max_curvature > hairpin_threshold

    if max_curvature > 0.001:
        if is_chicane or is_hairpin:
            lat_factor = max_lat_accel_factor
        else:
            lat_factor = 0.60

        max_lat_accel = lat_factor * max_acceleration

        if is_chicane:
            safety_factor = 0.66
        elif is_hairpin:
            safety_factor = 0.82
        else:
            safety_factor = 0.95

        velocity_for_curve = np.sqrt(max_lat_accel / max_curvature) * safety_factor
    else:
        velocity_for_curve = max_velocity

    speed_diff = current_velocity - velocity_for_curve
    brake_threshold = 2.0 if is_chicane else 10.0
    desired_velocity = velocity_for_curve

    if is_chicane:
        assumed_braking_accel = 0.32 * max_acceleration
    else:
        assumed_braking_accel = 0.95 * max_acceleration

    if speed_diff > brake_threshold and max_curvature_distance > 0.1:
        braking_accel = assumed_braking_accel
        required_brake_distance = (velocity_for_curve**2 - current_velocity**2) / (-2 * braking_accel)

        if is_chicane:
            base_margin = 1.6
        elif is_hairpin:
            base_margin = 1.3
        else:
            base_margin = 1.15

        distance_scale = np.clip(1.0 - (max_curvature_distance / max(required_brake_distance * 2.0, 1.0)), 0.0, 0.5)
        safety_margin = base_margin - 0.5 * distance_scale

        if max_curvature_distance > 40.0:
            safety_margin = max(1.0, safety_margin * 0.8)

        if max_curvature_distance < required_brake_distance * safety_margin:
            distance_ratio = max_curvature_distance / max(required_brake_distance, 1.0)
            desired_velocity = velocity_for_curve + (current_velocity - velocity_for_curve) * distance_ratio

        if not is_chicane:
            desired_velocity = velocity_for_curve
    else:
        desired_velocity = velocity_for_curve

    if compute_desired_velocity.prev_desired_velocity is None:
        smoothed = desired_velocity
    else:
        alpha = desired_velocity_smoothing_alpha
        smoothed = alpha * desired_velocity + (1.0 - alpha) * compute_desired_velocity.prev_desired_velocity
    compute_desired_velocity.prev_desired_velocity = smoothed

    desired_velocity = np.clip(smoothed, min_velocity, max_velocity)

    return desired_velocity

def controller(state: ArrayLike, parameters: ArrayLike, racetrack) -> ArrayLike:
    sx, sy, delta, v, phi = state
    position = np.array([sx, sy])

    wheelbase = parameters[0]
    max_velocity = parameters[5]
    max_acceleration = parameters[10]

    path = racetrack.centerline[:, :2]
    closest_idx = find_closest_point(position, path)

    delta_desired = pure_pursuit_steering(position, phi, path, wheelbase, v)
    velocity_desired = compute_desired_velocity(path, closest_idx, max_velocity, max_acceleration, v)

    return np.array([delta_desired, velocity_desired])

def lower_controller(state: ArrayLike, desired: ArrayLike, parameters: ArrayLike) -> ArrayLike:
    if not hasattr(lower_controller, 'velocity_error_integral'):
        lower_controller.velocity_error_integral = 0.0
        lower_controller.velocity_error_prev = 0.0
        lower_controller.steering_error_integral = 0.0
        lower_controller.steering_error_prev = 0.0

    dt = 0.1
    Kp_v = 2.0
    Ki_v = 0.1
    Kd_v = 0.5
    Kp_s = 4.0
    Ki_s = 0.06
    Kd_s = 0.3

    assert(desired.shape == (2,))

    _, _, delta, v, _ = state
    delta_desired, velocity_desired = desired

    max_steering_vel = parameters[9]
    max_acceleration = parameters[10]

    velocity_error = velocity_desired - v
    lower_controller.velocity_error_integral += velocity_error * dt
    velocity_error_derivative = (velocity_error - lower_controller.velocity_error_prev) / dt
    lower_controller.velocity_error_prev = velocity_error

    lower_controller.velocity_error_integral = np.clip(lower_controller.velocity_error_integral, -10.0, 10.0)

    a = Kp_v * velocity_error + Ki_v * lower_controller.velocity_error_integral + Kd_v * velocity_error_derivative
    a = np.clip(a, -max_acceleration, max_acceleration)

    steering_error = delta_desired - delta
    steering_error = np.arctan2(np.sin(steering_error), np.cos(steering_error))

    lower_controller.steering_error_integral += steering_error * dt
    steering_error_derivative = (steering_error - lower_controller.steering_error_prev) / dt
    lower_controller.steering_error_prev = steering_error

    lower_controller.steering_error_integral = np.clip(lower_controller.steering_error_integral, -5.0, 5.0)

    v_delta = Kp_s * steering_error + Ki_s * lower_controller.steering_error_integral + Kd_s * steering_error_derivative
    v_delta = np.clip(v_delta, -max_steering_vel, max_steering_vel)

    return np.array([v_delta, a])
