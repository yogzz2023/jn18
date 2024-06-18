import numpy as np
import math
import csv
import pandas as pd
from scipy.stats import chi2
import matplotlib.pyplot as plt

class CVFilter:
    def __init__(self):
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.Pf = np.eye(6)  # Filter state covariance matrix
        self.Sp = np.zeros((6, 1))
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time
        self.Z = np.zeros((3, 1))

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        self.Sf = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = time

    def initialize_measurement_for_filtering(self, x, y, z, mt):
        self.Z = np.array([[x], [y], [z]])
        self.Meas_Time = mt

    def predict_step(self, current_time):
        dt = current_time - self.Meas_Time
        Phi = np.eye(6)
        Phi[0, 3] = dt
        Phi[1, 4] = dt
        Phi[2, 5] = dt
        Q = np.eye(6) * self.plant_noise
        self.Sp = np.dot(Phi, self.Sf)
        self.Pf = np.dot(np.dot(Phi, self.Pf), Phi.T) + Q

    def update_step(self, report):
        Inn = report - np.dot(self.H, self.Sf)  # Calculate innovation using associated report
        S = np.dot(self.H, np.dot(self.Pf, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pf, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sf + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pf)

def sph2cart(az, el, r):
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z

def cart2sph(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    el = math.atan(z / np.sqrt(x ** 2 + y ** 2)) * 180 / np.pi
    az = math.atan2(y, x) * 180 / np.pi
    if az < 0:
        az += 360
    return r, az, el

def cart2sph2(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    el = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
    az = np.degrees(np.arctan2(y, x))
    az[az < 0] += 360
    return r, az, el

def read_and_group_measurements(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            mr = float(row[7])  # MR column
            ma = float(row[8])  # MA column
            me = float(row[9])  # ME column
            mt = float(row[10])  # MT column
            x, y, z = sph2cart(ma, me, mr)  # Convert spherical to Cartesian coordinates
            measurements.append((x, y, z, mt))
    
    measurements.sort(key=lambda x: x[3])  # Sort by mt
    grouped_measurements = []
    current_group = []
    for measurement in measurements:
        if current_group and abs(measurement[3] - current_group[-1][3]) >= 0.050:
            grouped_measurements.append(current_group)
            current_group = []
        current_group.append(measurement)
    if current_group:
        grouped_measurements.append(current_group)
    
    return grouped_measurements

def initialize_tracks(grouped_measurements, cov_inv):
    tracks = []
    track_ids = []

    for i, group in enumerate(grouped_measurements):
        if i == 0:
            for measurement in group:
                track_id = len(tracks)
                tracks.append([measurement])
                track_ids.append(track_id)
        else:
            for measurement in group:
                assigned = False
                for track_id, track in enumerate(tracks):
                    if chi_squared_test(measurement, track, cov_inv):
                        tracks[track_id].append(measurement)
                        assigned = True
                        break
                if not assigned:
                    track_id = len(tracks)
                    tracks.append([measurement])
                    track_ids.append(track_id)
    
    return tracks, track_ids

def chi_squared_test(measurement, track, cov_inv):
    distances = []
    for track_measurement in track:
        distance = mahalanobis_distance(np.array(track_measurement[:3]), np.array(measurement[:3]), cov_inv)
        distances.append(distance)
    min_distance = min(distances)
    return min_distance < chi2_threshold

def generate_clusters(tracks, measurements, cov_inv):
    clusters = {}
    for track_id, track in enumerate(tracks):
        clusters[track_id] = []
        for measurement in measurements:
            if chi_squared_test(measurement, track, cov_inv):
                clusters[track_id].append(measurement)
    return clusters

def is_valid_hypothesis(hypothesis):
    non_zero_hypothesis = [val for _, val in hypothesis if val != -1]
    return len(non_zero_hypothesis) == len(set(non_zero_hypothesis)) and len(non_zero_hypothesis) > 0

state_dim = 3  # 3D state (e.g., x, y, z)
chi2_threshold = chi2.ppf(0.95, df=state_dim)

def mahalanobis_distance(x, y, cov_inv):
    delta = y[:3] - x[:3]
    return np.sqrt(np.dot(np.dot(delta, cov_inv), delta))

def generate_hypotheses(clusters):
    hypotheses = []
    for track_id, cluster in clusters.items():
        num_tracks = len(cluster)
        base = len(cluster) + 1
        for count in range(base ** num_tracks):
            hypothesis = []
            for track_idx in range(num_tracks):
                report_idx = (count // (base ** track_idx)) % base
                hypothesis.append((track_id, report_idx - 1))
            if is_valid_hypothesis(hypothesis):
                hypotheses.append(hypothesis)
    return hypotheses

def calculate_joint_probabilities(hypotheses, tracks, clusters, cov_inv):
    probabilities = []
    for hypothesis in hypotheses:
        prob = 1.0
        for track_id, report_idx in hypothesis:
            if report_idx != -1:
                track = tracks[track_id]
                report = clusters[track_id][report_idx]
                distance = mahalanobis_distance(np.array(track[-1][:3]), np.array(report[:3]), cov_inv)
                prob *= np.exp(-0.5 * distance ** 2)
        probabilities.append(prob)
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()
    return probabilities

def find_max_associations(hypotheses, probabilities, clusters):
    max_associations = [-1] * len(clusters)
    max_probs = [0.0] * len(clusters)
    for hypothesis, prob in zip(hypotheses, probabilities):
        for track_id, report_idx in hypothesis:
            if report_idx != -1 and prob > max_probs[track_id]:
                max_probs[track_id] = prob
                max_associations[report_idx] = track_id
    return max_associations, max_probs

def update_filter_with_max_probability(hypotheses, probabilities, clusters, kalman_filter):
    max_associations, max_probs = find_max_associations(hypotheses, probabilities, clusters)
    updated_states = []
    for report_idx, track_id in enumerate(max_associations):
        if track_id != -1:
            report = clusters[track_id][report_idx]
            kalman_filter.update_step(np.array(report[:3]).reshape((3, 1)))
            filtered_state = kalman_filter.Sf.flatten()[:3]
            r, az, el = cart2sph(filtered_state[0], filtered_state[1], filtered_state[2])
            updated_states.append((report[3], r, az, el))
    return updated_states

def plot_track_data(updated_states):
    csv_file_predicted = "ttk_84_2.csv"
    df_predicted = pd.read_csv(csv_file_predicted)
    filtered_values_csv = df_predicted[['F_TIM', 'F_X', 'F_Y', 'F_Z']].values

    r1, az1, el1 = cart2sph2(filtered_values_csv[:, 1], filtered_values_csv[:, 2], filtered_values_csv[:, 3])
    plt.plot(filtered_values_csv[:, 0], r1, label='Actual Track', color='blue', linestyle='-')
    updated_states = np.array(updated_states)
    plt.plot(updated_states[:, 0], updated_states[:, 1], label='Predicted Track', color='red', linestyle='-')
    plt.xlabel('Time (s)')
    plt.ylabel('Range (m)')
    plt.legend()
    plt.show()

def main():
    kalman_filter = CVFilter()
    csv_file_path = 'ttk_84_2.csv'

    try:
        grouped_measurements = read_and_group_measurements(csv_file_path)
    except FileNotFoundError:
        print(f"File not found: {csv_file_path}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    if not grouped_measurements:
        print("No measurements found in the CSV file.")
        return

    cov_inv = np.linalg.inv(np.eye(state_dim))  # Example covariance inverse matrix

    # Track initialization with chi-squared test
    tracks, track_ids = initialize_tracks(grouped_measurements, cov_inv)

    updated_states = []

    for group_idx, track_group in enumerate(grouped_measurements):
        print(f"Processing group {group_idx + 1}/{len(grouped_measurements)}")

        for i, (x, y, z, mt) in enumerate(track_group):
            if i == 0:
                kalman_filter.initialize_filter_state(x, y, z, 0, 0, 0, mt)
            elif i == 1:
                prev_x, prev_y, prev_z = track_group[i-1][:3]
                dt = mt - track_group[i-1][3]
                vx = (x - prev_x) / dt
                vy = (y - prev_y) / dt
                vz = (z - prev_z) / dt
                kalman_filter.initialize_filter_state(x, y, z, vx, vy, vz, mt)
                break

        kalman_filter.predict_step(mt)

        # Generate clusters and hypotheses
        clusters = generate_clusters(tracks, track_group, cov_inv)
        hypotheses = generate_hypotheses(clusters)
        probabilities = calculate_joint_probabilities(hypotheses, tracks, clusters, cov_inv)
        updated_states += update_filter_with_max_probability(hypotheses, probabilities, clusters, kalman_filter)

    plot_track_data(updated_states)

if __name__ == "__main__":
    main()
