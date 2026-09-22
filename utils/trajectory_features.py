import numpy as np


def extract_trajectory_features(detailed_log):

    positions = np.array([
        step['position']
        for step in detailed_log
    ])

    orientations = np.array([
        step['orientation']
        for step in detailed_log
    ])

    contacts = np.array([
        step['contacts']
        for step in detailed_log
    ])

    torques = np.array([
        step['torques']
        for step in detailed_log
    ])

    # XY velocity
    velocities = np.diff(
        positions[:, :2],
        axis=0
    )

    speed = np.linalg.norm(
        velocities,
        axis=1
    )

    features = np.array([

        # final position
        positions[-1, 0],
        positions[-1, 1],

        # speed stats
        speed.mean(),
        speed.std(),
        speed.max(),

        # body stability
        positions[:, 2].mean(),
        positions[:, 2].std(),

        # orientation stability
        orientations[:, 0].std(),
        orientations[:, 1].std(),

        # contact duty factors
        *contacts.mean(axis=0),

        # torque usage
        np.abs(torques).mean(),
        torques.std(),

    ], dtype=np.float32)

    return features