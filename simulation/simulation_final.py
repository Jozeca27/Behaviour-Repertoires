import os
import pickle
import uuid
import time
from multiprocessing import Pool, cpu_count
from typing import Dict, Tuple

import numpy as np
import pybullet as p
import pybullet_data
import imageio.v2 as imageio

import config
from config import SIMULATION_CONFIG, URDF_PATH
from controllers.hexapod_controller_final import (
    RecurrentTorqueHexapodController
)


class HexapodTorqueSimulation:
    RANDOM_LOG_DIR = os.path.join(
        "results",
        "random_trajectories"
    )

    os.makedirs(RANDOM_LOG_DIR, exist_ok=True)

    def __init__(
        self,
        urdf_path: str,
        gui: bool = False,
        time_step: float = 1.0 / 240
    ):
        self.urdf_path = urdf_path
        self.gui = gui
        self.time_step = time_step

        self.robot_id = None
        self.joint_indices = []
        self.foot_link_indices = []

        self.physics_client = None

        self._transformer_bd_extractor = None

    # ==========================================================
    # TRANSFORMER BD
    # ==========================================================

    def _get_transformer_bd_extractor(self):

        # Lazy-load the appropriate transformer BD extractor
        # depending on the configured BD mode.
        if self._transformer_bd_extractor is None:

            if config.BD_MODE == 'transformer_8d':
                from utils.transformer_bd_8D import TransformerBDExtractor
            else:
                from utils.transformer_bd import TransformerBDExtractor

            self._transformer_bd_extractor = TransformerBDExtractor()

        return self._transformer_bd_extractor

    # ==========================================================
    # CONNECTION
    # ==========================================================

    def connect(self):

        if self.gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.setGravity(0, 0, -9.81)

        p.setTimeStep(self.time_step)

        p.setPhysicsEngineParameter(
            numSolverIterations=20,
            numSubSteps=2
        )

    # ==========================================================
    # DISCONNECT
    # ==========================================================

    def disconnect(self):

        if self.physics_client is not None:
            p.disconnect()
            self.physics_client = None

    # ==========================================================
    # RESET
    # ==========================================================

    def reset(self):

        p.resetSimulation()

        p.setGravity(0, 0, -9.81)

        plane_id = p.loadURDF("plane.urdf")

        p.changeDynamics(
            plane_id,
            -1,
            lateralFriction=1.2,
            spinningFriction=0.1,
            rollingFriction=0.1,
            restitution=0.0
        )

        # ------------------------------------------------------
        # Slight domain randomization
        # ------------------------------------------------------

        start_pos = [
            np.random.uniform(-0.01, 0.01),
            np.random.uniform(-0.01, 0.01),
            0.25
        ]

        start_orientation = p.getQuaternionFromEuler([
            np.random.uniform(-0.03, 0.03),
            np.random.uniform(-0.03, 0.03),
            np.random.uniform(-0.03, 0.03),
        ])

        self.robot_id = p.loadURDF(
            self.urdf_path,
            start_pos,
            start_orientation,
            flags=(
                p.URDF_USE_SELF_COLLISION
                |
                p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
            ),
        )

        num_joints = p.getNumJoints(self.robot_id)

        # ------------------------------------------------------
        # Randomized dynamics
        # ------------------------------------------------------

        for i in range(-1, num_joints):

            p.changeDynamics(
                self.robot_id,
                i,
                lateralFriction=np.random.uniform(1.0, 1.4),
                spinningFriction=0.1,
                rollingFriction=0.1,
                linearDamping=0.04,
                angularDamping=0.04,
                restitution=0.0,
            )

        # ------------------------------------------------------
        # Identify revolute joints
        # ------------------------------------------------------

        self.joint_indices = []

        for i in range(num_joints):

            joint_info = p.getJointInfo(
                self.robot_id,
                i
            )

            if joint_info[2] == p.JOINT_REVOLUTE:

                self.joint_indices.append(i)

        # ------------------------------------------------------
        # Identify feet
        # ------------------------------------------------------

        self.foot_link_indices = (
            self._identify_foot_links()
        )

        # ------------------------------------------------------
        # Stable initial pose
        # ------------------------------------------------------

        initial_angles = []

        for leg in range(6):

            coxa_angle = (
                -0.2
                if leg in [0, 2, 4]
                else 0.2
            )

            initial_angles.extend([
                coxa_angle,
                0.8,
                -1.3
            ])

        for i, joint_idx in enumerate(
            self.joint_indices
        ):

            if i < len(initial_angles):

                p.resetJointState(
                    self.robot_id,
                    joint_idx,
                    initial_angles[i]
                )

                p.changeDynamics(
                    self.robot_id,
                    joint_idx,
                    jointDamping=0.1
                )

        # ------------------------------------------------------
        # Disable default motors
        # ------------------------------------------------------

        for joint_idx in self.joint_indices:

            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_idx,
                controlMode=p.VELOCITY_CONTROL,
                force=0,
            )

    # ==========================================================
    # STEP
    # ==========================================================

    def step(self, joint_torques):

        for i, joint_idx in enumerate(self.joint_indices):

            torque = float(joint_torques[i])

            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_idx,
                controlMode=p.TORQUE_CONTROL,
                force=torque,
            )

        p.stepSimulation()

    # ==========================================================
    # STATE ACCESS
    # ==========================================================

    def get_base_position(self):

        pos, _ = p.getBasePositionAndOrientation(
            self.robot_id
        )

        return np.array(pos)

    # ----------------------------------------------------------

    def get_base_pose(self):

        pos, orn = p.getBasePositionAndOrientation(
            self.robot_id
        )

        return np.array(pos), np.array(orn)

    # ----------------------------------------------------------

    def get_base_velocity(self):

        lin_vel, ang_vel = p.getBaseVelocity(
            self.robot_id
        )

        return np.array(lin_vel), np.array(ang_vel)

    # ----------------------------------------------------------

    def get_joint_states(self):

        states = p.getJointStates(
            self.robot_id,
            self.joint_indices
        )

        positions = np.array([
            s[0] for s in states
        ])

        velocities = np.array([
            s[1] for s in states
        ])

        return positions, velocities

    # ----------------------------------------------------------

    def get_contact_states(self):

        contacts = np.zeros(6)

        for leg_idx, link_idx in enumerate(
            self.foot_link_indices
        ):

            if link_idx >= 0:

                pts = p.getContactPoints(
                    bodyA=self.robot_id,
                    linkIndexA=link_idx
                )

                if len(pts) > 0:

                    contacts[leg_idx] = 1.0

        return contacts.astype(np.float32)

    # ==========================================================
    # GRAVITY VECTOR
    # ==========================================================

    def get_gravity_vector_body(self):

        _, orn = self.get_base_pose()

        rot = np.array(
            p.getMatrixFromQuaternion(orn)
        ).reshape(3, 3)

        gravity_world = np.array([
            0,
            0,
            -1
        ])

        gravity_body = rot.T @ gravity_world

        return gravity_body.astype(np.float32)

    # ==========================================================
    # FOOT LINKS
    # ==========================================================

    def _identify_foot_links(self):

        num_joints = p.getNumJoints(
            self.robot_id
        )

        link_name_to_index = {}

        for i in range(num_joints):

            info = p.getJointInfo(
                self.robot_id,
                i
            )

            link_name = info[12].decode("utf-8")

            link_name_to_index[link_name] = i

        links = []

        for leg_idx in range(1, 7):

            links.append(
                link_name_to_index.get(
                    f"leg{leg_idx}_tibia",
                    -1
                )
            )

        return links

    # ==========================================================
    # CAMERA / GIF RECORDING
    # ==========================================================

    def _capture_frame(
        self,
        width=640,
        height=480,
        camera_distance=2.5,
        camera_yaw=45,
        camera_pitch=-25
    ):
        """
        Capture a frame from a camera that follows the robot.

        The camera target is the current robot position.
        """

        if self.robot_id is None:
            return None

        # ------------------------------------------------------
        # Robot position
        # ------------------------------------------------------

        robot_pos = self.get_base_position()

        target = [
            float(robot_pos[0]),
            float(robot_pos[1]),
            float(robot_pos[2]) + 0.15
        ]

        # ------------------------------------------------------
        # Camera matrices
        # ------------------------------------------------------

        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=target,
            distance=camera_distance,
            yaw=camera_yaw,
            pitch=camera_pitch,
            roll=0,
            upAxisIndex=2
        )

        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(width) / float(height),
            nearVal=0.1,
            farVal=100.0
        )

        # ------------------------------------------------------
        # Capture
        # ------------------------------------------------------

        _, _, rgba, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        # ------------------------------------------------------
        # Convert RGBA -> RGB
        # ------------------------------------------------------

        frame = np.array(
            rgba,
            dtype=np.uint8
        ).reshape(
            height,
            width,
            4
        )

        frame = frame[:, :, :3]

        return frame

    # ==========================================================

    def _save_gif(
        self,
        frames,
        output_path,
        fps=15
    ):
        """
        Save captured frames as a GIF.
        """

        if not frames:
            print(
                "  WARNING: No frames were captured."
            )
            return False

        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True
        )

        try:

            imageio.mimsave(
                output_path,
                frames,
                format="GIF",
                fps=fps,
                loop=0
            )

            return True

        except Exception as e:

            print(
                f"  ERROR saving GIF: {e}"
            )

            return False

    # ==========================================================
    # EVALUATION
    # ==========================================================

    def evaluate_controller(
        self,
        genome,
        duration=10.0,
        warmup_time=2.0,
        render=False,
        sample_interval=0.05,
        controller_interval=4,
        log_all_steps=False,

        # ------------------------------------------------------
        # Recording options
        # ------------------------------------------------------

        record=False,
        record_path=None,
        record_fps=15,
        record_width=640,
        record_height=480,
        record_camera_distance=2.5,
        record_camera_yaw=45,
        record_camera_pitch=-25,
    ):

        if self.physics_client is None:

            self.connect()

        self.reset()

        controller = (
            RecurrentTorqueHexapodController(genome)
        )

        controller.reset()

        start_pos = self.get_base_position()

        positions = []
        velocities = []
        torque_history = []
        contact_history = []
        angular_velocity_history = []
        transformer_state_sequence = []

        detailed_log = (
            []
            if log_all_steps
            else None
        )

        total_energy = 0.0

        num_steps = int(
            duration / self.time_step
        )

        warmup_steps = int(
            warmup_time / self.time_step
        )

        current_torques = np.zeros(18)

        # ======================================================
        # RECORDING SETUP
        # ======================================================

        frames = []

        if record and not record_path:

            raise ValueError(
                "record_path must be provided when record=True."
            )

        # Capture approximately record_fps frames per second.
        record_interval = max(
            1,
            int(
                1.0
                /
                (
                    self.time_step
                    *
                    record_fps
                )
            )
        )

        # ======================================================
        # SIMULATION LOOP
        # ======================================================

        for step_idx in range(num_steps):

            t = (
                step_idx
                *
                self.time_step
            )

            # ==================================================
            # CONTROLLER UPDATE
            # ==================================================

            if step_idx % controller_interval == 0:

                joint_pos, joint_vel = (
                    self.get_joint_states()
                )

                base_lin_vel, base_ang_vel = (
                    self.get_base_velocity()
                )

                contacts = (
                    self.get_contact_states()
                )

                gravity_body = (
                    self.get_gravity_vector_body()
                )

                controller_state = {

                    'joint_positions':
                        joint_pos,

                    'joint_velocities':
                        joint_vel,

                    'base_linear_velocity':
                        base_lin_vel,

                    'base_angular_velocity':
                        base_ang_vel,

                    'gravity_vector_body':
                        gravity_body,

                    'foot_contacts':
                        contacts,
                }

                current_torques = (
                    controller.get_torques(
                        controller_state
                    )
                )

            # ==================================================
            # APPLY TORQUES
            # ==================================================

            self.step(
                current_torques
            )

            # ==================================================
            # RENDER / RECORD
            # ==================================================

            if (
                render
                and self.gui
            ):

                # Keep GUI synchronized with real time.
                time.sleep(
                    self.time_step
                )

            # --------------------------------------------------
            # Ignore warm-up for measurements and recording
            # --------------------------------------------------

            if step_idx < warmup_steps:

                continue

            # ==================================================
            # RECORD FRAME
            # ==================================================

            if (
                record
                and
                step_idx >= warmup_steps
                and
                (
                    step_idx - warmup_steps
                ) % record_interval == 0
            ):

                frame = self._capture_frame(
                    width=record_width,
                    height=record_height,
                    camera_distance=record_camera_distance,
                    camera_yaw=record_camera_yaw,
                    camera_pitch=record_camera_pitch,
                )

                if frame is not None:

                    frames.append(frame)

            # ==================================================
            # STATE LOGGING
            # ==================================================

            base_pos, base_orn = (
                self.get_base_pose()
            )

            base_lin_vel, base_ang_vel = (
                self.get_base_velocity()
            )

            positions.append(
                base_pos
            )

            velocities.append(
                base_lin_vel
            )

            angular_velocity_history.append(
                base_ang_vel
            )

            contacts = (
                self.get_contact_states()
            )

            contact_history.append(
                contacts.copy()
            )

            torque_history.append(
                current_torques.copy()
            )

            # ==================================================
            # ENERGY
            # ==================================================

            _, joint_vel = (
                self.get_joint_states()
            )

            total_energy += (
                np.sum(
                    np.abs(
                        current_torques
                        *
                        joint_vel
                    )
                )
                *
                self.time_step
            )

            # ==================================================
            # TERMINATION
            # ==================================================

            roll, pitch, _ = (
                p.getEulerFromQuaternion(
                    base_orn
                )
            )

            if (
                abs(roll) > 1.6
                or
                abs(pitch) > 1.6
            ):

                break

            if np.any(
                np.isnan(base_pos)
            ):

                break

            # ==================================================
            # TRANSFORMER STATE
            # ==================================================

            if (
                config.BD_MODE
                in
                (
                    'transformer_latent',
                    'transformer_8d'
                )
            ):

                joint_pos, joint_vel = (
                    self.get_joint_states()
                )

                state_vec = np.concatenate([
                    joint_pos,
                    joint_vel,
                    current_torques,
                    contacts,
                    base_pos,
                    base_orn,
                    base_lin_vel,
                    base_ang_vel,
                ]).astype(
                    np.float32
                )

                transformer_state_sequence.append(
                    state_vec
                )

            # ==================================================
            # LOGGING
            # ==================================================

            if log_all_steps:

                detailed_log.append({

                    'time':
                        t,

                    'position':
                        base_pos.copy(),

                    'orientation':
                        base_orn.copy(),

                    'contacts':
                        contacts.copy(),

                    'torques':
                        current_torques.copy(),
                })

        # ======================================================
        # SAVE RECORDING
        # ======================================================

        if record:

            print(
                f"  Saving recording with "
                f"{len(frames)} frames..."
            )

            self._save_gif(
                frames,
                record_path,
                fps=record_fps
            )

            print(
                f"  Recording saved to:"
                f"\n    {record_path}"
            )

        # ======================================================
        # FITNESS
        # ======================================================

        if len(positions) == 0:

            return {

                'fitness':
                    -100.0,

                'behavior_descriptors':
                    np.zeros(2),

                'final_position':
                    None,

                'distance':
                    0.0,

                'energy':
                    0.0,

                'average_speed':
                    0.0,

                'stability_penalty':
                    0.0,
            }

        final_pos = positions[-1]

        displacement = (
            final_pos[:2]
            -
            start_pos[:2]
        )

        distance = np.linalg.norm(
            displacement
        )

        mean_velocity = np.mean(
            np.linalg.norm(
                velocities,
                axis=1
            )
        )

        energy_penalty = (
            config.ENERGY_WEIGHT
            *
            total_energy
        )

        stability_penalty = np.mean(
            np.linalg.norm(
                angular_velocity_history,
                axis=1
            )
        )

        # ======================================================
        # FITNESS MODE
        # ======================================================

        if config.FITNESS_MODE == "combined":

            fitness = (

                config.DISTANCE_WEIGHT
                *
                distance

                +

                config.AVG_SPEED_WEIGHT
                *
                mean_velocity

                -

                config.STABILITY_PENALTY_WEIGHT
                *
                stability_penalty

                -

                energy_penalty
            )

        elif config.FITNESS_MODE == "distance":

            fitness = distance

        elif config.FITNESS_MODE == "forward_velocity":

            fitness = (
                displacement[0]
                /
                config.SIMULATION_CONFIG["duration"]
            )

        elif config.FITNESS_MODE == "avg_speed":

            fitness = mean_velocity

        elif config.FITNESS_MODE == "stability":

            fitness = -stability_penalty

        else:

            raise ValueError(
                f"Unknown FITNESS_MODE: "
                f"{config.FITNESS_MODE}"
            )

        # ======================================================
        # DESCRIPTORS
        # ======================================================

        behavior_descriptors = (
            self._compute_behavior_descriptors(
                start_pos,
                final_pos,
                contact_history,
                transformer_state_sequence
            )
        )

        # ======================================================
        # RESULT
        # ======================================================

        result = {

            'fitness':
                float(fitness),

            'behavior_descriptors':
                behavior_descriptors,

            'final_position':
                final_pos,

            'distance':
                float(distance),

            'energy':
                float(total_energy),

            'average_speed':
                float(mean_velocity),

            'stability_penalty':
                float(stability_penalty),
        }

        if log_all_steps:

            result['detailed_log'] = (
                detailed_log
            )

        return result

    # ==========================================================
    # DESCRIPTORS
    # ==========================================================

    def _compute_behavior_descriptors(
        self,
        start_pos,
        final_pos,
        contact_history,
        transformer_state_sequence
    ):

        mode = config.BD_MODE

        # ------------------------------------------------------
        # BASELINE XY
        # ------------------------------------------------------

        if mode == 'xy_position':

            return np.array([
                final_pos[0],
                final_pos[1]
            ])

        # ------------------------------------------------------
        # CONTACT DESCRIPTOR
        # ------------------------------------------------------

        elif mode == 'foot_contacts':

            if len(contact_history) == 0:

                return np.zeros(6)

            contacts = np.array(
                contact_history
            )

            # Duty factor per leg
            return np.mean(
                contacts,
                axis=0
            )

        # ------------------------------------------------------
        # TRANSFORMER BDs
        # ------------------------------------------------------

        elif mode in (
            'transformer_latent',
            'transformer_8d'
        ):

            if len(
                transformer_state_sequence
            ) == 0:

                if mode == 'transformer_latent':

                    num_dims = len(
                        config.TRANSFORMER_BD_CONFIG.get(
                            'latent_indices',
                            [0, 1]
                        )
                    )

                else:

                    num_dims = (
                        config.TRANSFORMER_BD_CONFIG.get(
                            'bd_dim',
                            8
                        )
                    )

                return np.zeros(
                    num_dims
                )

            extractor = (
                self._get_transformer_bd_extractor()
            )

            states = np.array(
                transformer_state_sequence,
                dtype=np.float32
            )

            return extractor.compute_bd(
                states
            )

        else:

            raise ValueError(
                f"Unknown BD_MODE {mode}"
            )


# ==============================================================
# PARALLEL EVALUATION
# ==============================================================

TORQUE_GLOBAL_SIM = None


def init_torque_worker(sim_config):

    global TORQUE_GLOBAL_SIM

    TORQUE_GLOBAL_SIM = (
        HexapodTorqueSimulation(
            urdf_path=sim_config['urdf_path'],
            gui=False,
            time_step=sim_config['time_step'],
        )
    )

    TORQUE_GLOBAL_SIM.connect()


# ==============================================================
# SINGLE TORQUE EVALUATION
# ==============================================================

def evaluate_single_torque(genome):

    global TORQUE_GLOBAL_SIM

    logging_config = getattr(
        config,
        'LOGGING_CONFIG',
        {}
    )

    save_random_trajectories = (
        logging_config.get(
            'save_random_trajectories',
            True
        )
    )

    trajectory_save_probability = float(
        logging_config.get(
            'trajectory_save_probability',
            0.1
        )
    )

    log_this = (

        save_random_trajectories

        and

        trajectory_save_probability > 0.0

        and

        np.random.rand()
        <
        trajectory_save_probability
    )

    result = (
        TORQUE_GLOBAL_SIM.evaluate_controller(

            genome,

            duration=
                SIMULATION_CONFIG['duration'],

            warmup_time=
                SIMULATION_CONFIG['warmup_time'],

            sample_interval=
                SIMULATION_CONFIG['sample_interval'],

            controller_interval=
                SIMULATION_CONFIG['controller_interval'],

            log_all_steps=
                log_this
        )
    )

    # ==========================================================
    # SAVE RANDOM TRAJECTORY
    # ==========================================================

    if log_this:

        filename = (
            f"traj_{uuid.uuid4().hex}.pkl"
        )

        path = os.path.join(
            HexapodTorqueSimulation.RANDOM_LOG_DIR,
            filename
        )

        # Ensure behavior descriptors are explicitly saved.
        save_obj = {

            'result':
                result,

            'behavior_descriptors':
                result.get(
                    'behavior_descriptors'
                )
        }

        with open(
            path,
            'wb'
        ) as f:

            pickle.dump(
                save_obj,
                f
            )

    return (

        result['fitness'],

        result[
            'behavior_descriptors'
        ],

        result.get(
            'final_position'
        ),

        result.get(
            'distance',
            0.0
        ),

        result.get(
            'average_speed',
            0.0
        ),

        result.get(
            'energy',
            0.0
        ),

        result.get(
            'stability_penalty',
            0.0
        )
    )


# ==============================================================
# BATCH PARALLEL EVALUATION
# ==============================================================

def evaluate_batch_parallel_torque(
    genomes
):

    sim_config = {

        'urdf_path':
            URDF_PATH,

        'time_step':
            SIMULATION_CONFIG['time_step'],
    }

    with Pool(
        cpu_count(),
        initializer=init_torque_worker,
        initargs=(sim_config,),
    ) as pool:

        results = pool.map(
            evaluate_single_torque,
            genomes
        )

    fitnesses = np.array([
        r[0]
        for r in results
    ])

    bds = np.array([
        r[1]
        for r in results
    ])

    final_positions = [
        r[2]
        for r in results
    ]

    distances = np.array(
        [
            r[3]
            for r in results
        ],
        dtype=np.float32
    )

    average_speeds = np.array(
        [
            r[4]
            for r in results
        ],
        dtype=np.float32
    )

    energies = np.array(
        [
            r[5]
            for r in results
        ],
        dtype=np.float32
    )

    stability_penalties = np.array(
        [
            r[6]
            for r in results
        ],
        dtype=np.float32
    )

    return (

        fitnesses,

        bds,

        final_positions,

        distances,

        average_speeds,

        energies,

        stability_penalties
    )