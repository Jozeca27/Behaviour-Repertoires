import os
import pickle
import uuid
from multiprocessing import Pool, cpu_count
from typing import Dict, Tuple

import numpy as np
import pybullet as p
import pybullet_data

import config
from config import SIMULATION_CONFIG, URDF_PATH, GOAL_VECTORS
from controllers.hexapod_goal_controller import (GoalDirectedRecurrentTorqueHexapodController)


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

        # Lazy-load the appropriate transformer BD extractor implementation
        # depending on the configured BD mode. This supports both the
        # latent+PCA extractor and a learned 8D BD head extractor.
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
        )

        # Slight domain randomization
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

        self.joint_indices = []

        for i in range(num_joints):
            joint_info = p.getJointInfo(
                self.robot_id,
                i
            )
            if joint_info[2] == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)

        self.foot_link_indices = (
            self._identify_foot_links()
        )
        # Stable initial pose
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

        # Disable default motors
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
        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        return np.array(pos)

    def get_base_pose(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        return np.array(pos), np.array(orn)

    def get_base_velocity(self):
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id)
        return np.array(lin_vel), np.array(ang_vel)

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

    def get_contact_states(self):
        contacts = np.zeros(6)
        for leg_idx, link_idx in enumerate(self.foot_link_indices):

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
        rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        gravity_world = np.array([0,0,-1])
        gravity_body = rot.T @ gravity_world
        return gravity_body.astype(np.float32)

    # ==========================================================
    # FOOT LINKS
    # ==========================================================

    def _identify_foot_links(self):
        num_joints = p.getNumJoints(self.robot_id)
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
                link_name_to_index.get(f"link_{leg_idx}_3", -1)
            )

        return links

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
        goal_vector=None,
    ):

        if self.physics_client is None:
            self.connect()

        self.reset()

        controller = GoalDirectedRecurrentTorqueHexapodController(genome)

        controller.reset()

        start_pos = self.get_base_position()

        positions = []
        velocities = []
        torque_history = []
        contact_history = []
        angular_velocity_history = []
        transformer_state_sequence = []
        detailed_log = [] if log_all_steps else None
        total_energy = 0.0

        num_steps = int(duration / self.time_step)
        warmup_steps = int(warmup_time / self.time_step)
        current_torques = np.zeros(18)

        for step_idx in range(num_steps):
            t = step_idx * self.time_step

            # ==================================================
            # CONTROLLER UPDATE
            # ==================================================

            if step_idx % controller_interval == 0:
                joint_pos, joint_vel = (self.get_joint_states())

                base_lin_vel, base_ang_vel = (self.get_base_velocity())

                contacts = (self.get_contact_states())

                gravity_body = (self.get_gravity_vector_body())

                controller_state = {
                    'joint_positions':joint_pos,
                    'joint_velocities':joint_vel,
                    'base_linear_velocity':base_lin_vel,
                    'base_angular_velocity':base_ang_vel,
                    'gravity_vector_body':gravity_body,
                    'foot_contacts':contacts,
                }
                if goal_vector is not None:
                    controller_state["goal_vector"] = goal_vector

                current_torques = (controller.get_torques(controller_state))

            # ==================================================
            # APPLY TORQUES
            # ==================================================

            self.step(current_torques)

            # ==================================================
            # AFTER WARMUP
            # ==================================================

            if step_idx < warmup_steps:
                continue

            base_pos, base_orn = (self.get_base_pose())

            base_lin_vel, base_ang_vel = (self.get_base_velocity())

            positions.append(base_pos)

            velocities.append(base_lin_vel)

            angular_velocity_history.append(base_ang_vel)

            contacts = self.get_contact_states()

            contact_history.append(contacts.copy())

            torque_history.append(current_torques.copy())

            # ==================================================
            # ENERGY
            # ==================================================

            _, joint_vel = self.get_joint_states()

            total_energy += np.sum(np.abs(current_torques * joint_vel)) * self.time_step

            # ==================================================
            # TERMINATION
            # ==================================================

            roll, pitch, _ = (p.getEulerFromQuaternion(base_orn))

            if (
                abs(roll) > 1.6
                or
                abs(pitch) > 1.6
            ):
                break

            if np.any(np.isnan(base_pos)):
                break

            # ==================================================
            # TRANSFORMER STATE
            # ==================================================

            if (
                config.BD_MODE
                in
                ('transformer_latent', 'transformer_8d')
            ):

                joint_pos, joint_vel = (self.get_joint_states())

                state_vec = np.concatenate([
                    joint_pos,
                    joint_vel,
                    current_torques,
                    contacts,
                    base_pos,
                    base_orn,
                    base_lin_vel,
                    base_ang_vel,
                ]).astype(np.float32)

                transformer_state_sequence.append(state_vec)

            # ==================================================
            # LOGGING
            # ==================================================

            if log_all_steps:

                detailed_log.append({
                    'time': t,
                    'position':base_pos.copy(),
                    'orientation':base_orn.copy(),
                    'contacts':contacts.copy(),
                    'torques':current_torques.copy(),
                })

            # ==================================================
            # RENDER
            # ==================================================

            if render and self.gui:

                import time
                time.sleep(self.time_step)

        # ======================================================
        # FITNESS
        # ======================================================

        if len(positions) == 0:
            mode = config.BD_MODE
            if mode == 'xy_position':
                bd_dim = 2
            elif mode == 'duty_factor':
                bd_dim = 6
            elif mode == 'transformer_latent':
                bd_dim = len(config.TRANSFORMER_BD_CONFIG.get('latent_indices', [0, 1]))
            elif mode == 'transformer_8d':
                bd_dim = config.TRANSFORMER_BD_CONFIG.get('bd_dim', 8)
            else:
                raise ValueError(f"Unknown BD_MODE {mode}")

            return {
                'fitness': -100.0,
                'behavior_descriptors': np.zeros(bd_dim, dtype=np.float32),
                'final_position': None,
            }

        final_pos = positions[-1]
        displacement = (final_pos[:2] - start_pos[:2])

        goal_xy = np.asarray(goal_vector[:2], dtype=np.float32)

        goal_xy /= (
            np.linalg.norm(goal_xy)
            + 1e-8
        )

        directional_progress = np.dot(
            displacement,
            goal_xy
        )

        mean_velocity = np.mean(
            np.linalg.norm(
                velocities,
                axis=1
            )
        )

        energy_penalty = (
            config.ENERGY_WEIGHT * total_energy
        )

        stability_penalty = np.mean(
            np.linalg.norm(
                angular_velocity_history,
                axis=1
            )
        )

        fitness = (
            config.DIRECTION_WEIGHT * directional_progress
            + config.AVG_SPEED_WEIGHT * mean_velocity
            - config.STABILITY_PENALTY_WEIGHT * stability_penalty
            - energy_penalty
        )
        
        #fitness = directional_progress

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

        result = {
            'fitness': float(fitness),
            'behavior_descriptors': behavior_descriptors,
            'final_position': final_pos,
            'distance': float(directional_progress),
            'average_speed': float(mean_velocity),
            'energy': float(total_energy),
            'stability_penalty': float(stability_penalty),
        }

        if log_all_steps:
            result['detailed_log'] = detailed_log

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
            ], dtype=np.float32)

        # ------------------------------------------------------
        # DUTY FACTOR (fraction of stance/contact per leg)
        # ------------------------------------------------------

        elif mode == 'duty_factor':

            if len(contact_history) == 0:
                return np.zeros(6, dtype=np.float32)

            contacts = np.asarray(contact_history, dtype=np.float32)
            return np.mean(contacts, axis=0).astype(np.float32)

        # ------------------------------------------------------
        # TRANSFORMER-BASED BDs (latent PCA or learned 8D head)
        # ------------------------------------------------------

        elif mode in ('transformer_latent', 'transformer_8d'):

            if len(transformer_state_sequence) == 0:
                if mode == 'transformer_latent':
                    num_dims = len(config.TRANSFORMER_BD_CONFIG.get('latent_indices', [0, 1]))
                else:
                    num_dims = config.TRANSFORMER_BD_CONFIG.get('bd_dim', 8)

                return np.zeros(num_dims, dtype=np.float32)

            extractor = self._get_transformer_bd_extractor()
            states = np.array(transformer_state_sequence, dtype=np.float32)
            return extractor.compute_bd(states)

        else:
            raise ValueError(f"Unknown BD_MODE {mode}")


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


def evaluate_single_goal_conditioned(genome):

    global TORQUE_GLOBAL_SIM

    fitnesses = []
    run_results = []

    for goal in config.GOAL_VECTORS:

        result = TORQUE_GLOBAL_SIM.evaluate_controller(
            genome,
            duration=SIMULATION_CONFIG['duration'],
            warmup_time=SIMULATION_CONFIG['warmup_time'],
            sample_interval=SIMULATION_CONFIG['sample_interval'],
            controller_interval=SIMULATION_CONFIG['controller_interval'],
            goal_vector=goal,
        )

        fitnesses.append(result['fitness'])
        run_results.append(result)

    mean_fitness = float(np.mean(fitnesses))

    # ==================================================
    # BD AGGREGATION
    # ==================================================
    if config.BD_MODE == 'xy_position':

        bd = np.concatenate([
            np.array([r['final_position'][0], r['final_position'][1]])
            for r in run_results
        ]).astype(np.float32)

    else:

        transformer_bds = np.stack(
            [r['behavior_descriptors'] for r in run_results],
            axis=0
        )

        bd = np.mean(transformer_bds, axis=0).astype(np.float32)

    return (
        mean_fitness,
        bd,
        run_results[0]['final_position'],
        float(np.mean([r['distance'] for r in run_results])),
        float(np.mean([r['average_speed'] for r in run_results])),
        float(np.mean([r['energy'] for r in run_results])),
        float(np.mean([r['stability_penalty'] for r in run_results])),
    )

def evaluate_batch_parallel_goal_conditioned(genomes):

    sim_config = {
        'urdf_path': URDF_PATH,
        'time_step': SIMULATION_CONFIG['time_step'],
    }

    with Pool(
        cpu_count(),
        initializer=init_torque_worker,
        initargs=(sim_config,),
    ) as pool:

        results = pool.map(
            evaluate_single_goal_conditioned,
            genomes
        )

    fitnesses = np.array([r[0] for r in results])

    bds = np.array([r[1] for r in results])

    final_positions = [r[2] for r in results]

    distances = np.array([r[3] for r in results], dtype=np.float32)

    average_speeds = np.array([r[4] for r in results], dtype=np.float32)

    energies = np.array([r[5] for r in results], dtype=np.float32)

    stability_penalties = np.array([r[6] for r in results], dtype=np.float32)

    #print("bds.shape =", bds.shape)
    #print("bds.dtype =", bds.dtype)
    #print("first bd =", bds[0])
    #print("len first bd =", len(bds[0]))

    return (
        fitnesses,
        bds,
        final_positions,
        distances,
        average_speeds,
        energies,
        stability_penalties,
    )