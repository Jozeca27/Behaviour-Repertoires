import numpy as np


class RecurrentTorqueHexapodController:
    """
    Recurrent torque-based neural controller for hexapod locomotion.

    Features:
    - torque control
    - recurrent hidden state
    - contact sensing
    - torque smoothing
    - no explicit oscillator/clock
    """

    NUM_LEGS = 6
    JOINTS_PER_LEG = 3
    TOTAL_JOINTS = NUM_LEGS * JOINTS_PER_LEG

    CONTACT_INPUTS = NUM_LEGS

    # Inputs:
    # joint positions
    # joint velocities
    # base linear velocity
    # base angular velocity
    # gravity vector in body frame
    # foot contacts
    INPUT_DIM = (
        TOTAL_JOINTS +
        TOTAL_JOINTS +
        3 +
        3 +
        3 +
        CONTACT_INPUTS
    )

    HIDDEN_DIM = 32
    OUTPUT_DIM = TOTAL_JOINTS

    def __init__(self, genome: np.ndarray):

        self.genome = np.asarray(genome, dtype=np.float32)

        expected_size = self.get_genome_size()

        if self.genome.size != expected_size:
            raise ValueError(
                f"Genome size {self.genome.size} "
                f"does not match expected {expected_size}"
            )

        self.hidden_state = np.zeros(self.HIDDEN_DIM, dtype=np.float32)

        self.prev_torques = np.zeros(self.OUTPUT_DIM, dtype=np.float32)

        self.max_torque = self._build_torque_scales()

        self._unpack_genome()

    @classmethod
    def get_genome_size(cls):

        in_dim = cls.INPUT_DIM
        hid_dim = cls.HIDDEN_DIM
        out_dim = cls.OUTPUT_DIM

        # recurrent layer:
        # input -> hidden
        # hidden -> hidden
        # hidden bias
        recurrent_params = (
            (in_dim * hid_dim) +
            (hid_dim * hid_dim) +
            hid_dim
        )

        # hidden -> output
        output_params = (
            (hid_dim * out_dim) +
            out_dim
        )

        return recurrent_params + output_params

    @classmethod
    def random_genome(cls):

        rng = np.random.default_rng()

        hid_dim = cls.HIDDEN_DIM

        # Xavier init
        limit_in = np.sqrt(6.0 / (cls.INPUT_DIM + hid_dim))
        limit_rec = np.sqrt(6.0 / (hid_dim + hid_dim))
        limit_out = np.sqrt(6.0 / (hid_dim + cls.OUTPUT_DIM))

        w_in = rng.uniform(
            -limit_in,
            limit_in,
            size=(cls.INPUT_DIM, hid_dim)
        ).flatten()

        w_rec = rng.uniform(
            -limit_rec,
            limit_rec,
            size=(hid_dim, hid_dim)
        ).flatten()

        b_h = np.zeros(hid_dim, dtype=np.float32)

        w_out = rng.uniform(
            -limit_out,
            limit_out,
            size=(hid_dim, cls.OUTPUT_DIM)
        ).flatten()

        b_out = np.zeros(cls.OUTPUT_DIM, dtype=np.float32)

        genome = np.concatenate([
            w_in,
            w_rec,
            b_h,
            w_out,
            b_out
        ])

        return genome.astype(np.float32)

    @classmethod
    def random_genome_scales(cls):

        rng = np.random.default_rng()

        scale = rng.choice([0.25, 0.5, 1.0, 2.0])

        w_in = rng.uniform(-scale, scale,
                        size=(cls.INPUT_DIM, cls.HIDDEN_DIM)).flatten()

        w_rec = rng.uniform(-scale, scale,
                            size=(cls.HIDDEN_DIM, cls.HIDDEN_DIM)).flatten()

        b_h = rng.uniform(-0.2, 0.2, size=cls.HIDDEN_DIM)

        w_out = rng.uniform(-scale, scale,
                            size=(cls.HIDDEN_DIM, cls.OUTPUT_DIM)).flatten()

        b_out = rng.uniform(-0.2, 0.2, size=cls.OUTPUT_DIM)

        return np.concatenate([
            w_in,
            w_rec,
            b_h,
            w_out,
            b_out
        ]).astype(np.float32)

    @classmethod
    def random_genome_random(cls):

        rng = np.random.default_rng()

        strategy = rng.integers(4)

        if strategy == 0:
            scale = 0.25
        elif strategy == 1:
            scale = 0.5
        elif strategy == 2:
            scale = 1.0
        else:
            scale = 2.0

        w_in = rng.uniform(-scale, scale,
                        size=(cls.INPUT_DIM, cls.HIDDEN_DIM)).flatten()

        w_rec = rng.uniform(-scale, scale,
                            size=(cls.HIDDEN_DIM, cls.HIDDEN_DIM)).flatten()

        w_out = rng.uniform(-scale, scale,
                            size=(cls.HIDDEN_DIM, cls.OUTPUT_DIM)).flatten()

        b_h = rng.uniform(-0.5, 0.5, cls.HIDDEN_DIM)
        b_out = rng.uniform(-0.5, 0.5, cls.OUTPUT_DIM)

        return np.concatenate([
            w_in,
            w_rec,
            b_h,
            w_out,
            b_out
        ]).astype(np.float32)

    def reset(self):
        """
        Reset hidden dynamics at episode start.
        """

        self.hidden_state[:] = 0.0
        self.prev_torques[:] = 0.0

    def _build_torque_scales(self):

        scales = np.zeros(self.TOTAL_JOINTS, dtype=np.float32)

        for leg in range(self.NUM_LEGS):

            base = leg * 3

            scales[base + 0] = 2.0
            scales[base + 1] = 4.0
            scales[base + 2] = 5.0

        return scales

    def _unpack_genome(self):

        idx = 0

        # input -> hidden
        size = self.INPUT_DIM * self.HIDDEN_DIM

        self.w_in = self.genome[idx:idx + size].reshape(
            self.INPUT_DIM,
            self.HIDDEN_DIM
        )

        idx += size

        # recurrent
        size = self.HIDDEN_DIM * self.HIDDEN_DIM

        self.w_rec = self.genome[idx:idx + size].reshape(
            self.HIDDEN_DIM,
            self.HIDDEN_DIM
        )

        idx += size

        # hidden bias
        self.b_h = self.genome[idx:idx + self.HIDDEN_DIM]

        idx += self.HIDDEN_DIM

        # hidden -> output
        size = self.HIDDEN_DIM * self.OUTPUT_DIM

        self.w_out = self.genome[idx:idx + size].reshape(
            self.HIDDEN_DIM,
            self.OUTPUT_DIM
        )

        idx += size

        # output bias
        self.b_out = self.genome[idx:idx + self.OUTPUT_DIM]

    def _build_input(self, robot_state: dict):

        joint_pos = np.asarray(
            robot_state.get(
                'joint_positions',
                np.zeros(self.TOTAL_JOINTS)
            ),
            dtype=np.float32
        )

        joint_vel = np.asarray(
            robot_state.get(
                'joint_velocities',
                np.zeros(self.TOTAL_JOINTS)
            ),
            dtype=np.float32
        )

        base_lin = np.asarray(
            robot_state.get(
                'base_linear_velocity',
                np.zeros(3)
            ),
            dtype=np.float32
        )

        base_ang = np.asarray(
            robot_state.get(
                'base_angular_velocity',
                np.zeros(3)
            ),
            dtype=np.float32
        )

        # Better than Euler angles:
        gravity_vector = np.asarray(
            robot_state.get(
                'gravity_vector_body',
                np.array([0, 0, -1])
            ),
            dtype=np.float32
        )

        foot_contacts = np.asarray(
            robot_state.get(
                'foot_contacts',
                np.zeros(self.NUM_LEGS)
            ),
            dtype=np.float32
        )

        # normalization
        joint_pos_n = np.tanh(joint_pos)
        joint_vel_n = np.tanh(0.1 * joint_vel)

        base_lin_n = np.tanh(0.25 * base_lin)
        base_ang_n = np.tanh(0.25 * base_ang)

        x = np.concatenate([
            joint_pos_n,
            joint_vel_n,
            base_lin_n,
            base_ang_n,
            gravity_vector,
            foot_contacts
        ])

        return x.astype(np.float32)

    def get_torques(
        self,
        robot_state: dict,
        smoothing: float = 0.0
    ) -> np.ndarray:
        """
        Compute smoothed torque commands.

        smoothing:
            0.0 = no smoothing
            0.9 = very smooth
        """

        x = self._build_input(robot_state)

        # recurrent hidden dynamics
        self.hidden_state = np.tanh(
            x @ self.w_in +
            self.hidden_state @ self.w_rec +
            self.b_h
        )

        raw_output = np.tanh(
            self.hidden_state @ self.w_out +
            self.b_out
        )

        torques = self.max_torque * raw_output

        # exponential smoothing
        torques = (
            smoothing * self.prev_torques +
            (1.0 - smoothing) * torques
        )

        self.prev_torques = torques.copy()

        # safety clipping
        torques = np.clip(
            torques,
            -self.max_torque,
            self.max_torque
        )

        return torques.astype(np.float32)