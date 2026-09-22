"""
Hexapod neural-network controller.

The genome directly encodes all weights and biases of a small MLP policy.
"""
import numpy as np


class HexapodController:
    """Feedforward neural-network controller for the hexapod."""

    NUM_LEGS = 6
    JOINTS_PER_LEG = 3
    TOTAL_JOINTS = NUM_LEGS * JOINTS_PER_LEG

    # Observation = joints + velocities + base kinematics + time features
    INPUT_DIM = TOTAL_JOINTS + TOTAL_JOINTS + 3 + 3 + 3 + 2
    HIDDEN_DIM = 16
    OUTPUT_DIM = TOTAL_JOINTS

    def __init__(self, genome: np.ndarray):
        self.genome = np.asarray(genome, dtype=np.float32)
        self.total_joints = self.TOTAL_JOINTS

        expected_size = self.get_genome_size()
        if self.genome.size != expected_size:
            raise ValueError(
                f"Invalid genome size {self.genome.size}. Expected {expected_size} for NN controller."
            )

        self.base_pose = self._build_base_pose()
        self.max_delta = self._build_action_scales()
        self._unpack_genome()

        self.joint_names = []
        for i in range(1, 7):
            self.joint_names.extend([
                f'leg{i}_coxa_joint',
                f'leg{i}_femur_joint',
                f'leg{i}_tibia_joint'
            ])

    @classmethod
    def get_genome_size(cls) -> int:
        """Return total number of NN parameters (weights + biases)."""
        in_dim = cls.INPUT_DIM
        hid_dim = cls.HIDDEN_DIM
        out_dim = cls.OUTPUT_DIM
        return (in_dim * hid_dim + hid_dim) + (hid_dim * out_dim + out_dim)

    @classmethod
    def random_genome(cls) -> np.ndarray:
        """Xavier-like random initialization for stable initial policies."""
        rng = np.random.default_rng()

        fan_in_1, fan_out_1 = cls.INPUT_DIM, cls.HIDDEN_DIM
        fan_in_2, fan_out_2 = cls.HIDDEN_DIM, cls.OUTPUT_DIM
        limit_1 = np.sqrt(6.0 / (fan_in_1 + fan_out_1))
        limit_2 = np.sqrt(6.0 / (fan_in_2 + fan_out_2))

        w1 = rng.uniform(-limit_1, limit_1, size=(fan_in_1, fan_out_1)).flatten()
        b1 = np.zeros(fan_out_1, dtype=np.float32)
        w2 = rng.uniform(-limit_2, limit_2, size=(fan_in_2, fan_out_2)).flatten()
        b2 = np.zeros(fan_out_2, dtype=np.float32)

        return np.concatenate([w1, b1, w2, b2]).astype(np.float32)

    def _unpack_genome(self):
        """Decode flattened genome into matrix/bias tensors for the MLP."""
        idx = 0

        w1_size = self.INPUT_DIM * self.HIDDEN_DIM
        self.w1 = self.genome[idx:idx + w1_size].reshape(self.INPUT_DIM, self.HIDDEN_DIM)
        idx += w1_size

        self.b1 = self.genome[idx:idx + self.HIDDEN_DIM]
        idx += self.HIDDEN_DIM

        w2_size = self.HIDDEN_DIM * self.OUTPUT_DIM
        self.w2 = self.genome[idx:idx + w2_size].reshape(self.HIDDEN_DIM, self.OUTPUT_DIM)
        idx += w2_size

        self.b2 = self.genome[idx:idx + self.OUTPUT_DIM]

    def _build_base_pose(self) -> np.ndarray:
        """Nominal standing pose used as action center."""
        pose = []
        for leg in range(6):
            coxa = -0.2 if leg in [0, 2, 4] else 0.2
            pose.extend([coxa, 0.8, -1.3])
        return np.array(pose, dtype=np.float32)

    def _build_action_scales(self) -> np.ndarray:
        """Per-joint motion amplitude around the base pose."""
        scales = np.zeros(self.total_joints, dtype=np.float32)
        for leg in range(6):
            base = leg * 3
            scales[base + 0] = 0.2
            scales[base + 1] = 0.4
            scales[base + 2] = 0.5
        return scales

    def _build_input(self, t: float, robot_state: dict) -> np.ndarray:
        """Construct normalized observation vector for the policy network."""
        joint_pos = np.asarray(robot_state.get('joint_positions', np.zeros(self.total_joints)), dtype=np.float32)
        joint_vel = np.asarray(robot_state.get('joint_velocities', np.zeros(self.total_joints)), dtype=np.float32)
        base_lin = np.asarray(robot_state.get('base_linear_velocity', np.zeros(3)), dtype=np.float32)
        base_ang = np.asarray(robot_state.get('base_angular_velocity', np.zeros(3)), dtype=np.float32)
        base_euler = np.asarray(robot_state.get('base_euler', np.zeros(3)), dtype=np.float32)

        # Lightweight normalization for numeric stability during evolution.
        joint_pos_n = np.tanh(joint_pos)
        joint_vel_n = np.tanh(0.5 * joint_vel)
        base_lin_n = np.tanh(0.5 * base_lin)
        base_ang_n = np.tanh(0.2 * base_ang)
        base_euler_n = np.tanh(base_euler)

        gait_freq_hz = 1.0
        phase = 2.0 * np.pi * gait_freq_hz * t
        time_features = np.array([np.sin(phase), np.cos(phase)], dtype=np.float32)

        return np.concatenate([
            joint_pos_n,
            joint_vel_n,
            base_lin_n,
            base_ang_n,
            base_euler_n,
            time_features,
        ])

    def get_joint_angles(self, t: float, robot_state: dict = None, ramp_up_duration: float = 0.1) -> np.ndarray:
        """
        Compute target joint angles from NN policy.

        Args:
            t: Current simulation time in seconds.
            robot_state: Dict with robot kinematics used as NN input.
            ramp_up_duration: Smoothly ramps action magnitude at episode start.
        """
        if robot_state is None:
            robot_state = {}

        x = self._build_input(t, robot_state)
        h = np.tanh(x @ self.w1 + self.b1)
        y = np.tanh(h @ self.w2 + self.b2)

        #if t < ramp_up_duration:
        #    action_scale = (t / ramp_up_duration) ** 2
        #else:
        action_scale = 1.0

        target = self.base_pose + action_scale * (self.max_delta * y)
        return target.astype(np.float32)


class CPGController:
    """
    Alternative: Central Pattern Generator (CPG) based controller
    Uses coupled oscillators for more biological gait generation
    """
    
    def __init__(self, genome: np.ndarray):
        """
        Initialize CPG controller
        
        Args:
            genome: Parameter vector encoding CPG parameters
        """
        self.genome = genome
        self.num_legs = 6
        self.joints_per_leg = 3
        
        # Oscillator states (amplitude and phase for each joint)
        self.oscillator_states = np.zeros((self.num_legs * self.joints_per_leg, 2))
        
        # Parse genome for coupling weights and intrinsic frequencies
        self._parse_genome()
    
    def _parse_genome(self):
        """Extract CPG parameters from genome"""
        # This is a simplified version
        # In practice, you'd have coupling weights between oscillators
        pass
    
    def step(self, dt: float) -> np.ndarray:
        """
        Update oscillator states and return joint angles
        
        Args:
            dt: Time step
            
        Returns:
            Joint angles
        """
        # CPG dynamics would go here
        # For now, placeholder
        return np.zeros(18)
