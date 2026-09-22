import numpy as np

from controllers.hexapod_controller_final import RecurrentTorqueHexapodController


class GoalDirectedRecurrentTorqueHexapodController(
    RecurrentTorqueHexapodController
):
    """
    Recurrent torque-based controller with an explicit goal input.

    The goal can be provided either as:
    - goal_position + base_position, or
    - goal_vector

    If nothing is provided, the goal input defaults to zeros, so the
    controller still behaves like the base controller.
    """

    GOAL_INPUTS = 3

    INPUT_DIM = (
        RecurrentTorqueHexapodController.INPUT_DIM + GOAL_INPUTS
    )

    def _build_input(self, robot_state: dict):

        base_input = super()._build_input(robot_state)

        goal_vector = robot_state.get('goal_vector', None)

        if goal_vector is None:

            goal_position = robot_state.get('goal_position', None)
            base_position = robot_state.get('base_position', None)

            if goal_position is not None and base_position is not None:
                goal_vector = np.asarray(goal_position, dtype=np.float32) - np.asarray(
                    base_position,
                    dtype=np.float32,
                )
            else:
                goal_vector = np.zeros(3, dtype=np.float32)

        goal_vector = np.asarray(goal_vector, dtype=np.float32)

        if goal_vector.size != self.GOAL_INPUTS:
            raise ValueError(
                f"goal_vector must have shape ({self.GOAL_INPUTS},)"
            )

        goal_vector_n = np.tanh(0.25 * goal_vector)

        return np.concatenate([
            base_input,
            goal_vector_n,
        ]).astype(np.float32)