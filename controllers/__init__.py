"""
Controllers module initialization
"""
from .hexapod_controller import HexapodController, CPGController
from .hexapod_goal_controller import GoalDirectedRecurrentTorqueHexapodController

__all__ = [
	'HexapodController',
	'CPGController',
	'GoalDirectedRecurrentTorqueHexapodController',
]
