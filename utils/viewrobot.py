import os
import sys
import time

import pybullet as p
import pybullet_data

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import URDF_PATH

# Start GUI
physicsClient = p.connect(p.GUI)

# Optional: better camera defaults
p.resetDebugVisualizerCamera(
    cameraDistance=1.5,
    cameraYaw=50,
    cameraPitch=-35,
    cameraTargetPosition=[0, 0, 0]
)

# Add search path for plane
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load ground
planeId = p.loadURDF("plane.urdf")

# Load your hexapod
robot = p.loadURDF(
    URDF_PATH,
    basePosition=[0, 0, 0.2],
    useFixedBase=False
)

print("Hexapod loaded successfully!")
print("Use mouse to rotate view. Press Ctrl+C or close window to exit.")

# Set gravity
p.setGravity(0, 0, -9.81)

# Run simulation
try:
    while True:
        p.stepSimulation()
        time.sleep(1./240.)
except KeyboardInterrupt:
    print("\nShutting down...")
finally:
    p.disconnect()