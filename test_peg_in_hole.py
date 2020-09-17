import redis
from sai2_environment.robot_env import RobotEnv
from sai2_environment.action_space import ActionSpace

import numpy as np
import time
import torch


def main():

    # Robot stuff
    action_space = ActionSpace.DELTA_EE_POSE_IMPEDANCE 
    #action_space = ActionSpace.ABS_JOINT_POSITION_IMPEDANCE
    blocking_action = True
    env = RobotEnv(name='peg_in_hole',
                   simulation=True,
                   action_space=action_space,
                   isotropic_gains=True,
                   render=False,
                   blocking_action=blocking_action,
                   rotation_axis=(0, 0, 1),
                   observation_type=dict(camera=1, q=0, dq=0, tau=0, x=1, dx=1))    
    env.reset()               

 
    # for i in range(0,4):
    #     test_a = np.array([0,0,-0.05,0,0,0])
    #     o2, r, d, _ = env.step(test_a)
    #     test_a = np.array([0.05,0,0,0,0,0])
    #     o2, r, d, _ = env.step(test_a)
    test_a = np.array([0.12,0,-0.2,0,0,0])
    o2, r, d, _ = env.step(test_a)
    test_a = np.array([0.02,0,-0.06,0,0,0])
    o2, r, d, _ = env.step(test_a)
    test_a = np.array([0.04,0,-0.05,0,0,0])
    o2, r, d, _ = env.step(test_a)
    test_a = np.array([0.02,0,0,0,0,0])

    o2, r, d, _ = env.step(test_a)
    test_a = np.array([0.035,0,-0.04,0,0,0])

    o2, r, d, _ = env.step(test_a)
    test_a = np.array([0,0,0.12,0,0,0])
    o2, r, d, _ = env.step(test_a)


    print("Done")

if __name__ == "__main__":
    main()