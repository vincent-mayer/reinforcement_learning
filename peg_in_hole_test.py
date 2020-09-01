import redis
from sai2_environment.robot_env import RobotEnv
from sai2_environment.action_space import ActionSpace

import numpy as np
import time
from PIL import Image
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
                   
    o, ep_ret, ep_len = env.reset(), 0, 0
    test_a = np.array([0,0,-0.3,0,0,0])
    test_b = np.array([0,0,-0.02,0,0,0])
    o2, r, d, _ = env.step(test_a)
    time.sleep(2)
    o2_wait,_,_,_ = env.step(np.array([0,0,0,0,0,0]))
    for i in range(0,100):
        x = o2[1]
        o2, r, d, _ = env.step(test_b)



if __name__ == "__main__":
    main()