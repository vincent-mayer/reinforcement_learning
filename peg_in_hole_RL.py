from sai2_environment.robot_env import RobotEnv
from sai2_environment.action_space import ActionSpace

import torch

from sai2_environment.reinforcement_learning.utils.run_utils import setup_logger_kwargs
from sai2_environment.reinforcement_learning.rl_algos import sac


def main():
    # If Debug mode don't log
    debug = False
    # Parameters for MLP actor
    hid = 256
    l = 2
    #*** Stuff from OpenAI ***#
    logger_kwargs = setup_logger_kwargs("peg_in_hole_test", 0, datestamp=True, data_dir='logs/') # Vars: exp_name, seed
    ac_kwargs = dict(hidden_sizes=[hid]*l)
    torch.set_num_threads(torch.get_num_threads())

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
    # Run SAC
    sac(env, logger_kwargs = logger_kwargs, debug=debug, ac_kwargs=ac_kwargs)

if __name__ == "__main__":
    main()