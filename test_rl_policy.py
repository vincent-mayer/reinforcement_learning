import time
import os
import os.path as osp
import torch
from sai2_environment.reinforcement_learning import EpochLogger
from sai2_environment.robot_env import RobotEnv
from sai2_environment.action_space import ActionSpace

def load_policy_and_env(fpath, itr='last', deterministic=False):
    # handle which epoch to load from
    if itr=='last':
        # check filenames for epoch (AKA iteration) numbers, find maximum value

        pytsave_path = osp.join(fpath, 'pyt_save')
        # Each file in this folder has naming convention 'modelXX.pt', where
        # 'XX' is either an integer or empty string. Empty string case
        # corresponds to len(x)==8, hence that case is excluded.
        saves = [int(x.split('.')[0][5:]) for x in os.listdir(pytsave_path) if len(x)>8 and 'model' in x]

        itr = '%d'%max(saves) if len(saves) > 0 else ''

    else:
        assert isinstance(itr, int), \
            "Bad value provided for itr (needs to be int or 'last')."
        itr = '%d'%itr

    # load the get_action function
    get_action = load_pytorch_policy(fpath, itr, deterministic)
    env = None

    return env, get_action


def load_pytorch_policy(fpath, itr, deterministic=False):
    """ Load a pytorch policy saved with Spinning Up Logger."""
    
    fname = osp.join(fpath, 'pyt_save', 'model'+itr+'.pt')
    print('\n\nLoading from %s.\n\n'%fname)

    model = torch.load(fname)

    # make function for producing an action given a single state
    def get_action(x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            action = model.act(x)
        return action

    return get_action


def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True):

    logger = EpochLogger()
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        a = get_action(o)
        o, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


def main():

    fpath = '/home/vincent/sai2/sai2_environment/sai2_environment/logs/2020-08-28_peg_in_hole_test/2020-08-28_21-44-34-peg_in_hole_test_s0'
    length = 300
    episodes = 3
    norender = True
    iteration = 'last'#>=0 or last
    deterministic = False

    # Robot Env
    action_space = ActionSpace.DELTA_EE_POSE_IMPEDANCE 
    blocking_action = True
    env = RobotEnv(name='peg_in_hole',
                   simulation=True,
                   action_space=action_space,
                   isotropic_gains=True,
                   render=False,
                   blocking_action=blocking_action,
                   rotation_axis=(0, 0, 1),
                   observation_type=dict(camera=1, q=0, dq=0, tau=0, x=1, dx=1))    

    _ , get_action = load_policy_and_env(fpath, 
                                          iteration,
                                          deterministic)
    run_policy(env, get_action, length, episodes, not(norender))

if __name__ == "__main__":
    main()