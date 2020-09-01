#*******************+++++++++++++++++++++++++++++++ Dependencies +++++++++++++++++++++++++**********************#
from copy import deepcopy
import itertools
import numpy as np
import time

import torch
from torch.optim import Adam
from torch.distributions.normal import Normal
import torch.nn as nn
import torch.nn.functional as F

from sai2_environment.reinforcement_learning.utils.logx import EpochLogger

import matplotlib.pyplot as plt
import sai2_environment.reinforcement_learning.networks as networks
from sai2_environment.reinforcement_learning.utils.frame_stack import FrameStack
#*********************++++++++++++++++++++++++++++ Variables ++++++++++++++++++++++++++++++***********************#
LOG_STD_MAX = 2
LOG_STD_MIN = -20

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

frame_stack_size = 4
succ_env_itr = 0
succ_test_env_itr = 0
#*********************++++++++++++++++++++++++++++ SAC core +++++++++++++++++++++++++++++++************************#
def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = torch.as_tensor(act_limit, dtype = torch.float32, device = device)

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        #obs_dim = observation_space.shape[0]
        obs_dim = observation_space['proprioception']
        act_dim = action_space.shape[0]
        act_limit = action_space.high

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            print("TEST ACTION, MLPActorCritic Act: \n {}".format(a))
            return a.cpu().numpy()


class CnnEncoder(nn.Module):
    """
    """
    def __init__(self, img_channels = 3, feature_dim=128, params = None, img_size = 128):
        super(CnnEncoder, self).__init__()
        channels = [img_channels] + [32,32,32,32,32,32]
        kernel_sizes = [3,3,3,3,3,3]
        stride_size = [2,1,1,1,1,1]
        padding_sizes= [0,0,0,0,0,0]
        # for 84 x 84 inputs
        OUT_DIM = {2: 39, 4: 35, 6: 31}
        # for 64 x 64 inputs
        OUT_DIM_64 = {2: 29, 4: 25, 6: 21}
        # for 128 x 128 inputs
        OUT_DIM_128 = {2: 61,4: 57,6: 53}

        if img_size == 84:
            self.out_dim = OUT_DIM[6]
        elif img_size == 64:
            self.out_dim = OUT_DIM_64[6]
        elif img_size == 128:
            self.out_dim = OUT_DIM_128[6]
        else:
            raise NotImplementedError
        
        h_dim = channels[-1]*self.out_dim*self.out_dim
        self.output_logits = False
        self.encoder = networks.make_cnn(params, channels, kernel_sizes, stride_size, padding_sizes)
        self.output_layer = nn.Linear(h_dim, feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.feature_dim = feature_dim
    
    def get_output_dim(self):
        return self.feature_dim

    def forward_conv(self, x):
        x = x / 255.
        x = self.encoder(x)
        #return x.view(x.size(0), -1)
        return x.reshape(x.size(0), -1)

    def forward(self, x, detach = False):
        x = self.forward_conv(x)
        if detach:
            x = x.detach()
        h_out = self.output_layer(x)
        h_norm = self.layer_norm(h_out)
        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm)
        return out
    
    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        i = 0
        for module in self.encoder.modules():
            if type(module) == nn.Conv2d:
                tie_weights(src=source.encoder[i], trg=self.encoder[i])
            if type(module) != nn.Sequential:
                i += 1


 # ----------------- Own Implementation of CONV Actor critic) -------------------------------------------------- #
class CNNActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        #obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high
        
        self.cnn = CnnEncoder(observation_space['camera'][0]*observation_space['camera'][1])
        self.cnn = self.cnn.to(device)
        # Obs dim is output of conv network
        obs_dim = self.cnn.get_output_dim()
        # append robot state
        obs_dim += observation_space["proprioception"][0]
        # build policy and value functions, make them work with CNN output
        self.actor = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit).to(device)
        self.critic_1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(device)
        self.critic_2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(device)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            feats = self.cnn(torch.as_tensor(obs[0], dtype=torch.float32, device=device).unsqueeze(0))
            state = torch.as_tensor(obs[1], dtype=torch.float32, device = device)
            obs = torch.cat([feats.squeeze(0),state],-1)
            a, _ = self.actor(obs, deterministic, False)
            return a.cpu().numpy()
    
    def pi(self, obs, deterministic=False, with_logprob=True):
        feats = self.cnn(obs[0])
        state = obs[1]
        obs = torch.cat([feats,state],-1)
        return self.actor(obs, deterministic, with_logprob)

    def q1(self, obs, act):
        feats = self.cnn(obs[0])
        state = obs[1]
        obs = torch.cat([feats,state],-1)
        return self.critic_1(obs,act)

    def q2(self, obs, act):
        feats = self.cnn(obs[0])
        state = obs[1]
        obs = torch.cat([feats,state],-1)
        return self.critic_2(obs,act)




    
#***********************+++++++++++++++++ REPLAY BUFFER FROM sac.py ++++++++++++++++++++***************************#
class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_cam_buf = np.zeros(combined_shape(size, obs_dim[0]), dtype=np.float32)
        self.obs_robot_buf = np.zeros(combined_shape(size, obs_dim[1]), dtype=np.float32)
        self.obs2_cam_buf = np.zeros(combined_shape(size, obs_dim[0]), dtype=np.float32)
        self.obs2_robot_buf = np.zeros(combined_shape(size, obs_dim[1]), dtype=np.float32)

        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_cam_buf[self.ptr] = obs[0]
        self.obs_robot_buf[self.ptr] = obs[1]
        self.obs2_cam_buf[self.ptr] = next_obs[0]
        self.obs2_robot_buf[self.ptr] = next_obs[1]
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=[np.array([self.obs_cam_buf[idxs]]),
                          np.array([self.obs_robot_buf[idxs]])],
                     obs2=[np.array([self.obs2_cam_buf[idxs]]),
                          np.array([self.obs2_robot_buf[idxs]])],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: [torch.as_tensor(v[0], dtype=torch.float32, device=device),
                    torch.as_tensor(v[1], dtype=torch.float32, device=device)]
                    if k in ['obs', 'obs2']
                    else torch.as_tensor(v, dtype=torch.float32, device=device)
                    for k,v in batch.items()}

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#+++++++++++++++++++++++++++++++++++++++++++++++ THIS IS WHERE THE MAGIC HAPPENS ++++++++++++++++++++++++++++++++++++++++++++#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

def sac(env_fn, actor_critic=CNNActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e4), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1, debug=False):

    test = 1
    if test == 1:
        # Current training settings
        steps_per_epoch = 600
        epochs = 100
        replay_size=int(1e4)
        lr=1e-3
        batch_size = 128
        start_steps = 1000
        update_after = 300
        update_every = 150
        num_test_episodes = 2
        max_ep_len = 200
    elif test == 2:
        # Quick test parameters
        epochs = 1
        steps_per_epoch = 20
        num_test_episodes = 2
        update_every = 5
        update_after = 10
        max_ep_len = 15
        start_steps = 5

    #***************+++++++++++++++++++++++++++++++++++* FUNCTION DEFINITIONS **++++++++++++++++++++++++++++++++++++++++++++++******************
        # Set up function for computing SAC Q-losses

    ######### THIS FUNCTION DOES THE UPDATE HANDLING ##########
    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        if not debug:
            logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Record things
        if not debug:
            logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    ###### This function uses the ac_limit when calling ac.q1, ac.q1 etc ########
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        for k in 0,1: o[k] = o[k].squeeze(); o2[k] = o2[k].squeeze() #Remove 1 dimension

        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                      Q2Vals=q2.detach().cpu().numpy())

        return loss_q, q_info
    ##################################################################################


    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o = data['obs']
        for k in 0,1: o[k] = o[k].squeeze() # Squeeze to remove 1 dimension
        pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy())
        return loss_pi, pi_info


    def get_action(o, deterministic=False):
        #return ac.act(torch.as_tensor(o, dtype=torch.float32), deterministic)
        return ac.act(o, deterministic)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                a = get_action(o, True)
                # Test step the env
                o, r, d, _ = test_env.step(a)
                if r != 0:
                    print(" Action from CNN ACTOR: TEST REWARD: {}\n".format(r))
                    print("Current test length: {}\n".format(ep_len))
                    succ_test_env_itr += 1
                ep_ret += r
                ep_len += 1
            if not debug:
                logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # ******************+++++++++++++++++++++++++++++++++++++++++++++++ SAC MAIN START HERE +++++++++++++++++++++++++++++++++++++++++++*******************#

    if not debug:
        logger = EpochLogger(**logger_kwargs)
        logger.save_config(locals())


    torch.manual_seed(seed)
    np.random.seed(seed)

    # Wrap the environment when more than 1 frame should be used
    if frame_stack_size > 1:
        env = FrameStack(env_fn, frame_stack_size)
        test_env = env
    else:
        env, test_env = env_fn, env_fn

    obs_space = env.observation_space
    act_space = env.action_space
 
    obs_dim = (obs_space["camera"],obs_space["proprioception"]) #obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound! <-- DOES NOT ASSUME SHARED LIMITS ANYMORE
    act_limit = env.action_space.high

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
        
    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.critic_1.parameters(), ac.critic_2.parameters(), ac.cnn.parameters())
    pi_params = itertools.chain(ac.actor.parameters(), ac.cnn.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(count_vars(module) for module in [ac.actor, ac.critic_1, ac.critic_2, ac.cnn])
    if not debug:
        logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d, \t cnn: %d\n'%var_counts)

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(pi_params, lr=lr)
    q_optimizer = Adam(q_params, lr=lr)

    # Set up model saving
    if not debug:
        logger.setup_pytorch_saver(ac)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy.
        action_from_actor = False
        action_from_sampler = False
        if t > start_steps:
            a = get_action(o)
            action_from_actor = True
            
        else:
            action_from_sampler = True
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        
        # Monitor the stiffness values and action command
        if r != 0:
            if action_from_actor:
                print("the action came from ACTOR!! \n")
            elif action_from_sampler:
                print("the action came from SAMPLER!! \n")
            #print("Current Action command: \n {} \n".format(a))
            print("\n REWARD: {}\n".format(r))
            succ_env_itr += 1

            
        print("Current step: {}\n Current Epoch:{}\n".format(t,(t+1) // steps_per_epoch))
        # Monitor observation
        """
        plt.imshow(np.transpose(o2[0],(1,2,0)))
        plt.show()
        """
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            if not debug:
                logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                if not debug:
                    logger.save_state({'env': env}, None)
            print("+++++++++++++ Test Agent +++++++++++++")
            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            if not debug:
                print("-------------------------LOGGING EPOCH {}-------------------------".format(epoch))
                logger.log_tabular('Epoch', epoch)
                logger.log_tabular('Successful env interacts', succ_env_itr)
                logger.log_tabular('Successful test env interacts', succ_test_env_itr)
            # logger.log_tabuöar('Amount of Crashes', epoch=epoch)
                logger.log_tabular('EpRet', with_min_and_max=True, epoch=epoch)
                logger.log_tabular('TestEpRet', with_min_and_max=True, epoch=epoch)
                logger.log_tabular('EpLen', average_only=True, epoch=epoch)
                logger.log_tabular('TestEpLen', average_only=True, epoch=epoch)
                logger.log_tabular('TotalEnvInteracts', t, epoch=epoch)
                logger.log_tabular('Q1Vals', with_min_and_max=True, epoch=epoch)
                logger.log_tabular('Q2Vals', with_min_and_max=True, epoch=epoch)
                logger.log_tabular('LogPi', with_min_and_max=True, epoch=epoch)
                logger.log_tabular('LossPi', average_only=True, epoch=epoch)
                logger.log_tabular('LossQ', average_only=True, epoch=epoch)
                logger.log_tabular('Time', time.time()-start_time, epoch=epoch)
                logger.dump_tabular()   

            

