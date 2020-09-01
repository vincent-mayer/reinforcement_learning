from sai2_environment.tasks.task import Task
import numpy as np

#Reward based on Vision and Touch paper: https://arxiv.org/pdf/1907.13098.pdf

class PegInHole(Task):
    def __init__(self, task_name, redis_client, camera_handler, simulation=True):
        self._task_name = task_name
        self._client = redis_client
        self._simulation = simulation
        self.camera_handler = camera_handler
        self.CURRENT_POS_KEY = "sai2::ReinforcementLearning::current_position"
        self.GOAL_POS_KEY  = "sai2::ReinforcementLearning::move_object_to_target::goal_position"
        if simulation:
            self.hole_pos = self._client.redis2array(self._client.get(self.GOAL_POS_KEY))
            self.hole_pos[2] += 0.06 # adjust z-position such that it gives top of the hole
            self.current_peg_pos = self.get_current_peg_pos() # returns bottom of peg
            # Hyperparameters for reward
            self.lamda = 7 
            self.ca = 1
            self.ci = 2
            self.cr = 1
            self.epsilon1 = 0.03 # maximal reward in alignment stage when peg closer than 5cm to goal
            self.epsilon2 = 0.01 # success, when only 0.5cm away from max insertion depth
            self.hd = 0.03 #depth of the hole is 3cm, peg can be inserted approx. 2.5cm
        else:
            #setup the things that we need in the real world
            self.goal_pos = None
            self.hole_pos = None

    def compute_reward(self):
        if self._simulation:
            self.current_peg_pos = self.get_current_peg_pos()
            diff_ee_hole = self.current_peg_pos - self.hole_pos # called s in the paper
            done = False
            # Staged reward depending on phase of task
            if np.linalg.norm(diff_ee_hole) <= self.epsilon1: # Alignment phase
                reward = 1 + self.ca * (1 - np.linalg.norm(diff_ee_hole)/self.epsilon1)
            elif (diff_ee_hole[2] < 0) and (np.linalg.norm(diff_ee_hole) <= self.epsilon1): # if z-position smaller zero, meaning alignment done and insertion starts
                reward = 2 + self.ci*(self.hd - np.abs(diff_ee_hole[2]))
            elif (np.abs(self.hd - diff_ee_hole[2]) <= self.epsilon2) and (np.linalg.norm(diff_ee_hole) <= self.epsilon1):
                reward = 5
                done = True
            else:
                reward = self.cr*(1-(np.tanh(self.lamda * np.linalg.norm(diff_ee_hole))))

        else:
            #TODO
            reward = 0
            done = False
        #print("REWARD", reward)
        return reward, done

    def euclidean_distance(self, x1, x2):
        return np.linalg.norm(x1 - x2)

    def get_current_peg_pos(self):
        return self._client.redis2array(self._client.get(self.CURRENT_POS_KEY)) # returns pos of the peg bottom

    def initialize_task(self):
        #nothing has to happen here
        return