from sai2_environment.tasks.task import Task
import numpy as np

PUSH_HORIZONTAL = 0
PUSH_VERTICAL = 1
np.set_printoptions(precision=3, suppress=True)


class MoveObjectToTarget(Task):
    def __init__(self, task_name, redis_client, camera_handler, simulation=True):
        self._task_name = task_name
        self._client = redis_client
        self._simulation = simulation
        self.camera_handler = camera_handler
        self.traj = []
        self.cumulative_reward = 0
        self.TARGET_OBJ_POSITION_KEY = "sai2::ReinforcementLearning::move_object_to_target::object_position"
        self.GOAL_POSITION_KEY = "sai2::ReinforcementLearning::move_object_to_target::goal_position"
        self.CURRENT_POS_KEY = "sai2::ReinforcementLearning::current_position"
        self.DESIRED_POS_KEY = "sai2::ReinforcementLearning::desired_position"

        if simulation:
            self.goal_position = self._client.redis2array(
                self._client.get(self.GOAL_POSITION_KEY))
                        
            self.current_obj_position = self.get_puck_position()
            self.last_obj_position = self.current_obj_position
            self.total_distance = self.euclidean_distance(
                self.goal_position, self.current_obj_position)
        else:
            # setup the things that we need in the real world
            # self.goal_position = None
            # self.current_obj_position = None
            # self.last_obj_position = None

            # new modify
            self.current_obj_distance = self.camera_handler.grab_distance()
            self.last_obj_distance = self.current_obj_distance
            self.total_distance = self.camera_handler.grab_distance()

    def initialize_task(self):
        self.cumulative_reward = 0
        if self._simulation:
            self.goal_position = self._client.redis2array(
                self._client.get(self.GOAL_POSITION_KEY))
            self.current_obj_position = self.get_puck_position()
            self.last_obj_position = self.current_obj_position
            self.total_distance = self.euclidean_distance(
                self.goal_position, self.current_obj_position)-0.04
            self.traj = self.plan_optimal_trajectory()
        else:
            self.total_distance = self.camera_handler.grab_distance()

    def compute_reward(self):
        """
        There is a total of 10 reward per episde. 
        1 for pushing the object to the goal and 9 for completing the task.
        Reward is normalized by the initial distance.
        """
        done = False
        reward = 0
        if self._simulation:
            self.last_obj_position = self.current_obj_position
            self.current_obj_position = self.get_puck_position()
            d0 = self.euclidean_distance(
                self.goal_position, self.last_obj_position) - 0.04
            d1 = self.euclidean_distance(
                self.goal_position, self.current_obj_position) - 0.04

            reward = (d0 - d1)/self.total_distance            
            done = self.is_in_goal(self.current_obj_position)
            
        else:
            # reward = 0
            # TODO
            # new modify
            # When detecting no enough markers at the very beginning
            self.last_obj_distance = self.current_obj_distance
            self.current_obj_distance = self.camera_handler.grab_distance()
            d_last = self.last_obj_distance
            d_current = self.current_obj_distance
            if d_current == 1:
                reward = 0
            else:
                reward = (d_last - d_current)/self.total_distance
                # reward = d_current
            done = d_current < 0.04
        
        self.cumulative_reward += reward
        if done:
            reward += 1 - self.cumulative_reward
            reward += 9
        return reward, done

    def act_optimally(self):
        # only works for the moving target action space right now
        desired_pos = self.get_desired_position()
        ee_pos = self.get_ee_position()
        action = np.array([0, 0, 0, 0, 0])
        if self.traj:
            required_behavior = self.traj[0]
            required_position = required_behavior[:3]
            required_stiffness = required_behavior[3:]
            if (self.euclidean_distance(required_position, ee_pos) > 0.05):
                action_pos = required_position - desired_pos[:3]
                #TODO add stiffness
                action = np.concatenate((action_pos, np.array([0,0])))
            else:
                self.traj.pop(0)

        return action

    def plan_optimal_trajectory(self):
        puck_pos = self.get_puck_position()
        # first action behind the
        a1 = np.array([puck_pos[0], puck_pos[1] +
                       np.sign(puck_pos[1])*0.1, 0.15, 50, 0])
        # go down z direction
        a2 = np.array([puck_pos[0], puck_pos[1] +
                       np.sign(puck_pos[1])*0.1, 0.05, 50, 0])
        # go to middle of the workspace
        a3 = np.array([puck_pos[0], np.sign(puck_pos[1])*0.05, 0.05, 0, 0])
        # go up again
        a4 = np.array([puck_pos[0], np.sign(puck_pos[1])*0.05, 0.18, 0, 0])
        # go behind puck again
        a5 = np.array([puck_pos[0]-0.10, 0, 0.18, 0, 0])
        # go down z again
        a6 = np.array([puck_pos[0]-0.10, 0, 0.05, 0, 0])
        # push towards goal in (0.6,0,0)
        a7 = np.array([0.65, 0, 0.05, 0, 0])
        trajectory = [a1, a2, a3, a4, a5, a6, a7]

        return trajectory

    def euclidean_distance(self, x1, x2):
        return np.linalg.norm(x1 - x2)

    def is_in_goal(self, pos):
        return (pos[0]-self.goal_position[0])**2 + (pos[1]-self.goal_position[1])**2 <= 0.04**2

    def get_ee_position(self):
        return self._client.redis2array(self._client.get(self.CURRENT_POS_KEY))

    def get_puck_position(self):
        return self._client.redis2array(self._client.get(self.TARGET_OBJ_POSITION_KEY))

    def get_desired_position(self):
        return self._client.redis2array(self._client.get(self.DESIRED_POS_KEY))
