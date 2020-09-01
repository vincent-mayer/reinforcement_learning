import time
import cv2
import numpy as np
import pyrealsense2 as rs
from gym import spaces
from ipdb import set_trace
from scipy.spatial.transform import Rotation as Rot

from sai2_environment.client import RedisClient
from sai2_environment.action_space import *
from sai2_environment.utils import name_to_task_class, Timer
from sai2_environment.ranges import Range, RobotMinMaxScaler
from sai2_environment.camera_handler import CameraHandler
from sai2_environment.haptic_handler import HapticHandler


class RobotEnv(object):
    """
    The central wrapper around the robot control.
    """
    
    def __init__(self,
                 name='move_object_to_target',
                 simulation=True,
                 render=False,
                 action_space=ActionSpace.ABS_JOINT_POSITION_DYN_DECOUP,
                 isotropic_gains=True,
                 blocking_action=False,
                 action_frequency=20,
                 torque_seq_length=32,
                 camera_available=True,
                 camera_res = (128, 128),
                 rotation_axis=(True, True, True),
                 observation_type = dict(camera=1, q=1, dq=1, tau=32, x=0, dx=0)):

        # Defines which types of observations should be returned
        self.observation_type = observation_type

        self.camera_available = camera_available
        # connect to redis server
        hostname = "127.0.0.1" if simulation else "TUEIRSI-RL-001"
        self.env_config = {
            'simulation': simulation,
            'render': render,
            'camera_resolution': camera_res,
            'camera_frequency': 30,
            'hostname': hostname,
            'port': 6379,
            'blocking_action': blocking_action,
            'rotation_axis': rotation_axis,
            'torque_seq_length': torque_seq_length
        }

        # connect redis client
        self._client = RedisClient(config=self.env_config)
        self._client.connect()

        self.timer = Timer(frequency=action_frequency)
        self.start_time = time.time()

        # set action space to redis
        self._robot_action = get_robot_action(action_space, isotropic_gains,
                                              rotation_axis)
        #self._robot_action = RobotAction(action_space, isotropic_gains, rotation_axis=rotation_axis)

        self._client.init_action_space(self._robot_action)
        self._episodes = 0        

        self.action_space = self._robot_action.action_space

        self.haptic_handler = HapticHandler.getInstance(
            self._client, simulation, sensor_frequency=1000)
        self.camera_handler = CameraHandler.getInstance(
            self.env_config['camera_resolution'])

        self.scaler = RobotMinMaxScaler()

        if not self.env_config["simulation"] and self.camera_available:
            self.camera_handler.camera_thread.start()
        # ẃarm up camera

        if self.env_config["render"]:
            cv2.namedWindow('Simulator', cv2.WINDOW_NORMAL)

        time.sleep(1)

        cam, proprio, haptic = self._get_obs()
        self.observation_space = {
            "camera": cam.shape,
            "proprioception": proprio.shape,
            "haptic": (haptic[0].shape, haptic[1].shape)
        }
       
        # Validate dict entries
        observation_validations={
            "camera": lambda x: isinstance (x, int) and (x==0 or x==1),
            "q": lambda x: isinstance (x, int) and (x==0 or x==1),
            "dq": lambda x: isinstance (x, int) and (x==0 or x==1),
            "tau": lambda x: isinstance (x, int) and 0<=x<=1000,
            "x": lambda x: isinstance (x, int) and (x==0 or x==1),
            "dx": lambda x: isinstance (x, int) and (x==0 or x==1)}

        for k,v in observation_type.items():
            if not observation_validations[k](v):
                print("Key {} has wrong type or value, should be int".format(k))
                raise TypeError

        # TODO define what all the responsibilites of task are
        task_class = name_to_task_class(name)
        self.task = task_class(
            'tmp', self._client, camera_handler=self.camera_handler, simulation=simulation)

    def reset(self):
        self._client.reset(self._episodes)
        # TODO do we want to set it every time or keep one action space per experiment?
        if self._episodes != 0:
            self._client.set_action_space()

        if self._episodes % 10 == 0:
            print("Episode: {}; Elapsed Time: {} minutes".format(
                self._episodes, round((time.time()-self.start_time)/60), 4))

        self._episodes += 1
        self.task.initialize_task()
        return self._get_obs()

    def convert_image(self, im):
        return np.rollaxis(im, axis=2, start=0)/255.0

    def rotvec_to_quaternion(self, vec):
        quat = Rot.from_euler('zyx', vec).as_quat()
        #[w, x, y, z]
        idx = [3, 0, 1, 2]
        return quat[idx]

    def quaternion_to_rot(self, quaternion):
        return Rot.from_quat(quaternion).as_dcm()

    def step(self, action):
        assert action.shape == self._robot_action.action_space_size(
        ), "Action shape of {} not correct, expected shape {}".format(action.shape,
            self._robot_action.action_space_size())
        # build the full action if
        action = self._robot_action.build_full_command(action)

        # blocking action waits until the action is carried out and computes reward along the trajectory
        if self.env_config['blocking_action']:
            # first check if there is still something going on on the robot
            # print("Waiting for robot: {}".format(
            # self._client.action_complete()))
            self.take_action(action)
            time.sleep(0.01)
            sleep_counter = 0
            while not self._client.action_complete():
                sleep_counter += 1
                time.sleep(0.01)

            if sleep_counter > 0:
                reward, done = self._compute_reward()
            else:
                reward = 0; done = False
            

        # non-blocking does not wait and computes reward right away
        else:

            self.take_action(action)
            self.timer.wait_for_next_loop()

            reward, done = self._compute_reward()

        info = None
        obs = self._get_obs()  # has to be before the contact reset \!/        

        return obs, reward, done, info

    def take_action(self, action):
        return self._client.take_action(action)

    def act_optimally(self):
        action = self.task.act_optimally()
        return action

    def render(self, img):
        if self.env_config["render"]:
            cv2.imshow("Simulator", img)
            cv2.waitKey(1)

    def close(self):
        return 0

    def _compute_reward(self):
        reward, done = self.task.compute_reward()
        return reward, done

    def _get_obs(self):
        """
        camera_frame: im = (128,128)
        robot_state: (q,dq) = (14,)
        robot_state_cartesian: (x,dx) = (6,)
        haptic_feedback: (tau, contact) = ((7,n), (1,))
        """
        if self.env_config['simulation']:
            img = self._client.get_camera_frame() if self.observation_type['camera'] else 0
            camera_frame = self.convert_image(img)
        else:
            img = self.camera_handler.get_color_frame() if self.camera_available else 0
            camera_frame = self.convert_image(img)
                
        # retrieve robot state
        q, dq, x, dx = self._client.get_robot_state()        
        # normalize proprioception
        q = self.scaler.q_scaler.transform([q])[0]
        dq = self.scaler.dq_scaler.transform([dq])[0]

        #retrieve haptics
        #tau = self.haptic_handler.get_torques_matrix(n=self.env_config["torque_seq_length"])
        reversed__transposed_tau = np.array([0])
        contact = np.asarray([self.haptic_handler.contact_occured()])
        #normalize haptics
        #tau = self.scaler.tau_scaler.transform(tau)
        #reversed__transposed_tau = np.transpose(tau[::-1])

        #concatenate only state information which is demanded by observation_type
        normalized_robot_state = np.array([])
        if self.observation_type['q']: normalized_robot_state = np.concatenate((normalized_robot_state,q)) 
        if self.observation_type['dq']: normalized_robot_state = np.concatenate((normalized_robot_state,dq))
        if self.observation_type['x']: normalized_robot_state = np.concatenate((normalized_robot_state,x))
        if self.observation_type['dx']: normalized_robot_state = np.concatenate((normalized_robot_state,dx))

        normalized_haptic_feedback = (reversed__transposed_tau, contact)
        self.render(img)
        return camera_frame, normalized_robot_state, normalized_haptic_feedback
