from enum import Enum
import numpy as np
from gym.spaces import Box
from scipy.spatial.transform import Rotation as Rot
from ipdb import set_trace


class ActionSpace(Enum):
    NONE = 0
    """
    Digits Notation:
    | Space | Abs/Delta | Dyn decoup/ impedance |
    """
    #anisotropic: joints (7) Kp/stiffness (7)
    #isotropic: joints (7) Kp/stiffness (1)
    ABS_JOINT_POSITION_DYN_DECOUP = 111
    DELTA_JOINT_POSITION_DYN_DECOUP = 121
    ABS_JOINT_POSITION_IMPEDANCE = 112
    DELTA_JOINT_POSITION_IMPEDANCE = 122

    #joints (7)
    ABS_JOINT_TORQUE = 110
    DELTA_JOINT_TORQUE = 120

    #anisotropic: position (3) rotation (4) Kp/stiffness: translational (3) rotational (3)
    #isotropic: position (3) rotation (4) Kp/stiffness: translational (1) rotational (1)
    ABS_EE_POSE_DYN_DECOUP = 211
    DELTA_EE_POSE_DYN_DECOUP = 221
    ABS_EE_POSE_IMPEDANCE = 212
    DELTA_EE_POSE_IMPEDANCE = 222

    MT_EE_POSE_IMPEDANCE = 232


def get_robot_action(action_space_enum, isotropic_gains, rotation_axis):
    if action_space_enum.value // 10**2 % 10 == 1:
        return JointSpaceAction(action_space_enum)
    else:
        return TaskSpaceAction(action_space_enum,
                               isotropic_gains=isotropic_gains,
                               rotation_axis=rotation_axis)

class RobotAction(object):
    def __init__(self, action_space_enum):
        self.action_space_enum = action_space_enum

    def decode_action_space(self, action_space):
        i = action_space.value
        return i // 10**2 % 10, i // 10**1 % 10, i // 10**0 % 10

    def action_space_size(self):
        raise NotImplementedError()

    def reset_action(self):
        raise NotImplementedError()

    def build_full_command(self, action):
        raise NotImplementedError()


class TaskSpaceAction(RobotAction):
    def __init__(self,
                 action_space_enum,
                 isotropic_gains=False,
                 rotation_axis=(True, True, True)):
        super().__init__(action_space_enum)
        self.isotropic_gains = isotropic_gains
        self.rotation_axis = rotation_axis
        self.space_type, self.value_type, self.control_type = self.decode_action_space(
            self.action_space_enum)

        #either generate full quaternion or rotate around the specified axis
        self.pose_dim = 7 if sum(
            rotation_axis) == 3 else 7 - 4 + sum(rotation_axis)
        rot_dim = self.pose_dim - 3

        if self.value_type == 1:  #absolute position
            x_min = np.array([0.1, -0.45, 0.0])
            x_max = np.array([0.65, 0.45, 0.5])

            R_min = np.zeros((rot_dim, ))
            R_max = np.ones((rot_dim, ))

            if self.control_type == 1:  #dynamic decoupling
                #TODO validate these values
                kp_min = np.array([
                    0, 5
                ]) if self.isotropic_gains else np.array([0, 0, 0, 5, 5, 5])
                kp_max = np.array([500, 10
                                   ]) if self.isotropic_gains else np.array(
                                       [500, 500, 500, 10, 10, 10])

            else:  #impedance
                #first value is translational stiffness(0-500), second rotational (5-10)
                kp_min = np.array([
                    0, 5
                ]) if self.isotropic_gains else np.array([0, 0, 0, 5, 5, 5])
                kp_max = np.array([500, 10
                                   ]) if self.isotropic_gains else np.array(
                                       [500, 500, 500, 10, 10, 10])

        else:  #delta position
            x_min = np.array([-0.1, -0.1, -0.1])
            x_max = -1 * x_min

            R_min = -0.1 * np.ones((rot_dim, ))
            R_max = -1 * R_min

            if self.control_type == 1:  #dynamic decoupling
                kp_min = np.array([-10, -1
                                   ]) if self.isotropic_gains else np.array(
                                       [-10, -10, -10, -1, -1, -1])
                kp_max = np.array([
                    10, 1
                ]) if self.isotropic_gains else np.array([10, 10, 10, 1, 1, 1])
            else:  #impedance
                kp_min = np.array([-10, -1
                                   ]) if self.isotropic_gains else np.array(
                                       [-10, -10, -10, -1, -1, -1])
                kp_max = np.array([
                    10, 1
                ]) if self.isotropic_gains else np.array([10, 10, 10, 1, 1, 1])
        low = np.concatenate((x_min, R_min, kp_min))
        high = np.concatenate((x_max, R_max, kp_max))
        self.action_space = Box(low=low, high=high)

    def rotvec_to_quaternion(self, vec):
        quat = Rot.from_euler('xyz', vec).as_quat()
        #[w, x, y, z]
        idx = [3, 0, 1, 2]
        return quat[idx]

    def action_space_size(self):
        return self.action_space.shape

    def reset_action(self):
        return -1 * np.ones((13, ))

    def build_full_command(self, action):
        #depending on isotropic gains and/or the rotational axis we have to build the full command
        x = np.zeros(3)
        quat = np.zeros(4)
        kp = np.zeros(6)

        if self.isotropic_gains:
            kp_translation = action[-2] * np.ones(3)
            kp_rotation = action[-1] * np.ones(3)
            kp = np.concatenate((kp_translation, kp_rotation))
        else:
            kp = action[-6:]

        pose = action[:self.pose_dim]
        x = pose[:3]

        if sum(self.rotation_axis) == 3:
            quat = pose[3:]
        else:
            rot = np.zeros(3)
            vec = np.pi*pose[3:]
            idx = np.nonzero(np.asarray(self.rotation_axis))
            rot[idx] = vec
            quat = self.rotvec_to_quaternion(rot)

        return np.concatenate((x, quat, kp))


class JointSpaceAction(RobotAction):
    def __init__(self, action_space_enum):
        super().__init__(action_space_enum)
        self.space_type, self.value_type, self.control_type = self.decode_action_space(
            self.action_space_enum)

        #joint min and max values
        # https://frankaemika.github.io/docs/control_parameters.html
        if self.control_type == 0:  #torque controlled
            q_min = -1 * np.array([85, 85, 85, 85, 10, 10, 10])
            q_max = np.array([85, 85, 85, 85, 10, 10, 10])

            kp_min = np.array([])
            kp_max = np.array([])

        elif self.value_type == 1:  #absolute
            q_min = np.array([-2.7, -1.6, -2.7, -3.0, -2.7, 0.2, -2.7])
            q_max = np.array([2.7, 1.6, 2.7, -0.2, 2.7, 3.6, 2.7])

            if self.control_type == 1:  #dynamic decoupling
                kp_min = np.array([0, 0, 0, 0, 0, 0, 0])
                kp_max = np.array([500, 500, 500, 500, 500, 500, 500])

            else:  #impedance
                #temporary suggested values from Erfan
                kp_min = np.array([500, 500, 500, 500, 300, 200, 100])
                kp_max = np.array([1500, 1500, 1500, 1500, 1000, 500, 300])
        else:  #delta
            q_min = np.full((7, ), -0.1)
            q_max = -1 * q_min

            if self.control_type == 2:  #impedance
                kp_min = np.array([500, 500, 500, 500, 300, 200, 100])
                kp_max = np.array([1500, 1500, 1500, 1500, 1000, 500, 300])

        low = np.concatenate((q_min, kp_min))
        high = np.concatenate((q_max, kp_max))

        self.action_space = Box(low=low, high=high)

    def action_space_size(self):
        return self.action_space.shape

    def reset_action(self):
        return -1 * np.ones((14, ))

    def build_full_command(self, action):
        #since we dont do isotropic gains or rotational axis, we only return the action
        return action
