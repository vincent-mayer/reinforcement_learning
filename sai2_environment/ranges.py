import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Range:

    q = {"min": np.array([-2.7, -1.6, -2.7, -3.0, -2.7, 0.2, -2.7]),
         "max": np.array([2.7, 1.6, 2.7, -0.2, 2.7, 3.6, 2.7])}

    dq = {"min": - np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.610, 2.6100]),
          "max": np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.610, 2.6100])}

    tau = {"min": - np.array([85, 85, 85, 85, 10, 10, 10]),
           "max": np.array([85, 85, 85, 85, 10, 10, 10])}

class RobotMinMaxScaler(object):
     def __init__(self):
          self.q_scaler = MinMaxScaler()
          self.dq_scaler = MinMaxScaler()
          self.tau_scaler = MinMaxScaler()
          self.q_scaler.fit([Range.q['min'], Range.q['max']])
          self.dq_scaler.fit([Range.dq['min'], Range.dq['max']])
          self.tau_scaler.fit([Range.tau['min'], Range.tau['max']])
         