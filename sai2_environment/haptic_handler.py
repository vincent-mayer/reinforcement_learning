from sai2_environment.utils import Timer
from collections import deque
import numpy as np
import threading


class HapticHandler:
    __instance = None

    @staticmethod
    def getInstance(client, simulation, sensor_frequency=1000):
        if HapticHandler.__instance == None:
            HapticHandler(client, simulation, sensor_frequency)
        return HapticHandler.__instance

    def __init__(self, client, simulation, sensor_frequency=1000):
        if HapticHandler.__instance != None:
            raise Exception("This class: CameraHandler is a singleton!")
        else:
            HapticHandler.__instance = self
        self.client = client
        self.sensor_frequency = sensor_frequency
        self.haptic_thread = threading.Thread(
            name="haptic_thread", target=self.get_haptic_feedback)
        self.haptic_thread.start()
        self.torque_measurements = deque(maxlen=200)
        for i in range(100):
            self.torque_measurements.append(np.zeros((7,)))

        self.contact_event = False

    def get_haptic_feedback(self):
        timer = Timer(frequency=self.sensor_frequency)
        try:
            while True:
                timer.wait_for_next_loop()
                # read sensed contact
                if not self.contact_event:
                    self.contact_event = self.client.get_sensed_contact()

                # readtorques
                tau = self.client.get_torques()
                self.torque_measurements.append(tau)
        except KeyboardInterrupt:
            self.haptic_thread.join()

    def get_contact(self):
        while True:
            contact = self.client.get_contact_occurence()
            self.contact_event = True if contact.any() else self.contact_event
            #print("contact=", contact)

    def get_torques_matrix(self, n=32):
        
        return np.asarray([self.torque_measurements.pop() for i in range(n)])

    def contact_occured(self):
        #check if contact occured since the last time the function was called
        contact_since_last_call = self.contact_event
        self.contact_event = False
        return contact_since_last_call

    



