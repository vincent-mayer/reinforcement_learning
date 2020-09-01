

class RedisKeys(object):
    def __init__(self, simulation=True):
        #init keys to read/write to redis server
        self.ACTION_SPACE_KEY = "sai2::ReinforcementLearning::action_space"
        self.ACTION_KEY = "sai2::ReinforcementLearning::action"
        self.START_ACTION_KEY = "sai2::ReinforcementLearning::start_action"
        self.CAMERA_DATA_KEY  = "sai2::ReinforcementLearning::camera_data"
        self.ROBOT_IS_RESET_KEY = "sai2::ReinforcementLearning::robot_is_reset"      
        self.ACTION_COMPLETE_KEY =   "sai2::ReinforcementLearning::action_complete" 
        self.CURRENT_POS_KEY = "sai2::ReinforcementLearning::current_position"

        if simulation:
            self.JOINT_TORQUES_COMMANDED_KEY = "sai2::PandaApplication::actuators::fgc"
            self.JOINT_ANGLES_KEY  = "sai2::PandaApplication::sensors::q"
            self.JOINT_VELOCITIES_KEY = "sai2::PandaApplication::sensors::dq"

            self.SENSED_CONTACT_KEY = "sai2::PandaApplication::sensors::contact"

            self.HARD_RESET_SIMULATOR_KEY = "sai2::ReinforcementLearning::hard_reset_simulator"
            self.CURRENT_POS_KEY = "sai2::ReinforcementLearning::current_position"
            self.CURRENT_VEL_KEY = "sai2::ReinforcementLearning::current_velocity"

        else:        
            self.JOINT_TORQUES_COMMANDED_KEY = "sai2::FrankaPanda::actuators::fgc"
            self.JOINT_ANGLES_KEY  = "sai2::FrankaPanda::sensors::q"
            self.JOINT_VELOCITIES_KEY = "sai2::FrankaPanda::sensors::dq"
            self.SENSED_CONTACT_KEY = "sai2::FrankaPanda::sensors::model::contact"
            self.MASSMATRIX_KEY = "sai2::FrankaPanda::sensors::model::massmatrix"
            self.CORIOLIS_KEY = "sai2::FrankaPanda::sensors::model::coriolis"
            self.ROBOT_GRAVITY_KEY = "sai2::FrankaPanda::sensors::model::robot_gravity"
