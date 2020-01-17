import gym
import pybullet as p
import pybullet_data
import numpy as np
import time
class StackArmEnv(gym.Env):
    def __init__(self):
        self.physicsClient = p.connect(p.GUI) #or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) # for plane.urdf
        p.setGravity(0,0,-9.8)
        self.gripper_offset = 6
        self.zero_q = [0, 0, np.pi/2, 0, 0, 0.001*self.gripper_offset, 0.001*self.gripper_offset]#[0, 0, -np.pi/2, 0, np.pi/2, 12, 12]
        self.box_pos = [0.0, 0.0, 0.510775]
        self.box_orn = p.getQuaternionFromEuler([0,0,0])
        self.robotID = p.loadURDF("lynx2.urdf", flags=p.URDF_USE_SELF_COLLISION)
        self.boxID = p.loadURDF("box.urdf", basePosition=self.box_pos)
        self.planeID = p.loadURDF("plane.urdf")
        self.valid_joints = [1, 2, 4, 6, 7, 8, 9]  
        self.force = 200
        

    def applyAction(self, motorCommands):
        """
        motor commands are 6 dim 
        """
        # for i, n_joint in enumerate(self.valid_joints):
        #     p.setJointMotorControl2(self.robotID, n_joint, p.POSITION_CONTROL, targetPosition=motorCommands[i],force=self.force)
        p.setJointMotorControlArray(self.robotID, self.valid_joints, p.POSITION_CONTROL, targetPositions=motorCommands[:-1]+[motorCommands[-1]*0.001]*2, forces=[self.force]*7) # motor command[0:4] is for joint 1-5, command[5] is for the gripper, two gripper tips are given the same amount of movement in mm(need to be converted to m)


    def reset(self):
        p.resetBasePositionAndOrientation(self.boxID, self.box_pos, self.box_orn)
        for i, n_joint in enumerate(self.valid_joints):
            p.resetJointState(self.robotID, n_joint, self.zero_q[i])
            p.setJointMotorControl2(self.robotID, n_joint, p.POSITION_CONTROL, targetPosition=self.zero_q[i],force=self.force)

    def step(self, action):
        self.applyAction(action)
        box_pos = p.getLinkState(self.boxID, 0)[0:2]  
        reward = None
        done = False
        return box_pos, reward, done, None
        

if __name__ == "__main__":
    env = StackArmEnv()
    env.reset()
    print(p.getLinkState(env.robotID, 8))
    print(p.getLinkState(env.robotID, 9))
    for i in range (10000):
        p.stepSimulation()
        #setJointMotorControl2()
        #print(p.getDynamicsInfo(env.robotID, 8))
        time.sleep(.1)
        q = [0.1, 0.2, np.pi/2 + i * 0.001, 0.3, 0, 6] 
        env.applyAction(q)
     