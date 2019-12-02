import pybullet as p
import time
import pybullet_data
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0,0,0]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
#robotId = p.loadURDF("lynx2.urdf",cubeStartPos, cubeStartOrientation)
robotId = p.loadURDF("lynx2.urdf", flags=p.URDF_USE_SELF_COLLISION)

numJoints = p.getNumJoints(robotId)
print(numJoints)
names = []
for i in range(numJoints):
    names.append(p.getJointInfo(robotId, i)[1])
print(names)
# [b'base_to_robot', b'servo1', b'servo2', b'link2_servo', b'servo3', b'link3_servo', b'servo4', b'servo5', b'gripper_up', b'gripper_down']
# important ones: 1, 2, 4, 6, 7, (8,9) 
print(p.getLinkState(robotId, 8))
print(p.getLinkState(robotId, 9))
pos = [0.0, 0.263525, 0.24725]
boxId = p.loadURDF("box.urdf", basePosition=pos)
def reset(robotId):
    numJoints = p.getNumJoints(robotId)
    valid_joints = [1, 2, 4, 6, 7, 8, 9]
    for i in range(numJoints):
        if i in valid_joints:
            p.resetJointState(robotId, i, 0)
    
def applyAction(robotId, motorCommands):
    """
    motor commands are 6 dim 
    """
    valid_joints = [1, 2, 4, 6, 7, 8, 9]
    p.setJointMotorControlArray(robotId, valid_joints, p.POSITION_CONTROL, motorCommands[:-1]+[motorCommands[-1]*0.001]*2)

for i in range (10000):
    p.stepSimulation()
    #setJointMotorControl2()
    time.sleep(1./240.)
    cubePos, cubeOrn = p.getBasePositionAndOrientation(robotId)
    #print(cubePos,cubeOrn)
    
        
p.disconnect()