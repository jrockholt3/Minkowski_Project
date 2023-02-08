import numpy as np
import torch
from Agent import Agent
from Robot_Env import RobotEnv

env = RobotEnv()
agent = Agent(env.action_space)