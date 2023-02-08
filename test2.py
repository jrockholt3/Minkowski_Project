import numpy as np
from Robot_Env import RobotEnv
from Robot3D import robot_3link
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

robot = robot_3link()
bad_count = 0
bad_list = []
for i in range(1000):
    vec = (2*np.random.rand(3) - 1)
    mag = .4*np.random.rand() + .2
    goal = mag * .999 * (vec/np.linalg.norm(vec)) + np.array([0,0,.3])
    th = robot.reverse(goal=goal)
    if np.isnan(th[1,0]):
        bad_list.append(goal)
        bad_count += 1

print('bad_count', bad_count)
if bad_count > 0:
    bad_arr = np.vstack(bad_list)
    xx = bad_arr[:,0]
    yy = bad_arr[:,1]
    zz = bad_arr[:,2]
    fig = plt.figure
    ax = plt.axes(projection='3d')
    ax.scatter3D(xx,yy,zz,alpha=.5)
    plt.show()

robot.reverse(goal=goal,make_plot=True)