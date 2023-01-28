import Robot3D
import numpy as np
from Robot3D import robot_3link as Robot
from Robot3D import rand_object as Object
from Robot3D import workspace_limits
from Robot3D import workspace
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

obj = Object()
check = [[0,.01,-.01],[0,.01,0],[0,.01,.01],[0,0,.01],[0,-.01,.01],[0,-.01,0],[0,-.01,-.01],[0,0,-.01],[0,.01,-.01]]
r_int = int(obj.radius*100)
r = obj.radius
pos = obj.curr_pos
x_c,y_c = pos[0],pos[1]
pos = pos - np.array([r,0,0])
x = pos[0]
coord_list = []
coord_list.append(pos)
for i in range(-r_int,r_int+1):
    y_start = np.sqrt(abs(r**2 - (x-x_c)**2)) + y_c
    start = np.array([pos[0],y_start,pos[2]])
    pos = start
    done = False
    n = 0
#     print('starting at ', start-obj.curr_pos)
    while not done:
        n += 1
        found_next = False
        for j in range(1,len(check)):
            check_loc = pos + check[j] - obj.curr_pos
            prev_loc = pos + check[j-1] - obj.curr_pos
            if np.round(np.linalg.norm(check_loc),2) == np.round(r,2) and np.round(np.linalg.norm(prev_loc),2) > np.round(r,2):
                coord_list.append(pos+check[j])
                pos = pos + check[j]
#                 print('found next r =', np.round(check_loc,2))
                j = len(check) + 1
                found_next = True
        
        if np.all(np.round(start,2) == np.round(pos,2)) and found_next:
#             print('made it')
            done = True
        elif not found_next:
            done = True
        elif n > 1000:
            done = True
    
    pos = np.array([pos[0]+.01, obj.curr_pos[1], obj.curr_pos[2]])
    x = pos[0]

    if not found_next:
        print("was not able to iterate over slice")
        
coord_list.append(pos-np.array([.01,0,0]))        
            
coord_list = np.array(coord_list)
xx = coord_list[:,0]
yy = coord_list[:,1]
zz = coord_list[:,2]
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(xx,yy,zz)
ax.axes.set_xlim3d(left=obj.curr_pos[0] - 1.5*r, right=obj.curr_pos[0] + 1.5*r) 
ax.axes.set_ylim3d(bottom=obj.curr_pos[1] - 1.5*r, top=obj.curr_pos[1] + 1.5*r) 
ax.axes.set_zlim3d(bottom=obj.curr_pos[2] - 1.5*r, top=obj.curr_pos[2] + 1.5*r)
plt.show()