import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import Robot_Env
from Robot_Env import dt
from Robot3D import rand_object 
from Robot3D import workspace_limits as lims 
# Fixing random state for reproducibility


show_box = True

class Box():
    def __init__(self):
        a = .08
        self.x_arr = np.array([-a,a,a,-a,-a,-a,a,a,a,a,a,a,-a,-a,-a,-a])
        self.y_arr = np.array([a,a,-a,-a,a,a,a,a,a,-a,-a,-a,-a,-a,-a,a])
        self.z_arr = np.array([a,a,a,a,a,-a,-a,a,-a,-a,a,-a,-a,a,-a,-a])
        self.pos = np.array([0,0,0])

    def render(self,pos):
        return self.x_arr+pos[0], self.y_arr+pos[1], self.z_arr+pos[2]

def gen_centers(x,y,z):
    centers = []
    n = 5
    vec = np.array([x[1]-x[0], y[1]-y[0], z[1]-z[0]])
    slope = vec/np.linalg.norm(vec)
    ds = np.linalg.norm(vec)/n
    centers.append(np.array([x[0],y[0],z[0]]))
    for i in range(1,n+1):
        centers.append(slope*ds*i + centers[0])

    vec = np.array([x[2]-x[1], y[2]-y[1], z[2]-z[1]])
    slope = vec/np.linalg.norm(vec)
    ds = np.linalg.norm(vec)/n
    for i in range(1,n+1):
        centers.append(slope*ds*i + centers[n])
    
    return np.vstack(centers)

# Attaching 3D axis to the figure
fig = plt.figure()
ax = p3.Axes3D(fig)

# Fifty lines of random 3-D lines
env = Robot_Env.RobotEnv()
goal = env.goal
env.goal = np.array([-3*np.pi/4, goal[1], goal[2]])
obj = rand_object()
obj.dt = Robot_Env.dt
box = Box()

x_arr = []
y_arr = []
z_arr = []
x_arr2 = []
y_arr2 = []
z_arr2 = []
temp = env.robot.forward(th=env.start)
x_arr.append(temp[0,:])
y_arr.append(temp[1,:])
z_arr.append(temp[2,:])
temp = obj.curr_pos
obj.step()
x_arr2.append(temp[0])
y_arr2.append(temp[1])
z_arr2.append(temp[2])
x_box = []
y_box = []
z_box = []
done = False
while not done:
    _,th,done = env.step()
    temp = env.robot.forward(th=th)
    x_arr.append(temp[0,:])
    y_arr.append(temp[1,:])
    z_arr.append(temp[2,:])
    centers = gen_centers(temp[0,1:],temp[1,1:],temp[2,1:])
    temp = obj.curr_pos
    x_arr2.append(temp[0])
    y_arr2.append(temp[1])
    z_arr2.append(temp[2])
    obj.step()

    x_box.append(np.hstack([centers[:,0],temp[0]]))
    y_box.append(np.hstack([centers[:,1],temp[1]]))
    z_box.append(np.hstack([centers[:,2],temp[2]]))

x_arr = np.vstack(x_arr)
y_arr = np.vstack(y_arr)
z_arr = np.vstack(z_arr)

x_arr2 = np.vstack(x_arr2)
y_arr2 = np.vstack(y_arr2)
z_arr2 = np.vstack(z_arr2)

x_box = np.vstack(x_box)
y_box = np.vstack(y_box)
z_box = np.vstack(z_box)

line, = ax.plot([],[],[], 'bo-', lw=2)
line2, = ax.plot([],[],[], 'bo-',alpha=.3)
line3, = ax.plot([],[],[], 'bo-',alpha=.3)
line4, = ax.plot([],[],[], 'ro', lw=10)
line5, = ax.plot([],[],[], 'ro', lw=10, alpha=.3)
line6, = ax.plot([],[],[], 'ro', lw=10, alpha=.3)
line7, = ax.plot([],[],[], 'k-', alpha=.3)

j = int(np.round(x_arr.shape[0]))
def update(i):
    global j
    # set robot lines
    if not show_box:
        thisx = [x_arr[i,0],x_arr[i,1],x_arr[i,2],x_arr[i,3]]
        thisy = [y_arr[i,0],y_arr[i,1],y_arr[i,2],y_arr[i,3]]
        thisz = [z_arr[i,0],z_arr[i,1],z_arr[i,2],z_arr[i,3]]

        line.set_data_3d(thisx,thisy,thisz)
        n = 3
        if i > n-1:
            lastx = [x_arr[i-n,0],x_arr[i-n,1],x_arr[i-n,2],x_arr[i-n,3]]
            lasty = [y_arr[i-n,0],y_arr[i-n,1],y_arr[i-n,2],y_arr[i-n,3]]
            lastz = [z_arr[i-n,0],z_arr[i-n,1],z_arr[i-n,2],z_arr[i-n,3]]
            line2.set_data_3d(lastx,lasty,lastz)
        else:
            line2.set_data_3d(thisx,thisy,thisz)

        n = 6
        if i > n-1:
            lastx = [x_arr[i-n,0],x_arr[i-n,1],x_arr[i-n,2],x_arr[i-n,3]]
            lasty = [y_arr[i-n,0],y_arr[i-n,1],y_arr[i-n,2],y_arr[i-n,3]]
            lastz = [z_arr[i-n,0],z_arr[i-n,1],z_arr[i-n,2],z_arr[i-n,3]]
            line3.set_data_3d(lastx,lasty,lastz)
        else:
            line3.set_data_3d(thisx,thisy,thisz)

        # set object lines 
        objx,objy,objz = x_arr2[i],y_arr2[i],z_arr2[i]
        line4.set_data_3d(objx,objy,objz)
        n = 3
        if i > n-1:
            lastx = x_arr2[i-n]
            lasty = y_arr2[i-n]
            lastz = z_arr2[i-n]
            line5.set_data_3d(lastx,lasty,lastz)
        else:
            line5.set_data_3d(objx,objy,objz)

        n = 6
        if i > n-1:
            lastx = x_arr2[i-n]
            lasty = y_arr2[i-n]
            lastz = z_arr2[i-n]
            line6.set_data_3d(lastx,lasty,lastz)
        else:
            line6.set_data_3d(objx,objy,objz)
    else:
        # i is now used to iterate over a the objects at a center point in time
        if i%x_box[0,:].shape[0] == 0:
            j = j + 1
            j = j%x_arr.shape[0]

        thisx = [x_arr[j,0],x_arr[j,1],x_arr[j,2],x_arr[j,3]]
        thisy = [y_arr[j,0],y_arr[j,1],y_arr[j,2],y_arr[j,3]]
        thisz = [z_arr[j,0],z_arr[j,1],z_arr[j,2],z_arr[j,3]]

        line.set_data_3d(thisx,thisy,thisz)
        n = 3
        if j > n-1:
            lastx = [x_arr[j-n,0],x_arr[j-n,1],x_arr[j-n,2],x_arr[j-n,3]]
            lasty = [y_arr[j-n,0],y_arr[j-n,1],y_arr[j-n,2],y_arr[j-n,3]]
            lastz = [z_arr[j-n,0],z_arr[j-n,1],z_arr[j-n,2],z_arr[j-n,3]]
            line2.set_data_3d(lastx,lasty,lastz)
        else:
            line2.set_data_3d(thisx,thisy,thisz)

        n = 6
        if j > n-1:
            lastx = [x_arr[j-n,0],x_arr[j-n,1],x_arr[j-n,2],x_arr[j-n,3]]
            lasty = [y_arr[j-n,0],y_arr[j-n,1],y_arr[j-n,2],y_arr[j-n,3]]
            lastz = [z_arr[j-n,0],z_arr[j-n,1],z_arr[j-n,2],z_arr[j-n,3]]
            line3.set_data_3d(lastx,lasty,lastz)
        else:
            line3.set_data_3d(thisx,thisy,thisz)

        # set object lines 
        objx,objy,objz = x_arr2[j],y_arr2[j],z_arr2[j]
        line4.set_data_3d(objx,objy,objz)
        n = 3
        if j > n-1:
            lastx = x_arr2[j-n]
            lasty = y_arr2[j-n]
            lastz = z_arr2[j-n]
            line5.set_data_3d(lastx,lasty,lastz)
        else:
            line5.set_data_3d(objx,objy,objz)

        n = 6
        if j > n-1:
            lastx = x_arr2[j-n]
            lasty = y_arr2[j-n]
            lastz = z_arr2[j-n]
            line6.set_data_3d(lastx,lasty,lastz)
        else:
            line6.set_data_3d(objx,objy,objz)

        n = 3
        pos = np.array([x_box[j-n,i],y_box[j-n,i],z_box[j-n,i]])
        xi,yi,zi = box.render(pos)
        line7.set_data_3d(xi,yi,zi)

    return line, line2, line3, line4, line5, line6, line7

# Creating fifty line objects.
# NOTE: Can't pass empty arrays into 3d version of plot()
# lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]

# Setting the axes properties
ax.set_xlim3d(lims[0,:])
ax.set_xlabel('X')

ax.set_ylim3d(lims[1,:])
ax.set_ylabel('Y')

ax.set_zlim3d(lims[2,:])
ax.set_zlabel('Z')

ax.set_title('3D Test')

# Creating the Animation object
if show_box:
    N = x_box[0,:].shape[0]
    speed = dt*10000/2
else:
    N = x_arr.shape[0]
    speed = dt*1000

ani = animation.FuncAnimation(
    fig, update, N, interval=speed, blit=False)

plt.show()