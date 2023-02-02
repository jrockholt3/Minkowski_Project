def get_coord_list2(self, res = res, make_plot=False):
        # this function creates a coordinate list of points along
        # the object's surface based on a spherical representation, the radius
        # and the resolution of the workspace
        # the function starts at one end of the sphere then iterates over the x-axis
        # at each "slice" along the x-axis, it finds points on the surfaces
        def check_range(point, limits):
            if np.all(point >= limits[:,0]) and np.all(point<=limits[:,1]):
                return True 
            else:
                return False

        coord_list = []
        feat_list = []
        if np.all(np.abs(self.curr_pos)-self.radius <= workspace_limits[:,1]):
            check = [[0,res,-res],[0,res,0],[0,res,res],[0,0,res],[0,-res,res],[0,-res,0],[0,-res,-res],[0,0,-res],[0,res,-res]]
            r_int = int(np.round(self.radius/res))
            r = self.radius
            pos = self.curr_pos
            x_c,y_c = pos[0],pos[1]
            pos = pos - np.array([r,0,0])
            if check_range(pos, workspace_limits): 
                coord_list.append(np.hstack([self.t,quantize(pos)]))
                feat_list.append(np.array([1]))
            pos = pos + np.array([res,0,0])
            x = pos[0]
            for i in range(-r_int+1,r_int):
                y_start = np.sqrt(abs(r**2 - (x-x_c)**2)) + y_c
                start = np.array([pos[0],y_start,pos[2]])
                pos = start
                done = False
                n = 0
                while not done:
                    n += 1
                    found_next = False
                    for j in range(1,len(check)):
                        check_loc = pos + check[j] - self.curr_pos
                        prev_loc = pos + check[j-1] - self.curr_pos
                        if np.round(np.linalg.norm(check_loc),2) == np.round(r,2) and np.round(np.linalg.norm(prev_loc),2) > np.round(r,2):
                            pos = pos + check[j]
                            if check_range(pos, workspace_limits):
                                # coord_list.append(quantize(pos).reshape(1,len(pos)))
                                coord_list.append(np.hstack([self.t,quantize(pos)]))
                                feat_list.append(np.array([1]))
                            j = len(check) + 1
                            found_next = True
                    
                    if np.all(np.round(start,2) == np.round(pos,2)) and found_next:
                        done = True
                    elif not found_next:
                        done = True
                    elif n > 1000:
                        done = True
                        print('get_coord_list broke on over flow')
                

                pos = np.array([pos[0]+res, self.curr_pos[1], self.curr_pos[2]])
                x = pos[0]
            
            arr = self.curr_pos+np.array([r,0,0])
            if check_range(arr,workspace_limits):
                # coord_list.append(quantize(pos).reshape(1,len(pos)))
                coord_list.append(np.hstack([self.t,quantize(arr)]))
                feat_list.append(np.array([1]))

            if make_plot:
                coord_list2 = np.array(coord_list)
                xx = coord_list2[:,1]
                yy = coord_list2[:,2]
                zz = coord_list2[:,3]
                fig = plt.figure()
                ax = plt.axes(projection='3d')
                ax.plot3D(xx,yy,zz)
                ax.scatter3D(xx,yy,zz,alpha=.5)
                # ax.axes.set_xlim3d(left=self.curr_pos[0] - 1.5*r, right=self.curr_pos[0] + 1.5*r) 
                # ax.axes.set_ylim3d(bottom=self.curr_pos[1] - 1.5*r, top=self.curr_pos[1] + 1.5*r) 
                # ax.axes.set_zlim3d(bottom=self.curr_pos[2] - 1.5*r, top=self.curr_pos[2] + 1.5*r)
                plt.show()
        else:
            print('object out of workspace', self.curr_pos)


        return np.array(coord_list), np.array(feat_list)