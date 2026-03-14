# -*- coding:UTF-8 -*-
import os
import csv
import time
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import copy
import matplotlib.animation as animation

# import warnings
# warnings.simplefilter('ignore', np.RankWarning)


#修改了数据读取方式，直接传入
class LaneData:
    def __init__(self, lanedata):
        self.lane = lanedata.tolist()
    # bool function, return 1 if read data successfully, otherwise return 0 
    def read_data(self):
        readflag = 0
        if self.lane is not None:
            readflag = 1
        return readflag
    
class AllLaneData:
    #修改了一下输入参数，原来是5个文件名，现在直接传入类对象
    def __init__(self, ctrb, lcl, lbd, rcl, rbd):
        self.centerboundary_data = LaneData(ctrb)
        self.leftcenterline_data = LaneData(lcl)
        self.leftboundary_data = LaneData(lbd)
        self.rightcenterline_data = LaneData(rcl)
        self.rightboundary_data = LaneData(rbd)
    
    def read_all_data(self):
        if (self.centerboundary_data.read_data() == 0 or self.leftcenterline_data.read_data() == 0 or
            self.leftboundary_data.read_data() == 0 or self.rightcenterline_data.read_data() == 0 or 
            self.rightboundary_data.read_data() == 0):
            print("read all data failed!")
    
    def draw_all_data(self):
        plt.figure(1)
        plt.plot([row[0] for row in self.centerboundary_data.lane], 
                 [row[1] for row in self.centerboundary_data.lane], linestyle = '--', color = 'black', label='centerboundary')
        # plt.plot([row[0] for row in self.leftcenterline_data.lane], 
        #          [row[1] for row in self.leftcenterline_data.lane], linestyle = '--', color = 'blue', label='leftcenterline')
        plt.plot([row[0] for row in self.leftboundary_data.lane], 
                 [row[1] for row in self.leftboundary_data.lane], color = 'black', label='leftboundary')
        # plt.plot([row[0] for row in self.rightcenterline_data.lane], 
        #          [row[1] for row in self.rightcenterline_data.lane], linestyle = '--', color = 'blue', label='rightcenterline')
        plt.plot([row[0] for row in self.rightboundary_data.lane], 
                 [row[1] for row in self.rightboundary_data.lane], color = 'black', label='rightboundary')
        plt.title('Bend03 Road Scenario')
        plt.xlabel('X-axis (m)')
        plt.ylabel('Y-axis (m)')
        plt.axis('equal')
        plt.show(block=True)
        
class VehicleTrajData:
    def __init__(self, trajectory_state):
        self.WHEELBASE = 2.6 # check 2.875 wheelbase of model3, but use 2.6 in cec experiment VD
        # current_directory = os.getcwd()

        self.plan_param = [0,0,0,0] # vp, dyp, ve, dye
        self.index = trajectory_state[:,0].tolist()
        self.timestamp = trajectory_state[:,1].tolist()
        self.ego_state = trajectory_state[:,2:7].tolist()
        self.ngb_state = trajectory_state[:,7:].tolist()
        self.ego_curvature = []
        self.ngb_curvature = []
        self.ngb_steering_angle = []
        
    
    def cal_curvature_str_angle(self):
        self.ego_curvature = self.func_cal_curvature(self.ego_state)
        self.ngb_curvature = self.func_cal_curvature(self.ngb_state)
        self.ngb_steering_angle = self.func_cal_steering_angle(self.ngb_curvature)

        # self.ego_state = np.array(self.ego_state)
        # self.ego_state[:,-1] = self.func_cal_steering_angle(self.ego_curvature)
        # self.ego_state = self.ego_state.tolist()
    
    def func_cal_steering_angle(self, curvature):
        steering_angle = []
        for i in range(len(curvature)):
            steering_angle.append(np.arctan(self.WHEELBASE * curvature[i]))
        return steering_angle
    
    # calculate the curvature of the trajectory
    # def func_cal_curvature(self, state):
    #     dx = np.gradient([row[0] for row in state])
    #     dy = np.gradient([row[1] for row in state])
    #     ddx = np.gradient(dx)
    #     ddy = np.gradient(dy)
    #     curvature = []
    #     for i in range(len(ddx)):
    #         curvature.append((dx[i]*ddy[i] - dy[i]*ddx[i]) / (dx[i]**2 + dy[i]**2)**(3/2))
    #     return curvature
    
    # 二次多项式拟合求曲率
    def func_cal_curvature(self, state):
        """curvatue calculate

        Args:
            trajectory (np.array([x, y, speed])): the coordinate of trajectory

        Returns:
            np.array([cur1, cur2,...]): curvatue of every point on trajectory
        """
        #前后3m，共6m进行拟合
        CURV_RANGE = 3
        state = np.array(state)
        curvature = np.zeros((state.shape[0], 1))
        x = state[:,0]
        y = state[:,1]
        dis = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        dis = np.cumsum(dis)
        dis = np.insert(dis, 0, 0)

        # start_time = time.time()
        for i in range(state.shape[0]):
            # start_step_time = time.time()
            # step forward to curve_range to get maximum id_max
            id_max = i
            dis_curve = dis[i]
            while dis[id_max] - dis_curve < CURV_RANGE:
                id_max += 1
                if id_max >= state.shape[0]:
                    break
            # step backward to curve_range to get minimum id_min
            id_min = i
            while dis_curve - dis[id_min] < CURV_RANGE:
                id_min -= 1
                if id_min <= 0:
                    id_min = 0
                    break
            # get the interpolation that closes to each node
            x_close = x[id_min:id_max]
            y_close = y[id_min:id_max]
            if len(x_close) <= 3:
                curvature[i] = 0
                continue
            # 2 order poly fit
            param_t = dis[id_min:id_max] - dis[id_min]
            coef_matrix = np.zeros((len(x_close), 3))
            coef_matrix[:,0] = 1                 # Constant term
            coef_matrix[:,1] = param_t           # Coefficient of linear term
            coef_matrix[:,2] = param_t*param_t   # Coefficient of quadratic term

            # QR method to solve Least-Squares problem,  because np.linalg.solve can only solve square matrix Ax = b
            coef_x = np.linalg.lstsq(coef_matrix, x_close, rcond=None)[0]
            coef_y = np.linalg.lstsq(coef_matrix, y_close, rcond=None)[0]

            # curvature calculation through Curvature calculation formula of reference equations 
            x_dot = coef_x[1]   # x' = b + 2ct   but param_t = 0 and the start point of x_close
            x_ddot = 2 * coef_x[2]
            y_dot = coef_y[1]
            y_ddot = 2 * coef_y[2]
            curvature[i] = (x_dot*y_ddot - x_ddot*y_dot) / (x_dot**2 + y_dot**2)**(3./2)
            
            if np.abs(curvature[i]) < 0.00001:
                curvature[i] = 0
            # end_step_time = time.time()
            # print(f"Step {i} took {end_step_time - start_step_time} seconds")

        # end_time = time.time()
        # print(f"Total time of curvature calculate: {end_time - start_time} seconds")
        return curvature[:,0].tolist()
    def draw_data(self):
        plt.figure(1)
        plt.plot([row[0] for row in self.ego_state], [row[1] for row in self.ego_state], linestyle = '--', color = 'black', label='ego trajectory')
        plt.plot([row[0] for row in self.ngb_state], [row[1] for row in self.ngb_state], linestyle = '--', color = 'black', label='ngb trajectory')
        plt.title('Ego & Neighbor Vehicle Trajectory')
        plt.xlabel('X-axis (m)')
        plt.ylabel('Y-axis (m)')
        plt.axis('equal')
        plt.show(block=True)


class IndicatorData:
    def __init__(self, lane_data, vehtraj_data):
        # driver position offset & WHEELBASE is fixed
        self.driverDx = 1.7
        self.driverDy = 0.5
        self.WHEELBASE = 2.6
        self.lane_data = lane_data
        self.vehtraj_data = vehtraj_data
        self.drivertraj_data = self.get_traj_driver() # update driver trajectory
        self.ttc = []
        self.ttc_flags = []
        self.lateraloffset = []
        self.stlc = []
        self.ctlc = []
        self.ctad = []
        self.showtimecomsuing = False

    # convert the ego trajectory to driver trajectory
    def get_traj_driver(self):
        drivertraj_data = copy.deepcopy(self.vehtraj_data)
        for i in range(len(self.vehtraj_data.ego_state)):
            x0 = self.vehtraj_data.ego_state[i][0]
            y0 = self.vehtraj_data.ego_state[i][1]
            yaw0 = self.vehtraj_data.ego_state[i][2]
            speed0 = self.vehtraj_data.ego_state[i][3]
            str_angle0 = self.vehtraj_data.ego_state[i][4]
            x = x0 + self.driverDx * np.cos(yaw0) - self.driverDy * np.sin(yaw0)
            y = y0 + self.driverDx * np.sin(yaw0) + self.driverDy * np.cos(yaw0)
            drivertraj_data.ego_state[i] = [x, y, yaw0, speed0, str_angle0]
        return drivertraj_data

    # test ego trajectory and driver trajectory
    def draw_ego_driver_traj(self):
        plt.figure(1)
        plt.plot([row[0] for row in self.vehtraj_data.ego_state], [row[1] for row in self.vehtraj_data.ego_state], linestyle = '-.', linewidth = 0.5, color = 'red', label='ego trajectory')
        plt.plot([row[0] for row in self.vehtraj_data.ngb_state], [row[1] for row in self.vehtraj_data.ngb_state], linestyle = '-.', linewidth = 0.5, color = 'blue', label='ngb trajectory')
        plt.plot([row[0] for row in self.drivertraj_data.ego_state], [row[1] for row in self.drivertraj_data.ego_state], linestyle = '-.', linewidth = 0.5, color = 'green', label='driver trajectory')
        plt.title('Ego, Driver & Neighbor Vehicle Trajectory')
        plt.xlabel('X-axis (m)')
        plt.ylabel('Y-axis (m)')
        plt.axis('equal')
        plt.show(block=False)
        plt.waitforbuttonpress()
        plt.close()

    # draw the lane and vehicle trajectory
    def draw_lane_vehtraj(self):
        plt.figure(1)
        # add last pt to close the loop
        centerboundary = self.lane_data.centerboundary_data.lane
        leftboundary = self.lane_data.leftboundary_data.lane
        rightboundary = self.lane_data.rightboundary_data.lane
        centerboundary.append(centerboundary[0])
        leftboundary.append(leftboundary[0])
        rightboundary.append(rightboundary[0])
        
        plt.plot([row[0] for row in centerboundary], 
                 [row[1] for row in centerboundary], linestyle = '--', linewidth = 0.5, color = 'black', label='centerboundary')
        # plt.plot([row[0] for row in self.lane_data.leftcenterline_data.lane], 
        #          [row[1] for row in self.lane_data.leftcenterline_data.lane], linestyle = '--', linewidth = 0.5, color = 'blue', label='leftcenterline')
        plt.plot([row[0] for row in leftboundary], 
                 [row[1] for row in leftboundary], color = 'black', linewidth = 0.5, label='leftboundary')
        # plt.plot([row[0] for row in self.lane_data.rightcenterline_data.lane], 
        #          [row[1] for row in self.lane_data.rightcenterline_data.lane], linestyle = '--', linewidth = 0.5, color = 'blue', label='rightcenterline')
        plt.plot([row[0] for row in rightboundary], 
                 [row[1] for row in rightboundary], color = 'black', linewidth = 0.5, label='rightboundary')
        
        # plt.plot([row[0] for row in self.vehtraj_data.ego_state], [row[1] for row in self.vehtraj_data.ego_state], linestyle = '-.', linewidth = 0.5, color = 'red', label='ego trajectory')
        # plt.plot([row[0] for row in self.vehtraj_data.ngb_state], [row[1] for row in self.vehtraj_data.ngb_state], linestyle = '-.', linewidth = 0.5, color = 'blue', label='ngb trajectory')
        # plt.title('Bend03: Ego & Neighbor Vehicle Trajectory')
        plt.title('Bend03')
        plt.xlabel('X-axis (m)')
        plt.ylabel('Y-axis (m)')
        plt.axis('equal')
        plt.grid(True)
        plt.show(block=False)
        plt.waitforbuttonpress()
        plt.close()
    
    # 2d transform with single point
    def transform2d(self, xin, yin, tf_mat):
        xout = tf_mat[0][0] * xin + tf_mat[0][1] * yin + tf_mat[0][2]
        yout = tf_mat[1][0] * xin + tf_mat[1][1] * yin + tf_mat[1][2]
        return xout, yout

    # calculate the ttc
    def func_cal_ttc(self, traj_data, isfigure):
        # I notice that the ttc is switching + or - very fast and I think it is not correct
        print("Calculating ttc...")
        threshold = 1e-3  #对指标计算影响很大
        ind_ttc = []
        ind_ttc_flags = []
        dist_Ego2ngb = []
        if isfigure == 1:
            plt.ion()
            fig = plt.figure(2)
            ax = fig.add_subplot(111)
        
        for i in range(len(traj_data.ego_state)):
            dist_Ego2ngb.append(np.sqrt((traj_data.ego_state[i][0]-traj_data.ngb_state[i][0])**2 + 
                                        (traj_data.ego_state[i][1]-traj_data.ngb_state[i][1])**2))

        # 平滑距离
        window_size = 4# 设置窗口大小
        dist_Ego2ngb = pd.Series(dist_Ego2ngb)
        dist_Ego2ngb = dist_Ego2ngb.rolling(window_size).mean()
        dist_Ego2ngb = dist_Ego2ngb.bfill() #后一个非缺失值来填充缺失值

        t_ego = traj_data.timestamp
        t_ego_dot = np.diff(t_ego)
        t_ego_dot = np.append(t_ego_dot, t_ego_dot[-1])
        dist_Ego2ngb_dot = np.diff(dist_Ego2ngb)
        dist_Ego2ngb_dot = np.append(dist_Ego2ngb_dot, dist_Ego2ngb_dot[-1])

        # 平滑距离差分
        window_size = 4# 设置窗口大小
        dist_Ego2ngb_dot = pd.Series(dist_Ego2ngb_dot)
        dist_Ego2ngb_dot = dist_Ego2ngb_dot.rolling(window_size).mean()
        dist_Ego2ngb_dot = dist_Ego2ngb_dot.bfill() #后一个非缺失值来填充缺失值

        # relative velocity, Vr>0 means fading away; Vr<0 means closing
        Vr = [dist_Ego2ngb_dot[i] / t_ego_dot[i] for i in range(len(t_ego_dot))]
        for i in range(len(traj_data.ego_state)):
            #这里的判断逻辑做了修改
            if (dist_Ego2ngb[i] == 0):
                ttc_tmp = -2
                ind_ttc.append(ttc_tmp)
                ind_ttc_flags.append(-1) # -1 means invalid
            elif (np.abs(Vr[i]) <threshold):
                ttc_tmp = -1
                ind_ttc.append(ttc_tmp)
                ind_ttc_flags.append(0) # 0 means relative static
            elif (Vr[i] < 0):
                ttc_tmp = - dist_Ego2ngb[i] / Vr[i]
                ind_ttc.append(ttc_tmp)
                ind_ttc_flags.append(1) # 1 means closing
            elif (Vr[i] > 0):
                ttc_tmp = -1
                ind_ttc.append(ttc_tmp)
                ind_ttc_flags.append(2) # 2 means fading away
                
            # plot the animation
            if isfigure == 1:
                ax.clear()
                x_plot = [traj_data.ego_state[i][0], traj_data.ngb_state[i][0]]
                y_plot = [traj_data.ego_state[i][1], traj_data.ngb_state[i][1]]
                ax.plot([row[0] for row in self.lane_data.centerboundary_data.lane], [row[1] for row in self.lane_data.centerboundary_data.lane], 
                        linestyle = '-.', color = 'black', linewidth = 0.5, label='center boundary to ego')
                ax.plot([row[0] for row in self.lane_data.leftboundary_data.lane], [row[1] for row in self.lane_data.leftboundary_data.lane], 
                        linestyle = '-.', color = 'black', linewidth = 0.5, label='left boundary to ego')
                ax.plot([row[0] for row in self.lane_data.rightboundary_data.lane], [row[1] for row in self.lane_data.rightboundary_data.lane], 
                        linestyle = '-.', color = 'black', linewidth = 0.5, label='right boundary to ego')
                ax.plot(traj_data.ego_state[i][0], traj_data.ego_state[i][1], 'r*', label='ego vehicle')
                ax.plot(traj_data.ngb_state[i][0], traj_data.ngb_state[i][1], 'r*', label='ngb vehicle')
                if (Vr[i] < 0):
                    ax.plot(x_plot, y_plot, linestyle = '--', linewidth = 0.5, color = 'green', label='ego vehicle & neighbor vehicle')
                    ax.text(traj_data.ego_state[i][0], traj_data.ego_state[i][1]+8, 'now closing and ttc = %.4f' % ttc_tmp, ha='center', va='bottom', fontsize=10)
                elif (Vr[i] > 0):
                    ax.plot(x_plot, y_plot, linestyle = '--', linewidth = 0.5, color = 'blue', label='ego vehicle & neighbor vehicle')
                    ax.text(traj_data.ego_state[i][0], traj_data.ego_state[i][1]+8, 'now fading away and ttc = %.4f' % ttc_tmp, ha='center', va='bottom', fontsize=10)
                else:
                    ax.plot(x_plot, y_plot, linestyle = '--', linewidth = 0.5, color = 'yellow', label='ego vehicle & neighbor vehicle')
                    ax.text(traj_data.ego_state[i][0], traj_data.ego_state[i][1]+8, 'now relative static and ttc = infinite', ha='center', va='bottom', fontsize=10)
                
                ax.set_xlim(350, 600)
                ax.set_ylim(-360, -180)
                ax.set_title('Vehicle Trajectory Animation')
                ax.set_xlabel('X Axis (m)')
                ax.set_ylabel('Y Axis (m)')
                ax.axis('equal')
                ax.legend()
                fig.canvas.draw()
                fig.canvas.flush_events()
                time.sleep(0.01)
        
        print("Calculating ttc Done!")    
        return ind_ttc, ind_ttc_flags

    # calculate the lateral offset
    def func_cal_lateraloffset(self, traj_data, lanecenterline, isfigure):
        print("Calculating lateral offset...")
        lateraloffset = []
        lanectrline = lanecenterline
        if isfigure == 1:
            plt.ion()
            fig = plt.figure(2)
            ax = fig.add_subplot(111)
            
        for i in range(len(traj_data.ego_state)):
            # fine the nearest point on the lane centerline
            x0 = traj_data.ego_state[i][0]
            y0 = traj_data.ego_state[i][1]
            yaw0 = traj_data.ego_state[i][2]
            dist_Ego2lane = []
            for lane_pt in lanectrline:
                dist_Ego2lane.append(np.sqrt((x0-lane_pt[0])**2 + (y0-lane_pt[1])**2))
            min_dist_idx = dist_Ego2lane.index(min(dist_Ego2lane))
            
            # fine the nearest point and conduct interp
            if (min_dist_idx == 0):
                x2interp = np.array([lanectrline[min_dist_idx][0],lanectrline[min_dist_idx+1][0],lanectrline[min_dist_idx+2][0]])
                y2interp = np.array([lanectrline[min_dist_idx][1],lanectrline[min_dist_idx+1][1],lanectrline[min_dist_idx+2][1]])
                yaw2interp = np.array([lanectrline[min_dist_idx][2],lanectrline[min_dist_idx+1][2],lanectrline[min_dist_idx+2][2]])
            elif (min_dist_idx == len(lanectrline)-1):
                x2interp = np.array([lanectrline[min_dist_idx-2][0],lanectrline[min_dist_idx-1][0],lanectrline[min_dist_idx][0]])
                y2interp = np.array([lanectrline[min_dist_idx-2][1],lanectrline[min_dist_idx-1][1],lanectrline[min_dist_idx][1]])
                yaw2interp =  np.array([lanectrline[min_dist_idx-2][2],lanectrline[min_dist_idx-1][2],lanectrline[min_dist_idx][2]])
            else:
                x2interp = np.array([lanectrline[min_dist_idx-1][0],lanectrline[min_dist_idx][0],lanectrline[min_dist_idx+1][0]])
                y2interp = np.array([lanectrline[min_dist_idx-1][1],lanectrline[min_dist_idx][1],lanectrline[min_dist_idx+1][1]])
                yaw2interp = np.array([lanectrline[min_dist_idx-1][2],lanectrline[min_dist_idx][2],lanectrline[min_dist_idx+1][2]])
            
            # interp
            func_xy_interp = CubicSpline(x2interp, y2interp)
            func_xyaw_interp = CubicSpline(x2interp, yaw2interp)
            x_interp = np.linspace(x2interp[0], x2interp[-1], 1000)
            y_interp = func_xy_interp(x_interp)
            yaw_interp = func_xyaw_interp(x_interp)
            
            # calculate the lateral offset
            dist_Ego2interp = []
            for i in range(len(x_interp)):
                dist_Ego2interp.append(np.sqrt((x0-x_interp[i])**2 + (y0-y_interp[i])**2))
            min_dist_idx2interp = dist_Ego2interp.index(min(dist_Ego2interp))
            x_ego2interp = x_interp[min_dist_idx2interp]
            y_ego2interp = y_interp[min_dist_idx2interp]
            yaw_ego2interp = yaw_interp[min_dist_idx2interp]
            world2interplane_tf = [[np.cos(yaw_ego2interp), -np.sin(yaw_ego2interp), x_ego2interp],
                                      [np.sin(yaw_ego2interp), np.cos(yaw_ego2interp), y_ego2interp],
                                      [0, 0, 1]]
            interplane2world_tf = np.linalg.inv(world2interplane_tf)
            _, y_ego2lane = self.transform2d(x0, y0, interplane2world_tf)
            lateraloffset.append(y_ego2lane)
            
            # plot the animation
            if isfigure == 1:
                ax.clear()
                # ax.plot([row[0] for row in lanectrline], [row[1] for row in lanectrline], 
                #         linestyle = '-.', color = 'black', linewidth = 0.5, label='center lane to ego')
                ax.plot(x0, y0, 'ro', label='ego vehicle')
                ax.plot(x_ego2interp, y_ego2interp, 'go', label='nearest point on center lane')
                ax.plot(x_interp, y_interp, linestyle = '-.', linewidth = 0.5, color = 'black', label='fit center lane')
                ax.set_xlim(350, 600)
                ax.set_ylim(-360, -180)
                ax.set_title('Vehicle Trajectory Animation')
                ax.set_xlabel('X Axis (m)')
                ax.set_ylabel('Y Axis (m)')
                ax.axis('equal')
                ax.legend()
                fig.canvas.draw()
                fig.canvas.flush_events()
                time.sleep(0.01)
                
        if isfigure == 1:
            plt.ioff()
            plt.show()
        print("Calculating lateral offset Done!")
        return lateraloffset

    # predict the trajectory of ego vehicle with fixed steering_angle in 5 seconds
    # frontwheel steering_angle: rad
    def func_trajpredict(self, ego_pt, steering_angle):
        WHEELBASE = self.WHEELBASE
        trajpredict = []
        x0 = ego_pt[0]
        y0 = ego_pt[1]
        yaw0 = ego_pt[2]
        speed0 = ego_pt[3]
        t0 = 0.0
        steering_angle0 = steering_angle
        DT = 0.03 # time interval 0.03s
        PREDICTION_TIME = 7.0 # predict 7 instead of 5 seconds
        trajpredict.append([x0, y0, yaw0, speed0, t0])
        for _ in np.arange(0.0, PREDICTION_TIME, DT):
            x1 = x0 + speed0 * np.cos(yaw0) * DT
            y1 = y0 + speed0 * np.sin(yaw0) * DT
            yaw1 = yaw0 + speed0 / WHEELBASE * np.tan(steering_angle0) * DT
            yaw1 = np.arctan2(np.sin(yaw1), np.cos(yaw1))
            speed1 = speed0
            t1 = t0 + DT
            trajpredict.append([x1, y1, yaw1, speed1, t1])
            x0 = x1
            y0 = y1
            yaw0 = yaw1
            speed0 = speed1
            t0 = t1
        return trajpredict

    # calculate the ctad between ego vehicle and neighbor vehicle
    # method 1: parameter curve fitting
    # method 2: directsolvingtlc
    # method 3: gridmap
    def cal_ctad(self, vehtraj_data, whichmethod, isfigure):
        print("Calculating ctad...")
        ctad = []
        ctad_flag = []
        if isfigure == 1:
            plt.ion()
            fig = plt.figure(3)
            ax = fig.add_subplot(111)
            
        for idx in range(len(vehtraj_data.ego_state)):
            ego_pt = vehtraj_data.ego_state[idx]
            ngb_pt = vehtraj_data.ngb_state[idx]
            ngb_str_angle = vehtraj_data.ngb_steering_angle[idx]
            # ngb_str_angle = -1 * np.pi / 180 # 5 degree
            ego_trajpredict = self.func_trajpredict(ego_pt, ego_pt[4])
            ngb_trajpredict = self.func_trajpredict(ngb_pt, ngb_str_angle)
            
            if (whichmethod == 1):
                _, inter_pt, ngb_travel_pt, ctad_tmp, ctad_flag_tmp = self.func_ctad_paramcurvefitting(ego_trajpredict, ngb_trajpredict, 0)
            elif (whichmethod == 2):
                _, inter_pt, ngb_travel_pt, ctad_tmp, ctad_flag_tmp = self.func_ctad_directsolvingtlc(ego_trajpredict, ngb_trajpredict, 0)
            elif (whichmethod == 3):
                _, inter_pt, ngb_travel_pt, ctad_tmp, ctad_flag_tmp = self.func_ctad_gridmap(ego_trajpredict, ngb_trajpredict, 0)
            else:
                print("Please input correct method number!")
                return
            
            ctad.append(ctad_tmp)
            ctad_flag.append(ctad_flag_tmp)
            
            # plot the animation
            if isfigure == 1:
                ax.clear()
                ax.plot([row[0] for row in ego_trajpredict], [row[1] for row in ego_trajpredict], 
                        linestyle = '-.', color = 'green', linewidth = 0.5, label='ego trajectory')
                ax.plot([row[0] for row in ngb_trajpredict], [row[1] for row in ngb_trajpredict], 
                        linestyle = '-.', color = 'blue', linewidth = 0.5, label='ngb trajectory')
                if (ctad_tmp > 0):
                    ax.plot(inter_pt[0], inter_pt[1], marker = '*', color = 'red', label='intersection point')
                    ax.plot(ngb_travel_pt[0], ngb_travel_pt[1], marker = '*', color = 'blue', label='ngb travel point')
                ax.text(ego_pt[0], ego_pt[1]+5, 'ctad = %.4fs' % ctad_tmp, ha='center', va='bottom', fontsize=10)
                ax.text(ego_pt[0], ego_pt[1]+3, 'ctad_flag = %.4fs' % ctad_flag_tmp, ha='center', va='bottom', fontsize=10)
                ax.set_xlim(350, 600)
                ax.set_ylim(-360, -180)
                ax.set_title('Vehicle Trajectory Animation')
                ax.set_xlabel('X Axis (m)')
                ax.set_ylabel('Y Axis (m)')
                ax.axis('equal')
                ax.legend()
                fig.canvas.draw()
                fig.canvas.flush_events()
                time.sleep(0.0001)
                
        if isfigure == 1:
            plt.ioff()
            plt.show()
            
        print("Calculating ctad Done!")
        return ctad, ctad_flag

    # ATTENTION: solve ctad using parameter curve fitting method

    # ATTENTION: solve ctad using direct distance solving method
    def func_ctad_directsolvingtlc(self, trajdata1, trajdata2, isfigure):
        traj1 = np.array(trajdata1)[:, :2]
        traj2 = np.array(trajdata2)[:,:2]

        speed_ego = trajdata1[0][3]
        speed_ngb = trajdata2[0][3]
        DIST_THRESHOLD = 5 * 1e-2 # 5cm
        exist_inter = False
        mdist_pt_to_traj2 = []
        inter_points = []
        idx_inter_in_traj1 = []
        idx_inter_in_traj2 = []
        counter = 0
        # calculate the distance from start to each point in trajdata2
        # 计算相邻行之间的差值
        diff = traj2[1:] - traj2[:-1]
        diff = np.vstack((diff, diff[-1,:]))
        dist2_ngb_one = np.sqrt(np.sum(diff**2, axis=-1))
        # 计算累积和
        dist2_ngb = np.cumsum(dist2_ngb_one)
        
        # 计算所有点对之间的距离  
        distances = np.sum((traj1[:, None, :] - traj2[None, :, :]) ** 2, axis=-1)
        #返回一维列表，轨迹上所有点，到轨迹2的最短距离
        min_dist = np.min(distances, axis=-1)

        for i in range(traj1.shape[0]):
            if min_dist[i] < DIST_THRESHOLD:
                mdist_pt_to_traj2.append(min_dist[i])
                # find the intersection point using the middle point between two points
                inter_pt2 = traj2[np.argmin(distances[i])]
                inter_pt = (traj1[i] + inter_pt2) / 2
                inter_points.append(inter_pt)
                idx_inter_in_traj2.append(np.argmin(distances[i]))
                idx_inter_in_traj1.append(i)
                exist_inter = True
            counter += 1
                        
        if exist_inter == True:
            # select the smallest inter point
            idx_min = mdist_pt_to_traj2.index(min(mdist_pt_to_traj2))
            inter_pt = inter_points[idx_min]
            inter_in_traj1 = trajdata1[idx_inter_in_traj1[idx_min]]
            tlc = inter_in_traj1[4] - trajdata1[0][4]
            
            # calculate the distance from ngb to intersection point
            time_series = np.array([row[4] for row in trajdata2])
            ngb_travel_idx = np.argmin(np.abs(time_series - tlc))
            ngb_travel_pt = trajdata2[ngb_travel_idx]
            ngb_inter_idx = idx_inter_in_traj2[idx_min]
            ngb_inter_pt = trajdata2[ngb_inter_idx]
            dist2_ngb2inter = abs(dist2_ngb[ngb_inter_idx] - dist2_ngb[ngb_travel_idx])
            is_ngb_behind = 1 if ngb_inter_idx > ngb_travel_idx else -1 if ngb_inter_idx < ngb_travel_idx else 0

            # calulate projection speed on intersection point
            ngb_inter_yaw = trajdata2[ngb_inter_idx][2]
            ego_inter_yaw = trajdata1[idx_inter_in_traj1[idx_min]][2]
            speed_proj_ego = speed_ego * np.cos(abs(ego_inter_yaw - ngb_inter_yaw))
            delta_speed = abs(speed_proj_ego - speed_ngb)
            ctad_tmp = dist2_ngb2inter / delta_speed
            
            if ((is_ngb_behind > 0) & (speed_ngb > speed_proj_ego)) | ((is_ngb_behind < 0) & (speed_ngb < speed_proj_ego)):
                # 1 means closing
                ctad_flags = 1
                ctad_closing_factor = 1
            elif ((is_ngb_behind > 0) & (speed_ngb < speed_proj_ego)) | ((is_ngb_behind < 0) & (speed_ngb > speed_proj_ego)):
                # 2 means fading away
                ctad_flags = 2
                ctad_closing_factor = -1
            else:
                # 0 means relative static
                ctad_flags = 0
                ctad_closing_factor = 0
            # ctad = ctad_closing_factor * ctad_tmp
            ctad = ctad_tmp
        else:
            inter_pt = [-1, -1]
            ngb_travel_pt = [-1, -1]
            tlc = -1
            ctad = -1
            ctad_flags = -1 # -1 means invalid
            
        if isfigure == 1:
            plt.figure(2)
            plt.plot([row[0] for row in trajdata1], [row[1] for row in trajdata1], linestyle = '--', linewidth = 0.5, color = 'blue', label='ego trajectory')
            plt.plot([row[0] for row in trajdata2], [row[1] for row in trajdata2], linestyle = '--', linewidth = 0.5, color = 'green', label='ngb trajectory')
            if exist_inter == True:
                plt.plot(inter_pt[0], inter_pt[1], 'r*', label='intersection point')
                # plt.plot(inter_in_traj1[0], inter_in_traj1[1], 'g*', label='intersection point in traj1')
                plt.plot(ngb_travel_pt[0], ngb_travel_pt[1], 'b*', label='ngb travel point')
                plt.plot(ngb_inter_pt[0], ngb_inter_pt[1], 'g*', label='ngb intersection point')
            plt.title('Using distance loop: Ego & Neighbor Vehicle Trajectory')
            plt.xlabel('X-axis (m)')
            plt.ylabel('Y-axis (m)')
            plt.legend()
            plt.axis('equal')
            plt.grid(True)
            plt.show()       
        return tlc, inter_pt, ngb_travel_pt, ctad, ctad_flags

    # ATTENTION: solve ctad using grid map method
    def func_ctad_gridmap(self, trajdata1, trajdata2, isfigure):
        speed_ego = trajdata1[0][3]
        speed_ngb = trajdata2[0][3]        
        exist_inter = False
        GRID_SIZE_X = 70 # 70m
        GRID_SIZE_Y = 50 # 20m
        GRID_RESOLUTION = 0.01 # 0.1m
        traj_min_x1, traj_min_y1 = np.min([[row[0], row[1]] for row in trajdata1], axis=0)
        traj_max_x1, traj_max_y1 = np.max([[row[0], row[1]] for row in trajdata1], axis=0)
        traj_min_x2, traj_min_y2 = np.min([[row[0], row[1]] for row in trajdata2], axis=0)
        traj_max_x2, traj_max_y2 = np.max([[row[0], row[1]] for row in trajdata2], axis=0)
        traj_min_x, traj_min_y = np.min([[traj_min_x1, traj_min_y1], [traj_min_x2, traj_min_y2]], axis=0)
        traj_max_x, traj_max_y = np.max([[traj_max_x1, traj_max_y1], [traj_max_x2, traj_max_y2]], axis=0)
        # check if the trajectory is out of the grid map
        grid_bound_x = traj_max_x - traj_min_x
        grid_bound_y = traj_max_y - traj_min_y
        if (grid_bound_x > GRID_SIZE_X or grid_bound_y > GRID_SIZE_Y):
            # print('The trajectory is out of the grid map!')
            GRID_SIZE_X = int(grid_bound_x) + 10
            GRID_SIZE_Y = int(grid_bound_y) + 10
            # return
        
        grid_traj1 = np.zeros((int(GRID_SIZE_X / GRID_RESOLUTION), int(GRID_SIZE_Y / GRID_RESOLUTION)))
        grid_traj2 = np.zeros((int(GRID_SIZE_X / GRID_RESOLUTION), int(GRID_SIZE_Y / GRID_RESOLUTION)))
        grid_traj3 = np.zeros((int(GRID_SIZE_X / GRID_RESOLUTION), int(GRID_SIZE_Y / GRID_RESOLUTION)))
        # track the time
        speed_traj1 = -1 * np.ones((int(GRID_SIZE_X / GRID_RESOLUTION), int(GRID_SIZE_Y / GRID_RESOLUTION)))
        time_traj1 = -1 * np.ones((int(GRID_SIZE_X / GRID_RESOLUTION), int(GRID_SIZE_Y / GRID_RESOLUTION)))
        speed_traj2 = -1 * np.ones((int(GRID_SIZE_X / GRID_RESOLUTION), int(GRID_SIZE_Y / GRID_RESOLUTION)))
        time_traj2 = -1 * np.ones((int(GRID_SIZE_X / GRID_RESOLUTION), int(GRID_SIZE_Y / GRID_RESOLUTION)))
        yaw_traj1 = -1 * np.ones((int(GRID_SIZE_X / GRID_RESOLUTION), int(GRID_SIZE_Y / GRID_RESOLUTION)))
        yaw_traj2 = -1 * np.ones((int(GRID_SIZE_X / GRID_RESOLUTION), int(GRID_SIZE_Y / GRID_RESOLUTION)))
        
        # drop traj1 points to grid
        for pt in trajdata1:
            x0 = pt[0] - traj_min_x
            y0 = pt[1] - traj_min_y
            yaw0 = pt[2] # yaw
            speed0 = pt[3] # speed
            t0 = pt[4] # time
            xgrid = round(x0 / GRID_RESOLUTION)
            ygrid = round(y0 / GRID_RESOLUTION)
            grid_traj1[xgrid, ygrid] = 1
            time_traj1[xgrid, ygrid] = t0
            speed_traj1[xgrid, ygrid] = speed0
            yaw_traj1[xgrid, ygrid] = yaw0
            # error handling
            x_err = x0 - xgrid * GRID_RESOLUTION
            y_err = y0 - ygrid * GRID_RESOLUTION
        # line up dropped points
        for i in range(len(trajdata1) - 1):
            x0, y0 = trajdata1[i][0] - traj_min_x, trajdata1[i][1] - traj_min_y
            x1, y1 = trajdata1[i+1][0] - traj_min_x, trajdata1[i+1][1] - traj_min_y
            yaw0, yaw1 = trajdata1[i][2], trajdata1[i+1][2]
            t0, t1 = trajdata1[i][4], trajdata1[i+1][4]
            speed0, speed1 = trajdata1[i][3], trajdata1[i+1][3]
            x_vals = np.linspace(x0 / GRID_RESOLUTION, x1 / GRID_RESOLUTION, 100)
            y_vals = np.linspace(y0 / GRID_RESOLUTION, y1 / GRID_RESOLUTION, 100)
            yaw_vals = np.linspace(yaw0, yaw1, 100)
            t_vals = np.linspace(t0, t1, 100)
            speed_vals = np.linspace(speed0, speed1, 100)
            grid_points = np.column_stack((x_vals, y_vals))
            grid_points = np.round(grid_points).astype(int)
            grid_traj1[grid_points[:, 0], grid_points[:, 1]] = 1
            yaw_traj1[grid_points[:, 0], grid_points[:, 1]] = yaw_vals
            time_traj1[grid_points[:, 0], grid_points[:, 1]] = t_vals
            speed_traj1[grid_points[:, 0], grid_points[:, 1]] = speed_vals

        # drop traj2 points to grid
        for pt in trajdata2:
            x0 = pt[0] - traj_min_x
            y0 = pt[1] - traj_min_y
            yaw0 = pt[2] # yaw
            speed0 = pt[3] # speed
            t0 = pt[4] # time
            xgrid = round(x0 / GRID_RESOLUTION)
            ygrid = round(y0 / GRID_RESOLUTION)
            grid_traj2[xgrid, ygrid] = 1
            time_traj2[xgrid, ygrid] = t0
            speed_traj2[xgrid, ygrid] = speed0
            yaw_traj2[xgrid, ygrid] = yaw0
        # line up dropped points
        for i in range(len(trajdata2) - 1):
            x0, y0 = trajdata2[i][0] - traj_min_x, trajdata2[i][1] - traj_min_y
            x1, y1 = trajdata2[i+1][0] - traj_min_x, trajdata2[i+1][1] - traj_min_y
            yaw0, yaw1 = trajdata2[i][2], trajdata2[i+1][2]
            t0, t1 = trajdata2[i][4], trajdata2[i+1][4]
            speed0, speed1 = trajdata2[i][3], trajdata2[i+1][3]
            x_vals = np.linspace(x0 / GRID_RESOLUTION, x1 / GRID_RESOLUTION, 100)
            y_vals = np.linspace(y0 / GRID_RESOLUTION, y1 / GRID_RESOLUTION, 100)
            yaw_vals = np.linspace(yaw0, yaw1, 100)
            t_vals = np.linspace(t0, t1, 100)
            speed_vals = np.linspace(speed0, speed1, 100)
            grid_points = np.column_stack((x_vals, y_vals))
            grid_points = np.round(grid_points).astype(int)
            grid_traj2[grid_points[:, 0], grid_points[:, 1]] = 1
            yaw_traj2[grid_points[:, 0], grid_points[:, 1]] = yaw_vals
            time_traj2[grid_points[:, 0], grid_points[:, 1]] = t_vals
            speed_traj2[grid_points[:, 0], grid_points[:, 1]] = speed_vals

        # combine two grids
        grid_traj3 = grid_traj1 + grid_traj2
        intersection_couter = np.sum(grid_traj3 == 2)
        if intersection_couter == 0:
            inter_idx = np.argwhere(grid_traj3 == 1)
            for idx in inter_idx:
                if idx[0] == 0 or idx[1] == 0 or idx[0] == grid_traj3.shape[0]-1 or idx[1] == grid_traj3.shape[1]-1:
                    continue
                inter_ul = grid_traj3[idx[0]-1, idx[1]-1]
                inter_up = grid_traj3[idx[0]-1, idx[1]]
                inter_ur = grid_traj3[idx[0]-1, idx[1]+1]
                inter_left = grid_traj3[idx[0], idx[1]-1]
                inter_right = grid_traj3[idx[0], idx[1]+1]
                inter_dl = grid_traj3[idx[0]+1, idx[1]-1]
                inter_down = grid_traj3[idx[0]+1, idx[1]]
                inter_dr = grid_traj3[idx[0]+1, idx[1]+1]
                inter_ngb_count = inter_ul + inter_up + inter_ur + inter_left + inter_right + inter_dl + inter_down + inter_dr
                if inter_ngb_count >= 5:
                    grid_traj3[idx[0], idx[1]] = 2
        
        if np.sum(grid_traj3 == 2) > 0:
            intersection_points = np.where(grid_traj3 == 2)
            intersection_points_x = intersection_points[0] * GRID_RESOLUTION + traj_min_x
            intersection_points_y = intersection_points[1] * GRID_RESOLUTION + traj_min_y
            intersection_pt_x = (intersection_points_x[0] + intersection_points_x[-1]) / 2
            intersection_pt_y = (intersection_points_y[0] + intersection_points_y[-1]) / 2
            intersection_pt = [intersection_pt_x, intersection_pt_y]
            # intersection_pt_x = np.median(intersection_points_x)
            # intersection_pt_y = np.median(intersection_points_y)
            
            # calculate the dlc and tlc
            intersection_idx = np.argwhere(grid_traj3 == 2)
            intersection_first_idx = intersection_idx[0]
            intersection_first_time = time_traj1[intersection_first_idx[0], intersection_first_idx[1]]
            dist_first2real = np.sqrt((intersection_points_x[0] - intersection_pt_x)**2 + (intersection_points_y[0] - intersection_pt_y)**2)
            time_first2real = dist_first2real / speed_traj1[intersection_first_idx[0], intersection_first_idx[1]]
            tlc = intersection_first_time + time_first2real
            exist_inter = True
            
            # calculate the distance from ngb to intersection point
            ngb_travel_time = intersection_first_time            
            ngb_interfirst_time = time_traj2[intersection_first_idx[0], intersection_first_idx[1]]
            ngb_inter_time = ngb_interfirst_time + time_first2real
            dist2_ngb2inter = abs(ngb_inter_time - ngb_travel_time) * speed_ngb
            is_ngb_behind = 1 if ngb_inter_time > ngb_travel_time else -1 if ngb_inter_time < ngb_travel_time else 0
            time_differs = np.abs(time_traj2 - ngb_travel_time)
            ngb_travel_idx = np.argwhere(time_differs == np.min(time_differs))
            ngb_travel_pt = [ngb_travel_idx[0][0] * GRID_RESOLUTION + traj_min_x, ngb_travel_idx[0][1] * GRID_RESOLUTION + traj_min_y]
            
            # calulate projection speed on intersection point
            ngb_inter_yaw = yaw_traj2[intersection_first_idx[0], intersection_first_idx[1]]
            ego_inter_yaw = yaw_traj1[intersection_first_idx[0], intersection_first_idx[1]]
            speed_proj_ego = speed_ego * np.cos(abs(ego_inter_yaw - ngb_inter_yaw))
            delta_speed = abs(speed_proj_ego - speed_ngb)
            ctad_tmp = dist2_ngb2inter / delta_speed
            
            if ((is_ngb_behind > 0) & (speed_ngb > speed_proj_ego)) | ((is_ngb_behind < 0) & (speed_ngb < speed_proj_ego)):
                # 1 means closing
                ctad_flags = 1
                ctad_closing_factor = 1
            elif ((is_ngb_behind > 0) & (speed_ngb < speed_proj_ego)) | ((is_ngb_behind < 0) & (speed_ngb > speed_proj_ego)):
                # 2 means fading away
                ctad_flags = 2
                ctad_closing_factor = -1
            else:
                # 0 means relative static
                ctad_flags = 0
                ctad_closing_factor = 0
            # ctad = ctad_closing_factor * ctad_tmp
            ctad = ctad_tmp
        else:
            intersection_pt = [-1, -1]
            ngb_travel_pt = [-1, -1]
            tlc = -1.0
            exist_inter = False
            ctad = -1.0
            ctad_flags = -1 # -1 means invalid
        
        if (isfigure==1):
            grid3_plot = np.flipud(grid_traj3.T)
            plt.figure(1)
            plt.imshow(grid3_plot, cmap='binary', interpolation='nearest')
            # plt.colorbar()
            plt.title('Grid Chessboard')
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.grid(False)
            plt.show()   
            
            plt.figure(2)
            plt.plot([row[0] for row in trajdata1], [row[1] for row in trajdata1], linestyle = '--', linewidth = 0.5, color = 'blue', label='ego trajectory')
            plt.plot([row[0] for row in trajdata2], [row[1] for row in trajdata2], linestyle = '--', linewidth = 0.5, color = 'green', label='ngb trajectory')
            if exist_inter == True:
                plt.plot(intersection_pt_x, intersection_pt_y, 'r*', label='intersection point')
                plt.plot(ngb_travel_pt[0], ngb_travel_pt[1], 'b*', label='ngb travel point')
            plt.title('Using gridmap: Ego & Neighbor Vehicle Trajectory')
            plt.xlabel('X-axis (m)')
            plt.ylabel('Y-axis (m)')
            plt.legend()
            plt.axis('equal')
            plt.grid(True)
            plt.show()
        
        return tlc, intersection_pt, ngb_travel_pt, ctad, ctad_flags

    # calculate the tlc and test the function using 3 methods
    # method 1: parameter curve fitting
    # method 2: directsolvingtlc
    # method 3: gridmap
    def cal_tlc(self, vehtraj_data, lanedata, whichmethod, isfigure):
        print("Calculating tlc...")
        lane2hit = lanedata
        stlc = []
        ctlc = []
        if isfigure == 1:
            plt.ion()
            fig = plt.figure(4)
            ax = fig.add_subplot(111)
        first_cost = 0
        last_cost = 0
        for ego_pt in vehtraj_data.ego_state:
            # start = time.time()
            # predict the trajectory of ego vehicle
            ego_trajpredict_straight = self.func_trajpredict(ego_pt, 0)
            ego_trajpredict_curve = self.func_trajpredict(ego_pt, ego_pt[4])
            # end1 = time.time()

            # find the nearest point to the ego vehicle
            dist_lane2ego = []
            for lane_pt in lane2hit:
                dist_lane2ego.append((ego_pt[0]-lane_pt[0])**2 + (ego_pt[1]-lane_pt[1])**2)
            nearest_idx = dist_lane2ego.index(min(dist_lane2ego))
            
            laneselect_idx = []
            count_idx = 0
            dist_laneforward = 0
            # forward 120m and select lane points
            while(dist_laneforward < 100):
                idx = nearest_idx + count_idx
                dist_laneforward += np.sqrt((lane2hit[idx+1][0]-lane2hit[idx][0])**2 + (lane2hit[idx+1][1]-lane2hit[idx][1])**2)
                count_idx += 1
                if lane2hit[idx][2] <= -np.pi/2:
                    break # over 90 degree, it means turn back
                else:
                    laneselect_idx.append(idx)
            
            laneselect = [lane2hit[idx] for idx in laneselect_idx]
            xlaneselect = [row[0] for row in laneselect]
            ylaneselect = [row[1] for row in laneselect]
            yawlaneselect = [row[2] for row in laneselect]
            func_lanexy_interp = CubicSpline(xlaneselect, ylaneselect)
            func_lanexyaw_interp = CubicSpline(xlaneselect, yawlaneselect)
            xlane_interp = np.linspace(xlaneselect[0], xlaneselect[-1], 1000)
            ylane_interp = func_lanexy_interp(xlane_interp)
            yawlane_interp = func_lanexyaw_interp(xlane_interp)
            lane_select_interp = [[xlane_interp[i], ylane_interp[i], yawlane_interp[i]] for i in range(len(xlane_interp))]
            
            # calculate the tlc between ego vehicle and lane boundary
            if whichmethod == 1:
                stlc_tmp, sinter_pt,fit_x_new,traj1_y_fit, fit2_x_new,traj2_y_fit= self.func_paramcurvefitting(ego_trajpredict_straight, lane_select_interp, 0)
                ctlc_tmp, cinter_pt,fit_x_new2,traj1_y_fit2, fit2_x_new2,traj2_y_fit2= self.func_paramcurvefitting(ego_trajpredict_curve, lane_select_interp, 0)
                # stlc_tmp, sinter_pt = self.func_paramcurvefitting(ego_trajpredict_straight, lane_select_interp, 0)
                # ctlc_tmp, cinter_pt= self.func_paramcurvefitting(ego_trajpredict_curve, lane_select_interp, 0)
            elif whichmethod == 2:
                stlc_tmp, sinter_pt = self.func_directsolvingtlc(ego_trajpredict_straight, lane_select_interp, 0)
                ctlc_tmp, cinter_pt = self.func_directsolvingtlc(ego_trajpredict_curve, lane_select_interp, 0)
            elif whichmethod == 3:
                stlc_tmp, sinter_pt = self.func_gridmap(ego_trajpredict_straight, lane_select_interp, 0)
                ctlc_tmp, cinter_pt = self.func_gridmap(ego_trajpredict_curve, lane_select_interp, 0)
            else:
                print("Please input correct method number!")
                return
            
            stlc.append(stlc_tmp)
            ctlc.append(ctlc_tmp)
            # end2 = time.time()
            # first_cost = first_cost+end1-start
            # last_cost = last_cost+end2-end1
            # plot the animation
            if isfigure == 1:
                ax.clear()
                # ax.plot([row[0] for row in lane2hit[nearest_idx:idx]], [row[1] for row in lane2hit[nearest_idx:idx]],
                #         linestyle = '-', color = 'blue', linewidth = 1.5, label='foward_path')

                # fit_x_new = np.array(fit_x_new)
                # traj1_y_fit = np.array(traj1_y_fit)
                # # traj2_y_fit = np.array(traj2_y_fit)
                # index = np.logical_and(traj1_y_fit > 0, traj1_y_fit < 250)
                # fit_x_new = fit_x_new[index]
                # traj1_y_fit = traj1_y_fit[index]
                # # traj2_y_fit = traj2_y_fit[index]

                # ax.plot(fit_x_new, traj1_y_fit,
                #         linestyle = '-', color = 'blue', linewidth = 1.5, label='ego_trajpredict_straight')
                # ax.plot(fit2_x_new2, traj2_y_fit2,
                # linestyle = '-', color = 'blue', linewidth = 1.5, label='lane_select_interp')
                # ax.plot(fit_x_new2, traj1_y_fit2,
                # linestyle = '-', color = 'blue', linewidth = 1.5, label='ego_trajpredict_curve')


                ax.plot([row[0] for row in lanedata], [row[1] for row in lanedata] , 
                        linestyle = '-.', color = 'black', linewidth = 0.5, label='laneboundary')
                ax.plot(ego_pt[0], ego_pt[1], 'ro', label='ego vehicle')
                ax.plot([row[0] for row in lane_select_interp], [row[1] for row in lane_select_interp], linestyle = '--', color = 'orange', linewidth = 1.5, label='laneboundary interp')
                ax.plot([row[0] for row in ego_trajpredict_straight], [row[1] for row in ego_trajpredict_straight], linestyle = '-.', linewidth = 0.5, color = 'green', label='ego trajectory predicted straight')
                ax.plot([row[0] for row in ego_trajpredict_curve], [row[1] for row in ego_trajpredict_curve], linestyle = '-.', linewidth = 0.5, color = 'blue', label='ego trajectory predicted curve')
                if (stlc_tmp > 0):
                    ax.plot(sinter_pt[0], sinter_pt[1], marker = '*', color = 'red', label='intersection point on straight trajectory')
                if (ctlc_tmp > 0):
                    ax.plot(cinter_pt[0], cinter_pt[1], marker = '*', color = 'red', label='intersection point on curve trajectory')
                # ax.plot(sinter_pt[0], sinter_pt[1], marker = '*', color = 'red', label='intersection point on straight trajectory')
                ax.text(400, -250, 'stlc = %.4fs' % stlc_tmp, ha='center', va='bottom', fontsize=10)
                ax.text(400, -250-5, 'ctlc = %.4fs' % ctlc_tmp, ha='center', va='bottom', fontsize=10)
                # ax.set_xlim(350, 600)
                # ax.set_ylim(-360, -180)

                ax.set_xlim([-200,200])
                ax.set_ylim([0,250])
                ax.set_title('Vehicle Trajectory Animation')
                ax.set_xlabel('X Axis (m)')
                ax.set_ylabel('Y Axis (m)')
                ax.axis('equal')
                ax.legend()
                fig.canvas.draw()
                fig.canvas.flush_events()
                time.sleep(0.0001)
        # print(f"first_cost={first_cost}, last_cost={last_cost}")
        if isfigure == 1:
            plt.ioff()
            plt.show()
        print("Calculating tlc Done!")
        return stlc, ctlc

    # calculate the tlc and test the function using 3 methods
    def test_func_intersection(self):
        self.showtimecomsuing = True
        ego_pt = self.vehtraj_data.ego_state[0]
        ngb_pt = self.vehtraj_data.ngb_state[0]
        # ego_pt = [468.077, -246.981, 0.04546, 11.8423, -0.03054]
        # ego_pt = [453.771, -248.642, 0.171717, 12.1635, -0.0153915]
        # ngb_pt = [472.824, -244.128, -0.0110827, 10.8843]
        ego_str_angle = ego_pt[4]
        ngb_str_angle = -1 * np.pi / 180 # 5 degree
        # ngb_str_angle = -0.03325
        ego_trajpredict = self.func_trajpredict(ego_pt, ego_str_angle)
        ngb_trajpredict = self.func_trajpredict(ngb_pt, ngb_str_angle)
        
        _, inter_pt, ngb_travel_pt, ctad, ctad_flag = self.func_ctad_paramcurvefitting(ego_trajpredict, ngb_trajpredict, 1)
        print("[method 1]: using func_ctad_paramcurvefitting ctad is: %0.6fs" % ctad)
        _, inter_pt, ngb_travel_pt, ctad, ctad_flag = self.func_ctad_directsolvingtlc(ego_trajpredict, ngb_trajpredict, 0)
        print("[method 2]: using func_ctad_directsolvingtlc ctad is: %0.6fs" % ctad)
        _, inter_pt, ngb_travel_pt, ctad, ctad_flag = self.func_ctad_gridmap(ego_trajpredict, ngb_trajpredict, 1)
        print("[method 3]: using func_ctad_gridmap ctad is: %0.6fs" % ctad)
        
        tlc1, _ = self.func_paramcurvefitting(ego_trajpredict, ngb_trajpredict, 0)
        print("[method 1]: using func_paramcurvefitting tlc is: %0.6fs" % tlc1)
        tlc2, _ = self.func_gridmap(ego_trajpredict, ngb_trajpredict, 0)
        print("[method 2]: using gridmap tlc is: %0.6fs" % tlc2)
        tlc3, _ = self.func_directsolvingtlc(ego_trajpredict, ngb_trajpredict, 0)
        print("[method 3]: using direct method tlc is: %0.6fs" % tlc3)
        self.showtimecomsuing = False

    # 计算多次样条拟合曲线，可用于环状曲线，即x先递增后递减，或相反
    def my_cubic_spline(self, trajdata):
        increasing = trajdata[0][0] < trajdata[1][0]
        last_x = trajdata[0][0]
        split_index = 0
        for i in range(1,len(trajdata)):
            if (trajdata[i][0]>last_x)!=increasing:
                split_index = i
                break
            last_x = trajdata[i][0]
        data1 = trajdata[:split_index]
        data2 = trajdata[split_index:]
        #CubicSpline拟合需要至少3个点
        if increasing==True:
            if len(data1)<3:
                cs1 = CubicSpline([row[0] for row in data2], [row[1] for row in data2])
                cs2 = None
            else:
                cs1 = CubicSpline([row[0] for row in data1], [row[1] for row in data1])
                cs2 = CubicSpline([row[0] for row in data2[::-1]], [row[1] for row in data2[::-1]])
        # x先递减后递增
        else:
            if len(data1)<3:
                cs1 = CubicSpline([row[0] for row in data2], [row[1] for row in data2])
                cs2 = None
            else:
                cs1 = CubicSpline([row[0] for row in data1[::-1]], [row[1] for row in data1[::-1]])
                cs2 = CubicSpline([row[0] for row in data2], [row[1] for row in data2])
        return cs1,cs2,split_index
    def find_intersection(self, trajdata1, trajdata2,DIST_THRESHOLD, speed):
        ygap_fit = [np.abs(trajdata1[1][idx] - trajdata2[1][idx]) for idx in range(len(trajdata1[0]))]
        dist1_fit = [np.sqrt((trajdata1[0][idx+1] - trajdata1[0][idx])**2 + (trajdata1[1][idx+1] - trajdata1[1][idx])**2) for idx in range(len(trajdata1)-1)]        
        min_idx = ygap_fit.index(min(ygap_fit))
        if (min(ygap_fit) < DIST_THRESHOLD):
            intersection_pt = [trajdata1[0][min_idx], trajdata1[1][min_idx]]
            # dlc = np.sum(dist1_fit[0:min_idx-1]) # sum distance from start to intersection point
            # tlc = dlc / speed
            tlc = trajdata1[4][min_idx]
            exist_inter = 1
        else:
            # if no intersection point, set tlc = -1
            intersection_pt = [-1, -1]
            exist_inter = 0
            tlc = -1
        return tlc, intersection_pt, exist_inter
    # ATTENTION: solve the intersection point of two trajectories, like ego vehicle and lane boundary
    # using parameter curve fitting method
    def func_paramcurvefitting(self, trajdata1, trajdata2, isfigure):
        func_starttime = time.time()
        speed = trajdata1[0][3]
        exist_inter = False
        DIST_THRESHOLD = 5e-2 # 5cm
        # fit new x and y
        x_min1 = min([row[0] for row in trajdata1])
        x_max1 = max([row[0] for row in trajdata1])
        x_min2 = min([row[0] for row in trajdata2])
        x_max2 = max([row[0] for row in trajdata2])
        x_min = min(x_min1, x_min2) 
        x_max = max(x_max1, x_max2)
        # # fit_x_new = np.linspace(x_min, x_max, 1000)

        # traj1_coeff = np.polyfit([row[0] for row in trajdata1], [row[1] for row in trajdata1], 5) # 5th order polynomial
        # traj2_coeff = np.polyfit([row[0] for row in trajdata2], [row[1] for row in trajdata2], 5) # 5th order polynomial
        # func_traj1 = np.poly1d(traj1_coeff)
        # func_traj2 = np.poly1d(traj2_coeff)

        #解决x不是单调问题
        traj_cs1, traj_cs2,traj_split_index= self.my_cubic_spline(trajdata1)
        road_cs1, road_cs2,road_split_index= self.my_cubic_spline(trajdata2)
        fit_x_new = np.linspace(x_min, x_max, 1000)
        
        #根据traj_split_index计算弧长
        dist1 = [np.sqrt((trajdata1[idx+1][0] - trajdata1[idx][0])**2 + (trajdata1[idx+1][1] - trajdata1[idx][1])**2) for idx in range(traj_split_index)] 
        dlc1 = np.sum(dist1)

        # 根据四种情况进行交点计算
        if road_cs2 is None:
            road_y_fit1 = road_cs1(fit_x_new)
            if traj_cs2 is None:
                traj1_y_fit = traj_cs1(fit_x_new)
                tlc, intersection_pt, exist_inter = self.find_intersection([fit_x_new, traj1_y_fit], [fit_x_new, road_y_fit1], DIST_THRESHOLD, speed)
            else:
                traj1_y_fit = traj_cs1(fit_x_new)
                traj2_y_fit = traj_cs2(fit_x_new)
                tlc1, intersection_pt1, exist_inter1 = self.find_intersection([fit_x_new, traj1_y_fit], [fit_x_new, road_y_fit1], DIST_THRESHOLD, speed)
                tlc2, intersection_pt2, exist_inter2 = self.find_intersection([fit_x_new, traj2_y_fit], [fit_x_new, road_y_fit1], DIST_THRESHOLD, speed)
                if exist_inter1:
                    tlc = tlc1
                    intersection_pt = intersection_pt1
                    exist_inter = exist_inter1
                else:
                    tlc = tlc2 +dlc1/speed
                    intersection_pt = intersection_pt2
                    exist_inter = exist_inter2
        else:
            road_y_fit1 = road_cs1(fit_x_new)
            road_y_fit2 = road_cs2(fit_x_new)
            if traj_cs2 is None:
                traj1_y_fit = traj_cs1(fit_x_new)
                tlc1, intersection_pt1, exist_inter1 = self.find_intersection([fit_x_new, traj1_y_fit], [fit_x_new, road_y_fit1], DIST_THRESHOLD, speed)
                tlc2, intersection_pt2, exist_inter2 = self.find_intersection([fit_x_new, traj1_y_fit], [fit_x_new, road_y_fit2], DIST_THRESHOLD, speed)
                if exist_inter1:
                    tlc = tlc1
                    intersection_pt = intersection_pt1
                    exist_inter = exist_inter1
                else:
                    tlc = tlc2
                    intersection_pt = intersection_pt2
                    exist_inter = exist_inter2
            else:
                traj1_y_fit = traj_cs1(fit_x_new)
                traj2_y_fit = traj_cs2(fit_x_new)
                tlc1, intersection_pt1, exist_inter1 = self.find_intersection([fit_x_new, traj1_y_fit], [fit_x_new, road_y_fit1], DIST_THRESHOLD, speed)
                tlc2, intersection_pt2, exist_inter2 = self.find_intersection([fit_x_new, traj1_y_fit], [fit_x_new, road_y_fit2], DIST_THRESHOLD, speed)
                if exist_inter1:
                    tlc = tlc1
                    intersection_pt = intersection_pt1
                    exist_inter = exist_inter1
                elif exist_inter2:
                    tlc = tlc2 +dlc1/speed
                    intersection_pt = intersection_pt2
                    exist_inter = exist_inter2
                else:
                    tlc3, intersection_pt3, exist_inter3 = self.find_intersection([fit_x_new, traj2_y_fit], [fit_x_new, road_y_fit1], DIST_THRESHOLD, speed)
                    tlc4, intersection_pt4, exist_inter4 = self.find_intersection([fit_x_new, traj2_y_fit], [fit_x_new, road_y_fit2], DIST_THRESHOLD, speed)
                    if exist_inter3:
                        tlc = tlc3 +dlc1/speed
                        intersection_pt = intersection_pt3
                        exist_inter = exist_inter3
                    else:
                        tlc = tlc4 +dlc1/speed
                        intersection_pt = intersection_pt4
                        exist_inter = exist_inter4
        
         
        if isfigure == 1:
            plt.figure(2)
            plt.plot([row[0] for row in trajdata1], [row[1] for row in trajdata1], linestyle = '--', linewidth = 0.5, color = 'blue', label='ego trajectory')
            plt.plot([row[0] for row in trajdata2], [row[1] for row in trajdata2], linestyle = '--', linewidth = 0.5, color = 'green', label='ngb trajectory')
            if exist_inter == True:
                plt.plot(intersection_pt[0], intersection_pt[1], 'r*', label='intersection point')
            plt.title('Using parametric curve fitting: Ego & Neighbor Vehicle Trajectory')
            plt.xlabel('X-axis (m)')
            plt.ylabel('Y-axis (m)')
            plt.legend()
            plt.axis('equal')
            plt.grid(True)
            plt.show()
            # time.sleep(0.01)
                    
        func_endtime = time.time()
        if self.showtimecomsuing == 1:
            print('func_paramcurvefitting time cost: %0.4fs' % (func_endtime - func_starttime))
        # if (tlc == -1):
        #     a = 0
        # 这个方法还有待完善不能直接使用find_intersection求tlc
        return tlc, intersection_pt ,fit_x_new,traj1_y_fit, fit_x_new,road_y_fit1
        # return tlc, intersection_pt 

    # ATTENTION: solve the intersection point of two trajectories, like ego vehicle and lane boundary
    # using direct distance solving method
    def func_directsolvingtlc(self, trajdata1, trajdata2, isfigure):
        func_starttime = time.time()
        DIST_THRESHOLD = 10 * 1e-2 # 10cm
        traj1 = np.array(trajdata1)[:, :2]
        traj2 = np.array(trajdata2)[:,:2]
        # 计算所有点对之间的距离，返回二维数组，每个元素是两个轨迹点之间的距离，行号代表轨迹1的点索引，列号代表轨迹2的点索引
        distances = np.sum((traj1[:, None, :] - traj2[None, :, :]) ** 2, axis=-1)
        # 找到距离最小的点对的索引,扁平索引，找到所有中最小的
        min_index_traj1, min_index_traj2 = np.unravel_index(np.argmin(distances), distances.shape)
        min_dist  = np.sqrt(distances[min_index_traj1, min_index_traj2])
        if min_dist < DIST_THRESHOLD:
            # 计算交点
            inter_pt = (traj1[min_index_traj1] + traj2[min_index_traj2]) / 2
            inter_pt = inter_pt.tolist()
            tlc = trajdata1[min_index_traj1][4] - trajdata1[0][4]
        else:
            inter_pt = [-1, -1]
            tlc = -1

        if isfigure == 1:
            plt.figure(2)
            # plt.scatter([row[0] for row in trajdata1], [row[1] for row in trajdata1], marker = '.', s = 0.5, color = 'red', label='ego trajectory')
            # plt.scatter([row[0] for row in trajdata2], [row[1] for row in trajdata2], marker = '.', s = 0.5, color = 'blue', label='ngb trajectory')
            plt.plot([row[0] for row in trajdata1], [row[1] for row in trajdata1], linestyle = '--', linewidth = 0.5, color = 'blue', label='ego trajectory')
            plt.plot([row[0] for row in trajdata2], [row[1] for row in trajdata2], linestyle = '--', linewidth = 0.5, color = 'green', label='ngb trajectory')
            plt.plot(inter_pt[0], inter_pt[1], 'r*', label='intersection point')
            plt.title('Using distance loop: Ego & Neighbor Vehicle Trajectory')
            plt.xlabel('X-axis (m)')
            plt.ylabel('Y-axis (m)')
            plt.legend()
            plt.axis('equal')
            plt.grid(True)
            plt.show()
        func_endtime = time.time()
        if self.showtimecomsuing == 1:
            print('func_directsolvingtlc time cost: %0.4fs' % (func_endtime - func_starttime))
        return tlc, inter_pt

    # ATTENTION: solve the intersection point of two trajectories, like ego vehicle and lane boundary
    # using grid map method
    def func_gridmap(self, trajdata1, trajdata2, isfigure):
        func_starttime = time.time()
        exist_inter = False
        GRID_SIZE_X = 70 # 70m
        GRID_SIZE_Y = 50 # 20m
        GRID_RESOLUTION = 0.01 # 0.1m
        traj_min_x1, traj_min_y1 = np.min([[row[0], row[1]] for row in trajdata1], axis=0)
        traj_max_x1, traj_max_y1 = np.max([[row[0], row[1]] for row in trajdata1], axis=0)
        traj_min_x2, traj_min_y2 = np.min([[row[0], row[1]] for row in trajdata2], axis=0)
        traj_max_x2, traj_max_y2 = np.max([[row[0], row[1]] for row in trajdata2], axis=0)
        traj_min_x, traj_min_y = np.min([[traj_min_x1, traj_min_y1], [traj_min_x2, traj_min_y2]], axis=0)
        traj_max_x, traj_max_y = np.max([[traj_max_x1, traj_max_y1], [traj_max_x2, traj_max_y2]], axis=0)
        # check if the trajectory is out of the grid map
        grid_bound_x = traj_max_x - traj_min_x
        grid_bound_y = traj_max_y - traj_min_y
        if (grid_bound_x > GRID_SIZE_X or grid_bound_y > GRID_SIZE_Y):
            print('The trajectory is out of the grid map!')
            GRID_SIZE_X = int(grid_bound_x) + 10
            GRID_SIZE_Y = int(grid_bound_y) + 10
            # return
        
        grid_traj1 = np.zeros((int(GRID_SIZE_X / GRID_RESOLUTION), int(GRID_SIZE_Y / GRID_RESOLUTION)))
        grid_traj2 = np.zeros((int(GRID_SIZE_X / GRID_RESOLUTION), int(GRID_SIZE_Y / GRID_RESOLUTION)))
        grid_traj3 = np.zeros((int(GRID_SIZE_X / GRID_RESOLUTION), int(GRID_SIZE_Y / GRID_RESOLUTION)))
        # track the time
        speed_traj1 = -1 * np.ones((int(GRID_SIZE_X / GRID_RESOLUTION), int(GRID_SIZE_Y / GRID_RESOLUTION)))
        time_traj1 = -1 * np.ones((int(GRID_SIZE_X / GRID_RESOLUTION), int(GRID_SIZE_Y / GRID_RESOLUTION)))
        
        # drop traj1 points to grid
        for pt in trajdata1:
            x0 = pt[0] - traj_min_x
            y0 = pt[1] - traj_min_y
            speed0 = pt[3] # speed
            t0 = pt[4] # time
            xgrid = round(x0 / GRID_RESOLUTION)
            ygrid = round(y0 / GRID_RESOLUTION)
            grid_traj1[xgrid, ygrid] = 1
            time_traj1[xgrid, ygrid] = t0
            speed_traj1[xgrid, ygrid] = speed0
            # error handling
            x_err = x0 - xgrid * GRID_RESOLUTION
            y_err = y0 - ygrid * GRID_RESOLUTION
        # line up dropped points
        for i in range(len(trajdata1) - 1):
            x0, y0 = trajdata1[i][0] - traj_min_x, trajdata1[i][1] - traj_min_y
            x1, y1 = trajdata1[i+1][0] - traj_min_x, trajdata1[i+1][1] - traj_min_y
            t0, t1 = trajdata1[i][4], trajdata1[i+1][4]
            speed0, speed1 = trajdata1[i][3], trajdata1[i+1][3]
            x_vals = np.linspace(x0 / GRID_RESOLUTION, x1 / GRID_RESOLUTION, 100)
            y_vals = np.linspace(y0 / GRID_RESOLUTION, y1 / GRID_RESOLUTION, 100)
            t_vals = np.linspace(t0, t1, 100)
            speed_vals = np.linspace(speed0, speed1, 100)
            grid_points = np.column_stack((x_vals, y_vals))
            grid_points = np.round(grid_points).astype(int)
            grid_traj1[grid_points[:, 0], grid_points[:, 1]] = 1
            time_traj1[grid_points[:, 0], grid_points[:, 1]] = t_vals
            speed_traj1[grid_points[:, 0], grid_points[:, 1]] = speed_vals
            
        # drop traj2 points to grid
        for pt in trajdata2:
            x0 = pt[0] - traj_min_x
            y0 = pt[1] - traj_min_y
            xgrid = round(x0 / GRID_RESOLUTION)
            ygrid = round(y0 / GRID_RESOLUTION)
            grid_traj2[xgrid, ygrid] = 1
        # line up dropped points
        for i in range(len(trajdata2) - 1):
            x0, y0 = trajdata2[i][0] - traj_min_x, trajdata2[i][1] - traj_min_y
            x1, y1 = trajdata2[i+1][0] - traj_min_x, trajdata2[i+1][1] - traj_min_y
            x_vals = np.linspace(x0 / GRID_RESOLUTION, x1 / GRID_RESOLUTION, 100)
            y_vals = np.linspace(y0 / GRID_RESOLUTION, y1 / GRID_RESOLUTION, 100)
            grid_points = np.column_stack((x_vals, y_vals))
            grid_points = np.round(grid_points).astype(int)
            grid_traj2[grid_points[:, 0], grid_points[:, 1]] = 1

        # combine two grids
        grid_traj3 = grid_traj1 + grid_traj2
        intersection_couter = np.sum(grid_traj3 == 2)
        if intersection_couter == 0:
            inter_idx = np.argwhere(grid_traj3 == 1)
            for idx in inter_idx:
                if idx[0] == 0 or idx[1] == 0 or idx[0] == grid_traj3.shape[0]-1 or idx[1] == grid_traj3.shape[1]-1:
                    continue
                inter_ul = grid_traj3[idx[0]-1, idx[1]-1]
                inter_up = grid_traj3[idx[0]-1, idx[1]]
                inter_ur = grid_traj3[idx[0]-1, idx[1]+1]
                inter_left = grid_traj3[idx[0], idx[1]-1]
                inter_right = grid_traj3[idx[0], idx[1]+1]
                inter_dl = grid_traj3[idx[0]+1, idx[1]-1]
                inter_down = grid_traj3[idx[0]+1, idx[1]]
                inter_dr = grid_traj3[idx[0]+1, idx[1]+1]
                inter_ngb_count = inter_ul + inter_up + inter_ur + inter_left + inter_right + inter_dl + inter_down + inter_dr
                if inter_ngb_count >= 6:
                    grid_traj3[idx[0], idx[1]] = 2
        
        if np.sum(grid_traj3 == 2) > 0:
            intersection_points = np.where(grid_traj3 == 2)
            intersection_points_x = intersection_points[0] * GRID_RESOLUTION + traj_min_x
            intersection_points_y = intersection_points[1] * GRID_RESOLUTION + traj_min_y
            intersection_pt_x = (intersection_points_x[0] + intersection_points_x[-1]) / 2
            intersection_pt_y = (intersection_points_y[0] + intersection_points_y[-1]) / 2
            intersection_pt = [intersection_pt_x, intersection_pt_y]
            # intersection_pt_x = np.median(intersection_points_x)
            # intersection_pt_y = np.median(intersection_points_y)
            
            # calculate the dlc and tlc
            intersection_idx = np.argwhere(grid_traj3 == 2)
            intersection_first_idx = intersection_idx[0]
            intersection_first_time = time_traj1[intersection_first_idx[0], intersection_first_idx[1]]
            dist_first2real = np.sqrt((intersection_points_x[0] - intersection_pt_x)**2 + (intersection_points_y[0] - intersection_pt_y)**2)
            time_first2real = dist_first2real / speed_traj1[intersection_first_idx[0], intersection_first_idx[1]]
            tlc = intersection_first_time + time_first2real
            exist_inter = True
        else:
            intersection_pt = [-1, -1]
            tlc = -1.0
            exist_inter = False
        
        if (isfigure==1):
            grid3_plot = np.flipud(grid_traj3.T)
            plt.figure(1)
            plt.imshow(grid3_plot, cmap='binary', interpolation='nearest')
            # plt.colorbar()
            plt.title('Grid Chessboard')
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.grid(False)
            plt.show()   
            
            plt.figure(2)
            plt.plot([row[0] for row in trajdata1], [row[1] for row in trajdata1], linestyle = '--', linewidth = 0.5, color = 'blue', label='ego trajectory')
            plt.plot([row[0] for row in trajdata2], [row[1] for row in trajdata2], linestyle = '--', linewidth = 0.5, color = 'green', label='ngb trajectory')
            if exist_inter == True:
                plt.plot(intersection_pt_x, intersection_pt_y, 'r*', label='intersection point')
            plt.title('Using gridmap: Ego & Neighbor Vehicle Trajectory')
            plt.xlabel('X-axis (m)')
            plt.ylabel('Y-axis (m)')
            plt.legend()
            plt.axis('equal')
            plt.grid(True)
            plt.show()     
        func_endtime = time.time()
        if self.showtimecomsuing == 1:
            print('func_gridmap time cost: %0.4fs' % (func_endtime - func_starttime))
        return tlc, intersection_pt

    def get_indicator(self, whichmethod):
        if whichmethod == 1:
            print("Using method 1: parameter curve fitting")
        elif whichmethod == 2:
            print("Using method 2: directsolvingtlc")
        elif whichmethod == 3:
            print("Using method 3: gridmap")
        else:
            print("Please input correct method number!!!")
            return
        
        ttc, ttc_flags = self.func_cal_ttc(self.drivertraj_data, 0)
        lateral_offset = self.func_cal_lateraloffset(self.vehtraj_data, self.lane_data.rightcenterline_data.lane, 0)
        ctad, ctad_flag = self.cal_ctad(self.vehtraj_data, whichmethod, 0)
        # left boundary
        left_stlc, left_ctlc = self.cal_tlc(self.vehtraj_data, self.lane_data.leftboundary_data.lane, whichmethod, 0)
        right_stlc, right_ctlc = self.cal_tlc(self.vehtraj_data, self.lane_data.rightboundary_data.lane, whichmethod, 0)
        stlc = max(left_stlc, right_stlc)
        ctlc = max(left_ctlc, right_ctlc)
        return ttc, ttc_flags, lateral_offset, ctad, ctad_flag, stlc, ctlc
