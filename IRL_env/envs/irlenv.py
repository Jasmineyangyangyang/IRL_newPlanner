import numpy as np
import time
from planner.global_road import natural_road_load #load global road csv file
from IRL_env.envs.planner.polyplan_States_cost_irl import Polyplanner  # for target_v, target_d planner
# from data.polyplan import PolyPlanner  # for K_J, K_D planner
from risk.risk_evaluation import risk_ind_cal
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')
import matplotlib.ticker as ticker
import matplotlib
# matplotlib.use('Agg') # change from 'TkAgg', which means interactive mode on
# matplotlib.use('TkAgg')

class RewardEnv():
    """
    A IRL driving environment with expert data.
    """
    def __init__(self, lane_id):
        self.gamma = 0.95
        self.lane_id = lane_id
        self.road_width = 3.75       # m
        self.vehicle_width = 2.08 # m  from https://www.tesla.com/ownersmanual/model3/zh_cn/GUID-56562137-FC31-4110-A13C-9A9FC6657BF0.html
        self.vehicle_wheelbase = 2.6 # m from https://www.tesla.com/ownersmanual/model3/zh_cn/GUID-56562137-FC31-4110-A13C-9A9FC6657BF0.html
        self.sample_time = 0.1  # frequency of the coordinate in csv file
        self.plan_time = 5  # seconds
        self.target_speed = 40/3.6
        self.planner = Polyplanner(natural_road_load(lane_id=self.lane_id), self.lane_id)
        self.trajectory_data_process = natural_road_load(lane_id=lane_id)
        self.ego_trajectory, self.road, self.obs_trajectory = self.trajectory_data_process.build_trajectory()  # cartesian coordinates
        self.speed_limit = 60/3.6   # m/s: set vehicle's speed limit by road information
        self.duration = len(self.ego_trajectory) - 3

        print('Calculating ego lane center...')
        self.road_center = []   # this is the Cartesian center of the lane which ego vehicle is driving.
        self.road_left = self.road[:,0:2]     # [x, y]
        self.road_right = self.road[:,2:4]    # [x, y]

        for i in range(len(self.road)):
            if lane_id == 0:     # outside
                center_left_x = self.road[i][0]
                center_left_y = self.road[i][1]
                center_right_x = self.road[i][4]
                center_right_y = self.road[i][5]
                
            elif lane_id == 1:   # inside
                center_left_x = self.road[i][4]
                center_left_y = self.road[i][5]
                center_right_x = self.road[i][2]
                center_right_y = self.road[i][3]

            road_center_x = (center_left_x + center_right_x) / 2.0
            road_center_y = (center_left_y + center_right_y) / 2.0
            self.road_center.append([road_center_x, road_center_y])
        self.road_center = np.array(self.road_center)  # Cartesian coordinate
        print('lane center has been stored in self.road_center [center_x, center_y]')
        print('----------------------------------------------------------------')
        print('Calculate Frenet corordinate of the road center...')
        self.road_center_frenet = np.zeros((self.road_center.shape[0], 3))  # longitudinal_distance, lateral_offset=0, yaw (rad)
        diff_x = np.diff(self.road_center[:,0])
        diff_y = np.diff(self.road_center[:,1])
        road_s_temp = np.cumsum(np.sqrt(np.square(diff_x) + np.square(diff_y)))  # remove the last one
        road_s = np.insert(road_s_temp, 0, 0.0)  # the start point s=0
        self.road_center_frenet[:,0] = road_s

        self.road_yaw = np.zeros(diff_x.shape[0]+1)
        for i in range(diff_x.shape[0]-1):
            self.road_yaw[i] = np.arctan2(diff_y[i], diff_x[i])    # rad
        self.road_yaw[-1] = self.road_yaw[-2]
        self.road_center_frenet[:,2] = self.road_yaw
        print('lane center has been stored in self.road_center_frenet [longitudinal, offset=0, yaw]')    

    def process_raw_trajectory(self, raw_trajectory):
        """convert from Cartesian coordinate to frenet coordinate

        Args:
            raw_trajectory (list): each line format: [expert_x, expert_y, self.expert_speed, self.lane_id]

        Returns:
            trajectory (list): each line format: [expert_longitudinal_dis, expert_lateral_offset, self.expert_speed]
        """
        raw_traj = np.array(raw_trajectory)
        # frenet_trajectory = np.zeros((raw_traj.shape[0], raw_traj.shape[1]-1)) # remove lane_id column
        expert_trajectory = np.zeros((raw_traj.shape[0], raw_traj.shape[1])) # remove lane_id column add accelerate 0.0
        for i in range(raw_traj.shape[0]):
            x = raw_traj[i][0]
            y = raw_traj[i][1]
            yaw = raw_traj[i][2]
            speed = raw_traj[i][3]    # m/s
            a = float(0.0)
            curvature = raw_traj[i][4] * -1 # turn right is negative
            # longitudinal_dis, lateral_offset = self.cartesian_to_frenet(x, y)
            # frenet_trajectory[i] = [longitudinal_dis, lateral_offset, speed]
            expert_trajectory[i] = [x, y, yaw, speed, a, curvature]
        expert_trajectory[:,2] = np.unwrap(expert_trajectory[:,2])
        return expert_trajectory    # cartesian trajcetory, array: [x, y, speed(m/s)]
    
    def cartesian_to_frenet(self, x, y):
        """convert from Cartesian coordinate to frenet coordinate

        Args:
            cartesian coordinate (float, float): x,y

        Returns:
            frenet coordinate (float, float): each line format: s=longitudinal_position, l=lateral_offset
        """
        # find closest point
        distances = np.sqrt((self.road_center[:,0]-x)**2 + (self.road_center[:,1]-y)**2)
        closest_index = np.argmin(distances)

        s_to_point = np.array([x - self.road_center[closest_index][0], y - self.road_center[closest_index][1]])
        s_direction = np.array([self.road_center[closest_index + 1][0] - self.road_center[closest_index][0],\
                                self.road_center[closest_index + 1][1] - self.road_center[closest_index][1]])
        d_direction = np.cross(s_direction, s_to_point)  # left = positive; right = negative
        if np.dot(s_direction, s_to_point) == 0:
            longitudinal_dis = self.road_center_frenet[closest_index][0]
            lateral_offset = distances[closest_index] * d_direction
            return longitudinal_dis, lateral_offset
        elif np.dot(s_direction, s_to_point) < 0:
            if closest_index == 0:
                longitudinal_dis = self.road_center_frenet[closest_index][0]
                lateral_offset = distances[closest_index] * np.sign(d_direction)
                return longitudinal_dis, lateral_offset
            else:
                near_close_index = closest_index - 1
                s1 = np.array([self.road_center[near_close_index][0], self.road_center[near_close_index][1]])
                s2 = np.array([self.road_center[closest_index][0], self.road_center[closest_index][1]])
                longitudinal_d0 = self.road_center_frenet[near_close_index][0]
        elif np.dot(s_direction, s_to_point) > 0:
            near_close_index = closest_index + 1
            s1 = np.array([self.road_center[closest_index][0], self.road_center[closest_index][1]])
            s2 = np.array([self.road_center[near_close_index][0], self.road_center[near_close_index][1]])
            longitudinal_d0 = self.road_center_frenet[closest_index][0]
        # Use the linear interpolation point(parameter t) between s1 and s2 to approximate the point on the road centerline, 
        # first find t and then calculate the coordinates, if you want to be accurate, you can refer to the link
        # https://zhuanlan.zhihu.com/p/514864431
        s_vector = s2 - s1  # [x2-x1, y2-y1]
        point_vector = np.array([x, y]) - s1
        t = np.dot(s_vector, point_vector) / (s_vector[0]**2 + s_vector[1]**2)  
        close_point = s1 + t*s_vector

        longitudinal_dis = longitudinal_d0 + t * np.sqrt(s_vector[0]**2 + s_vector[1]**2)
        lateral_offset = np.sqrt((x - close_point[0])**2 + (y - close_point[1])**2)*d_direction

        return longitudinal_dis, lateral_offset
    
    def frenet_to_cartesian(self, s, d):
        """Frenet to Cartesian

        Args:
            s (float): frenet s
            d (float): frenet d
        """
        # find closest point
        distances = np.sqrt((self.road_center_frenet[:,0]-s)**2 + (self.road_center_frenet[:,1]-d)**2)
        closest_index = np.argmin(distances)
        x0, y0 = self.road_center[closest_index]
        delta_s = s - self.road_center_frenet[closest_index][0]
        global_yaw = self.road_center_frenet[closest_index][2]

        x = x0 + delta_s * np.cos(global_yaw) - d * np.sin(global_yaw)
        y = y0 + delta_s * np.sin(global_yaw) + d * np.cos(global_yaw)
        return [x, y]

    def reset(self, reset_time=0):
        """reset the environment at a given time

        Args:
            reset_time (int, optional): trajectory index. Defaults to 1.
        """
        self._create_vehicles(reset_time)
        self.steps = 0
        self.time = 0
        self.reset_time = reset_time

    def _create_vehicles(self, reset_time):

        """create expert trajectories for ego vehicle 

        Args:
            reset_time (int): add segment[reset_time:] of trajectory for vehicle
        """
        self.expert_trajectory= self.process_raw_trajectory(self.ego_trajectory)[reset_time:]  # cartesian [x, y, yaw, speed(m/s), acceleration, curvature]
        
        # 创建邻车轨迹 numpy.array，不然怎么计算feature
        self.obstacle_trajectory = self.process_raw_trajectory(self.obs_trajectory)[reset_time:]  #ahead 30m

 
    def step(self, action=None):
        """Perform a MDP step action

        Args:
            action (tuple, optional): (target_lateral, target_speed, horizon seconds between start point to end point/5s). Defaults to None.
        """
        if self.expert_trajectory is None:
            raise NotImplementedError("The expert trajectory must be initialized in the environment implementation")

        self.features = self._simulate(action)

        obs = action
        terminal = self._is_terminal() 
        info = {
            # "velocity": self.expert_speed ,
            "action": action,
            "time": self.reset_time
        }
        return obs, self.features, terminal, info

    def _simulate(self, action):
        """Perform several steps of simulation with the planned trajectory

        Args:
            action (tuple, optional): (target_lateral, target_speed, horizon seconds between start point to end point/5s). Defaults to None.
        """
        trajectory_features = []

        # SIM_LOOP = 1700 # sample time = 0.01
        SIM_LOOP = 170 # sample time = 0.1
        if action is not None: # action contains sample goal
            for i in range(SIM_LOOP):  
                if i == 0:  # generate trajectory in first time in T
                    ego_x = self.expert_trajectory[0][0]
                    ego_y = self.expert_trajectory[0][1]
                    ego_yaw = self.expert_trajectory[0][2]
                    ego_speed = self.expert_trajectory[0][3]
                    ego_acc = self.expert_trajectory[0][4]
                    ego_curvature = self.expert_trajectory[0][5]
                    self.feature_trajectory = [[ego_x, ego_y, ego_yaw, ego_speed, ego_acc, ego_curvature]]
                    s, s_dot, s_ddot, l, l_dot, l_ddot = self.planner.calculate_frenet_coordinates(ego_x, ego_y, ego_yaw, ego_speed, ego_acc, ego_curvature)
                    self.feature_trajectory_frenet = [[s, s_dot, s_ddot, 0.0, l, l_dot, l_ddot, 0.0]]
                else:
                    path = self.planner.poly_trajectory(ego_x, ego_y, ego_speed, [action[0], action[1]], self.target_speed,
                                                        [], ego_yaw, ego_acc, ego_curvature)
                    ego_x = path.x[1]
                    ego_y = path.y[1]
                    ego_yaw = path.yaw[1]
                    ego_speed = path.speed[1]
                    ego_acc = path.a[1]
                    ego_curvature = path.c[1]
                    self.feature_trajectory.append([ego_x, ego_y, ego_yaw, ego_speed, ego_acc, ego_curvature])
                    self.feature_trajectory_frenet.append([path.s[1], path.s_dot[1], path.s_ddot[1], path.s_dddot[1],path.l[1], path.l_dot[1], path.l_ddot[1], path.l_dddot[1]])
                if self._is_terminal():
                    print("bend over! loop = ", i)
                    break
            self.feature_trajectory = np.array(self.feature_trajectory)
            self.feature_trajectory_frenet = np.array(self.feature_trajectory_frenet)
        else:  # expert trajectory
            self.feature_trajectory_temp = self.expert_trajectory
            self.feature_trajectory_frenet = np.zeros((self.feature_trajectory_temp.shape[0], 8))
            for i in range(self.feature_trajectory_temp.shape[0]):
                s, s_dot, s_ddot, l, l_dot, l_ddot = self.planner.calculate_frenet_coordinates(self.feature_trajectory_temp[i][0], self.feature_trajectory_temp[i][1], self.feature_trajectory_temp[i][2],
                                                                                               self.feature_trajectory_temp[i][3], self.feature_trajectory_temp[i][4], self.feature_trajectory_temp[i][5])
                self.feature_trajectory_frenet[i] = [s, s_dot, s_ddot, 0.0, l, l_dot, l_ddot, 0.0]
            self.feature_trajectory = self.feature_trajectory_temp  # [x, y, speed]

        features = self._features()  # calculate features need target_speed, this is features of one point on the trajecotory
        trajectory_features = features
        '''
        # -------for paper figure show---------#
        predict_action_0 = (0.0, 45/3.6, 5)
        predict_action_1 = (0.48, 65/3.6, 5)
        predict_action_2 = (-0.48, 60/3.6, 5)
        self.feature_trajectory_temp_0 = self.planner.trajectory_planner(predict_action_0[0], predict_action_0[1], predict_action_0[2])  #frenet coordinate based on self.road_center
        self.feature_trajectory_frenet_0 = np.zeros((len(self.feature_trajectory_temp_0[0].d), 8)) # [s, s_dot, s_ddot, s_ddd, l, l_dot, l_ddot, l_ddd]
        self.feature_trajectory_0 = np.zeros((len(self.feature_trajectory_temp_0[0].d), 3))        # [x, y, speed]
        for i in range(len(self.feature_trajectory_temp_0[0].d)):
            self.feature_trajectory_frenet_0[i] = [self.feature_trajectory_temp_0[0].s[i], self.feature_trajectory_temp_0[0].d[i]]
            self.feature_trajectory_0[i, 0:2] = self.frenet_to_cartesian(self.feature_trajectory_temp_0[0].s[i], self.feature_trajectory_temp_0[0].d[i])
        if (np.diff(self.feature_trajectory_0[:,0])**2 + np.diff(self.feature_trajectory_0[:,1])**2 >= 0).all():
            self.feature_trajectory_0[:-1, 2] = np.sqrt(np.diff(self.feature_trajectory_0[:,0])**2 + np.diff(self.feature_trajectory_0[:,1])**2) / 0.02
        else:
            print(np.diff(self.feature_trajectory_0[:,0])**2 + np.diff(self.feature_trajectory_0[:,1])**2)
        self.feature_trajectory_0[-1, 2] = self.feature_trajectory_0[-2, 2]
        #----------------#
        self.feature_trajectory_temp_1 = self.planner.trajectory_planner(predict_action_1[0], predict_action_1[1], predict_action_1[2])  #frenet coordinate based on self.road_center
        self.feature_trajectory_frenet_1 = np.zeros((len(self.feature_trajectory_temp_1[0].d), 2)) # [s, d]
        self.feature_trajectory_1 = np.zeros((len(self.feature_trajectory_temp_1[0].d), 3))        # [x, y, speed]
        for i in range(len(self.feature_trajectory_temp_1[0].d)):
            self.feature_trajectory_frenet_1[i] = [self.feature_trajectory_temp_1[0].s[i], self.feature_trajectory_temp_1[0].d[i]]
            self.feature_trajectory_1[i, 0:2] = self.frenet_to_cartesian(self.feature_trajectory_temp_1[0].s[i], self.feature_trajectory_temp_1[0].d[i])
        if (np.diff(self.feature_trajectory_1[:,0])**2 + np.diff(self.feature_trajectory_1[:,1])**2 >= 0).all():
            self.feature_trajectory_1[:-1, 2] = np.sqrt(np.diff(self.feature_trajectory_1[:,0])**2 + np.diff(self.feature_trajectory_1[:,1])**2) / 0.02
        else:
            print(np.diff(self.feature_trajectory_1[:,0])**2 + np.diff(self.feature_trajectory_1[:,1])**2)
        self.feature_trajectory_1[-1, 2] = self.feature_trajectory_1[-2, 2]
        #----------------#
        self.feature_trajectory_temp_2 = self.planner.trajectory_planner(predict_action_2[0], predict_action_2[1], predict_action_2[2])  #frenet coordinate based on self.road_center
        self.feature_trajectory_frenet_2 = np.zeros((len(self.feature_trajectory_temp_2[0].d), 2)) # [s, d]
        self.feature_trajectory_2 = np.zeros((len(self.feature_trajectory_temp_2[0].d), 3))        # [x, y, speed]
        for i in range(len(self.feature_trajectory_temp_2[0].d)):
            self.feature_trajectory_frenet_2[i] = [self.feature_trajectory_temp_2[0].s[i], self.feature_trajectory_temp_2[0].d[i]]
            self.feature_trajectory_2[i, 0:2] = self.frenet_to_cartesian(self.feature_trajectory_temp_2[0].s[i], self.feature_trajectory_temp_2[0].d[i])
        if (np.diff(self.feature_trajectory_2[:,0])**2 + np.diff(self.feature_trajectory_2[:,1])**2 >= 0).all():
            self.feature_trajectory_2[:-1, 2] = np.sqrt(np.diff(self.feature_trajectory_2[:,0])**2 + np.diff(self.feature_trajectory_2[:,1])**2) / 0.02
        else:
            print(np.diff(self.feature_trajectory_2[:,0])**2 + np.diff(self.feature_trajectory_2[:,1])**2)
        self.feature_trajectory_2[-1, 2] = self.feature_trajectory_2[-2, 2]
        #----------------#
        plt.figure()     # figsize=(10,6)
        ax = plt.axes()
        # ax.set_facecolor("mintcream")  #  darkgrey
        ax.tick_params(axis='y',
                 labelsize=8, # y轴字体大小设置
                #  color='r',    # y轴标签颜色设置  
                #  labelcolor='b', # y轴字体颜色设置
                 direction='in' # y轴标签方向设置
                  ) 
        ax.tick_params(axis='x',
                 labelsize=8, # y轴字体大小设置
                 direction='in' # y轴标签方向设置
                  ) 
        st = int(self.feature_trajectory_frenet[0, 0])-5
        plt.plot(np.linspace(st, st+65/3.6*5, 100), np.ones(100)*2.0, 'k--', linewidth=1.5)
        plt.plot(np.linspace(st, st+65/3.6*5, 100), np.zeros(100), '--', color='grey', linewidth=1.5)
        plt.plot(np.linspace(st, st+65/3.6*5, 100), np.ones(100)*-2.0, 'k', linewidth=1.5)
        # adjacent lane
        plt.plot(np.linspace(st, st+65/3.6*5, 100), np.ones(100)*4.0, '--', color='grey', linewidth=1.5)
        plt.plot(np.linspace(st, st+65/3.6*5, 100), np.ones(100)*6.0, 'k', linewidth=1.5)
        # trajectory
        plt.plot(self.feature_trajectory_frenet_0[:, 0], self.feature_trajectory_frenet_0[:, 1], color='indianred', linewidth=2, label='Probobility: 27.44%')
        plt.plot(self.feature_trajectory_frenet_1[:, 0], self.feature_trajectory_frenet_1[:, 1], color='orange', linewidth=2, label='Probobility: 16.62%')
        tmp = self.feature_trajectory_frenet_0[-1, 0]
        plt.plot(np.linspace(tmp, tmp+9/3.6*5, 100), np.zeros(100), color='indianred', linewidth=2)
        plt.plot(self.feature_trajectory_frenet_2[:, 0], self.feature_trajectory_frenet_2[:, 1], color='peru', linewidth=2, label='Probobility: 7.03%')

        # plt.plot(np.linspace(st, st+65/3.6*5, 100), np.ones(100)*0.96, '--', color='mediumseagreen',linewidth=1.5, label='safe boundary')
        # plt.plot(np.linspace(st, st+65/3.6*5, 100), np.ones(100)*-0.96, '--', color='mediumseagreen', linewidth=1.5)
        plt.plot(np.linspace(st+30, st+65/3.6*5, 100), np.ones(100)*4.0, color='slateblue', linewidth=1.7)  # label='obstacle trajectory'
        obs_veh = plt.Rectangle(xy=(st+21, 3.74), width=self.vehicle_length*2, height=self.vehicle_width*0.25, angle=0.0, color='slateblue', zorder=2) # xy: 左下角位置，width, height：长，宽，angle：逆时针旋转角度，color：设置颜色
        ego_veh = plt.Rectangle(xy=(self.feature_trajectory_frenet_0[0, 0]-5, -0.26), width=self.vehicle_length*2, height=self.vehicle_width*0.25, angle=0.0, color='indianred', zorder=2)
        ax.add_patch(obs_veh)
        ax.add_patch(ego_veh)
        plt.xlabel('Frenet S (m)', fontdict={'size':8})
        plt.ylabel('lateral offset (m)', fontdict={'size':8})
        plt.legend(loc=1, bbox_to_anchor=(0.95, 0.95), prop={'size': 8})
        # plt.savefig(f'./Images/predict_trajectory_594.tiff', dpi=300)     

        plt.figure()
        ax = plt.axes()
        # ax.set_facecolor("mintcream")  #  darkgrey
        ax.tick_params(axis='y',
                 labelsize=8, # y轴字体大小设置
                 direction='in' # y轴标签方向设置
                  ) 
        ax.tick_params(axis='x',
                 labelsize=8, # y轴字体大小设置
                 direction='in' # y轴标签方向设置
                  ) 
        plt.plot(self.feature_trajectory_frenet_0[:,0],self.feature_trajectory_0[:,2]*3.6, color='indianred', linewidth=2, label='Probability: 37.51%') 
        plt.plot(self.feature_trajectory_frenet_0[:, 0], self.feature_trajectory_1[:, 2]*3.6, color='orange', linewidth=2, label='Probobility: 8.51%')
        plt.plot(self.feature_trajectory_frenet_0[:, 0], self.feature_trajectory_2[:, 2]*3.6, color='peru', linewidth=2, label='Probobility: 8.27%')
        plt.ylim([39, 69])
        plt.xlabel('Frenet S (m)', fontdict={'size':8})
        plt.ylabel('Speed (m/s)', fontdict={'size':8})
        plt.legend(loc=1, prop={'size': 8})  # , bbox_to_anchor=(0.95, 0.95)
        plt.savefig(f'./Images/predict_trajectory_speed_594.tiff', dpi=300)

        plt.figure()
        ax = plt.axes()
        # ax.set_facecolor("mintcream")  #  darkgrey
        ax.tick_params(axis='y',
                 labelsize=8, # y轴字体大小设置
                 direction='in' # y轴标签方向设置
                  ) 
        ax.tick_params(axis='x',
                 labelsize=8, # y轴字体大小设置
                 direction='in' # y轴标签方向设置
                  ) 
        plt.plot(self.road[1250:, 0], self.road[1250:, 1], 'k', linewidth = 1.5)
        plt.plot(self.road[1250:, 2], self.road[1250:, 3], 'k', linewidth = 1.5)
        plt.plot(self.road[1250:, 4], self.road[1250:, 5], 'k--', linewidth = 1.5) 
        plt.plot(self.feature_trajectory_0[:,0],self.feature_trajectory_0[:,1], color='indianred', linewidth=2, label='Probability: 37.51%') 
        plt.plot(self.feature_trajectory_0[:, 0], self.feature_trajectory_1[:,1], color='orange', linewidth=2, label='Probobility: 8.51%')
        plt.plot(self.feature_trajectory_0[:, 0], self.feature_trajectory_2[:,1], color='peru', linewidth=2, label='Probobility: 8.27%')
        plt.ylim([125, 175])
        plt.axis('equal')
        plt.xlabel('Global X (m)', fontdict={'size':8})
        plt.ylabel('Global Y (m)', fontdict={'size':8})
        plt.legend(loc=1, prop={'size': 8})  # , bbox_to_anchor=(0.95, 0.95)
        plt.savefig(f'./Images/predict_trajectory_cartesian_594.tiff', dpi=300)
        # -------for paper figure show---------#
        '''
        return trajectory_features
    
    def _features(self):
        """
        Hand-crafted features
        :return:  the defined features of trajectory
        """
        # self.feature_trajectory_frenet: [s, s_dot, s_ddot, s_dddot, l, l_dot, l_ddot, l_dddot]
        ego_longitudinal_speeds = self.feature_trajectory_frenet[:,1]  # get speeds from trajectory planning
        ego_longitudinal_accs = self.feature_trajectory_frenet[:,2]    # get longitudinal acceleration from trajectory planning
        ego_longitudinal_jerks = self.feature_trajectory_frenet[:,3]    # get longitudinal jerk from trajectory planning
 
        ego_lateral_speeds = self.feature_trajectory_frenet[:,5]  # get lateral speeds from trajectory planning
        ego_lateral_accs = self.feature_trajectory_frenet[:,6]    # get lateral acceleration from trajectory planning
        ego_lateral_jerks = self.feature_trajectory_frenet[:,7]    # get lateral jerk from trajectory planning


        N_sample = self.feature_trajectory_frenet.shape[0]
        # travel efficiency   
        ego_efficiency = np.mean(np.abs(ego_longitudinal_speeds-self.speed_limit)) 
        MAX_SPEED_DIFF = abs(38/3.6 - 62/3.6)
        norm_ego_efficiency = ego_efficiency / MAX_SPEED_DIFF  # normalize by the maximum possible efficiency

        # comfort
        MAX_JERK = 2.5  # maximum jerk error[m/sss]
        MAX_ACC = 2.5  # maximum acceleration error [m/ss]
        # 1. 计算加速度的均方根 (RMS Acceleration), 归一化的积分项 (即归一化 RMS 的平方)
        # 物理意义：反映车辆在弯道中持续的侧倾感强度。值越大，说明过弯速度过快或转角过大。
        ego_lateral_acc_total = np.mean(np.square(ego_lateral_accs))
        ego_longitudinal_acc_total = np.mean(np.square(ego_longitudinal_accs))
        norm_ego_lateral_acc_total = ego_lateral_acc_total / (MAX_ACC ** 2)  # normalize by the maximum possible lateral acceleration
        norm_ego_longitudinal_acc_total = ego_longitudinal_acc_total / (MAX_ACC ** 2)  # normalize by the maximum possible longitudinal acceleration

        # 2. 计算急动度的均方根 (RMS Jerk) —— 【核心舒适度指标】
        # 物理意义：反映方向盘打得猛不猛，车辆侧向晃动和瞬态冲击感。值越大越容易晕车。
        ego_lateral_jerk_total = np.mean(np.square(ego_lateral_jerks))
        ego_longitudinal_jerk_total = np.mean(np.square(ego_longitudinal_jerks)) 
        norm_ego_lateral_jerk_total = ego_lateral_jerk_total / (MAX_JERK ** 2)  # normalize by the maximum possible lateral jerk
        norm_ego_longitudinal_jerk_total = ego_longitudinal_jerk_total / (MAX_JERK ** 2)  # normalize by the maximum possible longitudinal jerk

        # 3. 计算最大加速度和急动度 (Max Acceleration and Jerk),表示瞬态舒适度
        ego_longitudinal_acc_max = np.max(np.abs(ego_longitudinal_accs)) 
        ego_lateral_acc_max = np.max(np.abs(ego_lateral_accs)) 
        ego_longitudinal_jerk_max = np.max(np.abs(ego_longitudinal_jerks))
        ego_lateral_jerk_max = np.max(np.abs(ego_lateral_jerks))
        norm_ego_longitudinal_acc_max = ego_longitudinal_acc_max / MAX_ACC  
        norm_ego_lateral_acc_max = ego_lateral_acc_max / MAX_ACC
        norm_ego_longitudinal_jerk_max = ego_longitudinal_jerk_max / MAX_JERK
        norm_ego_lateral_jerk_max = ego_lateral_jerk_max / MAX_JERK

        # 4. risk indicator: total RP
        risk_cal_start_time = time.time()
        if self.feature_trajectory.shape[0] != self.obstacle_trajectory.shape[0]:
            print("trajectory length is not equal")
            min_length = min(self.feature_trajectory.shape[0], self.obstacle_trajectory.shape[0])
            self.feature_trajectory_local = self.feature_trajectory[:min_length]
            self.obstacle_trajectory_local = self.obstacle_trajectory[:min_length]
        ego_STLCs, ego_CTLCs, ego_CTTCs, ego_TADs = risk_ind_cal(self.feature_trajectory_local, self.obstacle_trajectory_local, self.road_left, self.road_right)
        risk_cal_end_time = time.time()
        print(f"risk features calculating costs {risk_cal_end_time - risk_cal_start_time} seconds")

        ### RP test ###
        RP_array = np.zeros_like(ego_STLCs)
        for i in range(RP_array.shape[0]):
            if self.lane_id == 1:  # 0 means outside center trajectory, 1 means inside center trajectory
                RP_array[i] = 1.0/ego_TADs[i] + 0.16*(2.17*(1.0/ego_STLCs[i]) + 0.33*(1.0/ego_CTTCs[i]))
            elif self.lane_id == 0: 
                RP_array[i] = 1.0/ego_CTLCs[i] + 0.08*(1.09*(1.0/ego_STLCs[i]) + 0.45*(1.0/ego_CTTCs[i]))
            
            if RP_array[i] < 0:
                print("RP_array wrong")
        # b, a = signal.butter(8, 0.2, 'lowpass')  # filter signal > 0.2*50/2=5 Hz
        # RP_array = signal.filtfilt(b, a, RP_array)
        RP_total = np.sum(RP_array)
        RP_max = np.max(RP_array)
        RP_mean = np.mean(RP_array)

        STLC_mean = np.mean(ego_STLCs)  
        CTLC_mean = np.mean(ego_CTLCs)  
        CTTC_mean = np.mean(ego_CTTCs) 
        CTAD_mean = np.mean(ego_TADs)
        ### RP test ###

        # 5. lateral position: mean absolute lateral offset
        
        # a. 带符号的均值 (Signed Mean Deviation) —— 【捕捉个性化偏置的核心, positive means left, negative means right
        norm_ego_lateral_offset = np.mean(self.feature_trajectory_frenet[:,4]) / (self.road_width / 2) 
        # b.  平方均值 (Mean Squared Deviation) —— 【捕捉横向稳定性】
        norm_ego_lateral_offset_squared = np.mean(np.square(self.feature_trajectory_frenet[:,4])) / (self.road_width / 2) ** 2 
        # c. 最大绝对值 (Max Absolute Deviation) —— 【安全边界兜底】
        norm_ego_lateral_offset_max = np.max(np.abs(self.feature_trajectory_frenet[:,4])) / (self.road_width / 2) 

        # ego vehicle human-likeness
        ego_likeness = self.calculate_expert_likeness()
        # ego_likeness, MDE = self.calculate_expert_likeness()

        # feature array
        features = np.array([ego_efficiency,
                             ego_longitudinal_acc_max, ego_lateral_acc_max,
                             ego_longitudinal_jerk_max, ego_lateral_jerk_max,
                             ego_lateral_acc_total, ego_longitudinal_acc_total,
                             ego_lateral_jerk_total, ego_longitudinal_jerk_total,
                             RP_total, RP_max, RP_mean,
                             STLC_mean, CTLC_mean, CTTC_mean, CTAD_mean,
                             norm_ego_lateral_offset, norm_ego_lateral_offset_squared, norm_ego_lateral_offset_max,
                             norm_ego_efficiency,
                             norm_ego_longitudinal_acc_max, norm_ego_lateral_acc_max,
                             norm_ego_longitudinal_jerk_max, norm_ego_lateral_jerk_max,
                             norm_ego_lateral_acc_total, norm_ego_longitudinal_acc_total,
                             norm_ego_lateral_jerk_total, norm_ego_longitudinal_jerk_total,
                             ego_likeness])

        return features

    def calculate_expert_likeness(self):
        original_traj = self.feature_trajectory[:, 0:2].reshape(-1, 2)
        # print(self.reset_time)
        len_min = min(original_traj.shape[0], self.expert_trajectory.shape[0])
        if self.expert_trajectory.shape[0] == 0:
            print(self.expert_trajectory)
        ego_traj = self.expert_trajectory[:len_min, 0:2].reshape(-1, 2)
        feat_traj = original_traj[:len_min]
        if ego_traj.shape[0] == 0:
            print(self.feature_trajectory.shape[0])
        ADE = np.mean(np.linalg.norm(feat_traj - ego_traj, axis=1))
        FDE = np.linalg.norm(feat_traj[-1] - ego_traj[-1])
        MDE = np.mean(np.abs(self.feature_trajectory_frenet[:,4]))   # for center_expert
        # MDE = np.mean(np.abs(self.feature_trajectory_frenet[:,1] - 0.96))   # for 0.25_expert
        # MDE = np.mean(np.abs(self.feature_trajectory_frenet[:,1] - (-0.96)))   # for 0.75_expert
        if MDE > 2.0:
            print(MDE)
        # #--- 第 1 步：定义权重并计算综合误差 ---
        # # 权重的设定取决于你的研究侧重点：
        # # - 如果你更看重横向居中的拟合度（比如车道保持），调高 w_mde
        # # - 如果你更看重最终出弯目标点的一致性，调高 w_fde
        # # - 如果你要求轨迹每时每刻的时空都完全贴合，调高 w_ade
        # w_ade = 1.0  
        # w_fde = 0.5  # 通常终点误差会比平均误差大，为了平衡可以给个稍小的权重
        # w_mde = 2.0  # MDE 是纯横向指标，在弯道轨迹规划中往往最体现驾驶风格（个性化），建议给高权重

        # total_displacement_error = w_ade * ADE + w_fde * FDE + w_mde * MDE

        # # --- 第 2 步：将误差转换为 Likeness (0 到 1 的相似度得分) ---
        # # 引入灵敏度系数 alpha_likeness
        # # 假设你认为加权总误差达到 2.0 米时，两条轨迹就已经“完全不像”了（Likeness 降到约 0.05）
        # # 那么 2.0 * alpha = 3 -> alpha = 1.5
        # CRITICAL_ERROR = 2.0 
        # alpha_likeness = 3.0 / CRITICAL_ERROR

        # # 计算最终的 Likeness Score (完美重合为 1.0，越偏离越接近 0.0)
        # likeness_score = np.exp(-alpha_likeness * total_displacement_error)
        return MDE
    
    def _is_terminal(self):
        """
        The episode is over if the ego vehicle go off road or the time is out.
        """
        is_terminal = False
        if np.hypot(self.road_center[380, 0] - np.array(self.feature_trajectory)[-1,0], self.road_center[380, 1] - np.array(self.feature_trajectory)[-1,1]) <= 1.0:
            is_terminal = True
        return is_terminal 

    def render(self, num=0):
        """
        draw picture for test
        """
        # plot expert trajectory segment for paper show
        plt.figure()
        plt.rcParams['xtick.direction'] = 'in'  #将x周的刻度线方向设置向内
        plt.rcParams['ytick.direction'] = 'in'  #将y轴的刻度方向设置向内
        ax = plt.axes()
        ax.set_facecolor("grey")

        plt.plot(self.road[:, 0], self.road[:, 1], 'k', linewidth = 1.5)
        plt.plot(self.road[:, 2], self.road[:, 3], 'k', linewidth = 1.5)
        plt.plot(self.road[:, 4], self.road[:, 5], 'k--', linewidth = 1.5)  
        plt.plot(self.ego_trajectory[self.reset_time:, 0], self.ego_trajectory[self.reset_time:, 1], color='lime', label='expert trajectory') #teal
        plt.plot(self.obs_trajectory[self.reset_time:, 0], self.obs_trajectory[self.reset_time:, 1], color='m', label='obstacle trajectory')
        # plt.plot(self.ego_trajectory[73:323, 0], self.ego_trajectory[73:323, 1], color='lime', label='ego trajectory(start)') #teal
        # plt.plot(self.ego_trajectory[327:578, 0], self.ego_trajectory[327:578, 1], color='crimson', label='ego trajectory(middle)')
        # plt.plot(self.ego_trajectory[594:844, 0], self.ego_trajectory[594:844, 1], color='navy', label='ego trajectory(exit)')
        # plt.plot(self.obstacle_trajectory[73:323, 0], self.obstacle_trajectory[73:323, 1], color='m', label='obstacle trajectory')
        # plt.plot(self.obstacle_trajectory[327:578, 0], self.obstacle_trajectory[327:578, 1], color='m')
        # plt.plot(self.obstacle_trajectory[594:844, 0], self.obstacle_trajectory[594:844, 1], color='m')
        plt.xlim((-175, 10))
        plt.ylim((0, 170))
        plt.xlabel('Global X (m)',  fontdict={'family' : 'Times New Roman', 'size'   : 12})
        plt.ylabel('Global Y (m)',  fontdict={'family' : 'Times New Roman', 'size'   : 12})
        plt.legend(loc=0, prop={'size': 8})   # loc=0 means best
        plt.axis('equal')
        plt.savefig(f'./ResultRoadImages/trajectory_segment{num}.svg', dpi=300, bbox_inches='tight')

        # plot frenet trajectory
        plt.figure(figsize=(10,6))
        ax = plt.axes()
        ax.set_facecolor("grey")  # mintcream

        plt.plot(self.feature_trajectory_frenet[:, 0], self.feature_trajectory_frenet[:, 1], linewidth=3, label='planned trajectory')
        st = int(self.feature_trajectory_frenet[0, 0])
        plt.plot(np.linspace(st, st+70/3.6*5, 100), np.ones(100)*2.0, 'w--', linewidth=2, label='lane')
        plt.plot(np.linspace(st, st+70/3.6*5, 100), np.zeros(100), 'w--', linewidth=2)
        plt.plot(np.linspace(st, st+70/3.6*5, 100), np.ones(100)*-2.0, 'w--', linewidth=2)

        plt.plot(np.linspace(st, st+70/3.6*5, 100), np.ones(100)*0.96, '--', color='lime',linewidth=1.5, label='safe boundary')
        plt.plot(np.linspace(st, st+70/3.6*5, 100), np.ones(100)*-0.96, '--', color='lime', linewidth=1.5)
        plt.xlabel('Global Frenet X/m')
        plt.ylabel('Global lateral offset/m')
        plt.legend()

        plt.savefig(f'./Images/frenet_trajectory{num}.svg', dpi=300, bbox_inches='tight')

        # plot cartesian trajectory
        plt.figure(figsize=(10,6))
        ax = plt.axes()
        ax.set_facecolor("gainsboro")
        plt.plot(self.road[:, 0], self.road[:, 1], 'k', label='road')
        plt.plot(self.road[:, 2], self.road[:, 3], 'k')
        plt.plot(self.road[:, 4], self.road[:, 5], 'w--')
        plt.plot(self.feature_trajectory[:, 0], self.feature_trajectory[:, 1], color='lime', label='feature trajectory')
        plt.plot(self.obstacle_trajectory[:, 0], self.obstacle_trajectory[:, 1], color='m', label='obstacle trajectory')
        plt.xlabel('Global X/m')
        plt.ylabel('Global Y/m')
        plt.legend()
        plt.axis('equal')

        plt.savefig(f'./Images/cartesian_trajectory{num}.svg', dpi=300, bbox_inches='tight')

        # # plot features
        # plt.figure(figsize=(10,6))
        # ax = plt.axes()
        # x_labels = ['efficiency', 'lon_acc_max', 'lateral_acc_max', 'lon_jerk', 'RP_sum', 'risk_sum', 'risk_max', 'STLC', 'CTTC', 'ego_likeness']  # 准备上面指定的坐标轴的刻度对应替换的标签列表

        # plt.plot(x_labels, self.features, marker='o')
        # plt.xticks(rotation=30)
        # plt.xticks(fontsize=10.5)
        # plt.ylabel('feature values not normalized')
        
        # plt.savefig(f'./Images/features{num}.tiff', dpi=300)
        # plt.close('all')

    # def sampling_space(self):  # for target_v, target_d planner
    #     """
    #     The target sampling space (accelerate weight and time weight)
    #     """
    #     min_v = 40.0
    #     max_v = 55.0
    #     min_d = -0.8
    #     max_d = 0.8
    #     actions = []
    #     for vi in np.linspace(min_v, max_v, 8):
    #         for di in np.linspace(min_d, max_d, 10):
    #             actions.append([vi/3.6, di])
    #     actions = np.array(actions)
    #     return actions
    def sampling_space(self):  # for K_J, K_D planner
        """
        The target sampling space (accelerate weight[0,2] and distance weight[0,1])
        """
        actions = []
        for k_j in np.arange(0.0, 1.01, 0.1):
            for k_d in np.arange(0.0, 1.01, 0.1):
                actions.append([k_j, k_d])
        actions = np.array(actions)
        return actions


if __name__ == '__main__':
    lane_id = 1
    env = RewardEnv(lane_id)  
    env.reset(reset_time=0)  # resest_time<113
    action = (0.0, 0.5) # for target_v, target_d planner
    # action = (0.0, 1.0) # for K_J, K_D planner
    obs, features, terminal, info = env.step(action)  # use planner to generate expert trajectory
    # obs, features, terminal, info = env.step()    # use center line as expert trajectory
    env.render()

    # plt.figure(figsize=(10,6))
    # ax = plt.axes()
    # labels = ['','efficiency', 'lon_acc_max', 'lateral_acc_max', 'lon_jerk', 'STLC', 'CTTC', 'ego_likeness']  # 准备上面指定的坐标轴的刻度对应替换的标签列表
    # ax.set_xticklabels(labels, rotation=30, fontsize=10.5)
    # plt.plot(features, marker='o')
    # plt.ylabel('feature values not normalized')
    # plt.show()

