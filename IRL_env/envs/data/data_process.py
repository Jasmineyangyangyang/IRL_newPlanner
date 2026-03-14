import csv
import os
import numpy as np
import matplotlib.pyplot as plt

class natural_road_data():
    def __init__(self, lane_id):   
        self.ego_lane_id = lane_id # id=0 outside lane, id=1 inside lane
        self.expert_trajectory = []  # position_x, position_y, position_speed, position_lane_id
        self.road = []          # [left_lane_x,left_lane_y,right_lane_x,right_lane_y,center_lane_x,center_lane_y]
        self.obstacle_trajectory = []  # [x, y, V] km/h
    
    def read_from_csv(self, filepath):
        """return road ego and obstacle trajectory information

        Args:
            filepath (string): the location of your csv file
        """
        ego_filename = os.path.join(filepath, 'ego_trajectory.csv')
        # ego_filename = os.path.join(filepath, 'ego_trajectory_05.csv')
        print("Loading Data...")
        with open(ego_filename, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                if self.ego_lane_id == 0:
                    expert_x = float(row['left_center_x'])
                    expert_y = float(row['left_center_y'])
                else:
                    expert_x = float(row['right_center_x'])
                    expert_y = float(row['right_center_y'])
                expert_speed = float(42)
                # expert_speed = float(30)
                self.expert_trajectory.append([expert_x, expert_y, expert_speed, self.ego_lane_id])

        obs_filename = os.path.join(filepath, 'obs_trajectory.csv')
        with open(obs_filename, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                if self.ego_lane_id == 0:
                    obstacle_x = float(row['ad_right_x'])
                    obstacle_y = float(row['ad_right_y'])
                else:
                    obstacle_x = float(row['ad_left_x'])
                    obstacle_y = float(row['ad_left_y'])
                # obstacle_speed = float(42)
                obstacle_speed = float(30)
                self.obstacle_trajectory.append([obstacle_x, obstacle_y, obstacle_speed, 1 - self.ego_lane_id])

        road_filename = os.path.join(filepath, 'global_road.csv')
        print("Loading Data...")
        with open(road_filename, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                center_x = float(row['center_x'])
                center_y = float(row['center_y'])
                left_x = float(row['left_x'])
                left_y = float(row['left_y'])
                right_x = float(row['right_x'])
                right_y = float(row['right_y'])

                self.road.append([left_x, left_y, right_x, right_y, center_x, center_y])

        # input interest section of whole trajectory
        self.expert_trajectory = np.array(self.expert_trajectory)[:163,:]
        self.obstacle_trajectory = np.array(self.obstacle_trajectory)[11:174,:]
        # self.road = np.array(self.road)[479:2900,:]   
        self.road = np.array(self.road)[:600,:]    #lhl 计算时候TLC需要更长的道路信息
        print("Load data successfully!")

    def build_trajectory(self): # id=0 outside lane, id=1 inside lane  
        # filepath = '/Users/jiaxin/JiaxinCode/OASCode/Jtekt/RP_TD32312/env/data/processed'
        filepath = '/Users/jiaxin/JiaxinCode/OASCode/Jtekt/IRL_poly_newRP/IRL_env/envs/data/processed'
        self.read_from_csv(filepath)
        return self.expert_trajectory, self.road, self.obstacle_trajectory

if __name__ == '__main__':
    env_data = natural_road_data(lane_id=1)
    expert_trajectory, road, obstacle_trajectory = env_data.build_trajectory()
    ## for one round expert[0:4295], obstacle[0:4240]
    # plt.plot(expert_trajectory[:4295,0], expert_trajectory[:4295,1], color='g', label='expert trajectory')
    # plt.plot(obstacle_trajectory[0:4240,0], obstacle_trajectory[0:4240,1], color='m', label='obstacle trajectory')
    # plt.plot(road[:, 0], road[:, 1], color='k')
    # plt.plot(road[:, 2], road[:, 3], color='k')
    # plt.plot(road[:, 4], road[:, 5], 'k--', label='road center')
    # plt.xlabel('Global X/m')
    # plt.ylabel('Global Y/m')
    # plt.legend()
    # plt.show()

    # # for one circle expert[], obstacle[]
    # plt.plot(expert_trajectory[193:1076,0], expert_trajectory[193:1076,1], color='g', label='expert trajectory')
    # plt.plot(obstacle_trajectory[193:1076,0], obstacle_trajectory[193:1076,1], color='m', label='obstacle trajectory')
    # plt.plot(road[479:2900, 0], road[479:2900, 1], color='k')
    # plt.plot(road[479:2900, 2], road[479:2900, 3], color='k')
    # plt.plot(road[479:2900, 4], road[479:2900, 5], 'k--', label='road center')
    # plt.xlabel('Global X/m')
    # plt.ylabel('Global Y/m')
    # plt.legend()
    # plt.show()

    # for test
    plt.plot(expert_trajectory[:,0], expert_trajectory[:,1], 'g.', label='expert trajectory')
    plt.plot(obstacle_trajectory[:,0], obstacle_trajectory[:,1], 'm.', label='obstacle trajectory')
    plt.plot(road[:, 0], road[:, 1], color='k')
    plt.plot(road[:, 2], road[:, 3], color='k')
    plt.plot(road[:, 4], road[:, 5], 'k--', label='road center')
    plt.axis('equal')
    plt.xlabel('Global X/m')
    plt.ylabel('Global Y/m')
    plt.legend()
    plt.show()