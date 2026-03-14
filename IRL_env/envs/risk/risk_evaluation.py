
import numpy as np
from IRL_env.envs.planner.global_road import natural_road_load      # for debug
import time                                              # for debug
import matplotlib.pyplot as plt                          # for debug
from IRL_env.envs.risk.readdata import *
import pandas as pd
import math

def curvature_cal(trajectory):
    """curvatue calculate

    Args:
        trajectory (np.array([x, y, speed])): the coordinate of trajectory

    Returns:
        np.array([cur1, cur2,...]): curvatue of every point on trajectory
    """
    CURV_RANGE = 5


    curvature = np.zeros((trajectory.shape[0], 1))
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    dis = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    dis = np.cumsum(dis)
    dis = np.insert(dis, 0, 0)

    start_time = time.time()
    for i in range(trajectory.shape[0]):
        start_step_time = time.time()
        # step forward to curve_range to get maximum id_max
        id_max = i
        dis_curve = dis[i]
        while dis[id_max] - dis_curve < CURV_RANGE:
            id_max += 1
            if id_max >= trajectory.shape[0]:
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
        end_step_time = time.time()
        # print(f"Step {i} took {end_step_time - start_step_time} seconds")

    end_time = time.time()
    # print(f"Total time of curvature calculate: {end_time - start_time} seconds")
    return curvature



def road_process(road_left,road_right):
    # read land data
    #计算车道线的航向角，因为hjh接口车道数据x,y,theta
    theta = np.zeros((road_left.shape[0],1))
    theta[:-1,0] = np.arctan2(road_left[1:,1]-road_left[:-1,1], road_left[1:,0]-road_left[:-1,0]) 
    theta[-1,0] = theta[-2,0]
    road_left = np.concatenate((road_left, theta),axis=1)
    road_right = np.concatenate((road_right, theta),axis=1)
    centerboundary = (road_left + road_right) / 2
    leftcenterline = (road_left + centerboundary) / 2
    leftboundary = road_left
    rightcenterline = (road_right + centerboundary) / 2
    rightboundary = road_right
    return centerboundary, leftcenterline, leftboundary, rightcenterline,rightboundary
    
def distance_cal(trajectory):
    distance = np.zeros((trajectory.shape[0],))
    distance[0] = 0
    for i in range(1,trajectory.shape[0]):
        distance[i] = distance[i-1] + np.sqrt((trajectory[i,0]-trajectory[i-1,0])**2 + (trajectory[i,1]-trajectory[i-1,1])**2)
    return distance

def overtaking_process(center,traj_ego,traj_obs,ttc,ctad,length=2):
    # 自车超过临车length以上,ttc和ctad=0,超过距离在0-length,线性插值
    ttc = np.array(ttc)
    ctad = np.array(ctad)
    ttc_process = np.copy(ttc)
    ctad_process = np.copy(ctad)
    #找到自车和临车 离中心线最新的点
    distances_ego_center = np.sum((traj_ego[:,None,:]-center[None,:,:2])**2,axis=-1)
    distances_obs_center = np.sum((traj_obs[:,None,:]-center[None,:,:2])**2,axis=-1)
    index_ego = np.argmin(distances_ego_center,axis=-1)
    index_obs = np.argmin(distances_obs_center,axis=-1)
    dis_relative = np.zeros((index_ego.shape[0],))

    if np.any(index_ego>index_obs):
        dis_center = distance_cal(center)
        dis_ego = dis_center[index_ego] - dis_center[index_ego[0]]
        dis_obs = dis_center[index_obs] - dis_center[index_ego[0]]
        dis_relative = dis_ego-dis_obs
        ttc_process[dis_relative>=(length+0.1)] = 0
        ctad_process[dis_relative>=(length+0.1)] = 0
        
        # 处理反复超车被超的情况，返回result为两车相对距离0-length的索引
        result=[]
        indics = np.where((dis_relative>=0-0.1)& (dis_relative<=length+0.1))[0]  #两边扩大0.1
        temp=[indics[0]]
        for i in range(1,indics.shape[0]):
            if indics[i]==(indics[i-1]+1):
                temp.append(indics[i])
            else:
                result.append(temp)
                temp = [indics[i]-1,indics[i]]
        result.append(temp)

        for i in range(len(result)):
            #特殊情况：
            # 1.自车超越邻车，但超过距离小于Length,又被反超车  -> 不改变
            # 2.自车在邻车前面，两者相对距离缩小到小于Length，但是不低于0，自车又加速拉开距离 -> 零（-1）
            #情况2
            if abs(dis_relative[result[i][0]]-length)<0.1 and abs(dis_relative[result[i][-1]]-0)<0.1:
                ttc_process[result[i][0]:result[i][-1]] = ttc_process[result[i][0]]
                ctad_process[result[i][0]:result[i][-1]] = ctad_process[result[i][0]]
            # #情况1 不改变，直接注释掉
            # elif dis_relative[result[i][0]]<0 and dis_relative[result[i][-1]]<0:
            #     continue
            # 正常情况自车完成超车并距离大于length，或自车被超车  -> 线性插值
            if abs(dis_relative[result[i][0]]-0)<0.1 and abs(dis_relative[result[i][-1]]-length)<0.1:
                k_ttc = (-ttc_process[result[i][0]]+0)/(length)
                k_ctad = (-ctad_process[result[i][0]]+0)/(length)
                for j in range(len(result[i])):
                    ttc_process[result[i][j]] = ttc_process[result[i][0]]+k_ttc*dis_relative[result[i][j]]
                    ctad_process[result[i][j]] = ctad_process[result[i][0]]+k_ctad*dis_relative[result[i][j]]
            # 正常情况 自车被超车  -> 线性插值
            if abs(dis_relative[result[i][0]]-length)<0.1 and abs(dis_relative[result[i][-1]])<0.1:
                k_ttc = (ttc_process[result[i][-1]]-0)/(length)
                k_ctad = (ctad_process[result[i][-1]]-0)/(length)
                for j in range(len(result[i])):
                    ttc_process[result[i][j]] = 0+k_ttc*(length-dis_relative[result[i][j]])
                    ctad_process[result[i][j]] = 0+k_ctad*(length - dis_relative[result[i][j]])
    # # for debug
    # plt.figure()
    # plt.plot(ttc_process)
    # plt.plot(ttc)
    # plt.xlabel('trajectory index')
    # plt.ylabel('TTC/s')

    # plt.figure()
    # plt.plot(ctad_process)
    # plt.plot(ctad)
    # plt.xlabel('trajectory index')
    # plt.ylabel('TAD/s')
    # plt.figure()
    # plt.plot(dis_relative)
    # plt.xlabel('trajectory index')
    # plt.ylabel('distance_relative(m)') 
    # plt.show()
    return ttc_process,ctad_process

def trajectory_process(feature_trajectory,obstacle_trajectory,L,SAMPLE_T):
    #计算航向角
    feature_yaw = np.zeros(feature_trajectory.shape[0])  
    obstacle_yaw = np.zeros(obstacle_trajectory.shape[0])
    feature_yaw[:-1] = np.arctan2(feature_trajectory[1:, 1]-feature_trajectory[:-1, 1], \
                                  feature_trajectory[1:, 0]-feature_trajectory[:-1, 0])
    feature_yaw[-1] = feature_yaw[-2]
    obstacle_yaw[:-1] = np.arctan2(obstacle_trajectory[1:, 1]-obstacle_trajectory[:-1, 1], \
                                    obstacle_trajectory[1:, 0]-obstacle_trajectory[:-1, 0])
    obstacle_yaw[-1] = obstacle_yaw[-2]
    #平滑航向角
    window_size = 8 # 设置窗口大小
    feature_yaw = pd.Series(feature_yaw)
    smooth_ego_yaw = feature_yaw.rolling(window_size).mean()
    smooth_ego_yaw = smooth_ego_yaw.bfill() #后一个非缺失值来填充缺失值
    obstacle_yaw = pd.Series(obstacle_yaw)
    smooth_obstacle_yaw = obstacle_yaw.rolling(window_size).mean()
    smooth_obstacle_yaw = smooth_obstacle_yaw.bfill()

    #计算自车曲率
    feature_traj_curvature = curvature_cal(feature_trajectory) 

    #计算前轮转角
    feature_front_theta = np.arctan(L * feature_traj_curvature)

    # read trajectory data
    trajectory_state = np.zeros((feature_trajectory.shape[0],11))
    trajectory_state[:,0] = np.arange(feature_trajectory.shape[0]) #index
    trajectory_state[:,1] = trajectory_state[:,0]*SAMPLE_T #time
    trajectory_state[:,2] = feature_trajectory[:,0] #x
    trajectory_state[:,3] = feature_trajectory[:,1] #y
    trajectory_state[:,4] = smooth_ego_yaw #yaw
    trajectory_state[:,5] = feature_trajectory[:,2] #speed m/s ，注意传入单位，数据csv文件是km/h，测试的输入已经除3.6了，这里不用再除
    trajectory_state[:,6] = feature_front_theta[:,0]#front wheel angle
    trajectory_state[:,7] = obstacle_trajectory[:,0] #x
    trajectory_state[:,8] = obstacle_trajectory[:,1] #y
    trajectory_state[:,9] = smooth_obstacle_yaw #yaw
    trajectory_state[:,10] = obstacle_trajectory[:,2] #speed

    return trajectory_state

def risk_ind_cal(feature_trajectory, obstacle_trajectory, road_left, road_right):
    """calculate risk indicator 
    
    Args:
        feature_trajectory (np.array([x, y, speed])): Cartesian coordinate of ego trajectory
        obstacle_trajectory (np.array([x, y, speed])): Cartesian coordinate of adjacent vehicle trajectory
        road_left (np.array([x, y]): Cartesian coordinate of lane left trajectory
        roaf_right (np.array([x, y])): Cartesian coordinate of lane right trajectory
    """
    # TODO 要求传入的数据必须是间隔SAMPLE_T
    SAMPLE_T = 0.1    # s

    wheelbase = 2.6 # check 2.875 wheelbase of model3, but use 2.6 in cec experiment VD


    centerboundary, leftcenterline, leftboundary, rightcenterline,rightboundary = road_process(road_left,road_right)
    all_lane_data = AllLaneData(centerboundary, leftcenterline, 
                                leftboundary, rightcenterline,
                                rightboundary)
    # all_lane_data.read_all_data()
    # all_lane_data.draw_all_data()
    trajectory_state = trajectory_process(feature_trajectory,obstacle_trajectory,wheelbase,SAMPLE_T)
    vehtraj_data = VehicleTrajData(trajectory_state)
    vehtraj_data.cal_curvature_str_angle()
    # vehtraj_data.draw_data()
    # all_lane_data.read_all_data()
    # vehtraj_data.read_data()
    # calculate the indicator data
    indicator_data = IndicatorData(all_lane_data, vehtraj_data)


    # indicator_data.test_func_intersection()
    # indicator_data.draw_lane_vehtraj()
    # start = time.time()

    # 多进程加速，if need
    # with Pool(processes=3) as pool:
    #     ctad_task = pool.apply_async(indicator_data.cal_ctad, (indicator_data.vehtraj_data, 2, 0))
    #     left_stlc_task = pool.apply_async(indicator_data.cal_tlc, (indicator_data.vehtraj_data, all_lane_data.leftboundary_data.lane, 2,0))
    #     right_stlc_task = pool.apply_async(indicator_data.cal_tlc, (indicator_data.vehtraj_data, all_lane_data.rightboundary_data.lane, 2, 0))
    #     ctad, ctad_flag = ctad_task.get()
    #     left_stlc, left_ctlc = left_stlc_task.get()
    #     right_stlc, right_ctlc = right_stlc_task.get()
    start = time.time()
    ttc, ttc_flags = indicator_data.func_cal_ttc(indicator_data.drivertraj_data, 0)
    # lateral_offset = indicator_data.func_cal_lateraloffset(indicator_data.vehtraj_data, all_lane_data.rightcenterline_data.lane, 0)
    ctad, ctad_flag = indicator_data.cal_ctad(indicator_data.vehtraj_data, 2, 0)
    left_stlc, left_ctlc = indicator_data.cal_tlc(indicator_data.vehtraj_data, all_lane_data.leftboundary_data.lane, 2,0)
    right_stlc, right_ctlc = indicator_data.cal_tlc(indicator_data.vehtraj_data, all_lane_data.rightboundary_data.lane, 2, 0)
    ctlc = [left_ctlc[i] if left_ctlc[i] > right_ctlc[i] else right_ctlc[i] for i in range(len(left_ctlc))]
    stlc = [left_stlc[i] if left_stlc[i] > right_stlc[i] else right_stlc[i] for i in range(len(left_stlc))]
    # print(f"cost={time.time()-start}")

    ctlc = [math.inf if ctlc[i] == -1 else ctlc[i] for i in range(len(ctlc))]
    stlc = [math.inf if stlc[i] == -1 else stlc[i] for i in range(len(stlc))]

    # overtake process
    ittc = [0 if ttc[i] == -1 else 1/ttc[i] for i in range(len(ttc))]
    ictad = [0 if ctad[i] == -1 else 1/ctad[i] for i in range(len(ctad))]    

    length = 2 #超过多少米的距离，ttc和ctad=0
    ittc, ictad= overtaking_process(centerboundary,feature_trajectory[:,:2],obstacle_trajectory[:,:2],ittc,ictad,length)
    ttc = [math.inf if ittc[i] == 0 else 1/ittc[i] for i in range(len(ittc))]
    ctad = [math.inf if ictad[i] == 0 else 1/ictad[i] for i in range(len(ictad))]
    return stlc, ctlc, ttc, ctad



if __name__ == "__main__":
    lane_id=1
    sample_time = 0.1
    trajectory_data_process = natural_road_load(lane_id)
    ego_trajectory, road, obstacle_trajectory = trajectory_data_process.build_trajectory()
    trajectory = ego_trajectory[:,0:3]    # array([x, y, speed, lane_id])
    trajectory[:,2] = trajectory[:,2] / 3.6

    obstacle_trajectory = obstacle_trajectory[:,0:3]   # array([x, y, speed, lane_id])
    obstacle_trajectory[:,2] = obstacle_trajectory[:,2] / 3.6

    road_left = road[:,0:2]
    road_right = road[:,2:4]
    road_center = road[:,4:]

    timesteps = np.linspace(10, trajectory.shape[0]-60, num=15, dtype=np.int16)
    RP_list = []
    STLC_list = []
    CTLC_list = []
    CTTC_list = []
    CTAD_list = []

    for step in timesteps:
        print(f"step={step}")
        N_sample = int(trajectory[step:,:].shape[0])
        # read for: trajectory, obstacle_trajectory, road_left, road_right
        ego_STLCs, ego_CTLCs, ego_CTTCs, ego_TADs = \
        risk_ind_cal(trajectory[step:,:], obstacle_trajectory[step:, :], road_left, road_right)

        RP_array = np.zeros_like(ego_STLCs)
        for i in range(RP_array.shape[0]):
            if lane_id == 1:  # 0 means outside center trajectory, 1 means inside center trajectory
                RP_array[i] = 1.0/ego_TADs[i] + 0.16*(2.17*(1.0/ego_STLCs[i]) + 0.33*(1.0/ego_CTTCs[i]))
            elif lane_id == 0: 
                RP_array[i] = 1.0/ego_CTLCs[i] + 0.08*(1.09*(1.0/ego_STLCs[i]) + 0.45*(1.0/ego_CTTCs[i]))
            
            if RP_array[i] < 0:
                print("RP_array wrong")
        # b, a = signal.butter(8, 0.2, 'lowpass')  # filter signal > 0.2*50/2=5 Hz
        # RP_array = signal.filtfilt(b, a, RP_array)
        max_RP = np.max(RP_array)
        
        risk_exp = np.exp(-np.mean(RP_array))
        STLC_exp = np.exp(-np.mean(ego_STLCs))  
        CTLC_exp = np.exp(-np.mean(ego_CTLCs))
        CTTC_exp = np.exp(-np.mean(ego_CTTCs)) 
        CTAD_exp = np.exp(-np.mean(ego_TADs))
        
        RP_list.append(max_RP)
        STLC_list.append(STLC_exp)
        CTLC_list.append(CTLC_exp)
        CTTC_list.append(CTTC_exp)
        CTAD_list.append(CTAD_exp)
        print(f"trajectory's max_RP={max_RP}, STLC={STLC_exp}, CTLC={CTLC_exp}, CTTC={CTTC_exp}, CTAD={CTAD_exp}")
    
    print(f"max RP={max(RP_list)}, max STLC={max(STLC_list)}, max CTLC={max(CTLC_list)}, max CTTC={max(CTTC_list)}, max CTAD={max(CTAD_list)}")
    save_data = {
        'step': timesteps,
        'maxRP': RP_list,
        'maxSTLC': STLC_list,
        'maxCTLC': CTLC_list,
        'maxCTTC': CTTC_list,
        'maxCTAD': CTAD_list
    }
    #  存成pkl文件
    # 保存
    try:
        df = pd.DataFrame(save_data)
        file_path = os.path.join('./indicatorFile', 'risk_indicator_data.pkl')
        df.to_pickle(file_path)
        print(f"✅ 数据已保存: {file_path}")
    except Exception as e:
        print(f"❌ 保存失败: {e}")

    # # 读取 pickle 文件
    # df_loaded = pd.read_pickle('risk_indicator_data.pkl')
    # print(df_loaded.head())


    # plt.figure(figsize=(10,6))
    # ax = plt.axes()
    # ax.set_facecolor("grey")

    # plt.plot(road_left[:,0], road_left[:,1], 'w--', label='road')
    # plt.plot(road_center[:,0], road_center[:,1], 'w--')
    # plt.plot(road_right[:,0], road_right[:,1], 'w--')

    # plt.plot(trajectory[:,0], trajectory[:,1], color='lime')
    # plt.plot(obstacle_trajectory[:,0], obstacle_trajectory[:,1], color='lime')

    # plt.xlabel('Global X/m')
    # plt.ylabel('Global Y/m')

    # plt.legend()
    # plt.axis('equal')
    # plt.show()

    # x_axis = list(range(trajectory.shape[0]))
    # plt.figure(figsize=(10,6))
    # plt.plot(x_axis, ego_STLCs)  # 这个看着稍微正常一点，后面出弯后是直线所以是 'inf'
    # plt.xlabel('trajectory index')
    # plt.ylabel('STLC/s')


    # plt.figure(figsize=(10,6))
    # plt.plot(x_axis, ego_CTLCs)  # 中间一小段是inf吗？ 应该是后面直线段是inf吧？为啥最后直线段反而那么小？？  index=605:836这部分是inf
    # plt.xlabel('trajectory index')
    # plt.ylabel('CTLC/s')

    # plt.figure(figsize=(10,6))
    # plt.plot(x_axis, ego_CTTCs)  # 中间好像是空缺了一段，是inf吗？
    # plt.xlabel('trajectory index')
    # plt.ylabel('CTTC/s')

    # plt.figure(figsize=(10,6))
    # plt.plot(x_axis, ego_TADs)   # 为什么e13这么大的数字？？？，不正常吧
    # plt.xlabel('trajectory index')
    # plt.ylabel('TAD/s')

    # curvature = curvature_cal(trajectory)
    # plt.figure(figsize=(10,6))
    # plt.plot(x_axis, curvature, label='curve range = 40')
    # plt.ylim(-8.5e-3, 1e-3)

    # plt.xlabel('trajectory point index')
    # plt.ylabel('curvatue')
    # plt.legend()

    # plt.show()


'''
if __name__ == '__main__':
    lane_id=1
    trajectory_data_process = natural_road_load(lane_id)
    ego_trajectory, road, obstacle_trajectory, obstacle_trajectory03 = trajectory_data_process.build_trajectory()
    trajectory = ego_trajectory[:,0:3]    # array([x, y, speed, lane_id])
    trajectory[:,2] = trajectory[:,2] / 3.6

    obstacle_trajectory = obstacle_trajectory[:,0:3]   # array([x, y, speed, lane_id])
    obstacle_trajectory[:,2] = obstacle_trajectory[:,2] / 3.6

    obstacle_trajectory03 = obstacle_trajectory03[:,0:3]   # array([x, y, speed, lane_id])
    obstacle_trajectory03[:,2] = obstacle_trajectory03[:,2] / 3.6

    road_left = road[:,0:2]
    road_right = road[:,2:4]
    road_center = road[:,4:]

    plt.figure(figsize=(10,6))
    
    ax = plt.axes()
    ax.set_facecolor("grey")

    plt.plot(road_left[:,0], road_left[:,1], 'w--', label='road')
    plt.plot(road_center[:,0], road_center[:,1], 'w--')
    plt.plot(road_right[:,0], road_right[:,1], 'w--')

    plt.plot(trajectory[:,0], trajectory[:,1], color='lime')
    plt.plot(obstacle_trajectory[:,0], obstacle_trajectory[:,1], color='lime')
    plt.plot(obstacle_trajectory03[:,0], obstacle_trajectory03[:,1], color='purple')

    plt.xlabel('Global X/m')
    plt.ylabel('Global Y/m')

    plt.legend()
    plt.axis('equal')
    # plt.show()

    # read for: trajectory, obstacle_trajectory, road_left, road_right
    ego_STLCs, ego_CTLCs, ego_CTTCs, ego_TADs = \
    risk_ind_cal(trajectory, obstacle_trajectory, road_left, road_right)

    ego_STLCs03, ego_CTLCs03, ego_CTTCs03, ego_TADs03 = \
    risk_ind_cal(trajectory, obstacle_trajectory03, road_left, road_right)

    STLC = np.exp(-min(ego_STLCs))  
    CTTC = np.exp(-min(ego_CTTCs)) 

    ### RP test ###
    iSTLC = 1.0/min(ego_STLCs)
    iCTLC = 1.0/min(ego_CTLCs)
    iTTC = 1.0/min(ego_CTTCs)
    iCTAD = 1.0/min(ego_TADs)
    if lane_id == 1:  # 0 means outside center trajectory, 1 means inside center trajectory
        RP = iCTAD + 0.16*(2.17*iSTLC + 0.33*iTTC)
    elif lane_id == 0: 
        RP = iCTLC + 0.08*(1.09*iSTLC + 0.45*iTTC)
    print(f"trajectory's RP={RP}")
    #----------03#
    STLC03 = np.exp(-min(ego_STLCs03))  
    CTTC03 = np.exp(-min(ego_CTTCs03)) 
    ### RP test ###
    iSTLC03 = 1.0/min(ego_STLCs03)
    iCTLC03 = 1.0/min(ego_CTLCs03)
    iTTC03 = 1.0/min(ego_CTTCs03)
    iCTAD03 = 1.0/min(ego_TADs03)
    if lane_id == 1:  # 0 means outside center trajectory, 1 means inside center trajectory
        RP03 = iCTAD03 + 0.16*(2.17*iSTLC03 + 0.33*iTTC03)
    elif lane_id == 0: 
        RP03 = iCTLC03 + 0.08*(1.09*iSTLC03 + 0.45*iTTC03)
    print(f"trajectory's RP={RP03}")

    x_axis = list(range(trajectory.shape[0]))
    plt.figure(figsize=(10,6))
    plt.plot(x_axis, ego_STLCs)  # 这个看着稍微正常一点，后面出弯后是直线所以是 'inf'
    plt.plot(x_axis, ego_STLCs03, label='03')  # 这个看着稍微正常一点，后面出弯后是直线所以是 'inf'
    plt.xlabel('trajectory index')
    plt.ylabel('STLC/s')


    plt.figure(figsize=(10,6))
    plt.plot(x_axis, ego_CTLCs)  # 中间一小段是inf吗？ 应该是后面直线段是inf吧？为啥最后直线段反而那么小？？  index=605:836这部分是inf
    plt.plot(x_axis, ego_CTLCs03, label='03')
    plt.xlabel('trajectory index')
    plt.ylabel('CTLC/s')

    plt.figure(figsize=(10,6))
    plt.plot(x_axis, ego_CTTCs)  # 中间好像是空缺了一段，是inf吗？
    plt.plot(x_axis, ego_CTTCs03, label='03')
    plt.xlabel('trajectory index')
    plt.ylabel('CTTC/s')

    plt.figure(figsize=(10,6))
    plt.plot(x_axis, ego_TADs)   # 为什么e13这么大的数字？？？，不正常吧
    plt.plot(x_axis, ego_TADs03, label='03')
    plt.xlabel('trajectory index')
    plt.ylabel('TAD/s')

    curvature = curvature_cal(trajectory)
    plt.figure(figsize=(10,6))
    plt.plot(x_axis, curvature, label='curve range = 40')
    plt.ylim(-8.5e-3, 1e-3)

    plt.xlabel('trajectory point index')
    plt.ylabel('curvatue')
    plt.legend()

    plt.show()
'''






