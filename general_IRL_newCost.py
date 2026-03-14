import numpy as np
import csv
from tqdm import tqdm # 进度条库
import sys
sys.path.append("IRL_env/envs")
from IRL_env.envs.irlenv import RewardEnv
import pickle
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')
# matplotlib.use('Agg')

# parameters
n_iters = 200
lr = 0.05
lam = 0.01
# feature_num = 5 # first 5 features   # for maxEnt_IRL()
feature_num = [10, 11, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27] # maxEnt_IRL_new()
lane_id = 1     # 0 means outside center trajectory, 1 means inside center trajectory
render_env = False # to show sample trajectories(black) expert trajectory(red) and top3 optimal reward trajectory(blue) 
np.random.seed(0)
bufferfilepath = './features_buffer/buffer_2603_02.pkl'
maxEnt_training_log_filepath = './training_log/general_training_log_2603_02.csv'
maxEnt_theta_filepath = './theta_file/theta_2603_02.pkl'

def train_data_prepare():
    # Cache
    buffer = []
    expert_traj_features = []

    env = RewardEnv(lane_id)   
    # length to serve as reference, maybe in defferenct segments of curve has differenct reward function 
    # you can change the train_steps to improve this idea. 
    length = len(env.ego_trajectory)  # Cartesian trajectory, [x, y, speed, lane_id]
    timesteps = np.linspace(10, length-60, num=15, dtype=np.int16)
    train_steps = timesteps
    # train_steps = np.random.choice(timesteps, size=10, replace=False) # replace means can choose one repeat
    print("train_steps:", train_steps) ; # [ 10  16  23  29  36  43  49  56  63  69  76  83  89  96 103]
    
    pbar = tqdm(train_steps)
    # run until the road ends
    fig_num = 0
    for start in pbar:
        # env fun2. get trajectory for ego and adjacent: env.reset，cut trajectory from start time index to the end, so end=len(traj)-60
        pbar.set_description('calculate features from '+ str(start))
        env.reset(reset_time=start)

        # set up buffer of the scene 
        buffer_scene = []

        # target sampling space
        actions = env.sampling_space()  # array([acc_weight, distance_weight])

        # trajectory sampling  first set the simulate trajectory
        for action in actions:
            acc_weight, d_weight = action
            # sample a trajectory
            obs, features, terminated, info = env.step(action)
            if np.isnan(features).any():
                print("The features contains nan")
            # render env
            if render_env:
                fig_num += 1
                env.render(fig_num)  # render use to display vehicles state

            # get the features
            traj_features = features[:-1]   # 把最后一个likeness单独拿出来
            expert_likeness = features[-1]
            
            # add scene trajectories to buffer
            buffer_scene.append((acc_weight, d_weight, traj_features, expert_likeness))

            # set back to previous step
            env.reset(reset_time=start)
            print(f"cal done, acc_weight: {acc_weight}, d_weight: {d_weight}, expert_likeness: {expert_likeness}")
        # calculate expert trajectory feature
        env.reset(reset_time=start)
        _, features, _, _ = env.step()
        # eliminate invalid examples
        if features[-1] > 2.5:  # the last feature is expert trajectory likeness
            print(features[-1])
            continue
        # process data
        expert_features = features[:-1]
        # buffer_scene has lateral_offsets * arget_speeds number trajectories
        buffer_scene.append([0.0, 0.5, expert_features, features[-1]]) # expert reference speed is 42km/h
        # add to buffer
        expert_traj_features.append(expert_features)
        buffer.append(buffer_scene)# add expert trajectory to the last of buffer_scene
    
    # save buffer
    with open(bufferfilepath, 'wb') as f: # for K_J K_D，speed = 42km/h，laneid=1 ego在内道
        pickle.dump(buffer, f)

def maxEnt_IRL_newCost():
    # create training log
    with open(maxEnt_training_log_filepath, 'w', newline='') as csvfile:    # for K_J K_D laneid=0 offset=-0.5m
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['iteration', 'expert feature', 'trajectory feature', 'feature norm', 'weights', 'cal_expert_idx', 'expert likeness', 'mean expert likeness', 'likelihood'])

    with open(bufferfilepath, 'rb') as f:   # for K_J K_D laneid=0 offset=-0.5m
        buffer = pickle.load(f)
    
    # traj_features_temp = []
    # RP_total = []
    # RP_max = []
    # RP_mean = []
    # for buffer_scene in buffer:
    #     for traj in buffer_scene:
    #         traj_features_temp.append(traj[2])
    #         RP_total.append(traj[2][9])
    #         RP_max.append(traj[2][10])
    #         RP_mean.append(traj[2][11])
    # max_v = np.max(traj_features_temp, axis=0) 
    # min_v = np.min(traj_features_temp, axis=0)
    # plt.plot(RP_total, marker='o')

    # for scene in buffer:
    #     for traj in scene:
    #         for i in range(max_v.shape[0]):
    #             traj[2][i] = (traj[2][i] - min_v[i]) / (max_v[i] - min_v[i])

    #===============================#
    # normalize features 使用物理/理论界限归一化），而不是 min-max归一化，
    # 因为特征指标的“0”是有严格物理意义的。例如，加速度为 0 代表匀速，Jerk 为 0 代表舒适度最高。
    # 若 min jerk =-0.5, 做完 Min-Max 后，物理上的“0 Jerk”会被映射到类似 非 0 的位置。
    
    # RP_total的最大值为0.6,RP_max的最大值为0.018，,RP_mean的最大值为0.0069
    rp_alpha_max = 3.0 / 0.05
    rp_alpha_mean = 3.0 / 0.01
    for scene in buffer:
        for traj in scene:
            traj[2][10] = 1 - np.exp(-rp_alpha_max * traj[2][10])
            traj[2][11] = 1 - np.exp(-rp_alpha_mean * traj[2][11])
    
    expert_traj_features = []
    for buffer_scene in buffer:
        exp_feature_tmp = []
        for num in feature_num:
            exp_feature_tmp.append(buffer_scene[-1][2][num])
        expert_traj_features.append(np.array(exp_feature_tmp))  


    #### MaxEnt IRL ####
    print("Start training...")
    # initialize weights
    theta = np.random.normal(0, 0.05, size=len(feature_num))

    # iterations
    beta1 = 0.9; beta2 = 0.999; eps = 1e-8
    pm = None
    pv = None
    grad_log = []
    expert_likeness_log = []

    pbar = tqdm(range(n_iters))
    for iteration in pbar:
        pbar.set_description('iteration '+ str(iteration))

        feature_exp = np.zeros([len(feature_num)])
        expert_feature_exp = np.zeros([len(feature_num)])

        log_like_list = []
        iteration_expert_likeness = []
        top3_idx_backup = []
        num_traj = 0
        index = 0

        for scene in buffer:
            # compute on each scene
            scene_trajs = []
            for trajectory in scene:  # every scene has 36 trajectories, the last one is expert trajectory
                #---------------------#
                trajectory_feature_tmp = np.zeros(len(feature_num))
                for i in range(len(feature_num)):
                    trajectory_feature_tmp[i] = trajectory[2][feature_num[i]]
                reward = np.dot(trajectory_feature_tmp, theta)  # risk = RP_sum
                scene_trajs.append((reward, trajectory_feature_tmp, trajectory[3])) # # risk = RP  {3}

            # calculate probability of each trajectory
            rewards = [traj[0] for traj in scene_trajs] 
            max_reward = np.max(rewards)
            probs = [np.exp(reward - max_reward) for reward in rewards]  # list长36
            probs = probs / np.sum(probs)                   # list 长36  (36,)  
            log_like_list.append(np.log(probs[-1]/np.sum(probs)))                 # the last teajectory is expert trajectory

            # stroe expert likeness
            # likeness = [traj[2] for traj in scene_trajs]    # ADE
            # iteration_expert_likeness.append(np.mean(likeness))
            idx_fig = probs.argsort()[-3:][::-1]
            # iteration_expert_likeness.append(np.min([scene_trajs[i][-1] for i in idx_fig]))
            top3_idx_backup.append(idx_fig)
            iteration_expert_likeness.append(np.min([scene_trajs[i][-1] for i in idx_fig]))
            # iteration_expert_likeness.append([scene_trajs[idx_fig[0]][-1]])

            # calculate feature expectation with respect to the weights
            traj_features = np.array([traj[1] for traj in scene_trajs])   # (36, 6)
            feature_exp += np.dot(probs, traj_features) # feature expectation (211,)x(211,5)=（5, ）  公式（10）  210+1条轨迹
            
            # calculate expert trajectory features
            expert_feature_exp += expert_traj_features[index]
            # go to next trajectory
            num_traj += 1
            index += 1
        
        # compute gradient
        grad = expert_feature_exp - feature_exp - 2*lam*theta
        grad = np.array(grad, dtype=float)

        # update weights
        if pm is None:
            pm = np.zeros_like(grad)
            pv = np.zeros_like(grad)

        pm = beta1 * pm + (1 - beta1) * grad
        pv = beta2 * pv + (1 - beta2) * (grad*grad)
        mhat = pm / (1 - beta1**(iteration+1))
        vhat = pv / (1 - beta2**(iteration+1))
        update_vec = mhat / (np.sqrt(vhat) + eps)   # Adam梯度更新 https://blog.csdn.net/qq_32172681/article/details/102568789
        theta += lr * update_vec
        print(theta)

        # add to training log
        with open(maxEnt_training_log_filepath, 'a', newline='') as csvfile:  # for K_J K_D laneid = 1
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([iteration+1, np.array(expert_feature_exp/num_traj), np.array(feature_exp/num_traj), np.linalg.norm(expert_feature_exp/num_traj - feature_exp/num_traj),
                                theta, top3_idx_backup, iteration_expert_likeness, np.mean(iteration_expert_likeness), np.sum(log_like_list)/num_traj])
            csvfile.flush()   # flush the buffered data

    print("theta =", theta)
    with open(maxEnt_theta_filepath, 'wb') as f:  # for K_J K_D laneid=0 result: theta = [-2.90662447 -3.1720357  -2.98004635 -3.08564674  6.86630762]
        pickle.dump(theta, f)

def test_data_prepare():
    # create environment
    env = RewardEnv(lane_id=lane_id)  

    # Data collection
    # test_steps = np.array([ 61,  76,  85,  82,  13,  15, 101,  65,  97, 68])
    test_steps = np.array([63, 30, 28, 87, 13, 61, 82, 68, 51, 17])  # for target_v, targert_d planner
    print(test_steps)
    test_buffer = []
    # begin planning
    pbar = tqdm(test_steps)
    for start in pbar:
        pbar.set_description('calculate features from '+ str(start))
        # go to the scene
        env.reset(reset_time=start)

        # determine target sampling space
        actions = env.sampling_space()
                
        # set up buffer of the scene
        buffer_scene = []

        # lateral and speed trajectory sampling
        # print('start time: {}, sampling...'.format(start))
        for action in actions:
            # sample a trajectory
            acc_weight, d_weight = action
            obs, features, terminated, info = env.step(action)

            # render env
            if render_env:
                env.render()
                    
            # get the features
            traj_features = features[:-1]
            expert_likeness = features[-1]

            # add the trajectory to scene buffer
            buffer_scene.append([acc_weight, d_weight, traj_features, expert_likeness])

            # go back to original scene
    
        test_buffer.append(buffer_scene)    # with open('test_buffer_1009.pkl', 'wb') as f: # for center_expert

    # with open('test_buffer_2412_0.pkl', 'wb') as f: # for K_J K_D laneid = 1
    with open('test_buffer_2412_1.pkl', 'wb') as f: # for K_J K_D laneid = 0
    # with open('test_buffer_2501_0.pkl', 'wb') as f: # for target_v target_d laneid = 1
    # with open('test_buffer_2501_1.pkl', 'wb') as f: # for target_v target_d laneid = 0
        pickle.dump(test_buffer, f)
    return test_steps

def calculate_test_top3(test_steps): 
    test_steps = test_steps

    # # create training log
    # with open('general_testing_log_1009.csv', 'w', newline='') as csvfile:  
    #     # creating a csv writer object  
    #     csvwriter = csv.writer(csvfile)  
    #     # writing the fields  
    #     csvwriter.writerow(['scene', 'expert likeness', 'weights', 'max features', 'min features', 'FDE']) 
    #     csvfile.flush()   # flush the buffered data

    # with open('test_buffer_2501_0.pkl', 'rb') as f:    # for target_v target_d laneid = 1
    # with open('test_buffer_2412_0.pkl', 'rb') as f:  # for K_J K_D laneid = 1
    # with open('buffer_2502_1.pkl', 'rb') as f:    # for target_v target_d laneid = 0
    with open('buffer_2503_0.pkl', 'rb') as f:  # for K_J K_D laneid = 1
        test_buffer = pickle.load(f)
    
    # with open('theta_2501_1.pkl', 'rb') as f:  # for target_v target_d laneid = 0
    # with open('theta_2502_1.pkl', 'rb') as f:    # for target_v target_d laneid = 0
    with open('theta_2412_0.pkl', 'rb') as f:  # for K_J K_D laneid = 1
    # with open('theta_2412_1.pkl', 'rb') as f:  # for K_J K_D laneid = 0
        theta = pickle.load(f)
    
    # normalize features
    traj_features_temp = []
    for buffer_scene in test_buffer:
        for traj in buffer_scene:
            traj_features_temp.append(traj[2])
    max_v = np.max(traj_features_temp, axis=0)
    min_v = np.min(traj_features_temp, axis=0)

    for scene in test_buffer:
        for traj in scene:
            for i in range(max_v.shape[0]):
                traj[2][i] = (traj[2][i] - min_v[i]) / (max_v[i] - min_v[i])

    
    env = RewardEnv(lane_id=lane_id)  
    step_idx = 0
    for scene in test_buffer:
        buffer_scene = scene
        start = test_steps[step_idx]
        step_idx += 1

        env.reset(reset_time=start)
        
        # evaluate trajectories
        reward_HL = []
        for trajectory in buffer_scene:
            reward = np.dot(trajectory[2][0:5], theta)  # new features
            reward_HL.append([reward, trajectory[3]]) # reward, expert likeness

        # calculate probability of each trajectory
        rewards = [traj[0] for traj in reward_HL]
        probs = [np.exp(reward) for reward in rewards]
        probs = probs / np.sum(probs)

        # select trajectories to calculate expert likeness
        idx = probs.argsort()[-3:][::-1]
        HL = np.min([reward_HL[i][-1] for i in idx])

        # add to testing log
        # with open('general_testing_log_1009.csv', 'a', newline='') as csvfile:  
        #     csvwriter = csv.writer(csvfile) 
        #     csvwriter.writerow([start, HL, theta, max_v, min_v, [reward_HL[i][-1] for i in range(len(reward_HL))]])
        #     csvfile.flush()   # flush the buffered data
        
        # plt.plot([reward_HL[i][-1] for i in idx], marker='o')
        # plt.xlabel("top3 trajectory choosed by likelihood probs")
        # plt.ylabel("expert likeness") 
        # plt.savefig(f"./ResultImages/top3_trajectory_{start}.png")
        # plt.close()

        
        # plt.plot(probs, marker='o')
        # plt.xlabel("idx of all trajectories")
        # plt.ylabel("probability") 
        # plt.savefig(f"./ResultImages/choosen_probs_{start}.png")
        # plt.close()
        # env.reset(reset_time=start)
        print('------------------------')
        print('start = ', start)
        for i in idx:
            action = (buffer_scene[i][0], buffer_scene[i][1])
            print(f'action_idx ={i}, action = [{action}]')
            # obs, features, terminated, info = env.step(action)
            # env.render(start*i)

    print("done")

def calculate_test_top3_new(test_steps): 
    test_steps = test_steps

    # with open('buffer_2503_1.pkl', 'rb') as f:    # for K_J K_D laneid = 1
    # with open('test_buffer_2501_0.pkl', 'rb') as f:  # for target_v target_d laneid = 1
    with open('buffer_2503_24.pkl', 'rb') as f:    # for K_J K_D laneid = 0
    # with open('buffer_2502_1.pkl', 'rb') as f:  # for target_v target_d laneid = 0
        test_buffer = pickle.load(f)

    # with open('theta_2503_1.pkl', 'rb') as f:    # for target_v target_d laneid = 1
    with open('theta_2503_24.pkl', 'rb') as f:  # for target_v target_d laneid = 0
    # with open('theta_2412_0.pkl', 'rb') as f:    # for K_J K_D laneid = 1
    # with open('theta_2412_1.pkl', 'rb') as f:    # for K_J K_D laneid = 0
        theta = pickle.load(f)
    
    # normalize features
    traj_features_temp = []
    for buffer_scene in test_buffer:
        for traj in buffer_scene:
            traj_features_temp.append(traj[2])
    max_v = np.max(traj_features_temp, axis=0)
    min_v = np.min(traj_features_temp, axis=0)

    for scene in test_buffer:
        for traj in scene:
            for i in range(max_v.shape[0]):
                traj[2][i] = (traj[2][i] - min_v[i]) / (max_v[i] - min_v[i])

    env = RewardEnv(lane_id=lane_id)  
    step_idx = 0
    for scene in test_buffer:
        buffer_scene = scene
        start = test_steps[step_idx]
        step_idx += 1

        env.reset(reset_time=start)
        
        # evaluate trajectories
        reward_HL = []
        for trajectory in buffer_scene:
            trajectory_feature_tmp = np.zeros(len(feature_num))
            for i in range(len(feature_num)):
                    trajectory_feature_tmp[i] = trajectory[2][feature_num[i]]
            reward = np.dot(trajectory_feature_tmp, theta)  # new features
            reward_HL.append([reward, trajectory[3]]) # reward, expert likeness

        # calculate probability of each trajectory
        rewards = [traj[0] for traj in reward_HL]
        probs = [np.exp(reward) for reward in rewards]
        probs = probs / np.sum(probs)

        # select trajectories to calculate expert likeness
        idx = probs.argsort()[-3:][::-1]
        HL = np.min([reward_HL[i][-1] for i in idx])

        # add to testing log
        # with open('general_testing_log_2412_0.csv', 'a', newline='') as csvfile: 
        # with open('general_testing_log_2501_0.csv', 'a', newline='') as csvfile: 
        with open('general_testing_log_2503_24.csv', 'a', newline='') as csvfile:  
            csvwriter = csv.writer(csvfile) 
            csvwriter.writerow([start, HL, theta, max_v, min_v, [reward_HL[i][-1] for i in range(len(reward_HL))]])
            csvfile.flush()   # flush the buffered data
        
        plt.plot([reward_HL[i][-1] for i in idx], marker='o')
        plt.xlabel("top3 trajectory choosed by likelihood probs")
        plt.ylabel("expert likeness") 
        plt.savefig(f"./ResultImages/new_top3_trajectory_{start}.png")
        plt.close()

        plt.plot(probs, marker='o')
        plt.xlabel("idx of all trajectories")
        plt.ylabel("probability") 
        plt.savefig(f"./ResultImages/new_choosen_probs_{start}.png")
        plt.close()
        env.reset(reset_time=start)
        print('------------------------')
        print('start = ', start)
        for i in idx:
            action = (buffer_scene[i][0], buffer_scene[i][1])
            print(f'action_idx ={i}, action = [{action}]')
            # obs, features, terminated, info = env.step(action)
            # env.render(start*i)

    print("done")

def paper_figure_test_top3(show_steps):
    test_steps = show_steps
    test_idx = np.array([1, 4, 5])

    # with open('test_buffer_1009.pkl', 'rb') as f:  # for center_expert
    # with open('test_buffer_1030_0.pkl', 'rb') as f:  # for 0.25_expert
    with open('test_buffer_2410_2.pkl', 'rb') as f:  # for 0.75_expert
        test_buffer = pickle.load(f)
    show_buffer = []
    for idx_i in test_idx:
        show_buffer.append(test_buffer[idx_i])

    
    # with open('theta_1009.pkl', 'rb') as f: # for center_expert
    # with open('theta_1030_0.pkl', 'rb') as f:  # for 0.25_expert
    with open('theta_2410_2.pkl', 'rb') as f:  # for 0.75_expert
        theta = pickle.load(f)  

    env = RewardEnv(lane_id=lane_id)  
    step_idx = 0
    for scene in show_buffer:
        buffer_scene = scene
        start = test_steps[step_idx]
        step_idx += 1

        #----- #
        # env.reset(reset_time=73)
        # env.render()   
        #----- # 

        env.reset(reset_time=start)
        # normalize features
        traj_features_temp = []
        for traj in buffer_scene:
            traj_features_temp.append(traj[2])
        max_v = np.max(traj_features_temp, axis=0)
        min_v = np.min(traj_features_temp, axis=0)

        for traj in buffer_scene:
            for i in range(feature_num):
                if max_v[i] == 0:
                    traj[2][i] = 0
                else:
                    traj[2][i] /= max_v[i] 
        
        # evaluate trajectories
        reward_HL = []
        for trajectory in buffer_scene:
            reward = np.dot(trajectory[2][0:5], theta)
            reward_HL.append([reward, trajectory[3]]) # reward, expert likeness

        # calculate probability of each trajectory
        rewards = [traj[0] for traj in reward_HL]
        probs = [np.exp(reward) for reward in rewards]
        probs = probs / np.sum(probs)

        # select trajectories to calculate expert likeness
        idx = probs.argsort()[-3:][::-1]
        env.reset(reset_time=start)
        for i in idx:
            action = (buffer_scene[i][0], buffer_scene[i][1])
            obs, features, terminated, info = env.step(action)
            env.render(start)

    print("done")

if __name__ == "__main__":
    train_data_prepare()   # [ 10  16  23  29  36  43  49  56  63  69  76  83  89  96 103]

    # maxEnt_IRL_newCost()   # new features

    # test_steps = test_data_prepare()  # test steps [63 30 28 87 13 61 82 68 51 17]

    # test_steps = np.array([63, 30, 28, 87, 13, 61, 82, 68, 51, 17]) 
    # calculate_test_top3(test_steps)    # 前面是[0.2, 0.8] 12.7m/s, 后面是[0.3, 0.7] 12.7m/s
    # calculate_test_top3_new(test_steps)  # 前面是[0.6, 0.4] 12.7m/s, 后面是[0.2, 0.8] 12.7m/s

    # show_steps = np.array([61, 13, 101])
    # paper_figure_test_top3(show_steps)
