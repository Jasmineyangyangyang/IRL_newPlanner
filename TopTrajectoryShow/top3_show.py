import numpy as np
import csv
from tqdm import tqdm
import sys
sys.path.append("/home/jx/Yangjiaxin/IRL")
sys.path.append("/home/jx/Yangjiaxin/IRL/IRL_env/envs")
from IRL_env.envs.irlenv import RewardEnv
import pickle
import matplotlib.pyplot as plt

# parameters
n_iters = 200
lr = 0.05
lam = 0.01
feature_num = 6
lane_id = 1     # 0 means outside center trajectory, 1 means inside center trajectory
render_env = False # to show sample trajectories(black) expert trajectory(red) and top3 optimal reward trajectory(blue) 


# load pickle------------------------------###
# -----------------------------------------###
# -----------------------------------------###
# -----------------------------------------###
# with open('buffer.pkl', 'rb') as f:
#     buffer = pickle.load(f)
# expert_traj_features = []
# for buffer_scene in buffer:
#     expert_traj_features.append(buffer_scene[-1][2])

# # normalize features
# traj_features_temp = []
# for buffer_scene in buffer:
#     for traj in buffer_scene:
#         traj_features_temp.append(traj[2])
# max_v = np.max(traj_features_temp, axis=0)
# min_v = np.min(traj_features_temp, axis=0)
# # # normalize features
# # max_v = np.max([traj[2] for traj in buffer_scene for buffer_scene in buffer], axis=0)
# # min_v = np.min([traj[2] for traj in buffer_scene for buffer_scene in buffer], axis=0)

# for scene in buffer:
#     for traj in scene:
#         for i in range(feature_num):
#             traj[2][i] /= max_v[i]

# #### MaxEnt IRL ####
# print("Start training...")
# # initialize weights
# theta = np.random.normal(0, 0.05, size=feature_num)

# # iterations
# beta1 = 0.9; beta2 = 0.999; eps = 1e-8
# pm = None
# pv = None
# grad_log = []
# expert_likeness_log = []

# pbar = tqdm(range(n_iters))
# for iteration in pbar:
#     pbar.set_description('iteration '+ str(iteration))

#     feature_exp = np.zeros([feature_num])
#     expert_feature_exp = np.zeros([feature_num])

#     log_like_list = []
#     iteration_expert_likeness = []
#     num_traj = 0
#     index = 0

#     for scene in buffer:
#         # compute on each scene
#         scene_trajs = []
#         for trajectory in scene:  # every scene has 36 trajectories, the last one is expert trajectory
#             reward = np.dot(trajectory[2], theta)
#             scene_trajs.append((reward, trajectory[2], trajectory[3])) # reward, feature vector, expert likeness

#         # calculate probability of each trajectory
#         rewards = [traj[0] for traj in scene_trajs] 
#         probs = [np.exp(reward) for reward in rewards]  # list长36
#         probs = probs / np.sum(probs)                   # list 长36  (36,)  
#         log_like_list.append(probs[-1])                 # the last teajectory is expert trajectory

#         # stroe expert likeness
#         likeness = [traj[2] for traj in scene_trajs] 
#         iteration_expert_likeness.append(np.mean(likeness))

#         # calculate feature expectation with respect to the weights
#         traj_features = np.array([traj[1] for traj in scene_trajs])   # (36, 6)
#         feature_exp += np.dot(probs, traj_features) # feature expectation (36,)x(36,6)=（1, 6）  公式（10）  35+1条轨迹
        
#         # calculate expert trajectory features
#         expert_feature_exp += expert_traj_features[index]

#         # go to next trajectory
#         num_traj += 1
#         index += 1
    
#     # compute gradient
#     grad = expert_feature_exp - feature_exp - 2*lam*theta
#     grad = np.array(grad, dtype=float)

#     # update weights
#     if pm is None:
#         pm = np.zeros_like(grad)
#         pv = np.zeros_like(grad)

#     pm = beta1 * pm + (1 - beta1) * grad
#     pv = beta2 * pv + (1 - beta2) * (grad*grad)
#     mhat = pm / (1 - beta1**(iteration+1))
#     vhat = pv / (1 - beta2**(iteration+1))
#     update_vec = mhat / (np.sqrt(vhat) + eps)   # Adam梯度更新 https://blog.csdn.net/qq_32172681/article/details/102568789
#     theta += lr * update_vec



# # with open('theta.pkl', 'wb') as f:
# #     pickle.dump(theta, f)

# # load theta---------------------#
# # -------------------------------###
# # -------------------------------###
# # -------------------------------###
# with open('theta.pkl', 'rb') as f:
#     theta = pickle.load(f)  
# #### run test ####


# # create environment
# env = RewardEnv(lane_id=lane_id)  

# # Data collection
# # length = len(env.ego_trajectory)
# # timesteps = np.linspace(10, length-250, num=50, dtype=np.int16)
# # test_steps = np.random.choice(timesteps, size=10, replace=False) # replace means can choose one repeat
# test_steps = np.array([0, 250, 500])
# show_buffer = []
# # begin planning
# pbar = tqdm(test_steps)
# for start in pbar:
#     pbar.set_description('calculate features from '+ str(start))
#     # go to the scene
#     env.reset(reset_time=start)

#     # determine target sampling space
#     lateral_offsets, target_speeds = env.sampling_space()
            
#     # set up buffer of the scene
#     buffer_scene = []

#     # lateral and speed trajectory sampling
#     # print('start time: {}, sampling...'.format(start))
#     for lateral in lateral_offsets:
#         for target_speed in target_speeds:
#             # sample a trajectory
#             action = (lateral, target_speed, 5)
#             obs, features, terminated, info = env.step(action)

#             # render env
#             if render_env:
#                 env.render()
                    
#             # get the features
#             traj_features = features[:-1]
#             expert_likeness = features[-1]

#             # add the trajectory to scene buffer
#             buffer_scene.append([lateral, target_speed, traj_features, expert_likeness])

#             # go back to original scene
#             env.reset(reset_time=start)

#     show_buffer.append(buffer_scene)
# # save buffer
# with open('show_buffer.pkl', 'wb') as f:
#     pickle.dump(show_buffer, f)


# # load theta---------------------#
# # -------------------------------###
# # -------------------------------###
# # -------------------------------###
with open('theta.pkl', 'rb') as f:
    theta = pickle.load(f)  

with open('show_buffer.pkl', 'rb') as f:
    show_buffer = pickle.load(f)

top3_idx = []
test_steps = np.array([0, 250, 500])
step_idx = 0
env = RewardEnv(lane_id=lane_id)  
for scene in show_buffer:

    buffer_scene = scene

    start = test_steps[step_idx]
    step_idx += 1
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
                traj[2][i] /= max_v[i] + 1e-5
    
    # evaluate trajectories
    reward_HL = []
    for trajectory in buffer_scene:
        reward = np.dot(trajectory[2], theta)
        reward_HL.append([reward, trajectory[3]]) # reward, expert likeness

    # calculate probability of each trajectory
    rewards = [traj[0] for traj in reward_HL]
    probs = [np.exp(reward) for reward in rewards]
    probs = probs / np.sum(probs)

    # select trajectories to calculate expert likeness
    idx = probs.argsort()[-3:][::-1]
    top3_idx.append(idx)
    HL = np.min([reward_HL[i][-1] for i in idx])

    
    plt.plot([reward_HL[i][-1] for i in idx], marker='o')
    plt.xlabel("top3 trajectory choosed by likelihood probs")
    plt.ylabel("expert likeness") 
    # plt.savefig(f"./ResultImages/top3_trajectory_{start}.png")
    plt.savefig(f"./showImages/top3_trajectory_{start}.png")

    plt.plot(probs, marker='o')
    plt.xlabel("idx of all trajectories")
    plt.ylabel("probability") 
    # plt.savefig(f"./ResultImages/choosen_probs_{start}.png")
    plt.savefig(f"./showImages/choosen_probs_{start}.png")

    env.reset(reset_time=start)
    for i in idx:
        action = (buffer_scene[i][0], buffer_scene[i][1], 5)
        obs, features, terminated, info = env.step(action)
        env.render(start*i+6330)
with open('top3_idx.pkl', 'wb') as f:
    pickle.dump(top3_idx, f)


# print("done")

# 在planner中绘制idx图像吧