import sys
sys.path.append("/home/jx/Yangjiaxin/IRL")
sys.path.append("/home/jx/Yangjiaxin/IRL/IRL_env/envs")
import numpy as np
import copy
import pickle
from IRL_env.envs.irlenv import RewardEnv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def cartesian_show():
    with open('top3_idx.pkl', 'rb') as f:
        top3_idx = pickle.load(f)

    test_steps = np.array([0, 250, 500])
    step_idx = 0
    env = RewardEnv(lane_id=1)  
    env.reset(reset_time=0)

    plt.figure(figsize=(50,30))
    ax = plt.axes()
    ax.set_facecolor("grey")

    plt.plot(env.road[:, 0], env.road[:, 1], 'k', label='road')
    plt.plot(env.road[:, 2], env.road[:, 3], 'k')
    plt.plot(env.road[:, 4], env.road[:, 5], 'w--')
    plt.xlabel('Global X/m')
    plt.ylabel('Global Y/m')
    plt.axis('equal')

    lateral_offsets, target_speeds = env.sampling_space()
    actions = []
    for lateral in lateral_offsets:
        for target_speed in target_speeds:
            # sample a trajectory
            action = (lateral, target_speed, 5)
            actions.append(action)

    start_num = 0
    for start in test_steps:
        env.reset(reset_time=start)
        top3_idx_num = top3_idx[start_num]
        

        action_num = 0
        if start_num == 1:   # you can use "1/2/3" to get diffenrent part of the planned road
            for action in actions:
                env.step(action)  # you need to comment out some line first
                # obs, features, terminated, info = env.step(action)   
                
                if action_num == top3_idx_num[0]:
                    plt.plot(env.feature_trajectory[:, 0], env.feature_trajectory[:, 1], color='lime', linewidth=1.5, marker='*', label='top1 trajectory')
                    plt.plot(env.obstacle_trajectory[:, 0], env.obstacle_trajectory[:, 1], color='m', label='obstacle trajectory')
                elif action_num == top3_idx_num[1]:
                    print('top2')
                    # plt.plot(env.feature_trajectory[:, 0], env.feature_trajectory[:, 1], color='lime', linewidth=0.2, marker='o', label='top2 trajectory')
                elif action_num == top3_idx_num[2]:
                    print('top3')
                    # plt.plot(env.feature_trajectory[:, 0], env.feature_trajectory[:, 1], color='lime', linewidth=0.2, marker='o', label='top3 trajectory')
                else:
                    plt.plot(env.feature_trajectory[:, 0], env.feature_trajectory[:, 1], color='g', linewidth=0.5)

                action_num += 1

        start_num += 1
    plt.legend()
    plt.savefig('planner_show_Images/planner_show_1.png')
    plt.show()
    
def frenet_show():
    with open('top3_idx.pkl', 'rb') as f:
        top3_idx = pickle.load(f)

    test_steps = np.array([0, 250, 500])
    step_idx = 0
    env = RewardEnv(lane_id=1)  
    env.reset(reset_time=0)
    lateral_offsets, target_speeds = env.sampling_space()
    actions = []
    for lateral in lateral_offsets:
        for target_speed in target_speeds:
            # sample a trajectory
            action = (lateral, target_speed, 5)
            actions.append(action)

    plt.figure(figsize=(50,30))
    ax = plt.axes()
    ax.set_facecolor("grey")
    start_num = 0
    for start in test_steps:
        env.reset(reset_time=start)
        top3_idx_num = top3_idx[start_num]
        

        action_num = 0
        if start_num == 2:   # you can use "1/2/3" to get diffenrent part of the planned road
            for action in actions:
                env.step(action)  # you need to comment out some line first
                # obs, features, terminated, info = env.step(action)   
                if action_num == 0:
                    st = int(env.feature_trajectory_frenet[0, 0])
                    plt.plot(np.linspace(st, st+65/3.6*5, 100), np.ones(100)*2.0, 'w', linewidth=2, label='lane')
                    plt.plot(np.linspace(st, st+65/3.6*5, 100), np.zeros(100), 'w--', linewidth=2)
                    plt.plot(np.linspace(st, st+65/3.6*5, 100), np.ones(100)*-2.0, 'w', linewidth=2)

                    plt.plot(np.linspace(st, st+65/3.6*5, 100), np.ones(100)*0.96, '--', color='b',linewidth=1.5, label='safe boundary')
                    plt.plot(np.linspace(st, st+65/3.6*5, 100), np.ones(100)*-0.96, '--', color='b', linewidth=1.5)
                elif action_num == top3_idx_num[0]:
                    plt.plot(env.feature_trajectory_frenet[:, 0], env.feature_trajectory_frenet[:, 1], color='lime', linewidth=1.5, marker='*', label='top1 trajectory')
                elif action_num == top3_idx_num[1]:
                    print('top2')
                    # plt.plot(env.feature_trajectory_frenet[:, 0], env.feature_trajectory_frenet[:, 1], color='lime', linewidth=0.2, marker='o', label='top2 trajectory')
                elif action_num == top3_idx_num[2]:
                    print('top3')
                    # plt.plot(env.feature_trajectory_frenet[:, 0], env.feature_trajectory_frenet[:, 1], color='lime', linewidth=0.2, marker='o', label='top3 trajectory')
                else:
                    plt.plot(env.feature_trajectory_frenet[:, 0], env.feature_trajectory_frenet[:, 1], color='g', linewidth=0.5)

                action_num += 1

        start_num += 1

    plt.xlabel('Frenet S/m')
    plt.ylabel('Offset of single lane d/m')
    plt.legend()
    plt.savefig('planner_show_Images/planner_show_frenet_2.png')
    plt.show()

if __name__ == "__main__":
    # cartesian_show()
    frenet_show()




