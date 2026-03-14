import pickle
import matplotlib
import matplotlib.pyplot as plt



def theta_show():
    with open('theta.pkl', 'rb') as f:
        theta = pickle.load(f)  
    
    x_labels = ['mean(v)', 'max(a_longi)', 'max(jerk_long)', 'max(a_lateral)', 'exp(-STLC)', 'exp(-CTTC)']

    # fig, ax = plt.subplots()
    # ax.bar(x_labels, theta)
    # ax.set_xlabel('feature name', labelpad=15)
    # ax.set_ylabel('factor value')
    # ax.set_title('theta of local trajectory (T=5s)')


    # for i in range(len(theta)):
    #     ax.text(i, theta[i], str(theta[i]), ha='center', va='bottom')

    # ax.plot(x_labels, theta, '-o', color='black')
    # plt.savefig('theta.png')

    plt.show()





if __name__ == '__main__':
    theta_show()

