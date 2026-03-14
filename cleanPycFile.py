import os
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.pyc'):
            os.remove(os.path.join(root, file))

# #------------------------------------------------------# #
# import matplotlib.pyplot as plt
# x = list(range(1, 21))  # epoch array
# loss = [2 / (i**2) for i in x]  # loss values array
# plt.ion()
# for i in range(1, len(x)):
#     ix = x[:i]
#     iy = loss[:i]
#     plt.cla()
#     plt.plot(ix, iy)
#     plt.title("loss")
#     plt.xlabel("epoch")
#     plt.ylabel("loss")
#     plt.pause(0.5)
# plt.ioff()
# plt.show()


# #------------------------------------------------------# #
# Python code explaining 
# numpy.polyder()

# # importing libraries
# import numpy as np
# import pandas as pd

# # Constructing polynomial 
# p1 = np.poly1d([1, 2])
# p2 = np.poly1d([4, 9, 5, 4])

# print ("P1 : ", p1)
# print ("\n p2 : \n", p2)


# # Solve for x = 2 
# print ("\n\np1 at x = 2 : ", p1(2))
# print ("p2 at x = 2 : ", p2(2))

# a = np.polyder(p1, 1)
# b = np.polyder(p2, 1)
# print ("\n\nUsing polyder")
# print ("p1 derivative of order = 1 : \n", a)
# print ("p2 derivative of order = 1 : \n", b)

# a = np.polyder(p1, 2)
# b = np.polyder(p2, 2)
# print ("\n\nUsing polyder")
# print ("p1 derivative of order = 2 : ", a)
# print ("p2 derivative of order = 2 : ", b)
