import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

# fig, ax = plt.subplots(5, 2)
# fig.suptitle('my images')



for picture in glob.glob("*.png"):
    im = cv2.imread(picture, 0)
    im = np.max(im) - im
    
    l2 = np.atleast_1d(np.linalg.norm(im, ord=2))
    # l2[l2 == 0] = 1
    im = im / np.expand_dims(l2, 1)
    
# fig, axs = plt.subplots(5, 2)
# fig.suptitle('My Images')
# for i in range(10):
#     axs[i].imshow("{i}.png", cmap='gray')

fig = plt.figure()
for i in range(1, 11):
    ax1 = fig.add_subplot(2,2,i)
    k = i-1
    ax1.imshow("{k}.png", cmap='gray')
    