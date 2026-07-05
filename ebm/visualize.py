import numpy as np
import matplotlib.pyplot as plt

def imshow(img):
    # Normalize the image to [0, 1] for display
    img = img / 2 + 0.5  # unnormalize if images were in [-1, 1]
    npimg = img.numpy()
    # Transpose dimensions from (C, H, W) to (H, W, C) for matplotlib
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off') # Hide axes
