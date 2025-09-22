from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import numpy as np
import matplotlib.pyplot as plt
A = np.array([3, 2, 3, 2, 6, 10, 10, 8, 5, 3, 6, 7, 6, 4, 1, 0], dtype=float)
B = np.array([4, 6, 12, 11, 9, 7, 6, 3, 4, 4, 7, 8, 8, 5, 1, 0], dtype=float)
d, paths = dtw.warping_paths(A, B, penalty=1.5, psi=0)
best_path = dtw.best_path(paths)
dtwvis.plot_warpingpaths(A, B, paths, best_path)
plt.show()