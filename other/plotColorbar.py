# Sicheng He hschsc@umich.edu
# 06/17/2019
# truss result colorbar 
# area/stress/buckling

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import copy
from matplotlib import rc
import matplotlib.colors as plt_colors
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

Output_directory = './visualization/'

# stress
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111)
cm = matplotlib.cm.get_cmap('coolwarm_r')
xy = range(1)
z = xy

sc = plt.scatter(xy, xy, c=z, vmin=-1, vmax=1, s=35, cmap=cm)

cb = plt.colorbar(sc, ticks=[-1, 0, 1], orientation = 'horizontal')

cb.ax.set_xticklabels([r'$-\sigma_Y$', r'$0$', r'$\sigma_Y$'])  # horizontal colorbar

cb.ax.set_xlabel('stress')

plt.savefig(Output_directory + 'stress_bar.pdf', bbox_inches='tight') 



# buckling
import matplotlib.colors as mcolors
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
         'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
         cmap(np.linspace(minval, maxval, n)))
    return new_cmap

fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111)
cm = matplotlib.cm.get_cmap('coolwarm')
cm = truncate_colormap(cm, minval=0.5, maxval=1.0, n=-1)
xy = range(1)
z = xy
sc = plt.scatter(xy, xy, c=z, vmin=0, vmax=1, s=35, cmap=cm)

cb = plt.colorbar(sc, ticks=[0, 1], orientation = 'horizontal')

cb.ax.set_xticklabels([r'$\frac{\sigma}{-\gamma x}=0$',  r'$\frac{\sigma}{-\gamma x}=1$'])  # horizontal colorbar

cb.ax.set_xlabel('buckling')

plt.savefig(Output_directory + 'buckling_bar.pdf', bbox_inches='tight') 
