# Sicheng He hschsc@umich.edu
# 01/24/2018
# truss result visualization 
# area/stress/buckling

from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import copy
import utils as utils

# =================
# get data
# =================

filename_node = 'INPUT/data_nodes.dat'
filename_elem = 'INPUT/data_elems.dat'
filename_sol = 'INPUT/solution_315.dat'
filename_con = 'INPUT/data_constraints.dat'
ind_beg_elem = 7
ind_end_elem = 321
ind_beg_node = 327
ind_end_node = 548

length_scale = 100
yield_stress = 2.72

nodes, nodes_d, elems, cons, areas, rel_stress, bucklings = utils.preprocess(filename_node, filename_elem, filename_sol, filename_con,\
ind_beg_elem, ind_end_elem, ind_beg_node, ind_end_node, \
length_scale, yield_stress)

N_elem = len(areas)

# =================
# creating a block to give equal length axes
# =================
x_center = (max(nodes_d[:, 0]) + min(nodes_d[:, 0])) / 2
x_radius = (max(nodes_d[:, 0]) - min(nodes_d[:, 0])) / 2
y_center = (max(nodes_d[:, 1]) + min(nodes_d[:, 1])) / 2
y_radius = (max(nodes_d[:, 1]) - min(nodes_d[:, 1])) / 2
z_center = (max(nodes_d[:, 2]) + min(nodes_d[:, 2])) / 2
z_radius = (max(nodes_d[:, 2]) - min(nodes_d[:, 2])) / 2

alpha = 1.3
x_lb = x_center - x_radius * alpha
x_ub = x_center + x_radius * alpha
y_lb = y_center - y_radius * alpha
y_ub = y_center + y_radius * alpha
z_lb = z_center - z_radius * alpha
z_ub = z_center + z_radius * alpha

x_lb = min(min(x_lb, y_lb), z_lb)
x_ub = max(max(x_ub, y_ub), z_ub)
y_lb = x_lb
z_lb = x_lb
y_ub = x_ub
z_ub = x_ub






# =================
# plot stress
# =================

coolwarm_cmap = matplotlib.cm.get_cmap('coolwarm')
norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)

# view 1
fig = plt.figure(figsize=(16, 16))
ax = fig.add_subplot(111, projection='3d', proj_type = 'ortho')
# ax.set_aspect('equal')
ax.pbaspect = [1,1,1]

ax.set_xlim([0, 30])
ax.set_ylim([0, 30])
ax.set_zlim([-15, 15])
ax.grid(False)
ax._axis3don = False

# jig shape
for i in range(N_elem):

    ind1 = elems[i, 0]
    ind2 = elems[i, 1]

    xyz1 = nodes[ind1, :]
    xyz2 = nodes[ind2, :]

    r_loc = np.sqrt(areas[i]/np.pi)

    color = [192.0/255, 192.0/255, 192.0/255]
    utils.plotRod(xyz1, xyz2, r_loc, color, ax)

# deformed
for i in range(N_elem):

    ind1 = elems[i, 0]
    ind2 = elems[i, 1]

    xyz1 = nodes_d[ind1, :]
    xyz2 = nodes_d[ind2, :]

    r_loc = np.sqrt(areas[i]/np.pi)

    rgb_loc = matplotlib.colors.colorConverter.to_rgb(coolwarm_cmap(norm(rel_stress[i])))
    color = rgb_loc
    utils.plotRod(xyz1, xyz2, r_loc, color, ax)

# BC
for ind in cons:

    x_loc, y_loc, z_loc = nodes_d[ind, :]

    ax.plot([x_loc, x_loc], [y_loc, y_loc], [z_loc, z_loc], 'ko', markersize = 6)

ax.plot([x_lb, x_lb, x_lb, x_lb, x_ub, x_ub, x_ub, x_ub],
[y_lb, y_lb, y_ub, y_ub, y_ub, y_ub, y_lb, y_lb],
[z_lb, z_ub, z_ub, z_lb, z_lb, z_ub, z_ub, z_lb], alpha = 0)

ax.view_init(elev=90, azim=0)

fig.savefig("wing_stress_1.pdf", bbox_inches='tight') 

# view 2
ax = fig.add_subplot(111, projection='3d', proj_type = 'ortho')
# ax.set_aspect('equal')
ax.pbaspect = [1,1,1]

ax.set_xlim([0, 30])
ax.set_ylim([0, 30])
ax.set_zlim([-15, 15])
ax.grid(False)
ax._axis3don = False

# jig shape
for i in range(N_elem):

    ind1 = elems[i, 0]
    ind2 = elems[i, 1]

    xyz1 = nodes[ind1, :]
    xyz2 = nodes[ind2, :]

    r_loc = np.sqrt(areas[i]/np.pi)

    color = [192.0/255, 192.0/255, 192.0/255]
    utils.plotRod(xyz1, xyz2, r_loc, color, ax)

# deformed
for i in range(N_elem):

    ind1 = elems[i, 0]
    ind2 = elems[i, 1]

    xyz1 = nodes_d[ind1, :]
    xyz2 = nodes_d[ind2, :]

    r_loc = np.sqrt(areas[i]/np.pi)

    rgb_loc = matplotlib.colors.colorConverter.to_rgb(coolwarm_cmap(norm(rel_stress[i])))
    color = rgb_loc
    utils.plotRod(xyz1, xyz2, r_loc, color, ax)

# BC
for ind in cons:

    x_loc, y_loc, z_loc = nodes_d[ind, :]

    ax.plot([x_loc, x_loc], [y_loc, y_loc], [z_loc, z_loc], 'ko', markersize = 6)

ax.plot([x_lb, x_lb, x_lb, x_lb, x_ub, x_ub, x_ub, x_ub],
[y_lb, y_lb, y_ub, y_ub, y_ub, y_ub, y_lb, y_lb],
[z_lb, z_ub, z_ub, z_lb, z_lb, z_ub, z_ub, z_lb], alpha = 0)

ax.view_init(elev=0, azim=0)

fig.savefig("wing_stress_2.pdf", bbox_inches='tight') 

# view 3
ax = fig.add_subplot(111, projection='3d', proj_type = 'ortho')
# ax.set_aspect('equal')
ax.pbaspect = [1,1,1]

ax.set_xlim([0, 30])
ax.set_ylim([0, 30])
ax.set_zlim([-15, 15])
ax.grid(False)
ax._axis3don = False
ax.view_init(elev=30, azim=-90)

# jig shape
for i in range(N_elem):

    ind1 = elems[i, 0]
    ind2 = elems[i, 1]

    xyz1 = nodes[ind1, :]
    xyz2 = nodes[ind2, :]

    r_loc = np.sqrt(areas[i]/np.pi)

    color = [192.0/255, 192.0/255, 192.0/255]
    utils.plotRod(xyz1, xyz2, r_loc, color, ax)

# deformed
for i in range(N_elem):

    ind1 = elems[i, 0]
    ind2 = elems[i, 1]

    xyz1 = nodes_d[ind1, :]
    xyz2 = nodes_d[ind2, :]

    r_loc = np.sqrt(areas[i]/np.pi)

    rgb_loc = matplotlib.colors.colorConverter.to_rgb(coolwarm_cmap(norm(rel_stress[i])))
    color = rgb_loc
    utils.plotRod(xyz1, xyz2, r_loc, color, ax)

# BC
for ind in cons:

    x_loc, y_loc, z_loc = nodes_d[ind, :]

    ax.plot([x_loc, x_loc], [y_loc, y_loc], [z_loc, z_loc], 'ko', markersize = 6)

ax.plot([x_lb, x_lb, x_lb, x_lb, x_ub, x_ub, x_ub, x_ub],
[y_lb, y_lb, y_ub, y_ub, y_ub, y_ub, y_lb, y_lb],
[z_lb, z_ub, z_ub, z_lb, z_lb, z_ub, z_ub, z_lb], alpha = 0)

ax.view_init(elev=30, azim=-90)

fig.savefig("wing_stress_3.pdf", bbox_inches='tight') 





# =================
# plot buckling
# =================
# view 1
fig = plt.figure(figsize=(16, 16))
ax = fig.add_subplot(111, projection='3d', proj_type = 'ortho')
# ax.set_aspect('equal')
ax.pbaspect = [1,1,1]

ax.set_xlim([0, 30])
ax.set_ylim([0, 30])
ax.set_zlim([-15, 15])
ax.grid(False)
ax._axis3don = False

# jig shape
for i in range(N_elem):

    ind1 = elems[i, 0]
    ind2 = elems[i, 1]

    xyz1 = nodes[ind1, :]
    xyz2 = nodes[ind2, :]

    r_loc = np.sqrt(areas[i]/np.pi)

    color = [192.0/255, 192.0/255, 192.0/255]
    utils.plotRod(xyz1, xyz2, r_loc, color, ax)

# deformed
for i in range(N_elem):

    ind1 = elems[i, 0]
    ind2 = elems[i, 1]

    xyz1 = nodes_d[ind1, :]
    xyz2 = nodes_d[ind2, :]

    r_loc = np.sqrt(areas[i]/np.pi)

    rgb_loc = matplotlib.colors.colorConverter.to_rgb(coolwarm_cmap(norm(bucklings[i])))
    color = rgb_loc
    utils.plotRod(xyz1, xyz2, r_loc, color, ax)

# BC
for ind in cons:

    x_loc, y_loc, z_loc = nodes_d[ind, :]

    ax.plot([x_loc, x_loc], [y_loc, y_loc], [z_loc, z_loc], 'ko', markersize = 6)

ax.plot([x_lb, x_lb, x_lb, x_lb, x_ub, x_ub, x_ub, x_ub],
[y_lb, y_lb, y_ub, y_ub, y_ub, y_ub, y_lb, y_lb],
[z_lb, z_ub, z_ub, z_lb, z_lb, z_ub, z_ub, z_lb], alpha = 0)

ax.view_init(elev=90, azim=0)
fig.savefig("wing_buckling_1.pdf", bbox_inches='tight') 


# view 2
fig = plt.figure(figsize=(16, 16))
ax = fig.add_subplot(111, projection='3d', proj_type = 'ortho')
# ax.set_aspect('equal')
ax.pbaspect = [1,1,1]

ax.set_xlim([0, 30])
ax.set_ylim([0, 30])
ax.set_zlim([-15, 15])
ax.grid(False)
ax._axis3don = False

# jig shape
for i in range(N_elem):

    ind1 = elems[i, 0]
    ind2 = elems[i, 1]

    xyz1 = nodes[ind1, :]
    xyz2 = nodes[ind2, :]

    r_loc = np.sqrt(areas[i]/np.pi)

    color = [192.0/255, 192.0/255, 192.0/255]
    utils.plotRod(xyz1, xyz2, r_loc, color, ax)

# deformed
for i in range(N_elem):

    ind1 = elems[i, 0]
    ind2 = elems[i, 1]

    xyz1 = nodes_d[ind1, :]
    xyz2 = nodes_d[ind2, :]

    r_loc = np.sqrt(areas[i]/np.pi)

    rgb_loc = matplotlib.colors.colorConverter.to_rgb(coolwarm_cmap(norm(bucklings[i])))
    color = rgb_loc
    utils.plotRod(xyz1, xyz2, r_loc, color, ax)

# BC
for ind in cons:

    x_loc, y_loc, z_loc = nodes_d[ind, :]

    ax.plot([x_loc, x_loc], [y_loc, y_loc], [z_loc, z_loc], 'ko', markersize = 6)

ax.plot([x_lb, x_lb, x_lb, x_lb, x_ub, x_ub, x_ub, x_ub],
[y_lb, y_lb, y_ub, y_ub, y_ub, y_ub, y_lb, y_lb],
[z_lb, z_ub, z_ub, z_lb, z_lb, z_ub, z_ub, z_lb], alpha = 0)

ax.view_init(elev=0, azim=0)
fig.savefig("wing_buckling_2.pdf", bbox_inches='tight') 


# view 3
fig = plt.figure(figsize=(16, 16))
ax = fig.add_subplot(111, projection='3d', proj_type = 'ortho')
# ax.set_aspect('equal')
ax.pbaspect = [1,1,1]

ax.set_xlim([0, 30])
ax.set_ylim([0, 30])
ax.set_zlim([-15, 15])
ax.grid(False)
ax._axis3don = False

# jig shape
for i in range(N_elem):

    ind1 = elems[i, 0]
    ind2 = elems[i, 1]

    xyz1 = nodes[ind1, :]
    xyz2 = nodes[ind2, :]

    r_loc = np.sqrt(areas[i]/np.pi)

    color = [192.0/255, 192.0/255, 192.0/255]
    utils.plotRod(xyz1, xyz2, r_loc, color, ax)

# deformed
for i in range(N_elem):

    ind1 = elems[i, 0]
    ind2 = elems[i, 1]

    xyz1 = nodes_d[ind1, :]
    xyz2 = nodes_d[ind2, :]

    r_loc = np.sqrt(areas[i]/np.pi)

    rgb_loc = matplotlib.colors.colorConverter.to_rgb(coolwarm_cmap(norm(bucklings[i])))
    color = rgb_loc
    utils.plotRod(xyz1, xyz2, r_loc, color, ax)

# BC
for ind in cons:

    x_loc, y_loc, z_loc = nodes_d[ind, :]

    ax.plot([x_loc, x_loc], [y_loc, y_loc], [z_loc, z_loc], 'ko', markersize = 6)

ax.plot([x_lb, x_lb, x_lb, x_lb, x_ub, x_ub, x_ub, x_ub],
[y_lb, y_lb, y_ub, y_ub, y_ub, y_ub, y_lb, y_lb],
[z_lb, z_ub, z_ub, z_lb, z_lb, z_ub, z_ub, z_lb], alpha = 0)

ax.view_init(elev=30, azim=-90)
fig.savefig("wing_buckling_3.pdf", bbox_inches='tight') 