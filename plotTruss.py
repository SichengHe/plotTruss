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


coolwarm_cmap = matplotlib.cm.get_cmap('coolwarm')
norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)

nodes = np.loadtxt('./Wings/W_315/data_nodes.dat')
elems = np.loadtxt('./Wings/W_315/data_elems.dat').astype(int)
nodes_d = copy.deepcopy(nodes)

N_elem, __ = elems.shape

areas = np.zeros(N_elem)
stress = np.zeros(N_elem)
bucklings = np.zeros(N_elem)
yield_stress = 2.72

i = 0
j = 0
with open('solution_315') as f:
    for line in f:
        if (7<=j and j<=321):
            line_split = line.split()

            areas_loc = float(line_split[1])/10000 # m^2
            stress_loc = float(line_split[4]) 
            buckling_loc = float(line_split[5]) 

            if buckling_loc>0:
                buckling_loc = 0.0
            else:
                buckling_loc = -buckling_loc

            areas[i] = areas_loc
            stress[i] = stress_loc
            bucklings[i] =  buckling_loc

            print(line_split)

            i += 1
            
        j += 1

i = 0
j = 0

dys = np.zeros(548 - 327 + 1)
with open('solution_315') as f:
    for line in f:
        if (327<=j and j<=548):
            line_split = line.split()

            dys[i] = float(line_split[1])

            print(line_split)

            i += 1
            
        j += 1

dys /= 100
dys = dys.reshape((-1, 3))
nodes_d = nodes + dys


abs_stress = abs(stress)
rel_abs_stress = abs_stress/yield_stress
rel_stress = -stress/yield_stress


def plotRod(xyz1, xyz2, r, color, ax):

    # plot one rod
    dphi = np.pi/3 # use 6 triangular pyramid
    N = int(np.pi*2 / dphi)

    # get unit rod dir vec, v: (dx, dy, dz)
    x1 = xyz1[0]
    y1 = xyz1[1]
    z1 = xyz1[2]

    x2 = xyz2[0]
    y2 = xyz2[1]
    z2 = xyz2[2]

    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    dl = np.sqrt(dx**2 + dy**2 + dz**2)

    dx = dx/dl
    dy = dy/dl
    dz = dz/dl

    # perpendicular to v pick a line v_list[0]: (dx1, dy1, dz1)
    v_list = []
    if dz !=0:
        dx1 = 0
        dy1 = np.sqrt(dz**2/(dz**2 + dy**2))
        dz1 = -dy*dy1/dz

        v_list.append(np.array([dx1, dy1, dz1]))
    else:
        print('ERROR: Divide by zero -- dz=0!')
        exit()
        

    # Euler-Rodrigues formula
    a = np.cos(dphi/2)
    b = dx * np.sin(dphi/2)
    c = dy * np.sin(dphi/2)
    d = dz * np.sin(dphi/2)

    ER_mat = np.zeros((3, 3))

    ER_mat[0, 0] = a**2 + b**2 - c**2 - d**2
    ER_mat[0, 1] = 2*(b*c - a*d)
    ER_mat[0, 2] = 2*(b*d + a*c)
    ER_mat[1, 0] = 2*(b*c + a*d)
    ER_mat[1, 1] = a**2 + c**2 - b**2 -d**2
    ER_mat[1, 2] = 2*(c*d - a*b)
    ER_mat[2, 0] = 2*(b*d - a*c)
    ER_mat[2, 1] = 2*(c*d + a*b)
    ER_mat[2, 2] = a**2 + d**2 - b**2 - c**2

    # generate the vecs
    for i in xrange(N - 1):

        v_loc = v_list[i]
        v_list.append(ER_mat.dot(v_loc))

    # for convinience ...
    v_list.append(copy.deepcopy(v_list[0]))

    # scale to r
    for i in xrange(len(v_list)):
        v_list[i] *= r

    for i in xrange(len(v_list)):
        print(np.linalg.norm(v_list[i]))
 
    # plot
    # disk 1:
    for i in xrange(N):

        tri_x1 = x1
        tri_y1 = y1
        tri_z1 = z1

        tri_x2 = tri_x1 + v_list[i][0]
        tri_y2 = tri_y1 + v_list[i][1]
        tri_z2 = tri_z1 + v_list[i][2]

        tri_x3 = tri_x1 + v_list[i + 1][0]
        tri_y3 = tri_y1 + v_list[i + 1][1]
        tri_z3 = tri_z1 + v_list[i + 1][2]

        tri_x = [tri_x1, tri_x2, tri_x3]
        tri_y = [tri_y1, tri_y2, tri_y3]
        tri_z = [tri_z1, tri_z2, tri_z3]

        verts = [zip(tri_x, tri_y, tri_z)]

        poly3d = Poly3DCollection(verts)
        poly3d.set_facecolor(color)
        ax.add_collection3d(poly3d)

    # disk 2:
    for i in xrange(N):

        tri_x1 = x2
        tri_y1 = y2
        tri_z1 = z2

        tri_x2 = tri_x1 + v_list[i][0]
        tri_y2 = tri_y1 + v_list[i][1]
        tri_z2 = tri_z1 + v_list[i][2]

        tri_x3 = tri_x1 + v_list[i + 1][0]
        tri_y3 = tri_y1 + v_list[i + 1][1]
        tri_z3 = tri_z1 + v_list[i + 1][2]

        tri_x = [tri_x1, tri_x2, tri_x3]
        tri_y = [tri_y1, tri_y2, tri_y3]
        tri_z = [tri_z1, tri_z2, tri_z3]


        verts = [zip(tri_x, tri_y, tri_z)]

        poly3d = Poly3DCollection(verts)
        poly3d.set_facecolor(color)
        ax.add_collection3d(poly3d)

    # side
    for i in xrange(N):

        # from disk 1
        quad_x1 = x1 + v_list[i][0]
        quad_y1 = y1 + v_list[i][1]
        quad_z1 = z1 + v_list[i][2]

        quad_x2 = x1 + v_list[i + 1][0]
        quad_y2 = y1 + v_list[i + 1][1]
        quad_z2 = z1 + v_list[i + 1][2]

        # from disk 2
        quad_x3 = x2 + v_list[i][0]
        quad_y3 = y2 + v_list[i][1]
        quad_z3 = z2 + v_list[i][2]

        quad_x4 = x2 + v_list[i + 1][0]
        quad_y4 = y2 + v_list[i + 1][1]
        quad_z4 = z2 + v_list[i + 1][2]

        quad_x = [quad_x1, quad_x2, quad_x4, quad_x3]
        quad_y = [quad_y1, quad_y2, quad_y4, quad_y3]
        quad_z = [quad_z1, quad_z2, quad_z4, quad_z3]

        verts = [zip(quad_x, quad_y,quad_z)]

        poly3d = Poly3DCollection(verts)
        poly3d.set_facecolor(color)
        ax.add_collection3d(poly3d)



#  stress
# 1
fig = plt.figure(figsize=(30, 16))
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')

ax.set_xlim([0, 30])
ax.set_ylim([0, 30])
ax.set_zlim([-15, 15])
ax.grid(False)
ax._axis3don = False

# stress
for i in xrange(N_elem):

    ind1 = elems[i, 0]
    ind2 = elems[i, 1]

    xyz1 = nodes_d[ind1, :]
    xyz2 = nodes_d[ind2, :]

    r_loc = np.sqrt(areas[i]/np.pi)

    rgb_loc = matplotlib.colors.colorConverter.to_rgb(coolwarm_cmap(norm(rel_stress[i])))
    color = rgb_loc
    plotRod(xyz1, xyz2, r_loc, color, ax)

# jig shape
for i in xrange(N_elem):

    ind1 = elems[i, 0]
    ind2 = elems[i, 1]

    xyz1 = nodes[ind1, :]
    xyz2 = nodes[ind2, :]

    r_loc = np.sqrt(areas[i]/np.pi)

    color = [192.0/255, 192.0/255, 192.0/255]
    plotRod(xyz1, xyz2, r_loc, color, ax)


ax.view_init(elev=90, azim=0)

fig.savefig("wing_stress_1.pdf", bbox_inches='tight') 

# 2
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')

ax.set_xlim([0, 30])
ax.set_ylim([0, 30])
ax.set_zlim([-15, 15])
ax.grid(False)
ax._axis3don = False

for i in xrange(N_elem):

    ind1 = elems[i, 0]
    ind2 = elems[i, 1]

    xyz1 = nodes_d[ind1, :]
    xyz2 = nodes_d[ind2, :]

    r_loc = np.sqrt(areas[i]/np.pi)

    rgb_loc = matplotlib.colors.colorConverter.to_rgb(coolwarm_cmap(norm(rel_stress[i])))
    color = rgb_loc
    plotRod(xyz1, xyz2, r_loc, color, ax)

# jig shape
for i in xrange(N_elem):

    ind1 = elems[i, 0]
    ind2 = elems[i, 1]

    xyz1 = nodes[ind1, :]
    xyz2 = nodes[ind2, :]

    r_loc = np.sqrt(areas[i]/np.pi)

    color = [192.0/255, 192.0/255, 192.0/255]
    plotRod(xyz1, xyz2, r_loc, color, ax)

ax.view_init(elev=0, azim=0)

fig.savefig("wing_stress_2.pdf", bbox_inches='tight') 

# 3
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')

ax.set_xlim([0, 30])
ax.set_ylim([0, 30])
ax.set_zlim([-15, 15])
ax.grid(False)
ax._axis3don = False
ax.view_init(elev=30, azim=-90)

for i in xrange(N_elem):

    ind1 = elems[i, 0]
    ind2 = elems[i, 1]

    xyz1 = nodes_d[ind1, :]
    xyz2 = nodes_d[ind2, :]

    r_loc = np.sqrt(areas[i]/np.pi)

    rgb_loc = matplotlib.colors.colorConverter.to_rgb(coolwarm_cmap(norm(rel_stress[i])))
    color = rgb_loc
    plotRod(xyz1, xyz2, r_loc, color, ax)

# jig shape
for i in xrange(N_elem):

    ind1 = elems[i, 0]
    ind2 = elems[i, 1]

    xyz1 = nodes[ind1, :]
    xyz2 = nodes[ind2, :]

    r_loc = np.sqrt(areas[i]/np.pi)

    color = [192.0/255, 192.0/255, 192.0/255]
    plotRod(xyz1, xyz2, r_loc, color, ax)

ax.view_init(elev=30, azim=-90)

fig.savefig("wing_stress_3.pdf", bbox_inches='tight') 






#  buckling
# 1
fig = plt.figure(figsize=(30, 16))
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')

ax.set_xlim([0, 30])
ax.set_ylim([0, 30])
ax.set_zlim([-15, 15])
ax.grid(False)
ax._axis3don = False


for i in xrange(N_elem):

    ind1 = elems[i, 0]
    ind2 = elems[i, 1]

    xyz1 = nodes_d[ind1, :]
    xyz2 = nodes_d[ind2, :]

    r_loc = np.sqrt(areas[i]/np.pi)

    rgb_loc = matplotlib.colors.colorConverter.to_rgb(coolwarm_cmap(norm(bucklings[i])))
    color = rgb_loc
    plotRod(xyz1, xyz2, r_loc, color, ax)

# jig shape
for i in xrange(N_elem):

    ind1 = elems[i, 0]
    ind2 = elems[i, 1]

    xyz1 = nodes[ind1, :]
    xyz2 = nodes[ind2, :]

    r_loc = np.sqrt(areas[i]/np.pi)

    color = [192.0/255, 192.0/255, 192.0/255]
    plotRod(xyz1, xyz2, r_loc, color, ax)

ax.view_init(elev=90, azim=0)
fig.savefig("wing_buckling_1.pdf", bbox_inches='tight') 


# 2
fig = plt.figure(figsize=(30, 16))
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')

ax.set_xlim([0, 30])
ax.set_ylim([0, 30])
ax.set_zlim([-15, 15])
ax.grid(False)
ax._axis3don = False


for i in xrange(N_elem):

    ind1 = elems[i, 0]
    ind2 = elems[i, 1]

    xyz1 = nodes_d[ind1, :]
    xyz2 = nodes_d[ind2, :]

    r_loc = np.sqrt(areas[i]/np.pi)

    rgb_loc = matplotlib.colors.colorConverter.to_rgb(coolwarm_cmap(norm(bucklings[i])))
    color = rgb_loc
    plotRod(xyz1, xyz2, r_loc, color, ax)

# jig shape
for i in xrange(N_elem):

    ind1 = elems[i, 0]
    ind2 = elems[i, 1]

    xyz1 = nodes[ind1, :]
    xyz2 = nodes[ind2, :]

    r_loc = np.sqrt(areas[i]/np.pi)

    color = [192.0/255, 192.0/255, 192.0/255]
    plotRod(xyz1, xyz2, r_loc, color, ax)

ax.view_init(elev=0, azim=0)
fig.savefig("wing_buckling_2.pdf", bbox_inches='tight') 


# 3
fig = plt.figure(figsize=(30, 16))
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')

ax.set_xlim([0, 30])
ax.set_ylim([0, 30])
ax.set_zlim([-15, 15])
ax.grid(False)
ax._axis3don = False


for i in xrange(N_elem):

    ind1 = elems[i, 0]
    ind2 = elems[i, 1]

    xyz1 = nodes_d[ind1, :]
    xyz2 = nodes_d[ind2, :]

    r_loc = np.sqrt(areas[i]/np.pi)

    rgb_loc = matplotlib.colors.colorConverter.to_rgb(coolwarm_cmap(norm(bucklings[i])))
    color = rgb_loc
    plotRod(xyz1, xyz2, r_loc, color, ax)

# jig shape
for i in xrange(N_elem):

    ind1 = elems[i, 0]
    ind2 = elems[i, 1]

    xyz1 = nodes[ind1, :]
    xyz2 = nodes[ind2, :]

    r_loc = np.sqrt(areas[i]/np.pi)

    color = [192.0/255, 192.0/255, 192.0/255]
    plotRod(xyz1, xyz2, r_loc, color, ax)

ax.view_init(elev=30, azim=-90)
fig.savefig("wing_buckling_3.pdf", bbox_inches='tight') 