import meshpy.triangle as triangle
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as pt
from math import sqrt, pow, acos, pi,cos,sin
# from numpy.doc.constants import m


def round_trip_connect(start, end):
    return [(i, i + 1) for i in range(start, end)] + [(end, start)]

points = [(1, 0), (1, 1), (-1, 1), (-1, -1), (1, -1), (1, 0)]
facets = round_trip_connect(0, len(points) - 1)
# print(facets)

circ_start = len(points)
points.extend(
    (3 * np.cos(angle), 3 * np.sin(angle))
    for angle in np.linspace(0, 2 * np.pi, 30, endpoint=False)
)

facets.extend(round_trip_connect(circ_start, len(points) - 1))

def needs_refinement(vertices, area):
    bary = np.sum(np.array(vertices), axis=0) / 3
    max_area = 0.001 + (la.norm(bary, np.inf) - 1) * 0.01
    return bool(area > max_area)

info = triangle.MeshInfo()
info.set_points(points)
info.set_holes([(0, 0)])
info.set_facets(facets)

mesh = triangle.build(info, refinement_func=None)

mesh_points = np.array(mesh.points)
mesh_tris = np.array(mesh.elements)
# print(mesh_points)
# print(len(mesh_points))
#[35 23  3]
# print(mesh_tris)
# print(len(mesh_tris))


pt.triplot(mesh_points[:, 0], mesh_points[:, 1], mesh_tris)
pt.show()


# input is point array ex:p1=np.array([0,0]),output is the coordinate u and v
def triangle_local_coordinate(element):
    p1 = mesh_points[element[0]]
    p2 = mesh_points[element[1]]
    p3 = mesh_points[element[2]]
    # print(p1,p2,p3)
    u = np.subtract(p2, p1)
    b = np.subtract(p3, p1)
    # print(u,b)
    u1 = np.append(u,[0])
    b1 = np.append(b,[0])
    # print(u1,b1)
    n = np.cross(u1, b1)
    # print(n)
    v1 = np.cross(n, u)
    v = np.array(v1[0:-1])
    return u, v
# print(mesh_tris[1])
# print(mesh_points[22])
m1 = triangle_local_coordinate(mesh_tris[1])
# print(m1)
# all the triangle local coordinate
tri_loc_cor = list()
for i in range(len(mesh_tris)):
    m_i = triangle_local_coordinate(mesh_tris[i])
    # print(m_i)
    tri_loc_cor.append(m_i)

# print(tri_loc_cor)

# transform the points to m1 m2 m3
# input:u,v p1,p2,p3
def global2local(u,p1,p2,p3):
    r = sqrt(u[0] ** 2 + u[1] ** 2)
    sin_u = u[1] / r
    cos_u = u[0] / r
    m1 = np.zeros(2)
    m2_x = (p2[0]-p1[0])*cos_u +(p2[1]-p1[1])*sin_u
    m2_y = -(p2[0]-p1[0])*sin_u + (p2[1]-p1[1])*cos_u
    m3_x = (p3[0]-p1[0])*cos_u +(p3[1]-p1[1])*sin_u
    m3_y = -(p3[0]-p1[0])*sin_u + (p3[1]-p1[1])*cos_u
    m2 = np.array([m2_x,m2_y])
    m3 = np.array([m3_x,m3_y])
    return m1,m2,m3

def local2global(u,p1,m2,m3):
    r = sqrt(u[0] ** 2 + u[1] ** 2)
    sin_u = u[1] / r
    cos_u = u[0] / r
    p2_x = m2[0] * cos_u - m2[1] * sin_u + p1[0]
    p2_y = m2[0]*sin_u + m2[1]*cos_u+m1[1] +p1[1]
    p3_x = m3[0] * cos_u - m3[1] * sin_u + p1[0]
    p3_y = m3[0] * sin_u + m3[1] * cos_u + p1[1]
    p2 = np.array([p2_x,p2_y])
    p3 = np.array([p3_x ,p3_y])
    return p1,p2,p3

local_point = list()
for i in range(len(mesh_tris)):
    e_i = mesh_tris[i]
    p1 = mesh_points[e_i[0]]
    p2 = mesh_points[e_i[1]]
    p3 = mesh_points[e_i[2]]
    u = tri_loc_cor[i][0]
    e_local_i = global2local(u,p1,p2,p3)
    local_point.append(e_local_i)
print('111',local_point[1])

def compute_beta(localpoint):
    m1 = localpoint[0]
    m2 = localpoint[1]
    m3 = localpoint[2]
    m = np.array([m1, m2, m3])
    m_t = np.transpose(m)
    #print(m_t)
    one = np.array([[1,1,1]])
    #print(one)
    beta_inv = np.append(m_t, one, axis=0)
    beta = np.mat(beta_inv).I
    return beta
#beta is a matrix
Beta = list()
for i in range(len(local_point)):
    lp_i = local_point[i]
    beta_i = compute_beta(lp_i)
    Beta.append(beta_i)
print('bbb',Beta[1])

#intilaized the sigma,uniformly,try 1,1,1,1
Sigma = list()
sigma_og = np.array([[1,1],[1,1]])
for i in range(len(mesh_tris)):
    sigma_rd = (1/100)*(np.random.rand(2,2))
    sigma_i = sigma_og + sigma_rd
    Sigma.append(sigma_i)

print(Sigma[1])

# input force should be vector(3*1)
def func_m(vec) -> object:
    f = np.mat(vec)
    # print('f',f)
    if np.all(f == 0):
        m = np.array([0, 0])
    else:
        # print('f.t',f.T)
        # print('mff', np.dot(f.T, f))
        m = (1 / np.linalg.norm(f)) * np.dot(f.T, f)
        # print(m)
    return m


def compute_separate_stress_tensor(sigma):
    sigma_post = np.mat(np.array([[0,0],[0,0]]))
    sigma_nega = np.mat(np.array([[0,0],[0,0]]))
    for j in range(0, 2):
        e, v = np.linalg.eig(sigma)
        # print('engin',e,v)
        # print(v[0])
        v_j_norm = v[j] / np.linalg.norm(v[j])
        sigma_post = np.add(sigma_post, max(0, e[j]) * func_m(v_j_norm))
        sigma_nega = np.add(sigma_nega, min(0, e[j]) * func_m(v_j_norm))
    return sigma_post, sigma_nega
print(Sigma[1])
print('p,n',compute_separate_stress_tensor(Sigma[1]))

# cumpute A
def calc_area(p1, p2, p3):
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
    return 0.5 * abs(x2 * y3 + x1 * y2 + x3 * y1 - x3 * y2 - x2 * y1 - x1 * y3)

# A int, beta matrix(3*3),p(2*1)or(3*1),sigma matrix(2*2)
# def compute_force(beta, p1, p2, p3, sigma):
#     A = calc_area(p1, p2, p3)
#     print('A',A)
#     sigma = sigma
#     p = [p1, p2, p3]
#     beta = beta
#     sum = 0.0
#     for j in range(0, 3):
#         elem2 = 0
#         for k in range(0, 2):
#             for l in range(0, 2):
#                 elem2 += beta[j][l] * beta[i][k] * sigma[k][l]
#                 print(elem2)
#
# #         sum += p[j] * elem2
#
#     return -A * sum

def compute_force(beta, p1, p2, p3, sigma):
    force_i = list()
    p = [p1, p2, p3]
    A = calc_area(p1, p2, p3)
    # print('A',A)
    for i in range(0,3):
        elem_2 = 0
        for j in range(0,3):
            elem_1 = 0
            for k in range(0,2):
                for l in range(0,2):
                    elem_1 = elem_1 + beta[j,l] * beta[i,k]* sigma[k,l]
            elem_2 = elem_2 + p[j]*elem_1
        force_i.append(elem_2)
    return force_i

# e_i = mesh_tris[1]
# p1 = mesh_points[e_i[0]]
# p2 = mesh_points[e_i[1]]
# p3 = mesh_points[e_i[2]]
# beta_i = np.mat(Beta[1])
# print(beta_i)
# print('beta',beta_i[0,1])
# sigma_i = Sigma[1]
# print('sigma',sigma_i)
# force = compute_force(beta_i,p1,p2,p3,sigma_i)
# print('fff',force)




forces = list()
for i in range(len(mesh_tris)):
    e_i = mesh_tris[i]
    p1 = mesh_points[e_i[0]]
    p2 = mesh_points[e_i[1]]
    p3 = mesh_points[e_i[2]]
    beta_i = Beta[i]
    sigma_i = Sigma[i]
    for i in range(3):
        forces_e = list()
        forces_i = compute_force(beta_i,p1,p2,p3,sigma_i)
        forces_e.append(forces_i)
    forces.append(forces_e)
print('force',forces[1])

#zeta
forces_post = list()
forces_nega = list()
for i in range(len(mesh_tris)):
    e_i = mesh_tris[i]
    p1 = mesh_points[e_i[0]]
    p2 = mesh_points[e_i[1]]
    p3 = mesh_points[e_i[2]]
    beta_i = Beta[i]
    sigma_i = Sigma[i]
    sigma_i_pos, sigma_i_neg = compute_separate_stress_tensor(sigma_i)
    for i in range(3):
        forces_pos_i = list()
        forces_neg_i = list()
        fp_i = compute_force(beta_i,p1,p2,p3,sigma_i_pos)
        fn_i = compute_force(beta_i, p1, p2, p3, sigma_i_neg)
        forces_pos_i.append(fp_i)
        forces_neg_i.append(fn_i)
    forces_post.append(forces_pos_i)
    forces_nega.append(forces_neg_i)
print('force_pos',forces_post[9],'force_neg',forces_nega[9])
print('fp0',forces_post[9][0][0])


# input sigma-post and nega and compute the force post and nega then compute zeta
def compute_zeta(f_post, f_nega):
    f_post_sum = np.zeros((2, 2))
    f_nega_sum = np.zeros((2, 2))
    zeta_i = list()
    for f in f_post:
        f_post_sum += func_m(f)
    for f in f_nega:
        f_nega_sum += func_m(f)
    for i in range(0,3):
        zeta_p_i = np.mat((-func_m(f_post[0][i]) + f_post_sum + func_m(f_nega[0][i]) - f_nega_sum) / 2)
        zeta_i.append(zeta_p_i)
    return zeta_i
# print(forces_post[9][0])
zeta_9 = compute_zeta(forces_post[9],forces_nega[9])
print(zeta_9)

#for each element,we calculate the zeta for every point
zeta = list()
for i in range(len(mesh_tris)):
    f_post = forces_post[i]
    f_nega = forces_nega[i]
    zeta_i = compute_zeta(f_post,f_nega)
    zeta.append(zeta_i)

# print(zeta[9][0])
# vals, vects = np.linalg.eig(zeta[9][0])
# max_vals = max(vals)
# maxcol = list(vals).index(max(vals))
# max_vects = vects[maxcol]
# ge_point_i = [max_vals,max_vects]
# print('zeta-engin',vals,'11',vects)
# print(max(vals),'000',max_vects)
# print(vects[0])
# print('444',ge_point_i)

# determent whether we should generate the cracks
gc_point = list()
toa = 1.5
for i in range(len(mesh_tris)):
    zeta_i = zeta[i]
    e_i = mesh_tris[i]
    p1 = mesh_points[e_i[0]]
    p2 = mesh_points[e_i[1]]
    p3 = mesh_points[e_i[2]]
    p = [p1, p2, p3]

    for j in range(0,3):
        p_j = p[j]
        vals, vects = np.linalg.eig(zeta[i][j])
        max_vals = max(vals)
        maxcol = list(vals).index(max(vals))
        max_vects = vects[maxcol]
        if max_vals > toa:
            residual = max_vals - toa
            ge_point_i = [p_j,max_vals,max_vects,j,i,residual]
            print('gpi',ge_point_i)
            gc_point.append(ge_point_i)
print('point to generate the cracks',gc_point[3])

def get_vertical_vector(vec):
    """ Calculate the vertical vector of a two-dimensional vector """
    # assert isinstance(vec, list) and len(vec) == 2, r'The vector in the plane must be 2'
    return [vec[1], -vec[0]]
# generate points with point coordinate and direction
gc_pt_dr = list()
for point in gc_point:
    p = point[0]
    p1 = point[1]
    j = point[3]
    i = point[4]
    vec = np.array(point[2])
    vertical_vect = np.array(get_vertical_vector(vec[0]))
    print(vertical_vect)
    point_i = [p,p1,vertical_vect,j,i]
    gc_pt_dr.append(point_i)
print(gc_pt_dr)

#get the line function from node where we generate the cracks
def get_line_from_crack_point(p,vect):
    a0 = vect[1]
    b0 = -vect[0]
    c0 = vect[0]*p[1] - vect[1]*p[0]
    return a0,b0,c0
#get the line function from nodes who are in the same element
def get_line_from_node(p1,p2):
    a1 = p1[1] - p2[1]
    b1 = p2[0] - p1[0]
    c1 = p1[0]*p2[1] - p2[0]*p1[1]
    return a1,b1,c1

def get_cross_point(a0,b0,c0,a1,b1,c1):
    D = a0*b1 - a1*b0
    x = (b0*c1 - b1*c0)/D
    y = (a1*c0 - a0*c1)/D
    point = np.array([x,y])
    return point

#stop generate if theta is less than 15
def angle_of_vector(v1, v2):
    vector_prod = v1[0] * v2[0] + v1[1] * v2[1]
    length_prod = sqrt(pow(v1[0], 2) + pow(v1[1], 2)) * sqrt(pow(v2[0], 2) + pow(v2[1], 2))
    cos = vector_prod * 1.0 / (length_prod * 1.0 + 1e-6)
    return (acos(cos) / pi) * 180

#to get the cross point：input：gc_pt_dr [point,max-enginvalue,direction,point index in element,element index]
final_gene_point = list() #output: [node to generate cracks, cross point, node index in element,element index]
for points in gc_pt_dr:
    p = points[0]
    vect = points[2]
    a0,b0,c0 = get_line_from_crack_point(p,vect)
    i = points[4]
    e_i = mesh_tris[i]
    p1 = mesh_points[e_i[0]]
    p2 = mesh_points[e_i[1]]
    p3 = mesh_points[e_i[2]]
    p = [p1, p2, p3]
    j = points[3]
    pj = p[j]
    p_e = []
    for p_ in p:
        if np.all(p_ != p[j]):
            p_e.append(p_)
    a1,b1,c1 = get_line_from_node(p_e[0],p_e[1])
    pc = get_cross_point(a0,b0,c0,a1,b1,c1)
    print('pc',pc)

    #stop generate if theta is less than 15
    v_gene = [pc[0]- pj[0],pc[1]-pj[1]]
    v1 = [p_e[0][0]-pj[0],p_e[0][1]-pj[1]]
    v2 = [p_e[1][0] - pj[0], p_e[1][1] - pj[1]]
    print('v1',v1,'v2',v2)
    ang1 = angle_of_vector(v_gene,v1)
    ang2 = angle_of_vector(v_gene,v2)
    print('ag1',ang1,'ag2',ang2)
    if ang1 >= 15 and ang2 >= 15:
        pc_ = [p[j],pc,j,i]
        final_gene_point.append(pc_)
print(final_gene_point)


