# cython: infer_types=True

import numpy as np

cimport cython
from libc.math cimport sqrt

# Geometry.

def point_line_segment(double[::1] ss, double[::1] se, p):
    dr = np.subtract(se, ss)
    nx_line = np.dot(p - ss, dr) / norm_sq_one_vec(dr)
    nx_seg = np.minimum(1, np.maximum(0, nx_line))
    return ss + (dr[np.newaxis, :] * nx_seg[:, np.newaxis])


# Norms.


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double norm_sq_one_vec(double[::1] v):
    return v[0] * v[0] + v[1] * v[1]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double norm_one_vec(double[::1] v):
    return sqrt(norm_sq_one_vec(v))


@cython.boundscheck(False)
@cython.wraparound(False)
def norm_sq(double[:, ::1] v):
    nrm_sq = np.empty([v.shape[0]], dtype=np.float)
    cdef double[::1] nrm_sq_view = nrm_sq
    cdef long i
    for i in range(v.shape[0]):
        nrm_sq_view[i] = v[i, 0] * v[i, 0] + v[i, 1] * v[i, 1]
    return nrm_sq


@cython.boundscheck(False)
@cython.wraparound(False)
def norm(double[:, ::1] v):
    return np.sqrt(norm_sq(v))


# Periodic boundaries.


cdef inline long i_wrap(long i, long b):
    if i > b - 1:
        return i - b
    if i < 0:
        return i + b
    return i


cdef inline double x_wrap(double x, double l_half):
    if x > l_half:
        return x - 2 * l_half
    if x < -l_half:
        return x + 2 * l_half
    return x


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double dr_wrap_norm_sq(double[::1] src, double[::1] tgt, double l_half):
    dx = x_wrap(tgt[0] - src[0], l_half)
    dy = x_wrap(tgt[1] - src[1], l_half)
    return dx * dx + dy * dy


@cython.boundscheck(False)
@cython.wraparound(False)
def pairwise_norm_sq_wrapped(double[:, ::1] v, double l_half):
    cdef int n = v.shape[0]
    res = np.empty([n, n], dtype=np.float)
    cdef double [:, ::1] res_view = res
    cdef int i, j
    cdef double dr_sq
    for i in range(n):
        res[i, i] = 0.0
        for j in range(i + 1, n):
            dr_sq = dr_wrap_norm_sq(v[i], v[j], l_half)
            res[i, j] = dr_sq
            res[j, i] = dr_sq
    return res

def pairwise_norm_wrapped(double[:, ::1] v, double l_half):
    return np.sqrt(pairwise_norm_sq_wrapped(v, l_half))


# Unit vectors.


@cython.boundscheck(False)
@cython.wraparound(False)
def unitize_inplace(double[:, ::1] v):
    if v.shape[1] != 2:
        raise ValueError('Need second dimension to have length 2')
    cdef long n, d
    cdef double norm
    for n in range(v.shape[0]):
        norm = sqrt(v[n, 0] * v[n, 0] + v[n, 1] * v[n, 1])
        v[n, 0] /= norm
        v[n, 1] /= norm


@cython.boundscheck(False)
@cython.wraparound(False)
def unitize(v):
    vu = v.copy()
    unitize_inplace(vu)
    return vu


@cython.boundscheck(False)
@cython.wraparound(False)
def unitize_one_vec(double[::1] v):
    cdef double[::1] vu = v.copy()
    cdef double nrm = norm_one_vec(v)
    vu[0] /= nrm
    vu[1] /= nrm
    return vu


# Angles.


def rad_to_deg(rad):
    return rad * 180.0 / np.pi


def deg_to_rad(deg):
    return deg * np.pi / 180.0


def show_rad(rad):
    return f"{rad_to_deg(rad):.1f} deg"


# Vector.


def show_2d_vec(v):
    return f"({v[0]:.4f}, {v[1]:.4f})"


def rotate_2d_vec(v, th):
    vn = np.empty_like(v)
    cos_th = np.cos(th)
    sin_th = np.sin(th)
    vn[:, 0] = v[:, 0] * cos_th - v[:, 1] * sin_th
    vn[:, 1] = v[:, 0] * sin_th + v[:, 1] * cos_th
    return vn


# Clustering.

# def av_nearest_neighbor_ratio(l):
#     # Expected average nearest-neighbor ratio, for an empty square with periodic
#     # boundaries, is:
#     # l / sqrt(6)
#     d_mean_exp = l ** 2 / 6.0
#     d_mean_obs
#     return d_mean_obs / d_mean_exp
