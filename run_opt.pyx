# cython: infer_types=True

import numpy as np

cimport cython
cimport common as c
from libc.math cimport exp


def exp_repulse_v(d, v_0, d_0):
    return v_0 * np.exp(-d / d_0)


cdef inline double exp_repulse_v_(double d, double v_0, double d_0):
    return v_0 * exp(-d / d_0)


@cython.boundscheck(False)
@cython.wraparound(False)
def ag_ag_repulse_v(double v_0, double d_0, double[:, :, ::1] inters_dr, long[::1] intersi):
    cdef long n = inters_dr.shape[0]
    v = np.zeros([n, 2])
    cdef double[:, ::1] v_view = v
    cdef long i_src, i_tgt
    cdef double repulse_v, src_tgt_dist
    cdef double[::1] src_tgt_dr

    for i_src in range(n):
        for i_tgt in range(intersi[i_src]):
            src_tgt_dr = inters_dr[i_src, i_tgt]
            src_tgt_dist = c.norm_one_vec(src_tgt_dr)

            repulse_v = exp_repulse_v_(d=src_tgt_dist, v_0=v_0, d_0=d_0)

            v_view[i_src, 0] -= repulse_v * src_tgt_dr[0] / src_tgt_dist
            v_view[i_src, 1] -= repulse_v * src_tgt_dr[1] / src_tgt_dist
    return v
