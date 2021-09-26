# cython: infer_types=True

import math

import numpy as np
from libc.math cimport trunc

cimport cython
cimport common as c


cdef long M_MAX = 100

cdef long DIM = 2


@cython.boundscheck(False)
@cython.wraparound(False)
cdef fill_adjs(
        long i_a_src,
        long tgt_mx,
        long tgt_my,
        double[:, ::1] r,
        long[:, :, ::1] cl,
        long[:, ::1] cli,
        long[:, ::1] inters,
        double[:, :, ::1] inters_dr,
        long[::1] intersi,
        double l_half,
        double r_cut_sq
    ):
    cdef long i_cl_tgt_max = cli[tgt_mx, tgt_my]
    cdef double dx, dy
    cdef long i_a_tgt
    for i_cl_tgt in range(i_cl_tgt_max):
        i_a_tgt = cl[tgt_mx, tgt_my, i_cl_tgt]

        dx = c.x_wrap(r[i_a_tgt, 0] - r[i_a_src, 0], l_half)
        dy = c.x_wrap(r[i_a_tgt, 1] - r[i_a_src, 1], l_half)

        if i_a_tgt != i_a_src and (dx * dx + dy * dy) < r_cut_sq:
            inters[i_a_src, intersi[i_a_src]] = i_a_tgt

            inters_dr[i_a_src, intersi[i_a_src], 0] = dx
            inters_dr[i_a_src, intersi[i_a_src], 1] = dy

            intersi[i_a_src] += 1


cdef inline long get_m(double l, double r_cut):
    return min(int(l / r_cut), M_MAX)


def initialise_inters_structures(
    double l,
    double r_cut,
    long n,
):
    cdef long m = get_m(l, r_cut)

    cl = np.empty([m, m, n], dtype=np.long)
    cli = np.empty([m, m], dtype=np.long)

    inters = np.empty([n, n], dtype=np.long)
    intersi = np.empty([n], dtype=np.long)
    inters_dr = np.empty([n, n, DIM], dtype=np.double)

    return cl, cli, inters, intersi, inters_dr

@cython.boundscheck(False)
@cython.wraparound(False)
def get_inters(
        # Input data.
        double[:, ::1] r,
        double l,
        double r_cut,
        # Intermediate data structures (pre-allocated for speed).
        long[:, :, ::1] cl,
        long[:, ::1] cli,
        # Output data structures.
        long[:, ::1] inters,
        long[::1] intersi,
        double[:, :, ::1] inters_dr,
        ):
    if not (get_m(l, r_cut) == cl.shape[0] == cl.shape[1] == cli.shape[0] == cli.shape[1]):
        raise ValueError()
    if not (r.shape[0] == cl.shape[2]):
        raise ValueError()

    cdef long n = r.shape[0]
    cdef long m = cl.shape[0]

    # Bin agents into their 2D cell list indices.
    cdef long[:, ::1] inds = np.empty_like(r, dtype=np.int)
    cdef double l_half = l * 0.5
    cdef double grid_n = l / m
    for i in range(n):
        inds[i, 0] = int((r[i, 0] + l_half) / grid_n)
        inds[i, 1] = int((r[i, 1] + l_half) / grid_n)

    # Populate agent list at each cell list point.
    cl[:, :, :] = -1
    cli[:, :] = 0
    cdef long jx, jy
    for i in range(n):
        jx = inds[i, 0]
        jy = inds[i, 1]
        cl[jx, jy, cli[jx, jy]] = i
        cli[jx, jy] += 1

    # Populate agent neighbour list for each agent, by considering all agents in
    # adjacent cells, for each agent.
    cdef double r_cut_sq = r_cut * r_cut

    inters[:, :] = -1
    inters_dr[:, :, :] = -1
    intersi[:] = 0
    cdef long x_inc, x_dec, y_inc, y_dec, mx, my, i_cl_src, i_a_src
    for mx in range(m):
        x_inc = c.i_wrap(mx + 1, m)
        x_dec = c.i_wrap(mx - 1, m)
        for my in range(m):
            y_inc = c.i_wrap(my + 1, m)
            y_dec = c.i_wrap(my - 1, m)
            for i_cl_src in range(cli[mx, my]):
                i_a_src = cl[mx, my, i_cl_src]

                # The cell itself.
                fill_adjs(i_a_src, mx, my, r, cl, cli, inters, inters_dr, intersi, l_half, r_cut_sq)
                # East.
                fill_adjs(i_a_src, x_inc, my, r, cl, cli, inters, inters_dr, intersi, l_half, r_cut_sq)
                # West.
                fill_adjs(i_a_src, x_dec, my, r, cl, cli, inters, inters_dr, intersi, l_half, r_cut_sq)

                # North.
                fill_adjs(i_a_src, mx, y_inc, r, cl, cli, inters, inters_dr, intersi, l_half, r_cut_sq)
                # North-east.
                fill_adjs(i_a_src, x_inc, y_inc, r, cl, cli, inters, inters_dr, intersi, l_half, r_cut_sq)
                # North-west.
                fill_adjs(i_a_src, x_dec, y_inc, r, cl, cli, inters, inters_dr, intersi, l_half, r_cut_sq)

                # South.
                fill_adjs(i_a_src, mx, y_dec, r, cl, cli, inters, inters_dr, intersi, l_half, r_cut_sq)
                # South-east
                fill_adjs(i_a_src, x_inc, y_dec, r, cl, cli, inters, inters_dr, intersi, l_half, r_cut_sq)
                # South-west
                fill_adjs(i_a_src, x_dec, y_dec, r, cl, cli, inters, inters_dr, intersi, l_half, r_cut_sq)
