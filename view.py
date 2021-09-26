from sqlalchemy import select, desc
import pandas as pd
import numpy as np
import arrow

import pyglet
import pyglet.shapes as shapes
from pyglet.window import key

import db

engine = db.get_engine()
conn = engine.connect()

(run_id, created_at, params) = conn.execute(
    (
        select(
            db.run_table.c.id,
                db.run_table.c.created_at,
            db.run_table.c.params,
        )
        .order_by(desc(db.run_table.c.id))
    ),
).first()

created_at_arr = arrow.get(created_at)
print(f"Viewing run {run_id}")
print(f"Created at {created_at_arr.format()} ({created_at_arr.humanize()})")

seg_result = conn.execute(
    (
        select(
            db.seg_table.c.x1,
            db.seg_table.c.y1,
            db.seg_table.c.x2,
            db.seg_table.c.y2,
        )
        .where(db.seg_table.c.run_id == run_id)
        .order_by(db.seg_table.c.id)
    ),
).all()
print(f"Read {len(seg_result)} line segments")

d_seg = pd.DataFrame(
    seg_result,
    columns=['x1', 'y1', 'x2', 'y2'],
)

env_result = conn.execute(
    (
        select(
            db.env_table.c.id,
            db.env_table.c.step_view,
            db.env_table.c.step_sim,
            db.env_table.c.t_sim,
        )
        .where(db.env_table.c.run_id == run_id)
        .order_by(db.env_table.c.t_sim)
    ),
).all()
print(f"Read {len(env_result)} environment states")

d_env = (
    pd.DataFrame(
        env_result,
        columns=['env_id', 'step_view', 'step_sim', 't_sim']
    )
    # TODO: Fix properly, we write the same step_view twice if we resume from a
    # checkpoint, and have written outputs since the checkpoint.
    .drop_duplicates(subset=['step_view'], keep='first')
    .set_index('step_view', verify_integrity=True)
)
print(f"Environment states span {d_env.t_sim.min()} to {d_env.t_sim.max()}")
print(f"Latest environment state is at {d_env.step_sim.max()} steps")

agent_result = conn.execute(
    (
        select(
            db.agent_table.c.agent_id,
            db.agent_table.c.env_id,
            db.agent_table.c.rx,
            db.agent_table.c.ry,
            db.agent_table.c.ux,
            db.agent_table.c.uy,
            db.agent_table.c.vx,
            db.agent_table.c.vy,
        )
        .join(db.env_table)
        .where(db.env_table.c.run_id == run_id)
    ),
).all()

d_agent = pd.DataFrame(
    agent_result,
    columns=['agent_id', 'env_id', 'rx', 'ry', 'ux', 'uy', 'vx', 'vy'],
)
print(f"Read {d_agent.agent_id.nunique()} unique agents")

step_view_min, step_view_max = d_env.index.min(), d_env.index.max()


def at_pixel_res(a):
    return np.rint(a).astype(np.int)


def transform_len(sd, sl, pl):
    return at_pixel_res(pl * sd / sl)


def transform_coord(sd, sl, pl):
    return at_pixel_res(pl * ((sd + sl / 2.0) / sl))

# Drawing.

pl = 700

p_agent_repulsion_d_0 = transform_len(params.ag_repulse_d_0, params.l, pl)

print(f"s_agent_repulsion_d_0: {params.ag_repulse_d_0}")
print(f"p_agent_repulsion_d_0: {p_agent_repulsion_d_0}")

window = pyglet.window.Window(width=pl, height=pl)

fps_display = pyglet.window.FPSDisplay(window=window)

SETTINGS = {
    'step_view': step_view_min,
    'autostep_size': 1,
    'autostep_backwards': False,
    'autostep_enabled': False,
    'dirty': True,
}


def change_step(change):
    set_step(SETTINGS['step_view'] + change)


def set_step(new):
    cur = SETTINGS['step_view']
    SETTINGS['step_view'] = min(max(new, step_view_min), step_view_max)
    if SETTINGS['step_view'] != cur:
        SETTINGS['dirty'] = True


main_batch = pyglet.graphics.Batch()

seg_objs = []
for seg in d_seg.itertuples():
    seg_objs.append(shapes.Line(
        transform_coord(seg.x1, params.l, pl),
        transform_coord(seg.y1, params.l, pl),
        transform_coord(seg.x2, params.l, pl),
        transform_coord(seg.y2, params.l, pl),
        width=2,
        color=(255, 255, 255),
        batch=main_batch,
    ))


agent_objs = {
    agent_id: shapes.Circle(
        x=0,
        y=0,
        segments=5,
        radius=max(2, p_agent_repulsion_d_0),
        color=(200, 50, 50),
        batch=main_batch,
    )
    for agent_id in d_agent.agent_id.unique()
}

env_label = pyglet.text.Label(
    "[env label]",
    font_name="Times New Roman",
    font_size=12,
    x=20,
    y=20,
    anchor_x="left",
    anchor_y="center",
    batch=main_batch,
)

run_label = pyglet.text.Label(
    '[run label]',
    font_name="Times New Roman",
    font_size=12,
    x=20,
    y=pl - 20,
    anchor_x="left",
    anchor_y="center",
    batch=main_batch,
)


@window.event
def on_key_press(symbol, modifiers):
    if symbol == key.LEFT:
        change_step(-1)
    elif symbol == key.RIGHT:
        change_step(1)

    elif symbol == key.D:
        SETTINGS['autostep_backwards'] = not SETTINGS['autostep_backwards']

    elif symbol == key.SPACE:
        SETTINGS['autostep_enabled'] = not SETTINGS['autostep_enabled']

    elif symbol == key.UP:
        SETTINGS['autostep_size'] = max(int(round(2 * SETTINGS['autostep_size'])), 1)
    elif symbol == key.DOWN:
        SETTINGS['autostep_size'] = max(int(round(SETTINGS['autostep_size'] / 2)), 1)
    elif symbol == key.G:
        if modifiers & key.MOD_SHIFT:
            set_step(step_view_max)
        else:
            set_step(step_view_min)
    else:
        print(f"Unknown symbol: {symbol}")


@window.event
def on_draw():

    window.clear()

    fps_display.draw()
    main_batch.draw()

    # from cell_list import get_inters
    # # Draw nearby-agent connections, for debugging.
    # rr = np.array([[row.rx, row.ry] for row in d_agent.loc[lambda d: d.env_id == env_id].itertuples()])
    # insi, _, insdr = get_inters(rr, params.l, r_cut=0.1)
    # for i in range(len(rr)):
    #     sr_src = rr[i]
    #     pr_src = transform_coord(sr_src, params.l, pl)
    #     for inters_j in range(0, insi[i]):
    #         sr_dr = insdr[i, inters_j]
    #         sr_tgt_mic = sr_src + sr_dr
    #         pr_tgt_mic = transform_coord(sr_tgt_mic, params.l, pl)

    #         objs.append(shapes.Line(
    #             pr_src[0],
    #             pr_src[1],
    #             pr_tgt_mic[0],
    #             pr_tgt_mic[1],
    #             width=1,
    #             color=(255, 0, 0),
    #             batch=batch,
    #         ))

    SETTINGS['dirty'] = False


def auto_step(dt):
    if SETTINGS['autostep_enabled']:
        if SETTINGS['autostep_backwards']:
            change_step(-SETTINGS['autostep_size'])
            if SETTINGS['step_view'] == step_view_min:
                SETTINGS['autostep_enabled'] = False
        else:
            change_step(SETTINGS['autostep_size'])
            if SETTINGS['step_view'] == step_view_max:
                SETTINGS['autostep_enabled'] = False

    # if not SETTINGS['dirty']:
    #     return

    step_view = SETTINGS['step_view']

    step_env = d_env.loc[step_view]
    step_sim = step_env['step_sim']
    t_sim = step_env['t_sim']
    env_id = step_env['env_id']

    for row in d_agent.loc[lambda d: d.env_id == env_id].itertuples():
        sr = np.array([row.rx, row.ry])
        # sv = np.array([row.vx, row.vy])
        # su = np.array([row.ux, row.uy])

        # sv_u = c.unitize(sv)

        pr = transform_coord(sr, params.l, pl)

        circ = agent_objs[row.agent_id]
        circ.anchor_position = (pr[0], pr[1])

        # pu_u_tail = 10
        # pr_u_tail = at_pixel_res(pr - pu_u_tail * su)
        # objs.append(shapes.Line(
        #     pr_u_tail[0],
        #     pr_u_tail[1],
        #     pr[0],
        #     pr[1],
        #     width=1,
        #     color=(255, 255, 255),
        #     batch=batch,
        # ))

        # pu_v_tail = 8
        # pr_v_tail = at_pixel_res(pr - pu_v_tail * sv_u)
        # objs.append(shapes.Line(
        #     pr_v_tail[0],
        #     pr_v_tail[1],
        #     pr[0],
        #     pr[1],
        #     width=3,
        #     color=(50, 50, 255),
        #     batch=batch,
        # ))

        # objs.append(shapes.Circle(
        #     x=pr[0],
        #     y=pr[1],
        #     segments=10,
        #     radius=max(2, p_agent_repulsion_d_0),
        #     color=(200, 50, 50),
        #     batch=batch,
        # ))

        # objs.append(pyglet.text.Label(
        #     f"v: {c.show_2d_vec(sv)} s/t",
        #     font_name="Times New Roman",
        #     font_size=6,
        #     x=pr[0],
        #     y=pr[1],
        #     anchor_x="left",
        #     anchor_y="center",
        #     batch=batch,
        # ))

    direction_s = 'backward' if SETTINGS['autostep_backwards'] else 'forward'
    status_s = 'playing' if SETTINGS['autostep_enabled'] else 'paused'
    run_label.text = f"Viewing step size: {SETTINGS['autostep_size']}, {status_s}, {direction_s}"

    env_label.text = f"Sim time: {t_sim:.2f}, Sim step: {step_sim}, View step: {step_view}"


pyglet.clock.schedule_interval(auto_step, 0.01)

pyglet.app.run()
