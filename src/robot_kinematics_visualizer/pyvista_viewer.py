from __future__ import annotations

import gzip
import hashlib
import pathlib
import pickle
import time
from io import BytesIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import spatialmath as sm
import trimesh
import vtk
from PIL import Image

AXIS_COLORS = {"x": "red", "y": "green", "z": "blue"}

ROTATION_BG = "#bbdefb"
ROTATION_FG = "#0b3d91"
POSITION_BG = "#c8e6c9"
POSITION_FG = "#1b5e20"
HOMOGENEOUS_BG = "#ffe0b2"
HOMOGENEOUS_FG = "#bf360c"

ROBOT_OPACITY = 0.35
MATRIX_WORLD_HEIGHT = 0.10


def extract_link_meshes(robot):
    """Return mesh metadata for each link with renderable geometry."""
    meshes = []
    for link in robot.links:
        for geom in getattr(link, "geometry", []) or []:
            if getattr(geom, "stype", None) != "mesh":
                continue
            filename = getattr(geom, "filename", None)
            if not filename:
                continue
            try:
                tm = trimesh.load(filename, force="mesh")
            except Exception as exc:
                print(f"[viewer] could not load mesh {filename}: {exc}")
                continue
            scale = np.asarray(geom.scale, dtype=float)
            if scale.size == 3 and not np.allclose(scale, 1.0):
                tm.apply_scale(scale)
            meshes.append(
                {
                    "link": link,
                    "mesh": pv.wrap(tm),
                    "local_T": np.asarray(geom.T, dtype=float),
                    "color": tuple(float(c) for c in geom.color[:3]),
                }
            )
    return meshes


def compute_link_transforms(robot):
    """Return the world SE(3) transform for every link at the current q."""
    all_T = robot.fkine_all(robot.q)
    transforms = []
    for link in robot.links:
        T = all_T[link.number]
        transforms.append({"link": link, "T": T})
    return transforms


def _format_prefix(prefix):
    if prefix is None:
        return "$T = $"
    if isinstance(prefix, int):
        if prefix == 0:
            return "$T_0 = $"
        return f"$T_{{0,{prefix}}} = $"
    return f"$T_{{0,\\mathrm{{{prefix}}}}} = $"


_matrix_image_cache: dict = {}
_matrix_texture_cache: dict = {}
_MATRIX_CACHE_LIMIT = 8192
_MATRIX_FIG_DPI = 180
_MATRIX_FIG_SIZE = (4.6, 2.2)
_matrix_fig = None
_matrix_ax = None


def _disk_cache_signature():
    """Identifies the rendering config; changes invalidate the disk cache."""
    parts = (
        ("dpi", _MATRIX_FIG_DPI),
        ("figsize", _MATRIX_FIG_SIZE),
        ("rotation_bg", ROTATION_BG),
        ("position_bg", POSITION_BG),
        ("homogeneous_bg", HOMOGENEOUS_BG),
        ("rotation_fg", ROTATION_FG),
        ("position_fg", POSITION_FG),
        ("homogeneous_fg", HOMOGENEOUS_FG),
        ("v", 1),
    )
    return hashlib.sha256(repr(parts).encode()).hexdigest()[:16]


def _disk_cache_path():
    cache_dir = pathlib.Path(__file__).resolve().parents[2] / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"matrix_cache_{_disk_cache_signature()}.pkl.gz"


def _load_disk_cache():
    p = _disk_cache_path()
    if not p.exists():
        return 0
    try:
        with gzip.open(p, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            return 0
        _matrix_image_cache.update(data)
        return len(data)
    except Exception as exc:
        print(f"[viewer] cache load failed ({exc}); will re-render")
        return 0


def _save_disk_cache():
    p = _disk_cache_path()
    try:
        with gzip.open(p, "wb") as f:
            pickle.dump(dict(_matrix_image_cache), f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        print(f"[viewer] cache save failed: {exc}")


def _get_matrix_fig():
    global _matrix_fig, _matrix_ax
    if _matrix_fig is None:
        _matrix_fig = plt.figure(figsize=_MATRIX_FIG_SIZE, dpi=_MATRIX_FIG_DPI, facecolor="white")
        _matrix_ax = _matrix_fig.add_subplot(111)
    return _matrix_fig, _matrix_ax


def _render_matrix_image(transform, prefix=None, decimals=2):
    matrix = np.round(np.asarray(transform.A, dtype=float), decimals)
    matrix[np.abs(matrix) < 10 ** (-decimals)] = 0

    cache_key = (matrix.tobytes(), str(prefix), decimals)
    cached = _matrix_image_cache.get(cache_key)
    if cached is not None:
        return cached

    fig, ax = _get_matrix_fig()
    ax.clear()
    ax.set_facecolor("white")

    cell_w = 1.4
    cell_h = 1.0
    x0 = 2.1
    y0 = 0.0
    bracket_top = y0 + 4 * cell_h
    bracket_bot = y0

    for i in range(4):
        for j in range(4):
            x = x0 + j * cell_w
            y = y0 + (3 - i) * cell_h
            if i == 3:
                bg, fg = HOMOGENEOUS_BG, HOMOGENEOUS_FG
            elif j == 3:
                bg, fg = POSITION_BG, POSITION_FG
            else:
                bg, fg = ROTATION_BG, ROTATION_FG
            ax.add_patch(plt.Rectangle((x, y), cell_w, cell_h, facecolor=bg, edgecolor="none"))
            ax.text(
                x + cell_w / 2,
                y + cell_h / 2,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color=fg,
            )

    bracket_left_x = x0 - 0.18
    bracket_right_x = x0 + 4 * cell_w + 0.18
    tick = 0.28
    lw = 3.5
    bcolor = "black"
    ax.plot([bracket_left_x, bracket_left_x], [bracket_bot, bracket_top], color=bcolor, lw=lw)
    ax.plot(
        [bracket_left_x, bracket_left_x + tick], [bracket_top, bracket_top], color=bcolor, lw=lw
    )
    ax.plot(
        [bracket_left_x, bracket_left_x + tick], [bracket_bot, bracket_bot], color=bcolor, lw=lw
    )
    ax.plot([bracket_right_x, bracket_right_x], [bracket_bot, bracket_top], color=bcolor, lw=lw)
    ax.plot(
        [bracket_right_x - tick, bracket_right_x], [bracket_top, bracket_top], color=bcolor, lw=lw
    )
    ax.plot(
        [bracket_right_x - tick, bracket_right_x], [bracket_bot, bracket_bot], color=bcolor, lw=lw
    )

    ax.text(
        bracket_left_x - 0.15,
        y0 + 2 * cell_h,
        _format_prefix(prefix),
        ha="right",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="black",
    )

    ax.set_xlim(0, x0 + 4 * cell_w + 0.5)
    ax.set_ylim(-0.3, bracket_top + 0.3)
    ax.set_aspect("equal")
    ax.axis("off")

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05, facecolor="white")
    buf.seek(0)
    img = np.asarray(Image.open(buf).convert("RGBA")).copy()
    rgb = img[..., :3]
    pure_white = (rgb[..., 0] >= 252) & (rgb[..., 1] >= 252) & (rgb[..., 2] >= 252)
    img[pure_white, 3] = 0

    if len(_matrix_image_cache) > _MATRIX_CACHE_LIMIT:
        _matrix_image_cache.clear()
    _matrix_image_cache[cache_key] = img
    return img


def add_matrix_image(plotter, transform, anchor, prefix=None, world_height=MATRIX_WORLD_HEIGHT, offset=(0.10, 0.0, 0.04)):
    """Add a matplotlib-rendered matrix as an auto-billboarded textured plane."""
    img = _render_matrix_image(transform, prefix=prefix)
    h, w = img.shape[:2]
    aspect = w / h

    plane = pv.Plane(
        center=(0.0, 0.0, 0.0),
        direction=(0.0, 0.0, 1.0),
        i_size=world_height * aspect,
        j_size=world_height,
    )
    plane.texture_map_to_plane(inplace=True)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(plane)

    texture = pv.numpy_to_texture(img)

    follower = vtk.vtkFollower()
    follower.SetMapper(mapper)
    follower.SetTexture(texture)
    follower.SetCamera(plotter.camera)
    follower.SetPosition(*(np.asarray(anchor) + np.asarray(offset)))
    prop = follower.GetProperty()
    prop.LightingOff()
    prop.SetColor(1.0, 1.0, 1.0)
    plotter.renderer.AddActor(follower)

    return {
        "follower": follower,
        "texture": texture,
        "plane": plane,
        "anchor": np.asarray(anchor, dtype=float),
        "offset": np.asarray(offset, dtype=float),
    }


def _add_joint_label(plotter, anchor, text, offset_pixels=(20, 18)):
    actor = vtk.vtkBillboardTextActor3D()
    actor.SetInput(text)
    actor.SetPosition(float(anchor[0]), float(anchor[1]), float(anchor[2]))
    actor.SetDisplayOffset(int(offset_pixels[0]), int(offset_pixels[1]))
    prop = actor.GetTextProperty()
    prop.SetFontFamilyToCourier()
    prop.SetBold(True)
    prop.SetFontSize(14)
    prop.SetColor(0.0, 0.0, 0.0)
    prop.SetFrame(False)
    prop.SetBackgroundOpacity(0.0)
    plotter.renderer.AddActor(actor)
    return actor


def add_frame(
    plotter,
    transform,
    label=None,
    prefix=None,
    length=0.08,
    show_matrix=True,
    matrix_offset=(0.10, 0.0, 0.04),
    label_offset_pixels=(20, 18),
):
    """Draw an XYZ frame at the given SE(3) transform with optional labels.

    Returns a list of VTK actors created so callers can remove them later.
    """
    matrix = np.asarray(transform.A, dtype=float)
    origin = matrix[:3, 3]
    axes = {
        "x": matrix[:3, 0],
        "y": matrix[:3, 1],
        "z": matrix[:3, 2],
    }
    actors = []
    for name, direction in axes.items():
        arrow = pv.Arrow(
            start=origin,
            direction=direction,
            scale=length,
            tip_length=0.25,
            tip_radius=0.06,
            shaft_radius=0.02,
        )
        actor = plotter.add_mesh(arrow, color=AXIS_COLORS[name])
        actors.append(actor)

    if label is not None:
        actors.append(_add_joint_label(plotter, origin, label, offset_pixels=label_offset_pixels))

    if show_matrix:
        result = add_matrix_image(plotter, transform, origin, prefix=prefix, offset=matrix_offset)
        actors.append(result["follower"])

    return actors


def _add_legend(plotter):
    title_y = 0.965
    line_dy = 0.030
    plotter.add_text(
        "Matrix legend",
        position=(0.01, title_y),
        font_size=11,
        color="black",
        font="arial",
        viewport=True,
    )
    plotter.add_text(
        "T = [ R | p ; 0 0 0 1 ]",
        position=(0.01, title_y - line_dy),
        font_size=11,
        color="black",
        font="courier",
        viewport=True,
    )
    plotter.add_text(
        "blue  =  R (rotation 3x3)",
        position=(0.01, title_y - 2 * line_dy),
        font_size=11,
        color=ROTATION_FG,
        font="courier",
        viewport=True,
    )
    plotter.add_text(
        "green =  p (position 3x1)",
        position=(0.01, title_y - 3 * line_dy),
        font_size=11,
        color=POSITION_FG,
        font="courier",
        viewport=True,
    )
    plotter.add_text(
        "orange = [0 0 0 1] homogeneous row",
        position=(0.01, title_y - 4 * line_dy),
        font_size=11,
        color=HOMOGENEOUS_FG,
        font="courier",
        viewport=True,
    )


def _build_frame_skeleton(
    plotter,
    label,
    prefix,
    length,
    show_matrix,
    matrix_offset,
    label_offset_pixels=(20, 18),
):
    """Build XYZ arrows + label + matrix follower in canonical pose (identity).

    Returns a dict of actor handles meant to be moved each frame via
    user_matrix / SetPosition / SetTexture rather than rebuilt.
    """
    arrows = {}
    for axis_name, direction in (("x", (1.0, 0.0, 0.0)), ("y", (0.0, 1.0, 0.0)), ("z", (0.0, 0.0, 1.0))):
        arrow = pv.Arrow(
            start=(0.0, 0.0, 0.0),
            direction=direction,
            scale=length,
            tip_length=0.25,
            tip_radius=0.06,
            shaft_radius=0.02,
        )
        arrows[axis_name] = plotter.add_mesh(arrow, color=AXIS_COLORS[axis_name])

    label_actor = _add_joint_label(plotter, (0.0, 0.0, 0.0), label, offset_pixels=label_offset_pixels)

    follower = None
    if show_matrix:
        result = add_matrix_image(
            plotter, sm.SE3(), (0.0, 0.0, 0.0), prefix=prefix, offset=matrix_offset
        )
        follower = result["follower"]

    return {
        "arrows": arrows,
        "label": label_actor,
        "follower": follower,
        "prefix": prefix,
        "matrix_offset": np.asarray(matrix_offset, dtype=float),
    }


def _matrix_texture_for(transform, prefix, decimals=2):
    """Return a (cached) vtkTexture for the rounded transform/prefix combo."""
    matrix = np.round(np.asarray(transform.A, dtype=float), decimals)
    matrix[np.abs(matrix) < 10 ** (-decimals)] = 0
    key = (matrix.tobytes(), str(prefix), decimals)
    cached = _matrix_texture_cache.get(key)
    if cached is not None:
        return cached
    img = _render_matrix_image(transform, prefix=prefix, decimals=decimals)
    texture = pv.numpy_to_texture(img)
    if len(_matrix_texture_cache) > _MATRIX_CACHE_LIMIT:
        _matrix_texture_cache.clear()
    _matrix_texture_cache[key] = texture
    return texture


def _update_frame_skeleton(skeleton, transform):
    """Move/orient an existing frame skeleton to match `transform`."""
    T = np.asarray(transform.A, dtype=float)
    origin = T[:3, 3]

    for axis_actor in skeleton["arrows"].values():
        axis_actor.user_matrix = T

    skeleton["label"].SetPosition(float(origin[0]), float(origin[1]), float(origin[2]))

    follower = skeleton["follower"]
    if follower is not None:
        target = origin + skeleton["matrix_offset"]
        follower.SetPosition(float(target[0]), float(target[1]), float(target[2]))
        follower.SetTexture(_matrix_texture_for(transform, skeleton["prefix"]))


DEMO_POSES = [
    ("Pose 1 - Home / Identity", np.zeros(6)),
    ("Pose 2 - Shoulder sweep", np.deg2rad([90.0, -20.0, 20.0, 0.0, 0.0, 0.0])),
    ("Pose 3 - Elbow configuration", np.deg2rad([30.0, -70.0, 90.0, 0.0, 0.0, 0.0])),
    ("Pose 4 - Wrist articulation", np.deg2rad([30.0, -40.0, 50.0, 90.0, -45.0, 120.0])),
    ("Pose 5 - Near singularity", np.zeros(6)),
    ("Pose 6 - Folded pose", np.deg2rad([-90.0, 60.0, -80.0, 0.0, 45.0, 0.0])),
]
SEGMENT_DURATION_S = 5.0
TRANSITION_DURATION_S = 2.0


def _smoothstep(u):
    u = max(0.0, min(1.0, u))
    return u * u * (3.0 - 2.0 * u)


def _q_and_name_at(elapsed_s, poses=DEMO_POSES, seg=SEGMENT_DURATION_S, trans=TRANSITION_DURATION_S):
    cycle = seg * len(poses)
    t = elapsed_s % cycle
    idx = int(t // seg) % len(poses)
    in_seg = t - idx * seg
    name, target_q = poses[idx]
    if in_seg < trans:
        prev_q = poses[(idx - 1) % len(poses)][1]
        u = _smoothstep(in_seg / trans)
        q = prev_q + u * (target_q - prev_q)
    else:
        q = target_q
    return q.copy(), name


def show_robot_viewer(
    robot,
    frame_length=0.08,
    robot_opacity=ROBOT_OPACITY,
    animate=True,
    animate_interval_ms=33,
):
    plotter = pv.Plotter(window_size=(1280, 900))
    plotter.set_background("white")
    plotter.add_axes()

    # Static world frame + legend (never updated).
    add_frame(
        plotter,
        sm.SE3(),
        label="World",
        prefix=0,
        length=frame_length * 1.3,
        show_matrix=True,
        matrix_offset=(0.18, 0.0, -0.05),
    )
    _add_legend(plotter)

    mesh_actors = []
    for entry in extract_link_meshes(robot):
        mesh = entry["mesh"].copy()
        actor = plotter.add_mesh(
            mesh,
            color=entry["color"],
            opacity=robot_opacity,
            smooth_shading=True,
        )
        mesh_actors.append(
            {
                "actor": actor,
                "link_name": entry["link"].name,
                "local_T": np.asarray(entry["local_T"], dtype=float),
            }
        )

    matrix_visible_joints = {1, 2, 3, 4, 5}
    default_offset = (0.10, 0.0, 0.04)
    joint_matrix_offsets = {
        4: (0.10, 0.0, 0.17),
        5: (0.10, 0.0, -0.10),
    }

    joint_skeletons = []
    for j_idx in range(1, 7):
        joint_skeletons.append(
            _build_frame_skeleton(
                plotter,
                label=f"J{j_idx}",
                prefix=j_idx,
                length=frame_length,
                show_matrix=j_idx in matrix_visible_joints,
                matrix_offset=joint_matrix_offsets.get(j_idx, default_offset),
            )
        )

    tcp_skeleton = _build_frame_skeleton(
        plotter,
        label="TCP",
        prefix="TCP",
        length=frame_length * 1.5,
        show_matrix=True,
        matrix_offset=(0.10, 0.0, 0.17),
        label_offset_pixels=(28, 42),
    )

    def update_scene():
        link_transforms = compute_link_transforms(robot)
        transforms_by_link = {item["link"].name: item["T"] for item in link_transforms}

        for ma in mesh_actors:
            link_T = transforms_by_link.get(ma["link_name"])
            if link_T is None:
                continue
            user_T = np.asarray(link_T.A) @ ma["local_T"]
            ma["actor"].user_matrix = user_T

        joint_transforms = [entry["T"] for entry in link_transforms if entry["link"].isjoint]
        for j_idx, T in enumerate(joint_transforms[:6], start=1):
            _update_frame_skeleton(joint_skeletons[j_idx - 1], T)

        _update_frame_skeleton(tcp_skeleton, link_transforms[-1]["T"])

    update_scene()

    plotter.camera_position = "iso"
    plotter.camera.zoom(1.4)

    if not animate:
        plotter.show()
        return

    cycle_s = SEGMENT_DURATION_S * len(DEMO_POSES)
    n_steps = max(60, int(round(cycle_s * 1000.0 / animate_interval_ms)))

    # Centered top pose-name overlay.
    pose_text = plotter.add_text(
        DEMO_POSES[0][0],
        position="upper_edge",
        font_size=18,
        color="black",
        font="courier",
    )

    def _set_pose_text(name):
        try:
            pose_text.SetText(7, name)  # corner 7 == upper_edge for vtkCornerAnnotation
        except Exception:
            try:
                pose_text.SetInput(name)
            except Exception:
                pass

    # Try to load the matplotlib image cache from disk so we skip matplotlib
    # entirely on subsequent runs.
    loaded = _load_disk_cache()
    if loaded:
        print(f"[viewer] loaded {loaded} matrix images from disk cache")

    # Pre-render every animation frame so the live loop is a pure cache lookup.
    print(f"[viewer] pre-rendering {n_steps} animation frames "
          f"({cycle_s:.0f}s cycle, {len(DEMO_POSES)} poses)...")
    images_before = len(_matrix_image_cache)
    t_warm = time.time()
    for i in range(n_steps):
        t = i * animate_interval_ms / 1000.0
        q, _ = _q_and_name_at(t)
        robot.q = q
        link_transforms = compute_link_transforms(robot)
        joint_transforms = [e["T"] for e in link_transforms if e["link"].isjoint]
        for j_idx, T in enumerate(joint_transforms[:6], start=1):
            sk = joint_skeletons[j_idx - 1]
            if sk["follower"] is not None:
                _matrix_texture_for(T, sk["prefix"])
        _matrix_texture_for(link_transforms[-1]["T"], tcp_skeleton["prefix"])
        if i and i % max(1, n_steps // 10) == 0:
            print(f"  ...{int(100 * i / n_steps)}%")
    new_images = len(_matrix_image_cache) - images_before
    print(f"[viewer] prewarm done in {time.time() - t_warm:.1f}s "
          f"({len(_matrix_texture_cache)} textures, {new_images} new images rendered)")
    if new_images > 0:
        _save_disk_cache()
        print(f"[viewer] saved disk cache to {_disk_cache_path()}")

    q0, name0 = _q_and_name_at(0.0)
    robot.q = q0
    _set_pose_text(name0)
    update_scene()

    plotter.show(auto_close=False, interactive_update=True)
    t0 = time.time()
    last_name = name0
    try:
        while True:
            elapsed = time.time() - t0
            i = int(elapsed * 1000.0 / animate_interval_ms) % n_steps
            t = i * animate_interval_ms / 1000.0
            q, name = _q_and_name_at(t)
            robot.q = q
            update_scene()
            if name != last_name:
                _set_pose_text(name)
                last_name = name
            plotter.update(animate_interval_ms, force_redraw=True)
    except (KeyboardInterrupt, RuntimeError, AttributeError):
        pass
    plotter.close()
