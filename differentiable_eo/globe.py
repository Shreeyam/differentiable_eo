"""Globe rendering utilities for constellation visualization.

Supports 3-layer compositing:
  - Back layer: orbital elements behind Earth (vector, matplotlib)
  - Middle layer: Earth texture (raster, from Blender pre-render)
  - Front layer: orbital elements in front of Earth (vector, matplotlib)

Can also render a single composite matplotlib figure (vector orbits + raster earth)
suitable for both static figures and animation frames.
"""

import math
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from .constants import R_EARTH


def solve_kepler(M, e, tol=1e-10, max_iter=20):
    """Solve Kepler's equation M = E - e*sin(E) for eccentric anomaly E."""
    E = M  # initial guess
    for _ in range(max_iter):
        dE = (M - E + e * math.sin(E)) / (1 - e * math.cos(E))
        E += dE
        if abs(dE) < tol:
            break
    return E


def eci_xyz(inc_rad, raan_rad, ma_rad, r, ecc=0.0, argp_rad=0.0):
    """Convert orbital elements to ECI position.

    For circular orbits (ecc=0), r is the orbital radius and ma_rad is
    the angle in the orbital plane. For eccentric orbits, r is the
    semi-major axis, and the position is computed via Kepler's equation.

    Args:
        inc_rad: inclination (radians)
        raan_rad: RAAN (radians)
        ma_rad: mean anomaly (radians)
        r: orbital radius (circular) or semi-major axis (eccentric)
        ecc: eccentricity (default 0 for backward compatibility)
        argp_rad: argument of perigee (radians, default 0)
    """
    if ecc > 1e-6:
        # Solve Kepler's equation for eccentric anomaly
        E = solve_kepler(ma_rad, ecc)
        # True anomaly
        nu = 2.0 * math.atan2(
            math.sqrt(1 + ecc) * math.sin(E / 2),
            math.sqrt(1 - ecc) * math.cos(E / 2))
        # Radius at this true anomaly
        r_pos = r * (1 - ecc * math.cos(E))
        # Angle in orbital plane = true anomaly + argument of perigee
        angle = nu + argp_rad
    else:
        r_pos = r
        angle = ma_rad + argp_rad

    x_orb = r_pos * math.cos(angle)
    y_orb = r_pos * math.sin(angle)
    ci, si = math.cos(inc_rad), math.sin(inc_rad)
    cr, sr = math.cos(raan_rad), math.sin(raan_rad)
    x_inc, y_inc, z_inc = x_orb, y_orb * ci, y_orb * si
    return (x_inc * cr - y_inc * sr,
            x_inc * sr + y_inc * cr,
            z_inc)


def camera_direction(elev_deg, azim_deg):
    """Unit vector from origin toward camera (matplotlib convention)."""
    el_r = math.radians(elev_deg)
    az_r = math.radians(azim_deg)
    return np.array([math.cos(el_r) * math.cos(az_r),
                     math.cos(el_r) * math.sin(az_r),
                     math.sin(el_r)])


def is_occluded(pos, cam_dir, occluder_radius):
    """True if pos is behind the sphere of given radius from camera's perspective."""
    along = np.dot(pos, cam_dir)
    perp = np.linalg.norm(pos - along * cam_dir)
    return (along < 0) and (perp < occluder_radius)


def measure_ball_pixels(lim, fig_size=3.2, dpi=300, elev=25, azim=45):
    """Render a reference sphere and measure its pixel extent in the matplotlib frame.

    Returns (ball_diam, center_x, center_y, frame_size) in pixels.
    """
    fig = plt.figure(figsize=(fig_size, fig_size), facecolor='none')
    ax = fig.add_subplot(111, projection='3d', facecolor='none')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    u = np.linspace(0, 2 * math.pi, 60)
    v = np.linspace(0, math.pi, 30)
    xe = R_EARTH * np.outer(np.cos(u), np.sin(v))
    ye = R_EARTH * np.outer(np.sin(u), np.sin(v))
    ze = R_EARTH * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xe, ye, ze, alpha=1.0, color='red', linewidth=0)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    ax.axis('off')
    ax.view_init(elev=elev, azim=azim)

    fig.savefig('/tmp/_ball_ref.png', dpi=dpi, transparent=True)
    plt.close()

    img = np.array(Image.open('/tmp/_ball_ref.png'))
    alpha = img[:, :, 3]
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    ball_diam = max(rmax - rmin + 1, cmax - cmin + 1)
    cx_px = (cmin + cmax) / 2
    cy_px = (rmin + rmax) / 2
    frame_px = img.shape[0]
    return ball_diam, cx_px, cy_px, frame_px


def render_globe(ax_earth, ax3d, raans_deg, mas_deg, earth_img_path,
                 ball_diam, ball_cx_px, ball_cy_px, frame_px,
                 n_planes, n_sats_per_plane, alt_km=550.0, inc_deg=60.0,
                 elev=25, azim=45, earth_alpha=0.60, earth_shrink=27,
                 colors=None, ring_points=200, show_arrows=False,
                 prev_raans_deg=None, prev_mas_deg=None, arrow_scale=8.0,
                 eccs=None, argps_deg=None):
    """Render orbital elements on ax3d with earth image on ax_earth.

    Args:
        ax_earth: 2D matplotlib axes for the earth image (should be pre-positioned)
        ax3d: 3D matplotlib axes for orbital elements
        raans_deg: list of RAAN values per plane (degrees)
        mas_deg: list of mean anomaly values per satellite (degrees)
        earth_img_path: path to Blender-rendered earth PNG
        ball_diam, ball_cx_px, ball_cy_px, frame_px: from measure_ball_pixels()
        n_planes, n_sats_per_plane: constellation geometry
        alt_km: orbital altitude in km (or semi-major axis minus R_earth if eccentric)
        inc_deg: inclination in degrees
        elev, azim: camera angles
        earth_alpha: opacity of earth layer
        earth_shrink: pixels to shrink earth from measured ball diameter
        colors: per-plane colors (auto-generated if None)
        ring_points: number of points per orbital ring
        show_arrows: if True and prev_* provided, draw movement arrows
        prev_raans_deg, prev_mas_deg: previous state for arrows
        arrow_scale: scaling factor for movement arrows
        eccs: per-plane eccentricities (default None = circular)
        argps_deg: per-plane argument of perigee in degrees (default None = 0)
    """
    r_orbit = R_EARTH + alt_km
    inc_rad = math.radians(inc_deg)
    n_sats = n_planes * n_sats_per_plane
    ring_th = np.linspace(0, 2 * math.pi, ring_points)
    cam = camera_direction(elev, azim)

    # Default to circular if not specified
    if eccs is None:
        eccs = [0.0] * n_planes
    if argps_deg is None:
        argps_deg = [0.0] * n_planes

    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, n_planes))

    # ── Earth image ──────────────────────────────────────────────────────────
    earth_size_px = ball_diam - earth_shrink
    earth_size_frac = earth_size_px / frame_px
    earth_cx_frac = ball_cx_px / frame_px
    earth_cy_frac = 1.0 - ball_cy_px / frame_px

    ax_earth.clear()
    earth_img = Image.open(earth_img_path).convert('RGBA')
    earth_arr = np.array(earth_img, dtype=np.float32)
    earth_arr[:, :, 3] *= earth_alpha
    earth_arr = earth_arr.astype(np.uint8)
    ax_earth.imshow(earth_arr, interpolation='bilinear')
    ax_earth.axis('off')
    ax_earth.set_facecolor('none')

    # Reposition earth axes to match ball position, relative to the 3D axes bounds
    ax3d_pos = ax3d.get_position()
    # Map ball-in-frame fractions to figure coordinates via the 3D axes bounds
    earth_left = ax3d_pos.x0 + (earth_cx_frac - earth_size_frac / 2) * ax3d_pos.width
    earth_bottom = ax3d_pos.y0 + (earth_cy_frac - earth_size_frac / 2) * ax3d_pos.height
    earth_w = earth_size_frac * ax3d_pos.width
    earth_h = earth_size_frac * ax3d_pos.height
    ax_earth.set_position([earth_left, earth_bottom, earth_w, earth_h])

    # ── Orbital elements on 3D axes ──────────────────────────────────────────
    ax3d.clear()
    ax3d.patch.set_alpha(0)
    ax3d.set_facecolor('none')

    # Orbital rings
    for p in range(n_planes):
        raan_rad = math.radians(raans_deg[p])
        ecc_p = eccs[p]
        argp_p = math.radians(argps_deg[p])
        pts = np.array([eci_xyz(inc_rad, raan_rad, th, r_orbit,
                                ecc=ecc_p, argp_rad=argp_p) for th in ring_th])

        for seg_start in range(len(ring_th)):
            pos = pts[seg_start]
            occluded = is_occluded(pos, cam, R_EARTH)
            if occluded:
                seg_end = seg_start + 1
                while seg_end < len(ring_th) and is_occluded(pts[seg_end], cam, R_EARTH):
                    seg_end += 1
                seg = pts[seg_start:min(seg_end + 1, len(ring_th))]
                ax3d.plot(seg[:, 0], seg[:, 1], seg[:, 2], '-',
                          color=colors[p], alpha=0.25, lw=0.7, antialiased=True)
            else:
                seg_end = seg_start + 1
                while seg_end < len(ring_th) and not is_occluded(pts[seg_end], cam, R_EARTH):
                    seg_end += 1
                seg = pts[seg_start:min(seg_end + 1, len(ring_th))]
                ax3d.plot(seg[:, 0], seg[:, 1], seg[:, 2], '-',
                          color=colors[p], alpha=0.45, lw=0.9, antialiased=True)

    # Satellites
    sat_positions = []
    for i in range(n_sats):
        p = i // n_sats_per_plane
        pos = np.array(eci_xyz(inc_rad,
                               math.radians(raans_deg[p]),
                               math.radians(mas_deg[i]),
                               r_orbit,
                               ecc=eccs[p],
                               argp_rad=math.radians(argps_deg[p])))
        sat_positions.append(pos)
        occluded = is_occluded(pos, cam, R_EARTH)

        if occluded:
            ax3d.scatter(*pos, s=40, facecolors='none',
                         edgecolors=colors[p], linewidths=1.1,
                         depthshade=False, alpha=0.45)
        else:
            ax3d.scatter(*pos, s=40, color=colors[p],
                         edgecolors='k', linewidths=0.45,
                         depthshade=False, zorder=5)

    # Movement arrows
    if show_arrows and prev_raans_deg is not None and prev_mas_deg is not None:
        for i in range(n_sats):
            p = i // n_sats_per_plane
            prev_pos = np.array(eci_xyz(inc_rad,
                                        math.radians(prev_raans_deg[p]),
                                        math.radians(prev_mas_deg[i]),
                                        r_orbit))
            dx = (sat_positions[i][0] - prev_pos[0]) * arrow_scale
            dy = (sat_positions[i][1] - prev_pos[1]) * arrow_scale
            dz = (sat_positions[i][2] - prev_pos[2]) * arrow_scale
            length = math.sqrt(dx**2 + dy**2 + dz**2)
            if length > 5:
                ax3d.quiver(*sat_positions[i], dx, dy, dz,
                            color=colors[p], alpha=0.7,
                            arrow_length_ratio=0.2, linewidth=1.5)

    lim = r_orbit * 0.72
    ax3d.set_xlim(-lim, lim)
    ax3d.set_ylim(-lim, lim)
    ax3d.set_zlim(-lim, lim)
    ax3d.set_box_aspect([1, 1, 1])
    ax3d.axis('off')
    ax3d.view_init(elev=elev, azim=azim)


def setup_globe_axes(fig, position=None):
    """Create overlapping 2D (earth) + 3D (orbits) axes on a figure.

    Args:
        fig: matplotlib figure
        position: [left, bottom, width, height] for the axes (default full figure)

    Returns:
        (ax_earth, ax3d) tuple
    """
    if position is None:
        position = [0, 0, 1, 1]

    ax_earth = fig.add_axes(position, zorder=0)
    ax_earth.axis('off')
    ax_earth.set_facecolor('none')

    ax3d = fig.add_axes(position, projection='3d', zorder=1)
    ax3d.patch.set_alpha(0)
    ax3d.set_facecolor('none')

    return ax_earth, ax3d
