#!/usr/bin/env python3
"""
mc_slab_movie.py  (step-by-step live version)

Animated 2D Monte Carlo neutron transport in a slab with *incremental*
per-frame motion. A pool of live particles advances one free-flight step
at a time. In-slab segments fade toward a dark blue and then remain.
Absorption dots fade out and disappear.

Examples:
  # Slow, very readable live motion
  python mc_slab_movie.py --Sigma_t 6 --c 0.997 --thickness 1.5 \
      --N 800 --frames 600 --fps 20 --spawn-per-frame 3 --steps-per-frame 1

  # Denser, slower, lots of trails
  python mc_slab_movie.py --Sigma_t 10 --c 0.999 --thickness 2.0 \
      --N 1000 --frames 800 --fps 18 --spawn-per-frame 4 --steps-per-frame 1 --max-live 120

  # Pick a specific dark-blue target
  python mc_slab_movie.py --fade-target-color "#0b2a8f"
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from dataclasses import dataclass
from tqdm import tqdm


def _color_to_rgb_tuple(color):
    return mcolors.to_rgb(color)


@dataclass
class Particle:
    x: float
    y: float
    mu: float  # cos(theta)
    v: float   # sin(theta)
    alive: bool = True


class SimulationAnimator:
    def __init__(self, args, fig, ax_hud, ax_plot):
        self.args = args
        self.fig = fig
        self.ax_hud = ax_hud
        self.ax_plot = ax_plot

        # ---- Visual parameters ----
        self.historic_lw = 1.75
        self.active_lw = 2.2
        self.dot_size = 10

        self.active_rgb = _color_to_rgb_tuple('cyan')
        self.historic_start_rgb = _color_to_rgb_tuple('deepskyblue')
        self.exit_rgb = _color_to_rgb_tuple('limegreen')
        self.absorb_rgb = _color_to_rgb_tuple('red')

        # Trails fade toward this dark blue and then stay
        self.fade_target_rgb = _color_to_rgb_tuple(self.args.fade_target_color)

        # Per-frame fade
        self.fade_gamma_lines = 0.94
        self.fade_gamma_dots = 0.98

        # Lines will STOP fading once below this intensity (kept on canvas)
        self.line_hold_intensity = 0.40
        self.line_final_alpha = 0.85

        # Dots vanish when too dim
        self.min_intensity_dots = 0.02

        # RNG and counters
        self.rng = np.random.default_rng(self.args.seed)
        self.n_reflected = 0
        self.n_absorbed = 0
        self.n_transmitted = 0
        self.n_spawned = 0   # total particles created so far (<= N)
        self.n_completed = 0 # total finished

        # Live pool
        self.live_particles = []  # list[Particle]

        # HUD placeholders
        self.hud_text_title = None
        self.hud_text_header = None
        self.hud_reflected_val = None
        self.hud_absorbed_val = None
        self.hud_transmitted_val = None
        self.hud_completed_val = None
        self.hud_remaining_val = None
        self.init_artists = []

        # Fading registries
        self._fading_lines = []  # dict(artist, base_rgb, intensity, gamma)
        self._fading_dots = []

    # ---------- Fading utilities ----------
    def _register_fading_line(self, artist, base_rgb, start_intensity=1.0):
        self._fading_lines.append({
            "artist": artist,
            "base_rgb": base_rgb,
            "intensity": float(start_intensity),
            "gamma": self.fade_gamma_lines
        })

    def _register_fading_dot(self, artist, base_rgb, start_intensity=1.0):
        self._fading_dots.append({
            "artist": artist,
            "base_rgb": base_rgb,
            "intensity": float(start_intensity),
            "gamma": self.fade_gamma_dots
        })

    def _apply_color_line_blend(self, artist, base_rgb, intensity):
        """Blend from base_rgb toward fade_target_rgb; keep alpha >= floor."""
        br, bg, bb = base_rgb
        tr, tg, tb = self.fade_target_rgb
        t = 1.0 - max(0.0, min(1.0, intensity))  # 0 -> base, 1 -> target
        r = br * (1 - t) + tr * t
        g = bg * (1 - t) + tg * t
        b = bb * (1 - t) + tb * t
        alpha = max(self.line_final_alpha, min(1.0, intensity))
        artist.set_color((r, g, b, alpha))

    def _apply_color_dot(self, artist, base_rgb, intensity):
        r, g, b = base_rgb
        r *= intensity; g *= intensity; b *= intensity
        alpha = max(0.20, min(1.0, intensity))
        artist.set_color((r, g, b, alpha))

    def _update_faders(self):
        # Lines: fade until hold threshold, then freeze (keep on canvas)
        keep_lines = []
        for item in self._fading_lines:
            item["intensity"] *= item["gamma"]
            if item["intensity"] <= self.line_hold_intensity:
                try:
                    self._apply_color_line_blend(item["artist"], item["base_rgb"], 0.0)  # hit target color
                except Exception:
                    pass
                # Stop tracking this line; leave it drawn permanently
                continue
            # Still fading
            self._apply_color_line_blend(item["artist"], item["base_rgb"], item["intensity"])
            keep_lines.append(item)
        self._fading_lines = keep_lines

        # Dots: fade & remove when too dim
        keep_dots = []
        for item in self._fading_dots:
            item["intensity"] *= item["gamma"]
            if item["intensity"] < self.min_intensity_dots:
                try:
                    item["artist"].remove()
                except Exception:
                    pass
                continue
            self._apply_color_dot(item["artist"], item["base_rgb"], item["intensity"])
            keep_dots.append(item)
        self._fading_dots = keep_dots

    # ---------- Physics helpers ----------
    def _spawn_particles(self, n_to_spawn):
        """Create up to n_to_spawn new particles at (0,0) with forward hemisphere angles."""
        H = self.args.thickness
        n_avail = self.args.N - self.n_spawned
        n_to_make = max(0, min(n_to_spawn, n_avail))
        for _ in range(n_to_make):
            theta = self.rng.uniform(-np.pi/2, np.pi/2)  # initial towards +x
            p = Particle(
                x=0.0,
                y=0.0,
                mu=np.cos(theta),
                v=np.sin(theta),
                alive=True
            )
            self.live_particles.append(p)
        self.n_spawned += n_to_make

    def _do_one_step(self, p: Particle):
        """
        Advance particle p by one free-flight step.
        Draw the in-slab segment (fading), then handle interaction:
        scatter vs absorption, or exit (reflect/transmit with outside segment).
        Returns: True if the particle is finished this step.
        """
        H = self.args.thickness
        Sigma_t = self.args.Sigma_t
        c = self.args.c

        # Exponential free-flight
        d = -np.log(self.rng.random()) / Sigma_t
        x_new = p.x + d * p.mu
        y_new = p.y + d * p.v

        # If exits slab during this step, intersect plane x=0 or x=H
        if x_new >= H or x_new < 0:
            # Determine intersection point with boundary
            dx = d * p.mu
            dy = d * p.v
            # Solve p.x + t*mu = boundary
            if x_new >= H:
                x_boundary = H
                x_final_margin = self.args.right_margin
                self.n_transmitted += 1
            else:
                x_boundary = 0.0
                x_final_margin = self.args.left_margin
                self.n_reflected += 1

            t_hit = (x_boundary - p.x) / (p.mu if p.mu != 0 else 1e-12)
            y_hit = p.y + t_hit * p.v

            # 1) Draw inside segment up to boundary (active -> fades)
            self._plot_in_slab_segment([p.x, x_boundary], [p.y, y_hit], active=True)

            # 2) Draw outside segment in steady green
            y_final = y_hit + (x_final_margin - x_boundary) * (p.v / (p.mu if p.mu != 0 else 1e-12))
            (line_out,) = self.ax_plot.plot([x_boundary, x_final_margin], [y_hit, y_final],
                                            lw=self.historic_lw)
            lr, lg, lb = self.exit_rgb
            line_out.set_color((lr, lg, lb, 0.7))

            p.alive = False
            self.n_completed += 1
            return True

        # Still inside — draw in-slab segment from (x,y) to (x_new,y_new)
        self._plot_in_slab_segment([p.x, x_new], [p.y, y_new], active=True)

        # Move the particle to the new location
        p.x, p.y = x_new, y_new

        # Interaction at the end of the step: scatter or absorb
        if self.rng.random() < c:
            theta = self.rng.uniform(0, 2*np.pi)
            p.mu = np.cos(theta)
            p.v  = np.sin(theta)
            return False  # keep going next step
        else:
            # Absorbed: mark a fading dot
            (dot,) = self.ax_plot.plot(p.x, p.y, 'o', markersize=self.dot_size)
            self._apply_color_dot(dot, self.absorb_rgb, 1.0)
            self._register_fading_dot(dot, self.absorb_rgb, 1.0)
            self.n_absorbed += 1
            p.alive = False
            self.n_completed += 1
            return True

    # ---------- Drawing helpers ----------
    def _plot_in_slab_segment(self, xs, ys, active=False):
        """Plot a short in-slab segment that fades to dark blue and then stays."""
        (line,) = self.ax_plot.plot(xs, ys, lw=(self.active_lw if active else self.historic_lw))
        base = self.active_rgb if active else self.historic_start_rgb
        self._apply_color_line_blend(line, base, 1.0)      # start bright
        self._register_fading_line(line, base_rgb=base, start_intensity=1.0)

    # ---------- Matplotlib init/update ----------
    def init_plot(self):
        H = self.args.thickness
        title = f'Neutron Slab (live): H={H}, Sigma_t={self.args.Sigma_t}, c={self.args.c}'
        self.ax_plot.set_title(title)

        # Slab background
        self.ax_plot.axvspan(0, H, color='black', zorder=0)
        self.ax_plot.set_facecolor((0.05, 0.05, 0.07))
        # Beam indicator
        beam_x_pos = self.args.left_margin + (0 - self.args.left_margin) * 0.4
        arrow_dx = (0 - self.args.left_margin) * 0.2
        arr = self.ax_plot.arrow(beam_x_pos, 0, arrow_dx, 0,
                                 head_width=0.03, head_length=0.05,
                                 fc=self.active_rgb, ec=self.active_rgb, zorder=1)
        txt = self.ax_plot.text(beam_x_pos, 0.05, 'beam', color=self.active_rgb,
                                ha='center', va='bottom', zorder=1)
        self.init_artists += [arr, txt]

        self.ax_plot.set_xlabel('x')
        self.ax_plot.set_ylabel('y')
        self.ax_plot.set_xlim(self.args.left_margin, self.args.right_margin)
        self.ax_plot.set_ylim(-0.6, 0.6)
        self.ax_plot.grid(color='white', linestyle='--', linewidth=0.4, alpha=0.15)

        # HUD
        self.ax_hud.axis('off')
        self.ax_hud.set_ylim(0, 1)
        self.ax_hud.set_xlim(0, 1)
        hud_font = {'family': 'monospace', 'fontsize': 10}
        x_pos = 0.05
        self.hud_text_title = self.ax_hud.text(x_pos, 0.90, "Outcomes", fontweight='bold', **hud_font)
        self.hud_text_header = self.ax_hud.text(
            x_pos, 0.85, "{: <12}{: >7}{: >10}".format("Type", "Count", "Percent"), **hud_font
        )
        self.init_artists.append(self.ax_hud.text(x_pos, 0.83, "-"*30, **hud_font))
        self.hud_reflected_val = self.ax_hud.text(x_pos, 0.78, "", **hud_font)
        self.hud_absorbed_val = self.ax_hud.text(x_pos, 0.73, "", **hud_font)
        self.hud_transmitted_val = self.ax_hud.text(x_pos, 0.68, "", **hud_font)
        self.init_artists.append(self.ax_hud.text(x_pos, 0.63, "-"*30, **hud_font))
        self.hud_completed_val = self.ax_hud.text(x_pos, 0.58, "", **hud_font)
        self.hud_remaining_val = self.ax_hud.text(x_pos, 0.53, "", **hud_font)
        self.init_artists.extend([
            self.hud_text_title, self.hud_text_header,
            self.hud_reflected_val, self.hud_absorbed_val, self.hud_transmitted_val,
            self.hud_completed_val, self.hud_remaining_val
        ])
        return self.init_artists

    def _hud_update(self):
        done = self.n_completed
        remaining = self.args.N - self.n_spawned + len(self.live_particles)
        total_seen = max(1, done)  # avoid 0-div

        pct_ref = (self.n_reflected / total_seen) * 100.0
        pct_abs = (self.n_absorbed / total_seen) * 100.0
        pct_trn = (self.n_transmitted / total_seen) * 100.0

        fmt_row = "{: <12}{: >7}{: >9.2f}%"
        self.hud_reflected_val.set_text(fmt_row.format("Reflected", self.n_reflected, pct_ref))
        self.hud_absorbed_val.set_text(fmt_row.format("Absorbed", self.n_absorbed, pct_abs))
        self.hud_transmitted_val.set_text(fmt_row.format("Transmitted", self.n_transmitted, pct_trn))

        fmt_total = "{: <12}{: >7}"
        self.hud_completed_val.set_text(fmt_total.format("Completed", done))
        self.hud_remaining_val.set_text(fmt_total.format("In system:", remaining))

        return [self.hud_reflected_val, self.hud_absorbed_val,
                self.hud_transmitted_val, self.hud_completed_val, self.hud_remaining_val]

    def update_frame(self, frame_idx):
        # Spawn a few new particles each frame (respect N and max-live)
        space_for_live = max(0, self.args.max_live - len(self.live_particles))
        to_spawn = min(self.args.spawn_per_frame, space_for_live)
        self._spawn_particles(to_spawn)

        # Each live particle attempts a few steps this frame
        # (iterate over a snapshot list so we can remove finished safely)
        for _ in range(self.args.steps_per_frame):
            if not self.live_particles:
                break
            for p in list(self.live_particles):
                if not p.alive:
                    # Just in case (should be removed below)
                    continue
                finished = self._do_one_step(p)
                if finished:
                    try:
                        self.live_particles.remove(p)
                    except ValueError:
                        pass

        # Update fades (lines and dots)
        self._update_faders()

        # HUD text
        return self._hud_update()


# ---------- Argparse ----------
def parse_args():
    parser = argparse.ArgumentParser(
        description="2D Monte Carlo Neutron Slab Simulation Animator (live step-by-step)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Simulation
    parser.add_argument('--thickness', type=float, default=1.0, help="The slab thickness H.")
    parser.add_argument('--Sigma_t', type=float, default=10.0, help="Total cross-section Σt (mean free path = 1/Σt).")
    parser.add_argument('--c', type=float, default=0.99, help="Scattering ratio Σs/Σt.")
    parser.add_argument('--N', type=int, default=600, help="Total number of particles to simulate.")
    parser.add_argument('--seed', type=int, default=42, help="RNG seed.")

    # Live motion controls
    parser.add_argument('--spawn-per-frame', type=int, default=1,
                        help="New particles injected each frame (until N is reached).")
    parser.add_argument('--steps-per-frame', type=int, default=1,
                        help="How many free-flight steps each live particle attempts per frame.")
    parser.add_argument('--max-live', type=int, default=20,
                        help="Maximum number of concurrently active particles.")

    # Animation
    parser.add_argument('--frames', type=int, default=700, help="Total animation frames.")
    parser.add_argument('--fps', type=int, default=30, help="Frames per second.")
    parser.add_argument('--out', type=str, default="mc_slab_movie.mp4", help="Output .mp4 path.")

    # Figure/layout
    parser.add_argument('--dpi', type=int, default=150, help="DPI for output.")
    parser.add_argument('--fig-width', type=float, default=12, help="Figure width (inches).")
    parser.add_argument('--fig-height', type=float, default=7, help="Figure height (inches).")
    parser.add_argument('--left-margin', type=float, default=-0.5, help="Left x-limit of plot.")
    parser.add_argument('--right-margin', type=float, default=1.8, help="Right x-limit of plot.")
    parser.add_argument('--hud-width-ratio', type=float, default=0.4, help="HUD column width ratio.")

    # Fade target
    parser.add_argument('--fade-target-color', type=str, default='#0b2a8f',
                        help="Dark blue target that in-slab trails fade toward.")

    args = parser.parse_args()

    # Margins sanity
    if args.right_margin <= args.thickness:
        print(f"[warning] --right-margin ({args.right_margin}) <= --thickness ({args.thickness}); bumping.")
        args.right_margin = args.thickness + 0.5

    if args.left_margin >= 0:
        print(f"[warning] --left-margin ({args.left_margin}) is >= 0; setting to -0.5")
        args.left_margin = -0.5

    # Validate color
    try:
        _ = mcolors.to_rgb(args.fade_target_color)
    except ValueError:
        print(f"[warning] Unrecognized --fade-target-color '{args.fade_target_color}'. Falling back to '#0b2a8f'.")
        args.fade_target_color = '#0b2a8f'

    return args


# ---------- Main ----------
def main():
    args = parse_args()

    fig = plt.figure(figsize=(args.fig_width, args.fig_height), dpi=args.dpi)
    gs = gridspec.GridSpec(1, 2, width_ratios=[args.hud_width_ratio, 1])
    ax_hud = fig.add_subplot(gs[0, 0])
    ax_plot = fig.add_subplot(gs[0, 1])
    plt.tight_layout(pad=2.0)

    animator = SimulationAnimator(args, fig, ax_hud, ax_plot)

    pbar = tqdm(total=args.frames, desc="Saving animation", unit="frame", ncols=80, file=sys.stdout)

    def progress_callback(current_frame, total_frames):
        pbar.update(1)

    ani = animation.FuncAnimation(
        fig,
        animator.update_frame,
        frames=args.frames,
        init_func=animator.init_plot,
        blit=False
    )

    try:
        ani.save(
            args.out,
            fps=args.fps,
            dpi=args.dpi,
            progress_callback=progress_callback,
            writer='ffmpeg'
        )
        pbar.close()
        print(f"\nSaved: {args.out}")
    except Exception as e:
        pbar.close()
        print(f"\n[error] Failed to save animation: {e}", file=sys.stderr)
        print("Please ensure 'ffmpeg' is installed and on PATH.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
