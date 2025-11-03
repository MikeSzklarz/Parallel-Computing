#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, os, re, subprocess, sys, time
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BANNER = "-----------------------------------------------------------------------------------------------------------"
TIME_RX = re.compile(r"^COMP_TIME:\s*([0-9.]+)\s*$", re.M)

# ----------------------------- helpers -----------------------------


def run(cmd, timeout_sec):
    t0 = time.time()
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_sec,
    )
    wall = time.time() - t0
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed ({p.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{p.stdout}\n----\nstderr:\n{p.stderr}"
        )
    return p.stdout, p.stderr, wall


def parse_comp_time(stdout, fallback_wall):
    m = TIME_RX.search(stdout)
    if m:
        try:
            t = float(m.group(1))
            if t > 0:
                return t
        except Exception:
            pass
    return fallback_wall


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def int_linspace(n1, n2, num):
    if num <= 1:
        return [int(n2)]
    xs = np.linspace(n1, n2, num=num, dtype=float)
    vals = sorted({int(round(v)) for v in xs})
    if vals[0] != n1:
        vals = [n1] + [v for v in vals if v != n1]
    if vals[-1] != n2:
        vals = [v for v in vals if v != n2] + [n2]
    return vals


def int_range_inclusive(start, stop, step):
    if step <= 0:
        raise ValueError("step must be > 0")
    out, v = [], start
    while v <= stop:
        out.append(v)
        v += step
    return out


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def print_banner():
    print(BANNER)


def stamp_name(Ns, Is, Ps, label):
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    i_desc = f"{min(Is)}-{max(Is)}x{len(Is)}"
    p_desc = f"{min(Ps)}-{max(Ps)}x{len(Ps)}"
    n_desc = f"{min(Ns)}-{max(Ns)}x{len(Ns)}"
    return f"N[{n_desc}]_I[{i_desc}]_P[{p_desc}]_{label}_{ts}"


# ----------------------------- plotting -----------------------------


def _style():
    plt.style.use("seaborn-v0_8")
    plt.rcParams["figure.dpi"] = 120


def _group_by_I(rows):
    byI = {}
    for r in rows:
        byI.setdefault(r["iters"], []).append(r)
    for I in list(byI.keys()):
        byI[I] = sorted(byI[I], key=lambda x: (x["rows"], x["P"]))
    return byI


def _unique_sorted(seq):
    return sorted(list({x for x in seq}))


def plot_curves_by_N(rows_I, ykey, out_png, ylabel, title):
    _style()
    fig, ax = plt.subplots(figsize=(8, 5))
    Ns = _unique_sorted(r["rows"] for r in rows_I)
    # all P values present in this group (used for plotting ideal lines)
    Ps_all = _unique_sorted(r["P"] for r in rows_I)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, N in enumerate(Ns):
        grp = [r for r in rows_I if r["rows"] == N]
        grp = sorted(grp, key=lambda r: r["P"])
        Ps = [g["P"] for g in grp]
        ys = [g[ykey] for g in grp]
        ax.plot(Ps, ys, marker="o", color=colors[i % len(colors)], label=f"N={N}")
    # Add ideal reference lines for some y-keys
    if ykey == "speedup":
        # ideal speedup is linear with threads (S = P)
        if Ps_all:
            ax.plot(
                Ps_all, Ps_all, color="k", linestyle=":", linewidth=1.0, label="Ideal"
            )
    elif ykey == "efficiency":
        # ideal efficiency is 1.0 for all P
        if Ps_all:
            ax.plot(
                [min(Ps_all), max(Ps_all)],
                [1.0, 1.0],
                color="k",
                linestyle=":",
                linewidth=1.0,
                label="Ideal",
            )
    ax.set_xlabel("Threads (P)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_eff_heatmap(rows_I, out_png, title):
    _style()
    Ns = _unique_sorted(r["rows"] for r in rows_I)
    Ps = _unique_sorted(r["P"] for r in rows_I)
    mat = np.full((len(Ns), len(Ps)), np.nan, dtype=float)
    index = {(r["rows"], r["P"]): r for r in rows_I}
    for i, N in enumerate(Ns):
        for j, P in enumerate(Ps):
            r = index.get((N, P))
            if r is not None:
                mat[i, j] = r["efficiency"]
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        mat,
        aspect="auto",
        origin="lower",
        vmin=0.0,
        vmax=1.0,
        extent=[min(Ps) - 0.5, max(Ps) + 0.5, min(Ns) - 0.5, max(Ns) + 0.5],
        cmap="viridis",
    )
    ax.set_xlabel("Threads (P)")
    ax.set_ylabel("Grid size N")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Efficiency (S/P)")
    ax.set_xticks(Ps)
    yticks = Ns if len(Ns) <= 12 else Ns[:: max(1, len(Ns) // 12)]
    ax.set_yticks(yticks)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def iso_points_for_E(rows_I, E_target):
    perP = {}
    for r in rows_I:
        P, N = r["P"], r["rows"]
        perP.setdefault(P, {}).setdefault(N, []).append(r["efficiency"])
    pts = []
    for P, Ns in perP.items():
        avg_by_N = [(N, sum(v) / len(v)) for N, v in Ns.items()]
        feas = [N for (N, Eavg) in avg_by_N if Eavg >= E_target]
        if feas:
            pts.append((P, min(feas)))
    return sorted(pts, key=lambda x: x[0])


def plot_iso_curve(rows_I, E_target, out_png, out_csv):
    pts = iso_points_for_E(rows_I, E_target)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["P", "N_min"])
        for P, Nmin in pts:
            w.writerow([P, Nmin])
    _style()
    fig, ax = plt.subplots(figsize=(8, 5))
    if pts:
        Ps, Ns = zip(*pts)
        ax.plot(Ps, Ns, marker="o")
    ax.set_xlabel("Threads (P)")
    ax.set_ylabel(f"Smallest N with E ≥ {E_target:.2f}")
    ax.set_title(f"Iso-efficiency (E*={E_target:.2f})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_iso_surface(rows_I, out_png, title):
    _style()
    Ns = _unique_sorted(r["rows"] for r in rows_I)
    Ps = _unique_sorted(r["P"] for r in rows_I)
    Z = np.full((len(Ns), len(Ps)), np.nan, dtype=float)
    idx = {(r["rows"], r["P"]): r["efficiency"] for r in rows_I}
    for i, N in enumerate(Ns):
        for j, P in enumerate(Ps):
            if (N, P) in idx:
                Z[i, j] = idx[(N, P)]
    fig, ax = plt.subplots(figsize=(8, 5))
    Zm = np.ma.array(Z, mask=np.isnan(Z))
    cs = ax.contourf(Ps, Ns, Zm, levels=np.linspace(0, 1, 11), cmap="viridis")
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label("Efficiency (S/P)")
    ax.set_xlabel("Threads (P)")
    ax.set_ylabel("Grid size N")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def make_plots_full(
    summary_csv: Path,
    curves_dir: Path,
    heatmaps_dir: Path,
    iso_dir: Path,
    eff_start: float,
    eff_step: float,
):
    rows = []
    with open(summary_csv) as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(
                {
                    "rows": int(r["rows"]),
                    "iters": int(r["iters"]),
                    "P": int(r["P"]),
                    "t1_mean": float(r["t1_mean"]),
                    "t1_std": float(r["t1_std"]),
                    "tp_mean": float(r["tp_mean"]),
                    "tp_std": float(r["tp_std"]),
                    "speedup": float(r["speedup"]),
                    "efficiency": float(r["efficiency"]),
                    "trials": int(r["trials"]),
                }
            )

    byI = _group_by_I(rows)
    ensure_dir(curves_dir)
    ensure_dir(heatmaps_dir)
    ensure_dir(iso_dir)

    for I, rows_I in byI.items():
        # curves
        plot_curves_by_N(
            rows_I,
            "tp_mean",
            curves_dir / f"timing_vs_P_by_N_I{I}.png",
            "Time (s)",
            f"Time vs Threads — I={I}",
        )
        plot_curves_by_N(
            rows_I,
            "speedup",
            curves_dir / f"speedup_vs_P_by_N_I{I}.png",
            "Speedup (T(P=1)/T(P))",
            f"Speedup vs Threads — I={I}",
        )
        plot_curves_by_N(
            rows_I,
            "efficiency",
            curves_dir / f"efficiency_vs_P_by_N_I{I}.png",
            "Efficiency (S/P)",
            f"Efficiency vs Threads — I={I}",
        )
        # heatmap
        plot_eff_heatmap(
            rows_I,
            heatmaps_dir / f"efficiency_heatmap_I{I}.png",
            f"Efficiency Heatmap — I={I}",
        )
        # iso ladder
        E = eff_start
        while E <= 0.99 + 1e-12:
            plot_iso_curve(
                rows_I,
                E,
                iso_dir / f"isoefficiency_E={E:.2f}_I{I}.png",
                iso_dir / f"isoefficiency_E={E:.2f}_I{I}.csv",
            )
            E = round(E + eff_step, 2)
        # iso surface
        plot_iso_surface(
            rows_I,
            iso_dir / f"isoefficiency_surface_I{I}.png",
            f"Iso-efficiency Surface — I={I}",
        )


# ----------------------------- core sweep -----------------------------


def build_initial(make_exe, N, out_path: Path, timeout_sec: int):
    if not out_path.exists():
        run([make_exe, str(N), str(N), str(out_path)], timeout_sec)


def time_pthreads(pth_exe, init_path, out_path, iters, P, warmup, trials, timeout_sec):
    for _ in range(warmup):
        run(
            [pth_exe, str(iters), str(init_path), str(out_path), "-t", str(P)],
            timeout_sec,
        )
    times = []
    for _ in range(trials):
        out, err, wall = run(
            [pth_exe, str(iters), str(init_path), str(out_path), "-t", str(P)],
            timeout_sec,
        )
        t = parse_comp_time(out, wall)
        times.append(t)
    return times


def write_report(report_path: Path, summary_rows, Ns, Is, Ps, label):
    lines = []
    lines.append(BANNER)
    lines.append("")
    lines.append(f"Label        : {label}")
    lines.append(f"N values     : {Ns}")
    lines.append(f"I values     : {Is}")
    lines.append(f"P values     : {Ps}")
    lines.append(f"Points (N,I,P): {len(summary_rows)}")
    best = {}
    for r in summary_rows:
        key = (r["rows"], r["iters"])
        cur = best.get(key)
        if (cur is None) or (r["speedup"] > cur["speedup"]):
            best[key] = r
    lines.append("")
    lines.append("Best speedup per (N,I):")
    for key in sorted(best.keys()):
        r = best[key]
        lines.append(
            f"  N={r['rows']:4d} I={r['iters']:4d}: "
            f"max S={r['speedup']:.3f} at P={r['P']} "
            f"(Tp={r['tp_mean']:.4f}s, T1={r['t1_mean']:.4f}s)"
        )
    report_path.write_text("\n".join(lines))


# ----------------------------- main -----------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Single-file Pthreads stencil sweep with organized results/experiment directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # executables
    ap.add_argument(
        "--make_exe", default="../c_code/make-2d", help="Path to make-2d executable"
    )
    ap.add_argument(
        "--pth_exe",
        default="../c_code/stencil-2d-pth",
        help="Path to pthreads stencil executable",
    )

    # N sweep via inclusive integer linspace
    ap.add_argument("--N1", type=int, default=128)
    ap.add_argument("--N2", type=int, default=256)
    ap.add_argument(
        "--num_Ns", type=int, default=3, help="Number of N values (inclusive endpoints)"
    )

    # I sweep
    ap.add_argument("--I1", type=int, default=10)
    ap.add_argument("--I2", type=int, default=200)
    ap.add_argument("--Istep", type=int, default=20)

    # P sweep
    ap.add_argument("--P_start", type=int, default=1)
    ap.add_argument("--P_step", type=int, default=1)
    ap.add_argument("--P_max", type=int, default=8)

    # runs
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--timeout_sec", type=int, default=600)

    # outputs
    ap.add_argument(
        "--base_results", default="./results", help="Top-level results directory"
    )
    ap.add_argument("--label", default="bench")

    # iso ladder range "start step"
    ap.add_argument("--eff_range", type=str, default="0.29 0.05")

    args = ap.parse_args()

    # Prepare sweeps
    Ns = int_linspace(args.N1, args.N2, args.num_Ns)
    Is = int_range_inclusive(args.I1, args.I2, args.Istep)
    Ps = int_range_inclusive(args.P_start, args.P_max, args.P_step)

    # Build experiment directory layout
    exp_name = stamp_name(Ns, Is, Ps, args.label)
    exp_dir = ensure_dir(Path(args.base_results) / exp_name)
    data_dir = ensure_dir(exp_dir / "data")
    csv_dir = ensure_dir(exp_dir / "csv")
    plots_root = ensure_dir(exp_dir / "plots")
    curves_dir = ensure_dir(plots_root / "curves")
    heatmaps_dir = ensure_dir(plots_root / "heatmaps")
    iso_dir = ensure_dir(plots_root / "isoefficiency")

    raw_csv = csv_dir / "raw_runs.csv"
    summary_csv = csv_dir / "summary.csv"
    report_txt = exp_dir / "report.txt"

    # CSV header create
    write_csv(raw_csv, [], ["rows", "cols", "iters", "P", "trial", "tp_time"])

    total = len(Ns) * len(Is) * len(Ps)
    job_id = 0

    print_banner()
    print(f"Experiment directory: {exp_dir}")

    # Build C code: run `make clean` then `make` in the c_code directory
    # derived from the --make_exe path (default ../c_code). This mirrors
    # the behavior in run-all.py and ensures the required executables exist.
    try:
        make_dir = str(Path(args.make_exe).parent)
        print(f"Building C code in: {make_dir}")
        # First run `make clean` to ensure a fresh build
        print(f"Running: make -C {make_dir} clean")
        run(["make", "-C", make_dir, "clean"], args.timeout_sec)
        # Then run `make`
        print(f"Running: make -C {make_dir}")
        run(["make", "-C", make_dir], args.timeout_sec)
    except Exception as e:
        print("Failed to build C code:\n", e)
        sys.exit(1)

    # Sweep
    raw_rows = []
    for N in Ns:
        init_path = data_dir / f"initial.{N}x{N}.dat"
        build_initial(args.make_exe, N, init_path, args.timeout_sec)

        for I in Is:
            for P in Ps:
                job_id += 1
                print(
                    f"[{job_id:3d}/{total:3d}] N={N:4d} I={I:3d} P={P:2d} — trials={args.trials}"
                )
                out_path = data_dir / f"final.P{P}.N{N}.I{I}.dat"
                times = time_pthreads(
                    args.pth_exe,
                    init_path,
                    out_path,
                    I,
                    P,
                    args.warmup,
                    args.trials,
                    args.timeout_sec,
                )
                for t_idx, t in enumerate(times, 1):
                    raw_rows.append(
                        {
                            "rows": N,
                            "cols": N,
                            "iters": I,
                            "P": P,
                            "trial": t_idx,
                            "tp_time": t,
                        }
                    )

    # Write raw rows & build summary
    write_csv(raw_csv, raw_rows, ["rows", "cols", "iters", "P", "trial", "tp_time"])

    buckets = {}
    for r in raw_rows:
        key = (r["rows"], r["iters"], r["P"])
        buckets.setdefault(key, []).append(r["tp_time"])

    summary_rows = []
    for (N, I, P), vec in sorted(buckets.items()):
        tp_mean = float(np.mean(vec))
        tp_std = float(np.std(vec, ddof=0))
        base_vec = buckets.get((N, I, 1), None)
        if not base_vec:
            t1_mean = t1_std = speedup = eff = float("nan")
        else:
            t1_mean = float(np.mean(base_vec))
            t1_std = float(np.std(base_vec, ddof=0))
            speedup = (t1_mean / tp_mean) if tp_mean > 0 else float("nan")
            eff = (speedup / P) if P > 0 else float("nan")
        summary_rows.append(
            {
                "rows": N,
                "cols": N,
                "iters": I,
                "P": P,
                "t1_mean": t1_mean,
                "t1_std": t1_std,
                "tp_mean": tp_mean,
                "tp_std": tp_std,
                "speedup": speedup,
                "efficiency": eff,
                "trials": len(vec),
            }
        )

    summary_fields = [
        "rows",
        "cols",
        "iters",
        "P",
        "t1_mean",
        "t1_std",
        "tp_mean",
        "tp_std",
        "speedup",
        "efficiency",
        "trials",
    ]
    write_csv(summary_csv, summary_rows, summary_fields)

    # Plots
    eff_tokens = args.eff_range.strip().split()
    eff_start = float(eff_tokens[0]) if len(eff_tokens) >= 1 else 0.29
    eff_step = float(eff_tokens[1]) if len(eff_tokens) >= 2 else 0.05
    make_plots_full(summary_csv, curves_dir, heatmaps_dir, iso_dir, eff_start, eff_step)

    # Report
    write_report(report_txt, summary_rows, Ns, Is, Ps, args.label)

    # Final printout
    print("Done.")
    print("Experiment :", exp_dir)
    print("CSV        :", csv_dir)
    print("Plots      :", plots_root)
    print("  Curves   :", curves_dir)
    print("  Heatmaps :", heatmaps_dir)
    print("  Iso-Eff  :", iso_dir)
    # Mirror your example: list the main plot paths per I
    Is_present = sorted({r["iters"] for r in summary_rows})
    for I in Is_present:
        print_banner()
        print(f" {curves_dir}/timing_vs_P_by_N_I{I}.png")
        print(f" {curves_dir}/speedup_vs_P_by_N_I{I}.png")
        print(f" {curves_dir}/efficiency_vs_P_by_N_I{I}.png")
        print(f" {heatmaps_dir}/efficiency_heatmap_I{I}.png")
        print(f" {iso_dir}/isoefficiency_surface_I{I}.png")


if __name__ == "__main__":
    main()
