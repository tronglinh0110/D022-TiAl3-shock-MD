import numpy as np
import glob
from openpyxl import Workbook

dt = 0.001  # ps
nbins = 200

all_files = glob.glob("shock_load_*.dump")

# ---- collect & sort dumps by timestep ----
files_ts = []
for f in all_files:
    with open(f) as file:
        lines = file.readlines()
    ts = int(lines[1].strip())
    if 0 <= ts <= 6000:
        files_ts.append((ts, f))

files_ts.sort(key=lambda x: x[0])

times = []
zs_list = []

for timestep, f in files_ts:
    with open(f) as file:
        lines = file.readlines()

    data = []
    for line in lines[9:]:
        parts = line.split()
        z  = float(parts[4])
        vz = float(parts[7])
        data.append((z, vz))

    data = np.array(data, dtype=float)
    z = data[:, 0]
    vz = data[:, 1]

    zmin, zmax = z.min(), z.max()
    bins = np.linspace(zmin, zmax, nbins + 1)
    zc = 0.5 * (bins[:-1] + bins[1:])

    vz_prof = np.full(nbins, np.nan, dtype=float)
    for i in range(nbins):
        mask = (z >= bins[i]) & (z < bins[i + 1])
        if np.any(mask):
            vz_prof[i] = vz[mask].mean()

    valid = ~np.isnan(vz_prof)
    zc = zc[valid]
    vz_prof = vz_prof[valid]

    # ---- OPTIONAL: cut boundary regions to reduce false fronts ----
    # margin = 50.0  # Å
    # if len(zc) > 10:
    #     mask_core = (zc > (zc.min() + margin)) & (zc < (zc.max() - margin))
    #     zc = zc[mask_core]
    #     vz_prof = vz_prof[mask_core]

    dv = np.gradient(vz_prof, zc)
    ishock = int(np.argmax(np.abs(dv)))
    zs = float(zc[ishock])

    times.append(timestep * dt)
    zs_list.append(zs)

times = np.array(times, dtype=float)
zs_list = np.array(zs_list, dtype=float)

# ---- fit Us in window 0.4 - 6.0 ps ----
tmin, tmax = 0.4, 6.0
mask_fit = (times >= tmin) & (times <= tmax)

if mask_fit.sum() < 2:
    raise RuntimeError(f"Not enough points for fit in [{tmin}, {tmax}] ps. "
                       f"Got {mask_fit.sum()} points.")

coef = np.polyfit(times[mask_fit], zs_list[mask_fit], 1)
Us_A_per_ps = float(coef[0])   # slope
c_intercept = float(coef[1])

# goodness of fit (R^2)
z_fit = Us_A_per_ps * times[mask_fit] + c_intercept
ss_res = float(np.sum((zs_list[mask_fit] - z_fit) ** 2))
ss_tot = float(np.sum((zs_list[mask_fit] - np.mean(zs_list[mask_fit])) ** 2))
r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

print(f"Fit window: {tmin}–{tmax} ps (N={mask_fit.sum()})")
print(f"Us = {Us_A_per_ps:.6f} Å/ps  = {Us_A_per_ps*0.1:.6f} km/s")
print(f"Intercept c = {c_intercept:.6f} Å")
print(f"R^2 = {r2:.6f}")

# ---- write xlsx (include Us + fit window flag) ----
wb = Workbook()
ws = wb.active
ws.title = "shock_front"

ws.append(["time_ps", "z_shock_A", "in_fit_window"])
for t, z in zip(times, zs_list):
    ws.append([float(t), float(z), int(tmin <= t <= tmax)])

# put summary on top (optional)
ws2 = wb.create_sheet("fit_summary")
ws2.append(["tmin_ps", tmin])
ws2.append(["tmax_ps", tmax])
ws2.append(["Us_A_per_ps", Us_A_per_ps])
ws2.append(["Us_km_per_s", Us_A_per_ps * 0.1])
ws2.append(["Intercept_A", c_intercept])
ws2.append(["R2", r2])
ws2.append(["N_points", int(mask_fit.sum())])

wb.save("shock_front_position.xlsx")
print("Saved shock_front_position.xlsx")