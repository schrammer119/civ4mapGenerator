"""Diagnostic for Phase H.2: quantify why rainfall concentrates on 1-2 tiles.

Not a pass/fail test - prints distribution statistics and (optionally) plots
intermediate climate fields for a human to inspect before any rainfall-model
tuning is attempted. See FINISH_LINE_TODO.md Phase H.2.

Hypothesis under test: the atmospheric-thickness floor `max(0.1, H)` in
ClimateMap._calculate_thickness_field lets H approach zero for cold/high
tiles, so the topographic PV term `f0 * H_anomaly / H_total` spikes by
orders of magnitude at those tiles. That spike drives an extreme, localized
wind-speed (and therefore moisture) outlier, and the max-only normalization
used twice in the rainfall pipeline then collapses every other tile toward
zero relative to that outlier.
"""
import argparse
import os
import random
import sys

if sys.version_info[0] >= 3:
    xrange = range

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "tools"))
sys.path.insert(0, _REPO_ROOT)

if "--show" not in sys.argv and not os.environ.get("DISPLAY"):
    import matplotlib
    matplotlib.use("Agg")

from PlanetSim import *


def percentiles(values, ps):
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    out = {}
    for p in ps:
        idx = min(n - 1, int(p / 100.0 * (n - 1)))
        out[p] = sorted_vals[idx]
    return out


def pearson_corr(xs, ys):
    n = len(xs)
    mean_x = sum(xs) / float(n)
    mean_y = sum(ys) / float(n)
    cov = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in xrange(n))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return 0.0
    return cov / (var_x * var_y) ** 0.5


def describe(label, values):
    ps = percentiles(values, [0, 50, 75, 90, 95, 99, 100])
    median = ps[50]
    ratio = (ps[100] / median) if median > 0 else float('inf')
    print("%s: min=%.6g p50=%.6g p90=%.6g p99=%.6g max=%.6g  (max/median=%.3g)" % (
        label, ps[0], ps[50], ps[90], ps[99], ps[100], ratio))
    return ps


def diffuse_with_snapshots(cm, checkpoints):
    """Re-run _diffuse_moisture_iteratively's exact logic, but snapshot the
    concentration (max/median ratio and top-tile share) of RainfallMap at the
    given iteration checkpoints, to see how quickly/slowly it accumulates.

    Tests the hypothesis: does concentration saturate within a handful of
    iterations (horizon length irrelevant), or does it keep climbing across
    hundreds of iterations (long transport horizon draining the whole map into
    a few sink cells)?
    """
    mc_ = cm.mc
    max_iterations = mc_.rainfallMaxTransportDistance
    min_moisture_threshold = mc_.rainfallMinimumPrecipitation * 0.01

    base_temp = cm.rainfallConvectiveBaseTemp
    max_temp = cm.rainfallConvectiveMaxTemp
    temp_range = (max_temp - base_temp) if max_temp > base_temp else 1.0
    decline_rate = mc_.rainfallConvectiveDeclineRate
    min_factor = mc_.rainfallConvectiveMinFactor
    min_precip = mc_.rainfallMinimumPrecipitation

    snapshots = {}
    last_iteration = 0
    for iteration in xrange(max_iterations):
        new_moisture_grid = [0.0] * mc_.iNumPlots
        total_transported = 0.0

        for i in xrange(mc_.iNumPlots):
            current_moisture = cm._moisture_grid[i]
            if current_moisture <= min_moisture_threshold:
                cm.RainfallMap[i] += current_moisture
                continue

            temp_celsius = cm.TemperatureMap[i]
            conv_rate = cm._convective_rates[i]
            if temp_celsius <= base_temp:
                base_precip = 0.0
            elif temp_celsius <= max_temp:
                temp_factor = (temp_celsius - base_temp) / temp_range
                base_precip = current_moisture * conv_rate * temp_factor
            else:
                temp_excess = temp_celsius - max_temp
                decline_factor = temp_excess * decline_rate
                temp_factor = max(min_factor, 1.0 - decline_factor)
                base_precip = current_moisture * conv_rate * temp_factor

            local_precipitation = max(base_precip, min_precip)
            if local_precipitation >= current_moisture:
                cm.RainfallMap[i] += current_moisture
                continue

            cm.RainfallMap[i] += local_precipitation
            remaining_moisture = current_moisture - local_precipitation

            transport_data = cm._transport_weights[i]
            if not transport_data:
                continue

            for neighbour_id, transport_weight, precip_factor in transport_data:
                transported_amount = remaining_moisture * transport_weight
                transport_precipitation = transported_amount * precip_factor
                if transport_precipitation < transported_amount:
                    cm.RainfallMap[i] += transport_precipitation
                    final_transported = transported_amount - transport_precipitation
                    new_moisture_grid[neighbour_id] += final_transported
                    total_transported += final_transported
                else:
                    cm.RainfallMap[i] += transported_amount

        cm._moisture_grid = new_moisture_grid
        last_iteration = iteration + 1

        if (last_iteration in checkpoints) or (total_transported < min_moisture_threshold * mc_.iNumPlots):
            rain_sorted = sorted(cm.RainfallMap, reverse=True)
            total_rain = sum(cm.RainfallMap)
            top1_share = (rain_sorted[0] / total_rain) if total_rain > 0 else 0.0
            top10_share = (sum(rain_sorted[:10]) / total_rain) if total_rain > 0 else 0.0
            snapshots[last_iteration] = (total_transported, top1_share, top10_share)

        if total_transported < min_moisture_threshold * mc_.iNumPlots:
            break

    print("\nDiffusion ran %d iterations (cap: %d)" % (last_iteration, max_iterations))
    print("%10s %16s %12s %12s" % ("iteration", "total_transp", "top1_share", "top10_share"))
    for it in sorted(snapshots.keys()):
        total_transp, top1, top10 = snapshots[it]
        print("%10d %16.6g %11.1f%% %11.1f%%" % (it, total_transp, 100.0 * top1, 100.0 * top10))


def top_n_locations(mc, em, cm, values, n=10):
    indexed = sorted(xrange(len(values)), key=lambda i: values[i], reverse=True)[:n]
    print("%6s %6s %6s %10s %10s %10s %10s %8s" % (
        "rank", "x", "y", "value", "elev(m)", "temp(C)", "thick_m", "plot"))
    plot_names = {getattr(PlotTypes, name): name for name in
                  ("PLOT_OCEAN", "PLOT_LAND", "PLOT_HILLS", "PLOT_PEAK")}
    for rank, i in enumerate(indexed):
        x = i % mc.iNumPlotsX
        y = i // mc.iNumPlotsX
        thickness = cm._thickness_field_debug[i] if hasattr(cm, "_thickness_field_debug") else float('nan')
        print("%6d %6d %6d %10.6g %10.2f %10.2f %10.4f %8s" % (
            rank, x, y, values[i], em.aboveSeaLevelMap[i], cm.TemperatureMap[i],
            thickness, plot_names.get(em.plotTypes[i], "?")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None,
                         help="Fix the random seed so before/after runs use the same map.")
    parser.add_argument("--show", action="store_true",
                         help="Open matplotlib windows visualizing the intermediate fields.")
    parser.add_argument("--save", action="store_true",
                         help="Save the comparison figure to test-output/ instead of/in addition to showing it.")
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    gc = CyGlobalContext()
    mapCtx = gc.getMap()
    mc = MapConfig(gc, mapCtx)
    em = ElevationMap(mc)
    em.GenerateElevationMap()

    cm = ClimateMap(mc, em)
    cm.GenerateTemperatureMap()

    # Replicate _generate_wind_patterns but keep the intermediate thickness
    # field and topographic forcing term instead of discarding them.
    thickness_field = cm._calculate_thickness_field()
    cm._thickness_field_debug = thickness_field
    meridional_forcing = cm._calculate_meridional_forcing()
    cm._precalculate_pressure_gradient_winds()
    streamfunction = cm._solve_qg_streamfunction(thickness_field, meridional_forcing)
    cm.streamfunction = streamfunction
    cm._finalize_wind_extraction(streamfunction)

    print("Map size: %d x %d" % (mc.iNumPlotsX, mc.iNumPlotsY))
    peak_count = sum(1 for p in em.plotTypes if p == PlotTypes.PLOT_PEAK)
    print("Peak tiles: %d / %d" % (peak_count, mc.iNumPlots))

    describe("thickness_field (m)", thickness_field)
    floor_count = sum(1 for h in thickness_field if h <= 1.0)
    print("Tiles at/near the 0.1 floor (<=1.0 m): %d" % floor_count)

    topo_forcing = []
    coriolis_f0 = mc.qgCoriolisF0
    mean_layer_depth = mc.qgMeanLayerDepth
    for h in thickness_field:
        h_anomaly = mean_layer_depth - h
        topo_forcing.append(abs(coriolis_f0 * h_anomaly / h) if h > 0 else 0.0)
    describe("abs(topographic PV forcing) (1/s^2)", topo_forcing)

    describe("meridional forcing (1/s^2)", [abs(f) for f in meridional_forcing])

    print("\nGenerating rest of rainfall pipeline (wind speeds, moisture, precipitation)...")
    cm._precalculate_transport_data()
    describe("WindSpeeds (m/s)", cm.WindSpeeds)

    cm._set_dynamic_temperature_thresholds()
    cm._initialize_moisture_grid()
    moisture_snapshot = list(cm._moisture_grid)
    describe("moisture grid, normalized by max (0-1)", moisture_snapshot)

    diffuse_with_snapshots(cm, checkpoints=set([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]))
    cm._finalize_rainfall_map()
    describe("final RainfallMap, normalized by max (0-1)", cm.RainfallMap)

    land_rainfall = [cm.RainfallMap[i] for i in xrange(mc.iNumPlots)
                     if em.plotTypes[i] != PlotTypes.PLOT_OCEAN]
    negligible = sum(1 for v in land_rainfall if v < 0.01)
    print("Land tiles with rainfall < 1%% of max: %d / %d (%.1f%%)" % (
        negligible, len(land_rainfall), 100.0 * negligible / len(land_rainfall)))

    print("\nTop 10 tiles by WindSpeed:")
    top_n_locations(mc, em, cm, cm.WindSpeeds)

    print("\nTop 10 tiles by final RainfallMap:")
    top_n_locations(mc, em, cm, cm.RainfallMap)

    valid = xrange(mc.iNumPlots)
    corr_thickness_wind = pearson_corr(
        [1.0 / max(thickness_field[i], 0.1) for i in valid], [cm.WindSpeeds[i] for i in valid])
    corr_wind_rain = pearson_corr(list(cm.WindSpeeds), list(cm.RainfallMap))
    print("\nCorrelation(1/thickness, WindSpeed) = %.4f  (positive => thin-column tiles drive wind spikes)" %
          corr_thickness_wind)
    print("Correlation(WindSpeed, final RainfallMap) = %.4f  (positive => wind spikes drive rainfall spikes)" %
          corr_wind_rain)

    if args.show or args.save:
        import matplotlib.pyplot as plt
        import numpy as np
        shape = (mc.iNumPlotsY, mc.iNumPlotsX)

        # Panel A reproduces run_planetsim.py's rainfall plot exactly
        # (fixed clim=(0.0, 1.0)) to check whether that visualization choice,
        # rather than the underlying field, is what produces the "1-2 wet
        # tiles" appearance.
        rain_arr = np.array(cm.RainfallMap).reshape(shape)
        p99 = sorted(cm.RainfallMap)[int(0.99 * (len(cm.RainfallMap) - 1))]

        cm._calculate_percentiles()
        rain_pct_arr = np.array(cm.rainfall_percentiles).reshape(shape)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        im0 = axes[0].imshow(rain_arr, origin="lower", cmap="Blues", vmin=0.0, vmax=1.0)
        axes[0].set_title("RainfallMap, clim=(0,1)\n(matches run_planetsim.py)")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(rain_arr, origin="lower", cmap="Blues", vmin=0.0, vmax=p99)
        axes[1].set_title("Same RainfallMap, clim=(0, p99=%.3f)" % p99)
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(rain_pct_arr, origin="lower", cmap="Blues", vmin=0.0, vmax=1.0)
        axes[2].set_title("rainfall_percentiles\n(what biome placement actually uses)")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()

        if args.save:
            out_dir = os.path.join(_REPO_ROOT, "test-output")
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            out_path = os.path.join(out_dir, "rainfall_h2_comparison.png")
            plt.savefig(out_path, dpi=120)
            print("\nSaved comparison figure to %s" % out_path)
        if args.show:
            plt.show()
    else:
        print("\nVisualization disabled. Re-run with --show or --save to produce the diagnostic plots.")


if __name__ == "__main__":
    main()
