"""Diagnostic for Phase H.1: quantify coastline / plate-boundary co-location.

Not a pass/fail test - prints binned statistics for a human to inspect before
any elevationMap tuning is attempted. See FINISH_LINE_TODO.md Phase H.1.
"""
import argparse
import os
import random
import sys
from collections import deque

if sys.version_info[0] >= 3:
    xrange = range

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "tools"))
sys.path.insert(0, _REPO_ROOT)

from PlanetSim import *

CARDINAL_DIRS = (MapConfig.N, MapConfig.S, MapConfig.E, MapConfig.W)
COAST_BINS = [0, 1, 2, 3, 4, 5]


def multi_source_bfs(mc, sources):
    """4-connected BFS distance from a set of source tile indices, wrap-aware."""
    dist = [-1] * mc.iNumPlots
    queue = deque()
    for i in sources:
        dist[i] = 0
        queue.append(i)
    while queue:
        current = queue.popleft()
        for direction in CARDINAL_DIRS:
            neighbour = mc.neighbours[current][direction]
            if neighbour >= 0 and dist[neighbour] == -1:
                dist[neighbour] = dist[current] + 1
                queue.append(neighbour)
    return dist


def find_plate_boundary_tiles(mc, em):
    boundary = set()
    for i in xrange(mc.iNumPlots):
        plate_i = em.plateID[i]
        if plate_i >= mc.plateCount:
            continue
        for direction in xrange(1, 9):
            neighbour = mc.neighbours[i][direction]
            if neighbour < 0:
                continue
            plate_n = em.plateID[neighbour]
            if plate_n < mc.plateCount and plate_n != plate_i:
                boundary.add(i)
                break
    return boundary


def bin_index(distance, bins):
    for idx, edge in enumerate(bins):
        if distance == edge:
            return idx
    return len(bins)  # overflow bin: "bins[-1]+"


def summarize_by_distance(label, distances, land_mask, em, bins):
    bin_labels = [str(b) for b in bins] + ["%d+" % (bins[-1] + 1)]
    sums = [[0.0, 0.0, 0.0, 0, 0, 0] for _ in xrange(len(bin_labels))]
    # columns: prominence_sum, boundaryElev_sum, prelElev_sum, count, peak_count, hill_count

    for i in xrange(len(distances)):
        if not land_mask[i] or distances[i] < 0:
            continue
        b = bin_index(distances[i], bins)
        row = sums[b]
        row[0] += em.prominenceMap[i]
        row[1] += em.elevationBoundaryMap[i]
        row[2] += em.elevationPrelMap[i]
        row[3] += 1
        if em.plotTypes[i] == PlotTypes.PLOT_PEAK:
            row[4] += 1
        if em.plotTypes[i] == PlotTypes.PLOT_HILLS:
            row[5] += 1

    print("\n%s (land tiles only)" % label)
    print("%6s %8s %14s %12s %10s %10s %10s" % (
        "dist", "count", "mean_promin", "mean_bound", "mean_prel", "peak_pct", "hill_pct"))
    for b, bl in enumerate(bin_labels):
        prom_sum, bound_sum, prel_sum, count, peak_count, hill_count = sums[b]
        if count == 0:
            print("%6s %8d %14s %12s %10s %10s %10s" % (bl, 0, "-", "-", "-", "-", "-"))
            continue
        print("%6s %8d %14.4f %12.4f %10.4f %9.1f%% %9.1f%%" % (
            bl, count, prom_sum / count, bound_sum / count, prel_sum / count,
            100.0 * peak_count / count, 100.0 * hill_count / count))


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None,
                         help="Fix the random seed so before/after runs use the same map.")
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    gc = CyGlobalContext()
    mapCtx = gc.getMap()
    mc = MapConfig(gc, mapCtx)
    em = ElevationMap(mc)
    em.GenerateElevationMap()

    ocean_tiles = [i for i in xrange(mc.iNumPlots) if em.plotTypes[i] == PlotTypes.PLOT_OCEAN]
    land_mask = [p != PlotTypes.PLOT_OCEAN for p in em.plotTypes]

    boundary_tiles = find_plate_boundary_tiles(mc, em)

    coast_dist = multi_source_bfs(mc, ocean_tiles)
    boundary_dist = multi_source_bfs(mc, boundary_tiles)

    print("Map size: %d x %d, plates: %d" % (mc.iNumPlotsX, mc.iNumPlotsY, mc.plateCount))
    print("Ocean tiles: %d, plate-boundary tiles: %d" % (len(ocean_tiles), len(boundary_tiles)))

    summarize_by_distance("Stats vs. distance-to-coast", coast_dist, land_mask, em, COAST_BINS)
    summarize_by_distance("Stats vs. distance-to-plate-boundary", boundary_dist, land_mask, em, COAST_BINS)

    # Structural co-location: do plate boundaries sit where coastlines end up,
    # independent of the boundary mountain-building term itself?
    valid = [i for i in xrange(mc.iNumPlots) if coast_dist[i] >= 0 and boundary_dist[i] >= 0]
    corr_dist = pearson_corr([float(coast_dist[i]) for i in valid], [float(boundary_dist[i]) for i in valid])
    corr_prel = pearson_corr([float(boundary_dist[i]) for i in valid], [em.elevationPrelMap[i] for i in valid])
    print("\nCorrelation(coast_dist, boundary_dist) = %.4f  (near +1 => boundaries and coastlines co-locate)" % corr_dist)
    print("Correlation(boundary_dist, elevationPrelMap pre-mountain baseline) = %.4f  "
          "(positive => baseline elevation rises away from boundaries, i.e. buoyancy pulls boundaries toward sea level)" % corr_prel)

    # Enrichment: are PEAK tiles disproportionately at the coast vs random land tiles?
    land_at_coast = sum(1 for i in xrange(mc.iNumPlots) if land_mask[i] and coast_dist[i] <= 1)
    land_total = sum(1 for i in xrange(mc.iNumPlots) if land_mask[i])
    peak_at_coast = sum(1 for i in xrange(mc.iNumPlots)
                         if em.plotTypes[i] == PlotTypes.PLOT_PEAK and coast_dist[i] <= 1)
    peak_total = sum(1 for i in xrange(mc.iNumPlots) if em.plotTypes[i] == PlotTypes.PLOT_PEAK)
    if land_total and peak_total:
        base_rate = land_at_coast / float(land_total)
        peak_rate = peak_at_coast / float(peak_total)
        print("\nLand within 1 tile of coast: %.1f%% of all land" % (100.0 * base_rate))
        print("Peaks within 1 tile of coast: %.1f%% of all peaks (enrichment x%.2f)" % (
            100.0 * peak_rate, peak_rate / base_rate if base_rate else float('nan')))

    # Boundary intensity distribution: does the existing 0.01 cutoff (see
    # _collect_boundary_interactions) ever actually distinguish "passive" from
    # "active" boundaries, or does nearly everything clear it?
    boundaries = em._collect_boundary_interactions()
    intensities = [b['intensity'] for b in boundaries]
    types = {}
    for b in boundaries:
        types[b['type']] = types.get(b['type'], 0) + 1

    print("\nBoundary interactions collected: %d (type breakdown: %s)" % (len(boundaries), types))
    if intensities:
        intensities_sorted = sorted(intensities)
        n = len(intensities_sorted)
        percentiles = [0, 10, 25, 50, 75, 90, 100]
        print("Intensity percentiles:")
        for p in percentiles:
            idx = min(n - 1, int(p / 100.0 * (n - 1)))
            print("  p%3d: %.4f" % (p, intensities_sorted[idx]))
        for threshold in [0.01, 0.05, 0.1, 0.2, 0.5]:
            below = sum(1 for x in intensities if x < threshold)
            print("  fraction below %.2f: %.1f%% (%d / %d)" % (
                threshold, 100.0 * below / n, below, n))

    passive_count = sum(1 for b in boundaries if b.get('passive'))
    collision_count = sum(1 for b in boundaries if b.get('collision'))
    print("\npassive_count=%d" % passive_count)
    print("collision_count=%d" % collision_count)


if __name__ == "__main__":
    main()
