import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from COM_mapGenerator import *

em = ElevationMap()
em.GenerateElevationMap()

for i in range(em.iNumPlots):
    if em.elevationMap[i] <= em.seaLevelThreshold:
        em.plotTypes[i] = PlotTypes.PLOT_OCEAN
    elif em.prominenceMap[i] > em.peakHeight:
        em.plotTypes[i] = PlotTypes.PLOT_PEAK
    elif em.prominenceMap[i] > em.hillHeight:
        em.plotTypes[i] = PlotTypes.PLOT_HILLS
    else:
        em.plotTypes[i] = PlotTypes.PLOT_LAND

Z = np.array(em.continentID).reshape(em.iNumPlotsY, em.iNumPlotsX)
U = np.array(em.continentU).reshape(em.iNumPlotsY, em.iNumPlotsX)
V = np.array(em.continentV).reshape(em.iNumPlotsY, em.iNumPlotsX)

fig, ax = plt.subplots()
p = ax.imshow(Z, origin='lower', cmap=mpl.cm.tab20)
ax.quiver(U, V)
ax.plot([x["x_centroid"] for x in em.seedList], [x["y_centroid"]
        for x in em.seedList], 'bo')
ax.plot([x["x"] for x in em.plumeList], [x["y"]
        for x in em.plumeList], 'rx')
ax.set_title('continentID')
fig.colorbar(p)

Z = np.array(em.elevationBaseMap).reshape(em.iNumPlotsY, em.iNumPlotsX)

fig, ax = plt.subplots()
p = ax.imshow(Z, origin='lower', cmap=mpl.cm.terrain)
ax.set_title('base elevation')
fig.colorbar(p)

Z = np.array(em.elevationVelMap).reshape(em.iNumPlotsY, em.iNumPlotsX)

fig, ax = plt.subplots()
p = ax.imshow(Z, origin='lower', cmap=mpl.cm.terrain)
ax.set_title('vel elevation')
fig.colorbar(p)

Z = np.array(em.elevationBuoyMap).reshape(em.iNumPlotsY, em.iNumPlotsX)

fig, ax = plt.subplots()
p = ax.imshow(Z, origin='lower', cmap=mpl.cm.terrain)
ax.set_title('buoy elevation')
fig.colorbar(p)

Z = np.array(em.elevationPrelMap).reshape(em.iNumPlotsY, em.iNumPlotsX)

fig, ax = plt.subplots()
p = ax.imshow(Z, origin='lower', cmap=mpl.cm.terrain)
ax.set_title('prel elevation')
fig.colorbar(p)

Z = np.array(em.elevationBoundaryMap).reshape(em.iNumPlotsY, em.iNumPlotsX)

fig, ax = plt.subplots()
p = ax.imshow(Z, origin='lower', cmap=mpl.cm.terrain)
ax.set_title('bound elevation')
fig.colorbar(p)

Z = np.array(em.elevationMap).reshape(em.iNumPlotsY, em.iNumPlotsX)

fig, ax = plt.subplots()
p = ax.imshow(Z, origin='lower', cmap=mpl.cm.terrain)
ax.set_title('elevation')
fig.colorbar(p)

sorted_list = sorted(em.elevationMap, reverse=True)
index = int(0.38 * len(em.elevationMap))
if index >= len(em.elevationMap):
    index = len(em.elevationMap) - 1
oceanH = sorted_list[index]

elev = [0 if x < oceanH else x for x in em.elevationMap]

Z = np.array(elev).reshape(em.iNumPlotsY, em.iNumPlotsX)
iPeaks = [i for i, x in enumerate(em.plotTypes) if x == PlotTypes.PLOT_PEAK]
iHills = [i for i, x in enumerate(em.plotTypes) if x == PlotTypes.PLOT_HILLS]

fig, ax = plt.subplots()
p = ax.imshow(Z, origin='lower', cmap=mpl.cm.terrain)
ax.plot([i % em.iNumPlotsX for i in iPeaks], [
        i // em.iNumPlotsX for i in iPeaks], "^", mec="0.7", mfc="0.7", ms=8)
ax.plot([i % em.iNumPlotsX for i in iHills], [
        i // em.iNumPlotsX for i in iHills], linestyle="", marker="$\\frown$", mec='tab:brown', mfc='tab:brown', ms=8)
ax.set_title('elevation w sealevel')
fig.colorbar(p)

Z = np.array(em.prominenceMap).reshape(em.iNumPlotsY, em.iNumPlotsX)

fig, ax = plt.subplots()
p = ax.imshow(Z, origin='lower', cmap=mpl.cm.terrain)
ax.set_title('prominence')
fig.colorbar(p)

plt.show()


1
