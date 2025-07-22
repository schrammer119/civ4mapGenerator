from CvPythonExtensions import *
import random
import math
from collections import deque


class ElevationMap:
    # macros
    L = 0
    N = 1
    S = 2
    E = 3
    W = 4
    NE = 5
    NW = 6
    SE = 7
    SW = 8

    def __init__(self):
        self.gc = CyGlobalContext()
        self.map = self.gc.getMap()
        self.iNumPlotsX = self.map.getGridWidth()
        self.iNumPlotsY = self.map.getGridHeight()
        self.iNumPlots = self.iNumPlotsX * self.iNumPlotsY
        self.wrapX = self.map.isWrapX()
        self.wrapY = self.map.isWrapY()

        # default user settings
        # self.seaLevelChange = self.gc.getSeaLevelInfo(
        #     self.map.getSeaLevel()).getSeaLevelChange()  # -8, 0, 6
        # self.desertPercentChange = self.gc.getClimateInfo(
        #     self.map.getClimate()).getDesertPercentChange()  # -10, 0, 20
        # self.jungleLatitude = self.gc.getClimateInfo(
        #     self.map.getClimate()).getJungleLatitude()  # 2, 5, 6
        # self.hillRange = self.gc.getClimateInfo(
        #     self.map.getClimate()).getHillRange()  # 5, 7
        # self.peakPercent = self.gc.getClimateInfo(
        #     self.map.getClimate()).getPeakPercent()  # 25, 35
        # self.snowLatitudeChange = self.gc.getClimateInfo(
        #     self.map.getClimate()).getSnowLatitudeChange()  # -0.1, -0.025, 0.0, 0.1
        # self.tundraLatitudeChange = self.gc.getClimateInfo(
        #     self.map.getClimate()).getTundraLatitudeChange()  # -0.15, -0.05, 0.0, 0.1
        # self.grassLatitudeChange = self.gc.getClimateInfo(
        #     self.map.getClimate()).getGrassLatitudeChange()  # 0.0
        # self.desertBottomLatitudeChange = self.gc.getClimateInfo(
        #     self.map.getClimate()).getDesertBottomLatitudeChange()  # -0.1, 0.0
        # self.desertTopLatitudeChange = self.gc.getClimateInfo(
        #     self.map.getClimate()).getDesertTopLatitudeChange()  # -0.1, -0.05, 0.0, 0.1
        # self.iceLatitude = self.gc.getClimateInfo(
        #     self.map.getClimate()).getIceLatitude()  # 0.9, 0.95
        # self.randIceLatitude = self.gc.getClimateInfo(
        #     self.map.getClimate()).getRandIceLatitude()  # 0.20, 0.25, 0.5
        self.seaLevelChange = 0
        self.desertPercentChange = 0
        self.jungleLatitude = 5
        self.hillRange = 5
        self.peakPercent = 25
        self.snowLatitudeChange = 0.0
        self.tundraLatitudeChange = 0.0
        self.grassLatitudeChange = 0.0
        self.desertBottomLatitudeChange = 0.0
        self.desertTopLatitudeChange = 0.0
        self.iceLatitude = 0.95
        self.randIceLatitude = 0.25

        # custom settings
        self.landPercent = 0.38
        self.coastPercent = 0.01
        # general smoothing radius
        self.PWSclimeSmoothingRadius = 4
        # mean annual land temperature, equator: 30C (lowland, away from coast) (Manaus, Brazil ~28C @100m)
        # mean annual land temperature, antarctica: -30C (lowland, away from coast) (Plateua, Antartica -56.7C @3,666m)
        self.minimumTemp = -20.77
        self.maximumTemp = 29.0
        # max / min water temps
        self.maxWaterTempC = 35.0
        self.minWaterTempC = -10.0
        # max elevation
        self.maxElev = 5.1
        # temperature lapse rate (C vs elevation)
        self.tempLapse = 1.3  # C/km
        # oceant currents
        self.currentAttenuation = 1.0
        self.currentAmplFactor = 10.0
        # winds
        self.tempGradientFactor = 0.2
        # rains
        # adjusts amount of rain, vs spread of moisture (higher concentrates the rain)
        self.rainOverallFactor = 0.008
        self.rainConvectionFactor = 0.07  # rain due to temperature
        self.rainOrographicFactor = 0.11  # rain due to elevation gradients
        self.rainFrontalFactor = 0.03  # rain due to temperature+wind gradients
        self.rainPerlinFactor = 0.05  # factor for random rainfall
        # River thresholds
        self.riverGlacierSourceFactor = 4.0
        self.minRiverBasin = 10
        self.riverLengthFactor = 4.0
        self.PWSriverthreshold = 1.0
        # lakes
        self.maxLakeSize = 9
        self.lakeSizeFactor = 0.25
        # continents
        self.contN = 15  # number of continental plates, wiki shows 15 major plates on earth
        self.contMinDens = 0.8  # minimum plate density (out of 1.0)
        self.contTwistAngle = -0.35
        self.contPlumeN = 15  # number of hotspot plume, wiki shows 9 major
        self.contSlideFactor = 0.4  # height of sliding faults compare to crushing faults
        self.contBoundaryRadius = 1.0  # radius of boundary anomalies
        self.contBoundaryLift = 0.2  # radius of boundary anomalies
        self.contBoundaryLiftRadius = 7  # radius of boundary anomalies
        self.contHotSpotPeriod = 100
        self.contHotSpotDecay = 6
        self.contHotSpotRadius = 3
        self.contHotSpotFactor = 0.18
        self.hotSpotMinIntensity = 0.01  # Minimum intensity threshold
        self.volcanicFieldRadius = 2.0   # Radius multiplier for volcanic fields
        # Intensity reduction across plate boundaries
        self.crossPlateIntensityFactor = 0.3
        self.volcanoSizeVariation = 0.3  # Random size variation (±30%)
        self.contDensFactor = 1.3  # factor for base height based on density
        self.contVelFactor = 4.0  # height change of plate due to velocity direction
        self.contBoundarySmooth = 3
        self.contBuoyHeightFactor = 0.9  # height factor
        self.contBoundaryFactor = 3.5  # height of sliding faults compare to crushing faults
        self.contPerlinFactor = 0.3  # weight of perlin noise on map
        # features
        self.minBarePeaks = 0.2  # minimum percentage of peaks without forests
        # chance of single forest square spreading to peak
        self.MountainForestChance = 0.08

        # maps
        self.continentID = [self.contN + 1] * (self.iNumPlots)
        self.seedList = list()
        self.continentU = [0.0] * self.iNumPlots
        self.continentV = [0.0] * self.iNumPlots
        self.plumeList = list()
        self.elevationBaseMap = [0.0] * self.iNumPlots
        self.elevationVelMap = [0.0] * self.iNumPlots
        self.elevationBuoyMap = [0.0] * self.iNumPlots
        self.elevationPrelMap = [0.0] * self.iNumPlots
        self.elevationBoundaryMap = [0.0] * self.iNumPlots
        self.elevationMap = [0.0] * self.iNumPlots
        self.prominenceMap = [0.0] * self.iNumPlots
        self.plotTypes = [0] * self.iNumPlots
        self.neighbours = {}
        self.dx_centroid = [0.0] * self.iNumPlots
        self.dy_centroid = [0.0] * self.iNumPlots

        return

    def GenerateElevationMap(self):

        # pre-calced neighbours of every tile
        self.neighbours = {}
        for i in range(self.iNumPlots):
            x = i % self.iNumPlotsX
            y = i // self.iNumPlotsX
            nlist = [self.GetNeighbor(
                x, y, dir) for dir in range(9)]
            self.neighbours[i] = nlist

        # 1) Generate continental plates
        self.improved_continent_growth()
        self.smooth_continent_edges()

        # update self.seedList with continent sizes, centroids, mass, and inertia
        for s in self.seedList:
            s["mass"] = s["size"]*s["plateDensity"]

        # continent moments
        for i in range(self.iNumPlots):
            x = i % self.iNumPlotsX
            y = i // self.iNumPlotsX

            k = self.continentID[i]
            dx = x - self.seedList[k]["x_centroid"]
            dy = y - self.seedList[k]["y_centroid"]

            # Handle wrapping
            if self.wrapX:
                if dx > self.iNumPlotsX / 2:
                    dx -= self.iNumPlotsX
                elif dx < -self.iNumPlotsX / 2:
                    dx += self.iNumPlotsX
            if self.wrapY:
                if dy > self.iNumPlotsY / 2:
                    dy -= self.iNumPlotsY
                elif dy < -self.iNumPlotsY / 2:
                    dy += self.iNumPlotsY

            self.seedList[k]["moment"] += self.seedList[k]["plateDensity"] * \
                (dx**2 + dy**2)

        # 2) Calculate continental plate velocities
        # hot spot plumes:
        xlist_shuffled = list(range(int(self.iNumPlotsX)))
        ylist_shuffled = list(range(int(self.iNumPlotsY)))
        random.shuffle(xlist_shuffled)
        random.shuffle(ylist_shuffled)
        for i in range(self.contPlumeN):
            # plume ID list: (ID,x,y)
            self.plumeList.append(
                {"ID": i, "x": xlist_shuffled[i], "y": ylist_shuffled[i],
                 "x_wrap_plus": x + self.iNumPlotsX if self.wrapX else None,
                 "x_wrap_minus": x - self.iNumPlotsX if self.wrapX else None,
                 "y_wrap_plus": y + self.iNumPlotsY if self.wrapY else None,
                 "y_wrap_minus": y - self.iNumPlotsY if self.wrapY else None})

        self.calculate_realistic_plate_velocities()

        # 3) Generate heigh maps based on plate tectonics
        # plate heights:
        self.elevationBaseMap = list(
            map(lambda x: 1.0 - self.seedList[x]["plateDensity"], self.continentID))
        self.elevationBaseMap = self.Normalize(self.elevationBaseMap)

        # gradient due to velocity
        for i in range(self.iNumPlots):
            x = i % self.iNumPlotsX
            y = i // self.iNumPlotsX
            id = self.continentID[i]

            # add in x dir
            if self.continentU[i] > 0:
                dir = self.E
                xx, yy = self.neighbours[i][dir]
                ii = yy * self.iNumPlotsX + xx
            elif self.continentU[i] < 0:
                dir = self.W
                xx, yy = self.neighbours[i][dir]
                ii = yy * self.iNumPlotsX + xx
            else:
                ii = -1
            while (ii >= 0) and (self.continentID[ii] == id):
                self.elevationVelMap[ii] += abs(self.continentU[i])
                xx, yy = self.neighbours[ii][dir]
                ii = yy * self.iNumPlotsX + xx

            # add in y dir
            if self.continentV[i] > 0:
                dir = self.N
                xx, yy = self.neighbours[i][dir]
                ii = yy * self.iNumPlotsX + xx
            elif self.continentV[i] < 0:
                dir = self.S
                xx, yy = self.neighbours[i][dir]
                ii = yy * self.iNumPlotsX + xx
            else:
                ii = -1
            while (ii >= 0) and (self.continentID[ii] == id):
                self.elevationVelMap[ii] += abs(
                    self.continentV[i])
                xx, yy = self.neighbours[ii][dir]
                ii = yy * self.iNumPlotsX + xx
        self.elevationVelMap = self.Normalize(self.elevationVelMap)

        maxr = max(self.d_centroid)
        self.elevationBuoyMap = self.Normalize(
            [maxr - x for x in self.d_centroid])

        self.elevationPrelMap = self.gaussian_blur_2d([self.contDensFactor * base + self.contVelFactor * vel + self.contBuoyHeightFactor *
                                                       buoy for base, vel, buoy in zip(self.elevationBaseMap, self.elevationVelMap, self.elevationBuoyMap)], radius=self.contBoundarySmooth)

        self.identify_and_process_boundaries()

        # Improved hot spot volcanic activity (stationary hot spots with moving plates)
        for plume in self.plumeList:
            plume_x = plume["x"]
            plume_y = plume["y"]
            plume_i = plume_y * self.iNumPlotsX + plume_x

            # Get the plate this hot spot is under
            plate_id = self.continentID[plume_i]
            plate_u = self.continentU[plume_i]
            plate_v = self.continentV[plume_i]

            # Calculate plate velocity magnitude for volcanic intensity
            plate_velocity = math.sqrt(plate_u**2 + plate_v**2)

            # Create volcanic chain by tracing plate movement backwards in time
            current_x = plume_x
            current_y = plume_y

            # Create the main volcanic center (most recent, strongest)
            base_intensity = self.contHotSpotFactor * \
                (1.0 + 0.5 * (1.0 - plate_velocity))
            base_radius = max(2, int(self.contHotSpotRadius *
                              (1.0 + 0.3 * (1.0 - plate_velocity))))
            self.AddMountain(current_x, current_y, base_intensity, base_radius)

            # Trace the volcanic chain backwards (where the plate came from)
            # Accumulate fractional movement to avoid rounding errors
            accumulated_x = float(current_x)
            accumulated_y = float(current_y)

            # Calculate plate rotation (creates curved chains like Hawaii)
            plate_angular_velocity = 0.02 * \
                (1.0 - plate_velocity)  # Slower plates rotate more
            if random.random() > 0.5:  # Random rotation direction
                plate_angular_velocity *= -1

            for step in range(1, self.contHotSpotDecay + 1):
                # Move backwards along plate motion vector
                time_step = step * self.contHotSpotPeriod

                # Apply rotation to velocity vector over time (creates arc)
                rotation_angle = plate_angular_velocity * time_step
                cos_rot = math.cos(rotation_angle)
                sin_rot = math.sin(rotation_angle)

                # Rotate the velocity vector
                rotated_u = plate_u * cos_rot - plate_v * sin_rot
                rotated_v = plate_u * sin_rot + plate_v * cos_rot

                # Calculate movement step (keep as float)
                dx = -rotated_u * time_step
                dy = -rotated_v * time_step

                # Accumulate fractional movement
                accumulated_x += dx
                accumulated_y += dy

                # Only convert to integer for map coordinates
                past_x = int(round(accumulated_x))
                past_y = int(round(accumulated_y))

                # Handle wrapping
                if self.wrapX:
                    past_x = past_x % self.iNumPlotsX
                if self.wrapY:
                    past_y = past_y % self.iNumPlotsY

                # Skip if outside map bounds (for non-wrapping maps)
                if not self.wrapX and (past_x < 0 or past_x >= self.iNumPlotsX):
                    continue
                if not self.wrapY and (past_y < 0 or past_y >= self.iNumPlotsY):
                    continue

                past_i = past_y * self.iNumPlotsX + past_x

                # Check if still on same plate (volcanic chains can cross plate boundaries)
                if self.continentID[past_i] != plate_id:
                    # Reduce intensity when crossing plate boundaries but don't stop completely
                    plate_boundary_factor = self.crossPlateIntensityFactor
                else:
                    plate_boundary_factor = 1.0

                # Calculate volcanic intensity based on age and distance from hot spot
                distance_from_hotspot = math.sqrt(
                    (past_x - plume_x)**2 + (past_y - plume_y)**2)

                # Age-based decay (older volcanoes are less active)
                age_decay = math.exp(-step / (self.contHotSpotDecay * 1.5))

                # Distance-based decay (further from hot spot = less activity)
                max_distance = self.contHotSpotPeriod * self.contHotSpotDecay / 2.0
                # Changed exponent and minimum
                distance_decay = max(
                    0.3, 1.0 - (distance_from_hotspot / max_distance)**0.5)

                # Velocity factor (faster plates create more spread out, lower intensity chains)
                velocity_factor = 1.0 / (1.0 + plate_velocity * 0.5)

                # Combined intensity
                volcano_intensity = (base_intensity * age_decay * distance_decay *
                                     velocity_factor * plate_boundary_factor)

                # Size varies with intensity but has minimum for realism
                volcano_radius = max(
                    1, int(base_radius * math.pow(volcano_intensity / base_intensity, 0.3)))

                # Add some randomization for natural variation (±20%)
                intensity_variation = 1.0 + (random.random() - 0.5) * 0.4
                radius_variation = max(
                    1, int(volcano_radius * (1.0 + (random.random() - 0.5) * self.volcanoSizeVariation)))

                # Only add volcano if intensity is significant enough
                if volcano_intensity * intensity_variation > base_intensity * self.hotSpotMinIntensity:
                    self.AddMountain(past_x, past_y,
                                     volcano_intensity * intensity_variation,
                                     radius_variation)

            # Add some additional smaller volcanic features around the main hot spot
            # (represents volcanic field/peripheral volcanism)
            for i in range(3):  # 3 additional small features
                angle = random.random() * 2 * math.pi
                distance = random.randint(1, max(2, base_radius // 2))

                satellite_x = plume_x + int(distance * math.cos(angle))
                satellite_y = plume_y + int(distance * math.sin(angle))

                # Handle wrapping
                if self.wrapX:
                    satellite_x = satellite_x % self.iNumPlotsX
                if self.wrapY:
                    satellite_y = satellite_y % self.iNumPlotsY

                # Skip if outside bounds
                if not self.wrapX and (satellite_x < 0 or satellite_x >= self.iNumPlotsX):
                    continue
                if not self.wrapY and (satellite_y < 0 or satellite_y >= self.iNumPlotsY):
                    continue

                # Smaller intensity for satellite volcanoes
                satellite_intensity = base_intensity * \
                    (0.2 + random.random() * 0.3)  # 20-50% of main
                self.AddMountain(satellite_x, satellite_y,
                                 satellite_intensity, 1)

        self.add_volcanic_field_clustering()

        # add all together
        for i in range(self.iNumPlots):
            self.elevationMap[i] = self.elevationPrelMap[i] + self.contBoundaryFactor * \
                self.elevationBoundaryMap[i]
        self.elevationMap = self.Normalize(self.elevationMap)

        # 4) Finish up elevation maps items
        # add some pepper, perlin noise:
        # perl = [x + y + z for x, y, z in zip(self.generate_perlin_grid(scale=4.0),
        #                                      self.generate_perlin_grid(
        #                                          scale=8.0),
        #                                      self.generate_perlin_grid(scale=16.0))]
        # perl = self.Normalize(perl)
        # for i in range(self.iNumPlots):
        #     self.elevationMap[i] = self.elevationMap[i] + \
        #         self.contPerlinFactor * perl[i]
        # self.elevationMap = self.Normalize(self.elevationMap)

        # Sea Level
        land = self.landPercent
        land -= self.seaLevelChange / 100.0
        self.seaLevelThreshold = self.FindValueFromPercent(
            self.elevationMap, land, descending=True)
        waterTiles = list()
        for i in range(self.iNumPlots):
            x = i % self.iNumPlotsX
            y = i // self.iNumPlotsX
            if self.elevationMap[i] < self.seaLevelThreshold:
                waterTiles.append(self.elevationMap[i])
        self.coastLevelThreshold = self.FindValueFromPercent(
            waterTiles, self.coastPercent, descending=True)

        # prominence map
        for i in range(self.iNumPlots):
            x = i % self.iNumPlotsX
            y = i // self.iNumPlotsX
            maxDiff = 0.0
            if self.elevationMap[i] > self.seaLevelThreshold:
                for direction in range(1, 5):
                    xx, yy = self.neighbours[i][direction]
                    ii = yy * self.iNumPlotsX + xx
                    if ii >= 0:
                        maxDiff = max(
                            maxDiff, self.elevationMap[i] - max(self.seaLevelThreshold, self.elevationMap[ii]))
            self.prominenceMap[i] = maxDiff
        self.prominenceMap = self.Normalize(self.prominenceMap)

        # Find the peak and hill heights
        peakPercent = (self.peakPercent / 100.0) * self.landPercent
        hillPercent = peakPercent + \
            (4.0 * self.hillRange / 100.0)
        peakMap = [x for x, y in zip(
            self.prominenceMap, self.elevationMap) if y > self.seaLevelThreshold]

        self.hillHeight = self.FindValueFromPercent(
            peakMap, hillPercent, True)
        self.peakHeight = self.FindValueFromPercent(
            peakMap, peakPercent, True)

        return

    def has_available_neighbors(self, i):
        """Check if a plot has any unclaimed neighbors that could potentially be grown to"""
        for xx, yy in self.neighbours[i]:
            if xx < 0 or yy < 0:
                continue
            ii = yy * self.iNumPlotsX + xx
            if ii >= 0 and self.continentID[ii] > self.contN:  # Unclaimed tile
                return True
        return False

    def improved_continent_growth(self):
        """
        Enhanced continent growth with more natural, organic shapes
        """
        # Initialize with multiple seed points per continent for more complex shapes
        gridFactor = 1
        xlist_shuffled = list(range(int(self.iNumPlotsX / gridFactor)))
        ylist_shuffled = list(range(int(self.iNumPlotsY / gridFactor)))
        random.shuffle(xlist_shuffled)
        random.shuffle(ylist_shuffled)

        # Create multiple seeds per continent for more complex shapes
        # 2-4 seeds per continent
        seeds_per_continent = 1  # 2 + random.randint(0, 2)
        queue = deque()

        for i in range(self.contN):
            # Primary seed
            main_x = xlist_shuffled[i] * gridFactor
            main_y = ylist_shuffled[i] * gridFactor
            main_ii = main_y * self.iNumPlotsX + main_x

            # Create continent data structure
            continent_data = {
                "ID": i,
                "seeds": [{"x": main_x, "y": main_y, "i": main_ii}],
                "growthFactor": 0.3 + 0.4 * random.random(),  # More controlled growth
                "plateDensity": self.contMinDens + (1 - self.contMinDens) * random.random(),
                "size": 1,
                "x_centroid": main_x,
                "y_centroid": main_y,
                "mass": 0,
                "moment": 0,
                # New properties for organic growth
                "roughness": 0.1 + 0.3 * random.random(),  # How jagged the edges are
                "anisotropy": 0.5 + random.random(),        # Directional growth preference
                "growth_angle": random.random() * 2 * math.pi,  # Preferred growth direction
            }

            self.seedList.append(continent_data)
            self.continentID[main_ii] = i
            # (plot_index, continent_id, generation)
            queue.append((main_ii, i, 0))

            # Add secondary seeds for more complex shapes
            for j in range(1, min(seeds_per_continent, len(xlist_shuffled) - self.contN)):
                if i + j * self.contN < len(xlist_shuffled):
                    sec_x = xlist_shuffled[i + j * self.contN] * gridFactor
                    sec_y = ylist_shuffled[i + j * self.contN] * gridFactor
                    sec_ii = sec_y * self.iNumPlotsX + sec_x

                    if self.continentID[sec_ii] > self.contN:  # Not claimed yet
                        continent_data["seeds"].append(
                            {"x": sec_x, "y": sec_y, "i": sec_ii})
                        self.continentID[sec_ii] = i
                        queue.append((sec_ii, i, 0))

        # Organic growth algorithm
        while queue:
            i, k, generation = queue.popleft()
            x = i % self.iNumPlotsX
            y = i // self.iNumPlotsX

            continent = self.seedList[k]

            # Calculate growth probability based on multiple factors
            base_growth = continent["growthFactor"]

            # Distance-based decay from nearest seed
            min_seed_dist = float('inf')
            for seed in continent["seeds"]:
                dx = x - seed["x"]
                dy = y - seed["y"]
                # Handle wrapping
                if self.wrapX and abs(dx) > self.iNumPlotsX / 2:
                    dx = dx - math.copysign(self.iNumPlotsX, dx)
                if self.wrapY and abs(dy) > self.iNumPlotsY / 2:
                    dy = dy - math.copysign(self.iNumPlotsY, dy)
                dist = math.sqrt(dx*dx + dy*dy)
                min_seed_dist = min(min_seed_dist, dist)

            # Growth probability decreases with distance from seeds
            distance_factor = math.exp(-min_seed_dist * 0.1)

            # Anisotropic growth (preferred direction)
            if min_seed_dist > 0:
                nearest_seed = min(continent["seeds"],
                                   key=lambda s: abs(x-s["x"]) + abs(y-s["y"]))
                dx = x - nearest_seed["x"]
                dy = y - nearest_seed["y"]
                angle_to_seed = math.atan2(dy, dx)
                angle_diff = abs(angle_to_seed - continent["growth_angle"])
                angle_diff = min(angle_diff, 2*math.pi - angle_diff)
                direction_factor = math.exp(
                    -continent["anisotropy"] * (angle_diff / math.pi))
            else:
                direction_factor = 1.0

            # Roughness factor (adds noise to edges)
            roughness_factor = 1.0 + \
                continent["roughness"] * (random.random() - 0.5)

            # Generation-based decay (prevents infinite growth)
            generation_factor = math.exp(-generation * 0.05)

            # Combined growth probability
            growth_prob = base_growth * distance_factor * \
                direction_factor * roughness_factor  # * generation_factor

            # Try to grow to neighbors
            neighbours = self.neighbours[i].copy()
            random.shuffle(neighbours)

            for xx, yy in neighbours:
                if xx < 0 or yy < 0:
                    continue

                ii = yy * self.iNumPlotsX + xx
                if ii >= 0 and self.continentID[ii] > self.contN:
                    if random.random() < growth_prob:
                        self.continentID[ii] = k
                        continent["size"] += 1

                        # Update centroid
                        if self.wrapX and abs(continent["x_centroid"] - xx) > self.iNumPlotsX/2:
                            xx += self.iNumPlotsX if xx < continent["x_centroid"] else - \
                                self.iNumPlotsX
                        if self.wrapY and abs(continent["y_centroid"] - yy) > self.iNumPlotsY/2:
                            yy += self.iNumPlotsY if yy < continent["y_centroid"] else - \
                                self.iNumPlotsY

                        continent["x_centroid"] = (
                            continent["x_centroid"] * (continent["size"]-1) + xx) / continent["size"]
                        continent["y_centroid"] = (
                            continent["y_centroid"] * (continent["size"]-1) + yy) / continent["size"]

                        if self.wrapX:
                            continent["x_centroid"] = continent["x_centroid"] % self.iNumPlotsX
                        if self.wrapY:
                            continent["y_centroid"] = continent["y_centroid"] % self.iNumPlotsY

                        queue.append((ii, k, generation + 1))

            if self.has_available_neighbors(i):
                queue.append((i, k, generation))

    # Also add some post-processing to smooth continent edges
    def smooth_continent_edges(self):
        """Post-process to create more natural coastlines"""
        changes = []

        for i in range(self.iNumPlots):
            current_id = self.continentID[i]

            # Count neighbors of same continent
            same_neighbors = 0
            total_neighbors = 0

            for xx, yy in self.neighbours[i]:
                if xx >= 0 and yy >= 0:
                    ii = yy * self.iNumPlotsX + xx
                    total_neighbors += 1
                    if self.continentID[ii] == current_id:
                        same_neighbors += 1

            # If isolated or mostly isolated, consider changing
            if total_neighbors > 0:
                isolation = 1.0 - (same_neighbors / total_neighbors)

                # Small chance to flip isolated cells
                if isolation > 0.6 and random.random() < 0.3:
                    # Find most common neighbor continent
                    neighbor_counts = {}
                    for xx, yy in self.neighbours[i]:
                        if xx >= 0 and yy >= 0:
                            ii = yy * self.iNumPlotsX + xx
                            neighbor_id = self.continentID[ii]
                            neighbor_counts[neighbor_id] = neighbor_counts.get(
                                neighbor_id, 0) + 1

                    if neighbor_counts:
                        new_id = max(neighbor_counts.items(),
                                     key=lambda x: x[1])[0]
                        if new_id != current_id:
                            changes.append((i, new_id))

        # Apply changes
        for i, new_id in changes:
            old_id = self.continentID[i]
            self.continentID[i] = new_id
            self.seedList[old_id]["size"] -= 1
            self.seedList[new_id]["size"] += 1

    def calculate_realistic_plate_velocities(self):
        """
        More realistic plate velocity calculation with edge boundary forces
        """
        # Initialize forces
        U = [0] * self.contN
        V = [0] * self.contN
        R = [0] * self.contN

        # Convert to per-plot velocities
        self.calculate_centroid_distances()

        # Add multiple force types for more realistic motion
        self.add_hotspot_forces(U, V, R)
        self.add_slab_pull_forces(U, V)
        self.add_plate_interaction_forces(U, V, R)
        self.apply_basal_drag(U, V, R)

        # Replace progressive edge damping with boundary forces
        self.apply_edge_boundary_forces(U, V, R)

        for i in range(self.iNumPlots):
            id = self.continentID[i]
            if id < self.contN:
                self.continentU[i] = U[id] - R[id] * self.dy_centroid[i]
                self.continentV[i] = V[id] + R[id] * self.dx_centroid[i]

    def add_hotspot_forces(self, U, V, R):
        """Add hotspot plume forces with distance limits"""
        for i in range(self.iNumPlots):
            id = self.continentID[i]
            if id >= self.contN:
                continue

            x = i % self.iNumPlotsX
            y = i // self.iNumPlotsX

            for plume in self.plumeList:
                dx, dy = self.get_wrapped_distance(
                    x, y, plume["x"], plume["y"])
                r_squared = dx*dx + dy*dy

                # Limit influence distance to prevent edge effects
                max_influence_dist = min(
                    self.iNumPlotsX, self.iNumPlotsY) * 0.3
                if r_squared > max_influence_dist*max_influence_dist:
                    continue

                if r_squared > 0:
                    r = math.sqrt(r_squared)
                    # Use more realistic force falloff
                    F = 1.0 / (1 + r*r*0.01)  # Softer falloff
                    Fx = F * dx / r
                    Fy = F * dy / r

                    # Scale by plate mass
                    mass_factor = 1.0 / max(self.seedList[id]["mass"], 1.0)
                    U[id] += Fx * mass_factor
                    V[id] += Fy * mass_factor

                    # Rotational component
                    moment_factor = 1.0 / max(self.seedList[id]["moment"], 1.0)
                    R[id] += (self.dx_centroid[i] * Fy -
                              self.dy_centroid[i] * Fx) * moment_factor

    def add_slab_pull_forces(self, U, V):
        """
        Add realistic slab pull forces based on subduction zone detection.
        Slab pull occurs when a denser oceanic plate subducts beneath a less dense plate.
        """
        # Detect subduction zones and calculate forces
        subduction_zones = self.detect_subduction_zones()

        # Apply slab pull forces for each detected subduction zone
        for zone in subduction_zones:
            self.apply_slab_pull_force(zone, U, V)

    def detect_subduction_zones(self):
        """
        Detect potential subduction zones by analyzing plate boundaries.
        Returns a list of subduction zone data structures.
        """
        subduction_zones = []
        boundary_segments = {}  # Track boundary segments for each plate pair

        # Scan all plots to find plate boundaries
        for i in range(self.iNumPlots):
            x = i % self.iNumPlotsX
            y = i // self.iNumPlotsX
            current_plate = self.continentID[i]

            if current_plate >= self.contN:
                continue

            # Check all 4 cardinal directions for boundaries
            for direction in [self.N, self.S, self.E, self.W]:
                xx, yy = self.neighbours[i][direction]
                if xx < 0 or yy < 0:
                    continue

                ii = yy * self.iNumPlotsX + xx
                if ii < 0 or ii >= self.iNumPlots:
                    continue

                neighbor_plate = self.continentID[ii]

                # Found a plate boundary
                if neighbor_plate != current_plate and neighbor_plate < self.contN:
                    plate_pair = tuple(sorted([current_plate, neighbor_plate]))

                    if plate_pair not in boundary_segments:
                        boundary_segments[plate_pair] = {
                            'segments': [],
                            'total_length': 0,
                            'avg_x': 0,
                            'avg_y': 0
                        }

                    # Add this boundary segment
                    boundary_segments[plate_pair]['segments'].append({
                        'x': x, 'y': y, 'direction': direction,
                        'plate1': current_plate, 'plate2': neighbor_plate
                    })

        # Analyze each boundary to determine if it's a subduction zone
        for plate_pair, boundary_data in boundary_segments.items():
            plate1_id, plate2_id = plate_pair
            plate1_density = self.seedList[plate1_id]["plateDensity"]
            plate2_density = self.seedList[plate2_id]["plateDensity"]

            # Calculate boundary statistics
            segments = boundary_data['segments']
            boundary_length = len(segments)

            if boundary_length < 3:  # Too short to be significant
                continue

            # Calculate average boundary position
            avg_x = sum(seg['x'] for seg in segments) / boundary_length
            avg_y = sum(seg['y'] for seg in segments) / boundary_length

            # Determine density difference and subduction direction
            density_diff = abs(plate1_density - plate2_density)
            min_density_diff = 0.05  # Minimum difference for subduction

            if density_diff >= min_density_diff:
                # Determine which plate subducts (denser plate goes under)
                if plate1_density > plate2_density:
                    subducting_plate = plate1_id
                    overriding_plate = plate2_id
                    density_contrast = plate1_density - plate2_density
                else:
                    subducting_plate = plate2_id
                    overriding_plate = plate1_id
                    density_contrast = plate2_density - plate1_density

                # Calculate subduction zone properties
                subduction_zones.append({
                    'subducting_plate': subducting_plate,
                    'overriding_plate': overriding_plate,
                    'density_contrast': density_contrast,
                    'boundary_length': boundary_length,
                    'avg_x': avg_x,
                    'avg_y': avg_y,
                    'segments': segments
                })

        return subduction_zones

    def apply_slab_pull_force(self, zone, U, V):
        """
        Apply slab pull force for a specific subduction zone.
        The force pulls the subducting plate toward the subduction zone.
        """
        subducting_plate = zone['subducting_plate']
        density_contrast = zone['density_contrast']
        boundary_length = zone['boundary_length']

        # Calculate the direction from subducting plate centroid to subduction zone
        plate_centroid_x = self.seedList[subducting_plate]["x_centroid"]
        plate_centroid_y = self.seedList[subducting_plate]["y_centroid"]

        # Get wrapped distance to subduction zone
        dx, dy = self.get_wrapped_distance(
            plate_centroid_x, plate_centroid_y,
            zone['avg_x'], zone['avg_y']
        )

        distance = math.sqrt(dx*dx + dy*dy)
        if distance < 1e-6:  # Avoid division by zero
            return

        # Normalize direction vector
        force_dir_x = dx / distance
        force_dir_y = dy / distance

        # Calculate force magnitude based on geological principles
        base_slab_pull = 2.0  # Base slab pull strength

        # Force scales with:
        # 1. Density contrast (more contrast = stronger pull)
        density_factor = density_contrast / 0.2  # Normalize to typical contrast

        # 2. Boundary length (longer subduction zones = more pull)
        length_factor = math.sqrt(boundary_length / 10.0)  # Normalize

        # 3. Distance decay (closer boundaries have stronger effect)
        max_influence_distance = min(self.iNumPlotsX, self.iNumPlotsY) * 0.4
        distance_factor = max(0.1, 1.0 - (distance / max_influence_distance))

        # 4. Plate age approximation (denser plates are typically older and sink better)
        age_factor = self.seedList[subducting_plate]["plateDensity"]

        # Calculate total force magnitude
        force_magnitude = (base_slab_pull * density_factor *
                           length_factor * distance_factor * age_factor)

        # Apply force scaled by plate mass (F = ma, so a = F/m)
        plate_mass = max(self.seedList[subducting_plate]["mass"], 1.0)
        force_per_mass = force_magnitude / plate_mass

        # Add force components
        U[subducting_plate] += force_per_mass * force_dir_x
        V[subducting_plate] += force_per_mass * force_dir_y

        # Optional: Add small counter-force to overriding plate (Newton's 3rd law)
        # This represents the resistance of the overriding plate
        overriding_plate = zone['overriding_plate']
        overriding_mass = max(self.seedList[overriding_plate]["mass"], 1.0)
        counter_force_factor = 0.1  # Much smaller counter-force

        U[overriding_plate] -= (force_per_mass * force_dir_x *
                                counter_force_factor * plate_mass / overriding_mass)
        V[overriding_plate] -= (force_per_mass * force_dir_y *
                                counter_force_factor * plate_mass / overriding_mass)

    def add_plate_interaction_forces(self, U, V, R):
        """Add forces from plate-plate interactions"""
        for i in range(self.contN):
            for j in range(i + 1, self.contN):
                # Distance between plate centroids
                dx, dy = self.get_wrapped_distance(
                    self.seedList[i]["x_centroid"], self.seedList[i]["y_centroid"],
                    self.seedList[j]["x_centroid"], self.seedList[j]["y_centroid"]
                )

                dist = math.sqrt(dx*dx + dy*dy)
                if dist > 0 and dist < min(self.iNumPlotsX, self.iNumPlotsY) * 0.4:
                    # Repulsive force (plates push each other away)
                    force_mag = 0.1 / (dist * dist + 1)
                    force_x = force_mag * dx / dist
                    force_y = force_mag * dy / dist

                    # Apply equal and opposite forces
                    mass_i = max(self.seedList[i]["mass"], 1.0)
                    mass_j = max(self.seedList[j]["mass"], 1.0)

                    U[i] += force_x / mass_i
                    V[i] += force_y / mass_i
                    U[j] -= force_x / mass_j
                    V[j] -= force_y / mass_j

    def apply_basal_drag(self, U, V, R):
        """Apply drag force to slow down motion"""
        drag_coefficient = 0.1
        for id in range(self.contN):
            speed = math.sqrt(U[id]*U[id] + V[id]*V[id])
            if speed > 0:
                drag_factor = 1.0 - drag_coefficient * speed
                drag_factor = max(0.1, drag_factor)  # Don't stop completely
                U[id] *= drag_factor
                V[id] *= drag_factor

            # Rotational drag
            R[id] *= (1.0 - drag_coefficient)

    def apply_edge_boundary_forces(self, U, V, R):
        """
        Apply forces from immovable edge boundaries instead of simple damping.
        Treats non-wrapped edges as infinite-mass immovable plates.
        """
        edge_influence_distance = min(self.iNumPlotsX, self.iNumPlotsY) * 0.25
        base_edge_force = 1.5  # Strength of edge repulsion

        for id in range(self.contN):
            centroid_x = self.seedList[id]["x_centroid"]
            centroid_y = self.seedList[id]["y_centroid"]
            plate_mass = max(self.seedList[id]["mass"], 1.0)

            # X-direction edge forces
            if not self.wrapX:
                # Left edge force
                dist_to_left = centroid_x
                if dist_to_left < edge_influence_distance:
                    # Force pointing away from left edge (rightward)
                    force_magnitude = base_edge_force * \
                        (1.0 - dist_to_left / edge_influence_distance)
                    U[id] += force_magnitude / plate_mass

                    # Add rotational component if plate is angled toward edge
                    if U[id] < 0:  # Moving toward left edge
                        rotation_force = -U[id] * 0.3  # Counter-rotation
                        R[id] += rotation_force / \
                            max(self.seedList[id]["moment"], 1.0)

                # Right edge force
                dist_to_right = self.iNumPlotsX - centroid_x
                if dist_to_right < edge_influence_distance:
                    # Force pointing away from right edge (leftward)
                    force_magnitude = base_edge_force * \
                        (1.0 - dist_to_right / edge_influence_distance)
                    U[id] -= force_magnitude / plate_mass

                    # Add rotational component if plate is angled toward edge
                    if U[id] > 0:  # Moving toward right edge
                        rotation_force = -U[id] * 0.3  # Counter-rotation
                        R[id] += rotation_force / \
                            max(self.seedList[id]["moment"], 1.0)

            # Y-direction edge forces
            if not self.wrapY:
                # Bottom edge force
                dist_to_bottom = centroid_y
                if dist_to_bottom < edge_influence_distance:
                    # Force pointing away from bottom edge (upward)
                    force_magnitude = base_edge_force * \
                        (1.0 - dist_to_bottom / edge_influence_distance)
                    V[id] += force_magnitude / plate_mass

                    # Add rotational component if plate is angled toward edge
                    if V[id] < 0:  # Moving toward bottom edge
                        rotation_force = -V[id] * 0.3  # Counter-rotation
                        R[id] += rotation_force / \
                            max(self.seedList[id]["moment"], 1.0)

                # Top edge force
                dist_to_top = self.iNumPlotsY - centroid_y
                if dist_to_top < edge_influence_distance:
                    # Force pointing away from top edge (downward)
                    force_magnitude = base_edge_force * \
                        (1.0 - dist_to_top / edge_influence_distance)
                    V[id] -= force_magnitude / plate_mass

                    # Add rotational component if plate is angled toward edge
                    if V[id] > 0:  # Moving toward top edge
                        rotation_force = -V[id] * 0.3  # Counter-rotation
                        R[id] += rotation_force / \
                            max(self.seedList[id]["moment"], 1.0)

    def get_wrapped_distance(self, x1, y1, x2, y2):
        """Calculate distance considering wrapping"""
        dx = x1 - x2
        dy = y1 - y2

        if self.wrapX and abs(dx) > self.iNumPlotsX / 2:
            dx = dx - math.copysign(self.iNumPlotsX, dx)
        if self.wrapY and abs(dy) > self.iNumPlotsY / 2:
            dy = dy - math.copysign(self.iNumPlotsY, dy)

        return dx, dy

    def calculate_centroid_distances(self):
        """Pre-calculate distances from each plot to its continent centroid"""
        self.dx_centroid = [0.0] * self.iNumPlots
        self.dy_centroid = [0.0] * self.iNumPlots
        self.d_centroid = [0.0] * self.iNumPlots

        for i in range(self.iNumPlots):
            x = i % self.iNumPlotsX
            y = i // self.iNumPlotsX
            id = self.continentID[i]

            if id < self.contN:
                dx, dy = self.get_wrapped_distance(
                    x, y,
                    self.seedList[id]["x_centroid"],
                    self.seedList[id]["y_centroid"]
                )
                self.dx_centroid[i] = dx
                self.dy_centroid[i] = dy
                self.d_centroid[i] = math.sqrt(
                    self.dx_centroid[i]**2 + self.dy_centroid[i]**2)

    def identify_and_process_boundaries(self):
        """Main method to identify and process all tectonic boundaries"""
        # Reset boundary map
        self.elevationBoundaryMap = [0.0] * self.iNumPlots

        # Collect all boundary interactions
        boundary_queue = []

        for i in range(self.iNumPlots):
            x = i % self.iNumPlotsX
            y = i // self.iNumPlotsX
            current_plate = self.continentID[i]

            # Check all neighbors for plate boundaries
            for direction_idx, direction_name in [(self.N, "NS"), (self.E, "EW"),
                                                  (self.NE, "NE"), (self.NW, "NW")]:
                xx, yy = self.neighbours[i][direction_idx]
                ii = yy * self.iNumPlotsX + xx

                if ii < 0 or ii >= self.iNumPlots:
                    continue

                neighbor_plate = self.continentID[ii]
                if neighbor_plate != current_plate:
                    # Calculate relative motion
                    u_diff = self.continentU[i] - self.continentU[ii]
                    v_diff = self.continentV[i] - self.continentV[ii]

                    # Determine boundary type based on relative motion
                    if direction_name == "NS":
                        convergent_motion = v_diff  # Positive means compression
                        transform_motion = abs(u_diff)
                    else:  # EW, NE, NW
                        convergent_motion = u_diff  # Positive means compression
                        transform_motion = abs(v_diff)

                    # Get plate density difference
                    plate_density_diff = (self.seedList[current_plate]["plateDensity"] -
                                          self.seedList[neighbor_plate]["plateDensity"])

                    # Determine primary boundary type and intensity
                    convergent_intensity = abs(convergent_motion)
                    transform_intensity = transform_motion

                    if convergent_intensity > transform_intensity * 1.5:
                        if convergent_motion > 0:
                            boundary_type = "crush"
                            intensity = convergent_intensity
                        else:
                            boundary_type = "rift"
                            intensity = convergent_intensity
                    else:
                        boundary_type = "slide"
                        intensity = transform_intensity

                    # Only process significant boundaries
                    if intensity > 0.01:
                        boundary_queue.append({
                            'tile': i,
                            'direction': direction_name,
                            'type': boundary_type,
                            'intensity': intensity,
                            'density_diff': plate_density_diff,
                            'neighbor_tile': ii
                        })

        # Process boundaries with dynamic profiles
        for boundary in boundary_queue:
            # Apply asymmetric effects
            self.apply_asymmetric_boundary_effects(
                boundary['tile'],
                boundary['direction'],
                boundary['type'],
                boundary['intensity'],
                boundary['density_diff']
            )

            # Add fractal roughness
            roughness_extent = max(2, min(6, int(boundary['intensity'] * 8)))
            self.add_fractal_boundary_roughness(
                boundary['tile'],
                boundary['type'],
                boundary['intensity'],
                roughness_extent
            )

            # Create transform faults for slide boundaries
            if boundary['type'] == "slide" and boundary['intensity'] > 0.1:
                self.create_transform_fault(
                    boundary['tile'],
                    boundary['neighbor_tile'],
                    boundary['intensity']
                )

        # Apply erosion effects
        self.apply_erosion_effects(boundary_age_factor=0.3)

        # Normalize the final boundary map
        self.elevationBoundaryMap = self.Normalize(
            self.elevationBoundaryMap)

    def apply_asymmetric_boundary_effects(self, i, boundary_dir, collision_type, collision_intensity, plate_density_diff):
        """Create asymmetric mountain ranges and rift valleys"""
        x = i % self.iNumPlotsX
        y = i // self.iNumPlotsX

        # Determine which side is the "overriding" plate
        overriding_side = plate_density_diff > 0

        # Apply effects on both sides of boundary with different intensities
        max_distance = 8 if collision_type == "crush" else 5

        for side_multiplier in [-1, 1]:  # Both sides of boundary
            for distance in range(1, max_distance):
                try:
                    offset_x, offset_y = self.get_offset_coords(
                        x, y, boundary_dir, distance * side_multiplier)
                    offset_i = offset_y * self.iNumPlotsX + offset_x

                    if offset_i < 0 or offset_i >= self.iNumPlots:
                        continue

                    # Check if we're still in a reasonable boundary zone
                    if abs(self.continentID[offset_i] - self.continentID[i]) > 1:
                        continue

                    # Generate elevation based on boundary type and side
                    base_elevation = self.generate_dynamic_boundary_profile(
                        collision_type, collision_intensity, distance, plate_density_diff
                    )

                    # Apply asymmetry for convergent boundaries
                    if collision_type == "crush":
                        if (side_multiplier > 0) == overriding_side:
                            # Overriding plate: more gradual rise, potential volcanic activity
                            base_elevation *= 0.8
                        else:
                            # Subducting plate: steeper profile, potential trench
                            base_elevation *= 1.2
                            if distance <= 2:
                                base_elevation -= collision_intensity * 0.2  # Trench effect

                    # Add natural variation
                    variation = 0.8 + 0.4 * random.random()
                    self.elevationBoundaryMap[offset_i] += base_elevation * variation

                except (IndexError, ZeroDivisionError):
                    continue

    def add_fractal_boundary_roughness(self, center_i, boundary_type, base_intensity, extent):
        """Add multi-scale noise to boundary features"""
        x = center_i % self.iNumPlotsX
        y = center_i // self.iNumPlotsX
        extent = max(2, min(extent, 6))  # Reasonable extent limits

        for i in range(-extent, extent + 1):
            for j in range(-extent, extent + 1):
                if i == 0 and j == 0:
                    continue

                target_x = x + i
                target_y = y + j

                # Handle wrapping
                if self.wrapX:
                    target_x = target_x % self.iNumPlotsX
                elif target_x < 0 or target_x >= self.iNumPlotsX:
                    continue

                if self.wrapY:
                    target_y = target_y % self.iNumPlotsY
                elif target_y < 0 or target_y >= self.iNumPlotsY:
                    continue

                target_i = target_y * self.iNumPlotsX + target_x
                if target_i < 0 or target_i >= self.iNumPlots:
                    continue

                distance = math.sqrt(i**2 + j**2)
                if distance > extent:
                    continue

                # Multiple octaves of noise for fractal complexity
                roughness = 0
                for octave in [1, 2, 4]:
                    noise_scale = octave * 0.1
                    noise_value = self.get_perlin_noise(
                        target_x * noise_scale, target_y * noise_scale)
                    roughness += noise_value / octave

                # Scale roughness by distance and boundary type
                distance_factor = 1.0 - (distance / extent)
                roughness_factor = distance_factor * base_intensity * 0.2

                if boundary_type == "crush":
                    roughness_factor *= 1.8  # Mountains are rougher
                elif boundary_type == "rift":
                    roughness_factor *= 0.8  # Rifts are smoother

                self.elevationBoundaryMap[target_i] += roughness * \
                    roughness_factor

    def create_transform_fault(self, start_i, end_i, intensity):
        """Create a linear transform fault with characteristic features"""
        start_x = start_i % self.iNumPlotsX
        start_y = start_i // self.iNumPlotsX
        end_x = end_i % self.iNumPlotsX
        end_y = end_i // self.iNumPlotsX

        # Calculate direction and length
        dx = end_x - start_x
        dy = end_y - start_y

        # Handle wrapping
        if self.wrapX and abs(dx) > self.iNumPlotsX / 2:
            dx = dx - math.copysign(self.iNumPlotsX, dx)
        if self.wrapY and abs(dy) > self.iNumPlotsY / 2:
            dy = dy - math.copysign(self.iNumPlotsY, dy)

        length = max(1, int(math.sqrt(dx**2 + dy**2)))
        if length == 0:
            return

        direction = math.atan2(dy, dx)

        # Create the main fault valley
        for step in range(length):
            progress = step / length

            fault_x = start_x + progress * dx
            fault_y = start_y + progress * dy

            # Add some natural meandering
            meander_amplitude = intensity * 0.3
            meander = meander_amplitude * \
                math.sin(step * 0.3) * math.sin(step * 0.1)
            fault_x += meander * math.cos(direction + math.pi/2)
            fault_y += meander * math.sin(direction + math.pi/2)

            # Ensure coordinates are within bounds
            fault_x = int(fault_x) % self.iNumPlotsX if self.wrapX else max(
                0, min(self.iNumPlotsX-1, int(fault_x)))
            fault_y = int(fault_y) % self.iNumPlotsY if self.wrapY else max(
                0, min(self.iNumPlotsY-1, int(fault_y)))

            fault_i = fault_y * self.iNumPlotsX + fault_x
            if fault_i < 0 or fault_i >= self.iNumPlots:
                continue

            # Create valley with intensity variation along length
            valley_intensity = intensity * \
                (0.6 + 0.4 * (1 - abs(progress - 0.5) * 2))
            self.elevationBoundaryMap[fault_i] -= valley_intensity * \
                (0.8 + 0.4 * random.random())

            # Add pressure ridges on sides
            for side in [-1, 1]:
                for ridge_dist in [1, 2]:
                    side_x = fault_x + side * ridge_dist * \
                        math.cos(direction + math.pi/2)
                    side_y = fault_y + side * ridge_dist * \
                        math.sin(direction + math.pi/2)

                    side_x = int(side_x) % self.iNumPlotsX if self.wrapX else max(
                        0, min(self.iNumPlotsX-1, int(side_x)))
                    side_y = int(side_y) % self.iNumPlotsY if self.wrapY else max(
                        0, min(self.iNumPlotsY-1, int(side_y)))

                    side_i = side_y * self.iNumPlotsX + side_x

                    if side_i >= 0 and side_i < self.iNumPlots:
                        ridge_height = intensity * \
                            (0.4 / ridge_dist) * (0.8 + 0.4 * random.random())
                        self.elevationBoundaryMap[side_i] += ridge_height

    def apply_erosion_effects(self, boundary_age_factor=0.5):
        """Simulate erosion and time effects on mountain ranges"""
        for i in range(self.iNumPlots):
            # Only erode elevated areas
            if self.elevationBoundaryMap[i] > 0:
                # Older boundaries are more eroded
                erosion_factor = 1.0 - (boundary_age_factor * 0.4)
                # Add some randomness for natural variation
                erosion_factor *= (0.7 + 0.6 * random.random())
                # Ensure we don't make things negative
                erosion_factor = max(0.3, erosion_factor)

                self.elevationBoundaryMap[i] *= erosion_factor

    def get_offset_coords(self, x, y, direction, distance):
        """Get coordinates offset by distance in given direction"""
        if direction == "NS":
            return x, (y + distance) % self.iNumPlotsY if self.wrapY else max(0, min(self.iNumPlotsY - 1, y + distance))
        elif direction == "EW":
            return (x + distance) % self.iNumPlotsX if self.wrapX else max(0, min(self.iNumPlotsX - 1, x + distance)), y
        elif direction == "NE":
            new_x = (x + distance) % self.iNumPlotsX if self.wrapX else max(0,
                                                                            min(self.iNumPlotsX - 1, x + distance))
            new_y = (y + distance) % self.iNumPlotsY if self.wrapY else max(0,
                                                                            min(self.iNumPlotsY - 1, y + distance))
            return new_x, new_y
        elif direction == "NW":
            new_x = (x - distance) % self.iNumPlotsX if self.wrapX else max(0,
                                                                            min(self.iNumPlotsX - 1, x - distance))
            new_y = (y + distance) % self.iNumPlotsY if self.wrapY else max(0,
                                                                            min(self.iNumPlotsY - 1, y + distance))
            return new_x, new_y
        else:
            return x, y

    def generate_dynamic_boundary_profile(self, boundary_type, intensity, distance_from_boundary, plate_density_diff):
        """Generate elevation based on geological processes"""
        base_elevation = 0

        if boundary_type == "rift":
            # Create a valley profile with raised shoulders
            if distance_from_boundary == 0:
                base_elevation = -intensity * 0.8  # Deep rift valley
            elif distance_from_boundary <= 2:
                base_elevation = intensity * 0.4 * \
                    (1 - distance_from_boundary / 2)  # Raised shoulders
            elif distance_from_boundary <= 5:
                base_elevation = intensity * 0.1 * \
                    math.exp(-distance_from_boundary / 3)  # Gradual falloff

        elif boundary_type == "crush":
            # Mountain building with foothills
            # Stronger boundaries create wider mountain ranges
            peak_distance = 1 + int(abs(intensity) * 3)
            if distance_from_boundary <= peak_distance:
                # Asymmetric profile based on plate density difference
                asymmetry_factor = 1.5 if plate_density_diff > 0 else 1.0
                base_elevation = intensity * \
                    (1 - (distance_from_boundary / peak_distance) ** asymmetry_factor)
            else:
                # Foothills extending further
                falloff_distance = distance_from_boundary - peak_distance
                base_elevation = intensity * 0.3 * \
                    math.exp(-falloff_distance / 4)

        elif boundary_type == "slide":
            # Transform boundaries create linear valleys with pressure ridges
            if distance_from_boundary == 0:
                base_elevation = -intensity * 0.4  # Fault valley
            elif distance_from_boundary <= 2:
                base_elevation = intensity * 0.3 * \
                    (1 + 0.5 * random.random())  # Pressure ridges
            else:
                base_elevation = intensity * 0.1 * \
                    math.exp(-distance_from_boundary / 2)

        return base_elevation

    def AddMountain(self, x, y, h, r):
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                radius = math.sqrt(dx**2+dy**2)
                if radius <= r:
                    xx = x + dx
                    if self.wrapX:
                        xx = xx % self.iNumPlotsX
                    yy = y + dy
                    if self.wrapY:
                        yy = yy % self.iNumPlotsY
                    i = yy * self.iNumPlotsX + xx
                    if 0 <= i < self.iNumPlots:
                        self.elevationBoundaryMap[i] += max(
                            0, h*(math.cos(math.pi*radius/r)+1.0)/2.0)

    def add_volcanic_field_clustering(self):
        """Add secondary volcanic activity near hot spots to create volcanic fields"""
        for plume in self.plumeList:
            plume_x = plume["x"]
            plume_y = plume["y"]

            # Create a small volcanic field around each hot spot
            field_radius = self.contHotSpotRadius * self.volcanicFieldRadius
            num_field_volcanoes = random.randint(2, 5)

            for _ in range(num_field_volcanoes):
                # Random position within field
                angle = random.random() * 2 * math.pi
                distance = random.random() * field_radius

                field_x = plume_x + int(distance * math.cos(angle))
                field_y = plume_y + int(distance * math.sin(angle))

                # Handle wrapping and bounds checking
                if self.wrapX:
                    field_x = field_x % self.iNumPlotsX
                if self.wrapY:
                    field_y = field_y % self.iNumPlotsY

                if not self.wrapX and (field_x < 0 or field_x >= self.iNumPlotsX):
                    continue
                if not self.wrapY and (field_y < 0 or field_y >= self.iNumPlotsY):
                    continue

                # Smaller volcanic features
                field_intensity = self.contHotSpotFactor * \
                    (0.1 + random.random() * 0.2)
                self.AddMountain(field_x, field_y, field_intensity, 1)

    def interpolate_2d_map(self, map, i, v, width, reverse):
        """
        Interpolate a 2D map stored as a flat list

        Args:
            map: Flat list representing 2D matrix
            width: Width of each row
            i: x-axis index
            v: y-axis value to interpolate
        """
        # Convert to row index and interpolation factor
        n_rows = len(map) // width
        row_float = v * (n_rows - 1)
        row_low = int(row_float)
        row_high = min(row_low + 1, n_rows - 1)
        frac = row_float - row_low

        # Get values from both rows
        if reverse:
            i = width - i - 1
        val_low = map[row_low * width + i]
        val_high = map[row_high * width + i]

        # Linearly interpolate
        return val_low + frac * (val_high - val_low)

    # def IsBelowSeaLevel(self, x, y):
    #     i = y * self.iNumPlotsX + x
    #     if self[i] < self.seaLevelThreshold:
    #         return True
    #     return False

    # def IsBelowCoastLevel(self, x, y):
    #     i = y * self.iNumPlotsX + x
    #     if self[i] < self.coastLevelThreshold:
    #         return True
    #     return False

    # def GetAltitudeAboveSeaLevel(self, x, y):
    #     i = y * self.iNumPlotsX + x
    #     if i == -1:
    #         return 0.0
    #     altitude = self[i]
    #     if altitude <= self.seaLevelThreshold:
    #         return 0.0
    #     altitude = (altitude - self.seaLevelThreshold) / \
    #         (1.0 - self.seaLevelThreshold)
    #     return altitude

    # def FillInLakes(self):

        # new

    def GetNeighbor(self, x, y, direction):
        xx, yy = x, y

        if direction == self.N:
            yy += 1
        elif direction == self.S:
            yy -= 1
        elif direction == self.E:
            xx += 1
        elif direction == self.W:
            xx -= 1
        elif direction == self.NE:
            xx += 1
            yy += 1
        elif direction == self.NW:
            xx -= 1
            yy += 1
        elif direction == self.SE:
            xx += 1
            yy -= 1
        elif direction == self.SW:
            xx -= 1
            yy -= 1

        if self.wrapY:
            yy = yy % self.iNumPlotsY
        elif (yy < 0) or (yy > self.iNumPlotsY - 1):
            return -1, -1

        if self.wrapX:
            xx = xx % self.iNumPlotsX
        elif (xx < 0) or (xx > self.iNumPlotsX - 1):
            return -1, -1

        return xx, yy

    def Normalize(self, ilist):
        min_val = min(ilist)
        max_val = max(ilist)
        if max_val - min_val == 0:
            return [(v) / (max_val) for v in ilist]
        else:
            return [(v - min_val) / (max_val - min_val) for v in ilist]

    def gaussian_blur_2d(self, grid, radius=2):
        sigmaList = [0.0, 0.32, 0.7, 1.12, 1.57, 2.05, 2.56, 3.09, 3.66, 4.25, 4.87, 5.53,
                     6.22, 6.95, 7.72, 8.54, 9.41, 10.34, 11.35, 12.44, 13.66, 15.02, 16.63, 18.65]
        sigma = sigmaList[radius]

        kernel = []
        sum_val = 0.0
        for i in range(-radius, radius + 1):
            val = math.exp(-(i ** 2) / (2 * sigma ** 2))
            kernel.append(val)
            sum_val += val
        # Normalize
        kernel = [v / sum_val for v in kernel]

        # Horizontal pass
        temp = [0.0] * self.iNumPlots
        for i in range(self.iNumPlots):
            x = i % self.iNumPlotsX
            y = i // self.iNumPlotsX
            sum = 0.0
            weight = 0.0
            for k in range(-radius, radius + 1):
                xx = x + k
                if self.wrapX:
                    xx = xx % self.iNumPlotsX
                if 0 <= xx < self.iNumPlotsX:
                    ii = y * self.iNumPlotsX + xx
                    sum += grid[ii] * kernel[k + radius]
                    weight += kernel[k + radius]
            temp[i] = sum / weight if weight > 0 else 0

        # Vertical pass
        result = [0.0] * self.iNumPlots
        for i in range(self.iNumPlots):
            x = i % self.iNumPlotsX
            y = i // self.iNumPlotsX
            sum = 0.0
            weight = 0.0
            for k in range(-radius, radius + 1):
                yy = y + k
                if self.wrapY:
                    yy = yy % self.iNumPlotsY
                if 0 <= yy < self.iNumPlotsY:
                    ii = yy * self.iNumPlotsX + x
                    sum += temp[ii] * kernel[k + radius]
                    weight += kernel[k + radius]
            result[i] = sum / weight if weight > 0 else 0

        return result

    class Perlin2D:
        def __init__(self, seed=None):
            self.p = list(range(256))
            if seed is not None:
                random.seed(seed)
            random.shuffle(self.p)
            self.p += self.p  # Repeat

        def noise(self, x, y):
            # Find unit grid cell containing point
            X = int(math.floor(x)) & 255
            Y = int(math.floor(y)) & 255
            # Relative x, y in cell
            xf = x - math.floor(x)
            yf = y - math.floor(y)
            # Fade curves
            u = self.fade(xf)
            v = self.fade(yf)
            # Hash coordinates of the 4 square corners
            aa = self.p[self.p[X] + Y]
            ab = self.p[self.p[X] + Y + 1]
            ba = self.p[self.p[X + 1] + Y]
            bb = self.p[self.p[X + 1] + Y + 1]
            # Add blended results from 4 corners
            x1 = self.lerp(self.grad(aa, xf, yf), self.grad(ba, xf - 1, yf), u)
            x2 = self.lerp(self.grad(ab, xf, yf - 1),
                           self.grad(bb, xf - 1, yf - 1), u)
            return (self.lerp(x1, x2, v) + 1) / 2  # Normalize to [0,1]

        def fade(self, t):
            # Perlin's fade function
            return t * t * t * (t * (t * 6 - 15) + 10)

        def lerp(self, a, b, t):
            return a + t * (b - a)

        def grad(self, hash, x, y):
            # Convert low 4 bits of hash code into 12 gradient directions
            h = hash & 7
            u = x if h < 4 else y
            v = y if h < 4 else x
            return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)

    def generate_perlin_grid(self, scale=10.0, seed=None):
        perlin = self.Perlin2D(seed)
        grid = []
        for y in range(self.iNumPlotsY):
            for x in range(self.iNumPlotsX):
                nx = x / scale
                ny = y / scale
                grid.append(perlin.noise(nx, ny))
        return grid

    def get_perlin_noise(self, x, y, seed=None):
        """Perlin noise scaled to approximate original frequency"""
        if not hasattr(self, '_perlin_instance'):
            self._perlin_instance = self.Perlin2D(seed)

        # Scale to match original frequency characteristics
        scale = 0.015  # Approximately matches x*57, y*113 frequency
        return self._perlin_instance.noise(x * scale, y * scale)

    def FindValueFromPercent(self, ilist, percent, descending=True):
        """
        Returns the value from ilist such that 'percent' of the elements are above (if descending=True)
        or below (if descending=False) that value.
        """
        sorted_list = sorted(ilist, reverse=descending)
        index = int(percent * len(ilist))
        if index >= len(ilist):
            index = len(ilist) - 1
        return sorted_list[index]


def generatePlotTypes():
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

    return em.plotTypes
