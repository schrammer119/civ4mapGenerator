# Vulture whitelist: names called by the Civ IV engine or by test mocks.
_ = None
_.getDescription
_.isAdvancedMap
_.isClimateMap
_.isSeaLevelMap
_.beforeInit
_.beforeGeneration
_.afterGeneration
_.addLakes
_.isNone

# Pending: game-provided climate settings (CvClimateInfo). Remove when the climate TODO is done.
_.desert_percent_change
_.jungle_latitude
_.snow_latitude_change
_.tundra_latitude_change
_.grass_latitude_change
_.desert_bottom_latitude_change
_.desert_top_latitude_change
_.ice_latitude
_.rand_ice_latitude
_.get_y_for_latitude

# Pending: F.2 landmass placement rule. Remove when the F.2 placement TODO is done.
_.continent_assignments
