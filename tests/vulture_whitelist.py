# Vulture whitelist: names called by the Civ IV engine or by test mocks.
_ = None
_.getDescription
_.isAdvancedMap
_.isClimateMap
_.isSeaLevelMap
_.beforeInit
_.beforeGeneration
_.afterGeneration
_.isNone

# Pending: game-provided climate settings (CvClimateInfo). Remove when the climate TODO is done.
_.desertPercentChange
_.jungleLatitude
_.snowLatitudeChange
_.tundraLatitudeChange
_.grassLatitudeChange
_.desertBottomLatitudeChange
_.desertTopLatitudeChange
_.iceLatitude
_.randIceLatitude
_.get_y_for_latitude

# Pending: F.2 landmass placement rule. Remove when the F.2 placement TODO is done.
_.continent_assignments
