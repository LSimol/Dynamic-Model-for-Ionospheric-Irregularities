import numpy as np

# Configuration Constants
SATELLITES = ['A', 'B']
YEARS = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
MONTHS = [f"{i:02d}" for i in range(1, 13)]

# Solar Activity Thresholds (F10.7 index)
SOLAR_THRESHOLDS = {
    'LSA': (0, 74),
    'MSA': (74, 106),
    'HSA': (106, 9999)
}

# Season DOY (Day of Year) Ranges
SEASONS = {
    'Summer': [(129, 221)],
    'Equinox': [(221, 313), (40, 129)],
    'Winter': [(0, 40), (313, 366)]
}

# Clock Angle Sectors (Degrees)
CLOCK_ANGLES = {
    'UR': (30, 90),
    'R':  (90, 150),
    'BR': (150, 210),
    'B':  (210, 270),
    'BL': (270, 330),
    'L':  (330, 390) # 330-360 and 0-30 handled by wrapping logic
}

# Grid Resolution
LAT_RES = 1
LON_RES = 1
NSIDE = 32 # Healpix resolution
