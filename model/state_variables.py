import numpy as np

population_state = np.dtype([
    ('t', np.float64),
    # Environment variables
    ('dogs', np.int64),
    ('cats', np.int64),
    ('snails', np.int64),
    ('fish', np.int64),
    # Variables tracking human population summary statistics
    ('humans', np.int64),
    ('humans_eligible_variance', np.float64),
    ('humans_positive', np.int64),
    ('mean_age', np.float64),
    ('latrine_coverage', np.float64),
    ('latrine_use', np.float64),
    ('eating', np.float64),
    ('eating_eligible', np.float64),
    ('humans_variance', np.float64),
    ('humans_maximum', np.int64),
    ('humans_epg_mean', np.float64),
    ('humans_epg_variance', np.float64),
    ('humans_epg_maximum', np.int64),
    ('humans_epg_none', np.int64),
    ('humans_epg_low', np.int64),
    ('humans_epg_medium', np.int64),
    ('humans_epg_high', np.int64),
    ('humans_epg_low_mean', np.float64),
    ('humans_epg_medium_mean', np.float64),
    ('humans_epg_high_mean', np.float64),
])

human_state = np.dtype([
    ('t', np.float64),
    ('sex', np.bool),
    ('age', np.float32),
    ('worms', np.int64),
    ('worm_days', np.float64),
    ('beta_hf', np.float64),
    ('beta_multiplier', np.float64),
    ('eating', np.bool),
    ('latrine', np.bool),
    ('latrine_use', np.bool),
    ('MDA_treatments', np.int64),
    ('epg', np.int64),
])

