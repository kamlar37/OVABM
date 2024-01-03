import numpy as np
import os
import pandas as pd

# Import age distributions and mortality
mortality_male = np.load(os.path.join(os.path.dirname(__file__), "mortality-male.npy"))
population_male = np.load(os.path.join(os.path.dirname(__file__), "age-distribution-male.npy"))
mortality_female = np.load(os.path.join(os.path.dirname(__file__), "mortality-female.npy"))
population_female = np.load(os.path.join(os.path.dirname(__file__), "age-distribution-female.npy"))


def worm_to_epg(worms, rnd = False):
    # epg = 20 *(worms)**(.82)
    epg = (20 *worms)**(.82)
    if rnd:
        epg = np.round(epg)
    return epg
def epg_to_worm(epg, rnd = True):
    worms = 1/20 * (epg)**(1/.82)
    if rnd:
        worms = np.round(worms)
    return worms

# # Worm transformation functions used by the model
# def epg_to_worm(epg, rnd = True):
#     # EPG for both functions is equal at 2993.573 EPG
#     threshold = 2993.5738952900006
#     worms = np.zeros(epg.shape)
#     worms[epg<threshold] = np.sqrt(epg[epg<threshold] + 1) - 1
#     worms[epg>=threshold] = epg[epg>=threshold]**1.75438 * 10**-4.3684210526
#     if rnd:
#         worms = np.round(worms)
#     return worms

# def worm_to_epg(worms, rnd = True):
#     # EPG for both functions is equal at 53.7227 worms
#     threshold = 53.7227
#     epg = np.zeros(worms.shape)
#     epg[worms<threshold] = worms[worms<threshold]**2 + 2*worms[worms<threshold]
#     epg[worms>=threshold] = 10**2.49 * worms[worms>=threshold]**0.57 
#     if rnd:
#         epg = np.round(epg)
#     return epg


data = pd.read_csv(os.path.join(os.path.dirname(__file__), "IDRC-2012.csv"), sep=';')
data['worms'] = epg_to_worm(data.ovepg)
worm_proportions_empirical = data[['id', 'worms']].groupby('worms').count()/data.shape[0]
worm_prevalence_empirical = (data.worms>0).sum()/data.shape[0]

data_animals = pd.read_csv('model/parameters/dogs-cats-buffalos.csv')
data_animals['worms'] = epg_to_worm(data_animals.OVEGG)
data_animals = data_animals[['ANIMAL', 'worms']].groupby('ANIMAL').mean()


# %% Set parameters ===================================================
N = 15000
p = {

    # Human population
    'N': N,
    'proportion_male': 0.5013,
    'population_distribution_male': population_male,
    'population_distribution_female': population_female,
    'mortality_male': mortality_male,
    'mortality_female': mortality_female,
    'pregnancy_age_bins': np.array([15*365, 20*365, 25*365, 30*365, 35*365, 40*365, 45*365, 50*365]),
    'pregnancy_proportions': np.array([0,  0.0942, 0.185, 0.157, 0.104, 0.059, 0.0235, 0.009, 0]),
    
    # Transformation functions
    'worms_to_epg_transformation': worm_to_epg, 
    'epg_to_worms_transformation': epg_to_worm, 

    # Initial worm distribution.
    'initial_worm_distribution_values': worm_proportions_empirical.index,
    'initial_worm_distribution_probabilities': worm_proportions_empirical.id,
    'min_initial_worm_eating_proportion': 1,
    'max_worms_per_person': 1e5,
    'minimum_age_for_worm_infection': 2*365,
    'epg_classification_bin_edges': [0., 1., 1e3, 1e4, 1e6], 
    
    # Latrine coverage
    'latrine_coverage': 0.4406,
    'latrine_change_timing': np.array([0]),
    'latrine_change_coverage': np.array([0]),
    'latrine_change_strategy': 'addition',

    # MDA
    'MDA_timing': np.array([0]),
    'MDA_coverage': np.array([0]),
    'MDA_strategy': "random",
    'MDA_minimum_age': 6*365,
    'fake_MDA': False,
    
    # Education
    'education_timing': np.array([0]),
    'education_coverage': np.array([0]),
    'education_efficacy': np.array([0]),
    'education_minimum_age': 6*365,
    'education_strategy': 'top',
    'educ_decay': 0.97,
    
    # Animal population
    'N_dogs': round(N/6),
    'initial_worms_per_dog': 0,
    'N_cats': round(N/6),
    'initial_worms_per_cat': 0,
    'N_snails': N*100,
    'initial_prevalence_snails': 0.0029,
    'N_fish': N*10,
    'initial_prevalence_fish': 0.2691,
    
    # Feces volume
    'feces_gram_per_day_human': 100,
    'feces_gram_per_day_dog': 100/6,
    'feces_gram_per_day_cat': 100/20,
    
    # Gamma
    'beta_hf_distribution': 'gamma',
    'log_beta_hf_mean': -0.321, # -1.1395422484531164
    'log_beta_hf_variance': 0.277, # -1.45
    'log_beta_df': -999,
    'log_beta_cf': -999,
    'log_beta_sx': -9.875,
    'log_beta_fs': -.856,
    
    # Mortalities: Daily rates
    'mu_ph': 1/10/365,
    'mu_pd': 1/4/365, #1-math.exp(-1/(2.2*365)),
    'mu_pc': 1/4/365,
    'mu_s': 1/365,
    'mu_f': 1/2.5/365,

    # Time steps
    'number_of_years': 150,
    'timesteps_per_year': 120,
    
    'align_beta_hf_to_initial_worm_burden': True,
    'verbose': False,
    'timer_log': False,
}
