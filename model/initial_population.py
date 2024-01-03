import numpy as np

from .utils import insert_humans_summary_statistics
from .state_variables import human_state, population_state

def beta_hf_gamma(p):
    gamma_shape = p.beta_hf_mean**2 / p.beta_hf_variance
    gamma_scale = p.beta_hf_variance / p.beta_hf_mean
    beta_hf = np.random.gamma(gamma_shape, gamma_scale, p.N)
    return beta_hf

def beta_hf_empirical_worms(p):
    # @todo: first transform, then draw, as described in ODD
    beta = np.random.choice(
        p.initial_worm_distribution_values,
        size=p.N,
        p=p.initial_worm_distribution_probabilities)

    beta[beta>0] = 10 ** (p.beta_empirical_worms_a + p.beta_empirical_worms_b * np.log10(beta[beta>0]))
    return beta

# Define mappings from string parameters to functions specified above
function_mapping = \
    {'gamma': beta_hf_gamma,
     'empirical_worms': beta_hf_empirical_worms}

def create_initial_population(p):

    pop = np.zeros(1, dtype=population_state)
    humans = np.zeros((p.N), dtype=human_state)

    # Generate human population ==========================================
    # Set gender ---------------------------------------------------------
    humans['sex'] = np.random.binomial(1, p.proportion_male, p.N)

    # Set age ------------------------------------------------------------
    # Draw age given probability vector with probablity for each day
    humans['age'][humans['sex']] = np.random.choice(
        range(p.population_distribution_male.size),
        humans['sex'].sum(),
        p=p.population_distribution_male)

    humans['age'][~humans['sex']] = np.random.choice(
        range(p.population_distribution_female.size),
        (~humans['sex']).sum(),
        p=p.population_distribution_female)

    # Set initial worm counts in humans according to provided distribution ----------------------------------
    initial_worm_distribution_values = p.initial_worm_distribution_values
    initial_worm_distribution_probabilities = p.initial_worm_distribution_probabilities

    idx_minimum_age = (humans['age'] >= p.minimum_age_for_worm_infection)
    humans['worms'][idx_minimum_age] = np.random.choice(
        initial_worm_distribution_values,
        idx_minimum_age.sum(),
        p=initial_worm_distribution_probabilities)
    

    humans['eating'] = True
    
    # If we need more eaters than implied by the initial populations, add these now despite worm count 0
    eaters_gap = p.min_initial_worm_eating_proportion - (1-p.initial_worm_distribution_probabilities[0])
    if eaters_gap > 0:
        idx = np.logical_and(~humans['eating'], idx_minimum_age)
        humans['eating'][idx] = np.random.binomial(1, min(1, eaters_gap/p.initial_worm_distribution_probabilities[0]), idx.sum())
            
    humans['epg'] = p.worms_to_epg_transformation(humans['worms'])
    
    # Set beta for indivudals ---------------------------
    beta_hf = function_mapping[p.beta_hf_distribution](p)

    # Align individual beta_hf with initial worm burden if option set to true in parameters 
    if p.align_beta_hf_to_initial_worm_burden:
        sorted_beta_hf = np.sort(beta_hf)
        idx_humans_sorted = np.argsort(humans['worms'])
        humans['beta_hf'][idx_humans_sorted] = sorted_beta_hf
    else:
        humans['beta_hf'] = beta_hf
    
    # Set initial worm days to 0 ----------------------------------
    humans['worm_days'] = np.zeros(p.N)

    # Set initial MDA treatment counter to 0 ----------------------------------
    humans['MDA_treatments'] = np.zeros(p.N)
    
    # initialize beta_multipilier to 1 for all indiviudals ----------------------------------------
    humans['beta_multiplier'] = np.ones(p.N)

    # Random latrine availability ----------------------------------------
    humans['latrine'] = np.random.binomial(1, p.latrine_coverage, p.N)
    humans['latrine_use'] = humans['latrine']

    # Update summary statistics of humans in population array ------------
    insert_humans_summary_statistics(pop, humans, p)

    # Animal population ==================================================
    pop['dogs'] = int(p.N_dogs * p.initial_worms_per_dog)
    pop['cats'] = int(p.N_cats * p.initial_worms_per_cat)
    pop['fish'] = int(p.N_fish * p.initial_prevalence_fish)
    pop['snails'] = int(p.N_snails * p.initial_prevalence_snails)
    
    return (humans, pop)
