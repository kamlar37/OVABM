import copy
import collections
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pdb
import scipy
import seaborn as sns
import time
import warnings

from contexttimer import Timer
from types import SimpleNamespace

from .state_variables import population_state, human_state
from .initial_population import create_initial_population
from .utils import My_timer, insert_humans_summary_statistics, getAlphaBeta, getShapeScale

def ov_model(p_dict, full_history=False, initial_population_only=False, checkpoint_directory=None, reset_worm_days=True):
    p = SimpleNamespace(**p_dict)
    
    if 'seed' in p_dict:
        np.random.seed(int(p.seed))
        
    # Create initial population in case there is no checkpointing
    if checkpoint_directory is None:
        humans, pop = create_initial_population(p)
    else:
        humans = np.load(os.path.join(checkpoint_directory, 'checkpoint_humans.npy'))
        if reset_worm_days:
            humans['worm_days'] = 0
        pop = np.load(os.path.join(checkpoint_directory, 'checkpoint_history.npy'))[-1]

    # Create vector containing all time points tracked in simulation
    number_of_timesteps = p.timesteps_per_year * p.number_of_years
    t = np.linspace(0, p.number_of_years * 365,
                    number_of_timesteps + 1)
    
    # Timestep size is equal to first nonzero time point
    dt = t[1]

    # Adjust mortality rates to timestep if timestep > 1 day
    if dt > 1:
        # Append ones to mortality rates
        # If people older than highest age provided by mortality file, then they will die with certainty
        p.mortality_female = np.append(p.mortality_female, np.ones(1000))
        p.mortality_male = np.append(p.mortality_male, np.ones(1000))

        remainder = dt%1

        survival_female = 1 - p.mortality_female
        survival_female_stack = [np.roll(survival_female, -i) for i in range(math.ceil(dt))]
        if remainder > 0:
            survival_female_stack[-1] = np.power(survival_female_stack[-1], remainder)
        survival_female = np.prod(np.vstack(survival_female_stack), axis=0)
        p.mortality_female = 1 - survival_female

        survival_male = 1 - p.mortality_male
        survival_male_stack = [np.roll(survival_male, -i) for i in range(math.ceil(dt))]
        if remainder > 0:
            survival_male_stack[-1] = np.power(survival_male_stack[-1], remainder)
        survival_male = np.prod(np.vstack(survival_male_stack), axis=0)
        p.mortality_male = 1 - survival_male

    if full_history:
        history_humans = np.zeros((number_of_timesteps + 1, p.N), dtype=human)
        history_humans[0] = humans

    history = np.zeros((number_of_timesteps + 1), dtype=population_state)
    history[0] = pop
    history['t'] = t

    if(initial_population_only):
        return pop, humans

    # Define tempory vectors for loop
    latrine_use_probability = np.zeros((p.N), dtype=np.float64)
    mortality = np.zeros((p.N), dtype=np.float64)
    MDA_adherence_probabilities = None
    education_adherences = None

    # Individuals keep gender when being "reborn", therefore can store gender index once outside loop
    idx_male = humans['sex']
    idx_female = ~humans['sex']
    

    for i in range(number_of_timesteps):

        # Initialize timer for logging purposes
        timer = My_timer()

        # Determine current time
        t = i*dt
        
        # Update timestep variable in humans vector
        humans['t'] = t

        # Create views of the population matrix for reading and updating
        pop = history[i].view()
        pop_next = history[i+1].view()

        # MDA submodule ========================================
        
        # Determine if MDA round happening in current time step.
        MDA_selection_index = (p.MDA_timing >= t) & (p.MDA_timing < t+dt-1e-6) # -1e-6 for numerical reasons

        # Multiple MDA rounds in one timestep not possible
        if MDA_selection_index.sum() > 1:
            raise ValueError("Multiple MDA rounds within a timestep not supported")

        # If yes, apply MDA
        if np.any(MDA_selection_index):
            # Assign MDA eligibility to everyone, then remove according to criteria
            MDA_eligible = np.ones(p.N)

            # 1) No MDA for pregnant women
            digits = np.digitize(humans['age'][idx_female],
                         p.pregnancy_age_bins)
            pregnancy_proportions = p.pregnancy_proportions[digits]
            not_pregnant = np.random.binomial(1, 1 - pregnancy_proportions, idx_female.sum())
            MDA_eligible[idx_female] = not_pregnant
            # 2) No MDA for children under minimum age
            MDA_eligible[humans['age'] < p.MDA_minimum_age] = 0
            
            # Distribute MDA among eligibles according to coverage level and distribution plan
            if p.MDA_strategy == "random":
                # pdb.set_trace()
                MDA = np.random.binomial(1, p=MDA_eligible * p.MDA_coverage[MDA_selection_index.squeeze()])


            elif p.MDA_strategy == 'gaussian_copula':
                if MDA_adherence_probabilities is None:
                    # Determine parameters of beta marignal given mean/variance
                    a, b = getAlphaBeta(p.MDA_coverage, p.MDA_variance)
                    # Determine shape/scale parameter of gamma distributed that generated individual betas
                    shape, scale = getShapeScale(p.beta_hf_mean, p.beta_hf_variance)
                    
                    # Gaussian copula                     
                    ux = scipy.stats.gamma.cdf(humans['beta_hf'], a=shape, scale=scale)
                    xx = scipy.stats.norm.ppf(ux)
                    rho = p.MDA_copula_correlation
                    yy = rho * xx + scipy.stats.norm.rvs(size=len(xx)) * math.sqrt(1-rho**2)
                    uy = scipy.stats.norm.cdf(yy)
                    MDA_adherence_probabilities = scipy.stats.beta.ppf(uy, a=a, b=b)

                MDA = np.random.binomial(1, p=MDA_eligible*MDA_adherence_probabilities)
    
            # Set worms to zero if MDA received
            if not p.fake_MDA:
                humans['worms'][MDA.squeeze().astype(np.bool)] = 0

            if p.education_strategy == 'with_mda':
                humans['beta_multiplier'][MDA.squeeze().astype(np.bool)] = \
                    humans['beta_multiplier'][MDA.squeeze().astype(np.bool)] * p.education_efficacy
                #print(humans['beta_multiplier'].mean())
                
            # Increase MDA  worms to zero if MDA received
            humans['MDA_treatments'][MDA.squeeze().astype(np.bool)] += 1

        timer.measure("MDA")

        # Education submodule ======================================== 
        
        # Previous education decays
        humans['beta_multiplier'] = 1 - (p.educ_decay**(1/p.timesteps_per_year)) * (1 - humans['beta_multiplier'])
        
        # Determine if education round happening in current time step.
        educ_selection_index = (p.education_timing >= t) & (p.education_timing < t+dt-1e-6) # -1e-6 for numerical reasons

        # Multiple education rounds in one timestep not possible
        if educ_selection_index.sum() > 1:
            raise ValueError("Multiple education rounds within a timestep not supported")

        # If yes, apply education
        if np.any(educ_selection_index):
            # Assign eudcation eligibility to everyone, then remove according to criteria
            educ_eligible = np.ones(p.N)

            # No eduaction for children under minimum age
            educ_eligible[humans['age'] < p.education_minimum_age] = False
            
            # Distribute MDA among eligibles according to coverage level and distribution plan

            # Modify beta_multiplier for people that received eduction
            if p.education_strategy == "random":
                
                educ = np.random.binomial(1, p=educ_eligible.astype(int) * p.education_coverage[educ_selection_index])
                
                humans['beta_multiplier'][educ.astype(np.bool)] = \
                    humans['beta_multiplier'][educ.astype(np.bool)] * p.education_efficacy[educ_selection_index]

            elif p.education_strategy == 'gaussian_copula':
                if education_adherences is None:
                    # Determine parameters of beta marignal given mean/variance
                    a, b = getAlphaBeta(p.education_efficacy_mean, p.education_efficacy_variance)
                    # Determine shape/scale parameter of gamma distributed that generated individual betas
                    shape, scale = getShapeScale(p.beta_hf_mean, p.beta_hf_variance)
                    
                    # Gaussian copula                     
                    ux = scipy.stats.gamma.cdf(humans['beta_hf'], a=shape, scale=scale)
                    xx = scipy.stats.norm.ppf(ux)
                    rho = p.education_copula_correlation
                    yy = rho * xx + scipy.stats.norm.rvs(size=len(xx)) * math.sqrt(1-rho**2)
                    uy = scipy.stats.norm.cdf(yy)
                    education_adherences = scipy.stats.beta.ppf(uy, a=a, b=b)
                
                humans['beta_multiplier'][educ_eligible.astype(np.bool)] = humans['beta_multiplier'][educ_eligible.astype(np.bool)]* education_adherences[educ_eligible.astype(np.bool)]

        # Latrine submodule ======================================== 

        # Determine if eudcation change is happening in current time step.
        latrine_selection_index = (p.latrine_change_timing >= t) & (p.latrine_change_timing < t+dt-1e-6) # -1e-6 for numerical reasons

        # Multiple latrine changes rounds in one timestep not possible
        if latrine_selection_index.sum() > 1:
            raise ValueError("Multiple latrine changes within a timestep not supported")

        # If timed, apply latrine change
        if np.any(latrine_selection_index):
            
            if p.latrine_change_strategy == "random":
                humans['latrine'] = np.random.binomial(1, p=p.latrine_change_coverage[latrine_selection_index], size=p.N)
                
            elif p.latrine_change_strategy == "addition":
                latrine_index = humans['latrine']
                latrine_gap = p.latrine_change_coverage[latrine_selection_index] - latrine_index.mean()
                if latrine_gap>0:
                    gap_probability = latrine_gap / (1-humans['latrine'].mean())
                    humans['latrine'][~humans['latrine']] = np.random.binomial(1, p=gap_probability, size=(p.N-humans['latrine'].sum())).astype('bool')
            
            humans['latrine_use'] = humans['latrine']


        # Human worms submodule ========================================
 
        # Determine humans that have reached minimum age for consuming fish in this timestep
        idx_new_eaters = \
            np.all([humans['age'] >= p.minimum_age_for_worm_infection, 
                    humans['age'] < (p.minimum_age_for_worm_infection + dt)],
                   axis=0)
        if idx_new_eaters.sum()>0:
            humans['eating'][idx_new_eaters] = 1
        timer.measure("New eaters")

        humans_worms_old = humans['worms'].copy()
        # Mortality of parasites in humans
        humans['worms'] -= np.random.binomial(humans['worms'], 1-np.exp(-dt * p.mu_ph))\
            .astype(np.int64)
        timer.measure("Parasite mortality")

         # Acquisition of new parasites in humans
        lbda_humans = (dt * humans[humans['eating']]['beta_hf']
                       * humans[humans['eating']]['beta_multiplier'] * pop['fish']/p.N_fish)
        humans['worms'][humans['eating']] += \
            np.random.poisson(lbda_humans).astype(np.int64)
        humans['worms'][humans['worms']>p.max_worms_per_person] = p.max_worms_per_person
        humans_worms_average_in_timestep = (humans_worms_old + humans['worms']) / 2
        humans['worm_days'] += humans_worms_average_in_timestep * dt
        timer.measure("Parasite acquisition")

        humans['epg'][humans['age']>=p.minimum_age_for_worm_infection] = \
            p.worms_to_epg_transformation(humans['worms'][humans['age']>=p.minimum_age_for_worm_infection])
        timer.measure("EPG calculation")


        # Demography submodule  ========================================

        # Assign mortality rates to humans according to age and gender
        mortality[idx_female] = p.mortality_female[np.rint(humans[idx_female]['age']).astype(int)].flatten()
        mortality[idx_male] = p.mortality_male[np.rint(humans[idx_male]['age'],).astype(int)].flatten()
        timer.measure("Assign natural death rates according to age")

        # Assign death to individuals based on rates
        deaths_age_idx = (np.random.binomial(1, 1-np.exp(-mortality))==1)
        timer.measure("Draw deaths from age")
        
        deaths = deaths_age_idx
        deaths = deaths>0
        number_of_deaths = deaths.sum()
        timer.measure("Sum deaths from age")

        # Human aging
        humans['age'] += dt
        timer.measure("Humans age")
        
        # Replace dead individuals with newborns
        if number_of_deaths > 0:
            humans['age'][deaths] = np.random.choice(math.ceil(dt), number_of_deaths)
            humans['worms'][deaths] = 0
            humans['epg'][deaths] = 0
            humans['eating'][deaths] = False
            humans['worm_days'][deaths] = 0
            humans['MDA_treatments'][deaths] = 0
            # humans['beta_multiplier'][deaths] = 1
        timer.measure("Replace dead humans")

        if full_history:
            history_humans[i+1] = humans



        # Nonhuman worms submodule ========================================  

        # Worms in dogs
        lambda_dogs = dt * pop['fish']/p.N_fish * p.beta_df * p.N_dogs

        pop_next['dogs'] = (pop['dogs']
                         + np.random.poisson(lambda_dogs)
                         - np.random.binomial(pop['dogs'], 1-np.exp(-dt * p.mu_pd)))
        timer.measure("Update dogs")

        # Worms  in cats
        lambda_cats =  dt * pop['fish']/p.N_fish * p.beta_cf * p.N_cats
        pop_next['cats'] = (pop['cats']
                         + np.random.poisson(lambda_cats)
                         - np.random.binomial(pop['cats'], 1-np.exp(-dt * p.mu_pc)))
        timer.measure("Update cats")


        # Intermediate hosts submodule ========================================

        # Update snails
        # Collect epg sums for definitive hosts. Work with mean for nonhumans, as we have limited
        # information on the distribution of worm burdern there
        eggs_humans_total = humans[(~humans['latrine_use'])]['epg'].sum() * p.feces_gram_per_day_human
        eggs_dogs_total = p.worms_to_epg_transformation(pop['dogs']/p.N_dogs, rnd = False) * p.N_dogs * p.feces_gram_per_day_dog  
        eggs_cats_total = p.worms_to_epg_transformation(pop['cats']/p.N_cats, rnd = False) * p.N_cats * p.feces_gram_per_day_cat

        lambda_snails = (dt
                         * (eggs_humans_total * p.beta_sh/p.N
                            + eggs_dogs_total * p.beta_sd/p.N_dogs
                            + eggs_cats_total * p.beta_sc/p.N_cats) 
                         * (p.N_snails - pop['snails']))

        pop_next['snails'] = (pop['snails']
                           + np.random.poisson(lambda_snails)
                           - np.random.binomial(pop['snails'], 1-np.exp(-dt * p.mu_s)))
        if pop_next['snails'] > p.N_snails:
            pop_next['snails'] = p.N_snails
        if pop_next['snails'] < 0:
            pop_next['snails'] = 0
        timer.measure("Update snails")

        # Update fish
        lbda_infections_fish = dt * pop['snails']/p.N_snails * p.beta_fs * (p.N_fish - pop['fish'])

        pop_next['fish'] = (pop['fish']
                           + np.random.poisson(lbda_infections_fish)
                           - np.random.binomial(pop['fish'], 1-np.exp(-dt * p.mu_f)))

        if pop_next['fish'] > p.N_fish:
            pop_next['fish'] = p.N_fish
        timer.measure("Update fish")
        
        
        # Calculate and store summary statistics for humans ========================================
        insert_humans_summary_statistics(pop_next, humans, p)
        timer.measure("Update human summary statistics")

        # Logging ========================================
        if p.timer_log:
            timer.print_log()
        if p.verbose and t % 365 == 0.:
            print(f"year {int(t/365)}/{p.number_of_years}", flush=True)

    if full_history:
        return (history, history_humans)

    return (history, humans)
