import numpy as np
import time
import warnings

# from .state_variables import human_state

def getAlphaBeta(mu, var):
    '''Calculates alpha and beta parameter for beta distribution
    with chosen mean mu and variance var'''
    sigma = np.sqrt(var)
    alpha = mu**2 * ((1 - mu) / sigma**2 - 1 / mu)
    beta = alpha * (1 / mu - 1)
    return alpha, beta

def getShapeScale(mu, var):
    '''Calculates shape and scale parameter for gamma distribution
    with chosen mean mu and variance var'''
    shape = mu**2 / var
    scale = var / mu
    return shape, scale 

class My_timer:
    def __init__(self):
        self.t = time.time()
        self.t0 = time.time()
        self.log_descr = [] 
        self.log_times = [] 

    def measure(self, descr):
        t_new = time.time()
        self.log_descr += [descr]
        t_elapsed = t_new - self.t
        self.log_times += [t_elapsed]
        self.t = t_new
        
    def print_log(self):
        print(f"Total time:  {time.time() - self.t0}")
        t_elapsed_total = sum(self.log_times)
        for i in range(len(self.log_descr)):
            print(f"{self.log_descr[i]}: {self.log_times[i]/t_elapsed_total}")



def insert_humans_summary_statistics(pop, humans, p):
    
    pop['humans'] = humans['worms'].sum()
    pop['mean_age'] = humans['age'].mean()
    pop['eating'] = humans['eating'].sum()
    pop['eating_eligible'] = (humans['age']>=p.minimum_age_for_worm_infection).sum()
    pop['latrine_coverage'] = humans['latrine'].sum()
    pop['latrine_use'] = humans['latrine_use'].sum()
    pop['humans_positive'] = (humans['worms']>0).sum()
    pop['humans_maximum'] = humans['worms'].max()
    pop['humans_variance'] = humans['worms'].var()
    pop['humans_eligible_variance'] = humans['worms'][humans['age']>=p.minimum_age_for_worm_infection].var()
    epg = humans[humans['age']>=p.minimum_age_for_worm_infection]['epg']
    pop['humans_epg_mean'] = epg.mean()
    pop['humans_epg_maximum'] = epg.max()
    pop['humans_epg_variance'] = epg.var()
    hist = np.histogram(epg, p.epg_classification_bin_edges)[0]
    pop['humans_epg_none'] = hist[0]
    pop['humans_epg_low'] = hist[1]
    pop['humans_epg_medium'] = hist[2]
    pop['humans_epg_high'] = hist[3]
    
    epg_groups = np.digitize(epg, p.epg_classification_bin_edges)

    # Contextmaganger prevent printing runtime warnings in case of empty categories
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pop['humans_epg_low_mean'] = epg[epg_groups==2].mean()
        pop['humans_epg_medium_mean'] = epg[epg_groups==3].mean()
        pop['humans_epg_high_mean'] =  epg[epg_groups==4].mean()