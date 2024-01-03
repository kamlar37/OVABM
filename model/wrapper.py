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

from .core import ov_model
from .utils import My_timer, insert_humans_summary_statistics

def f(k, M, P):
    return 1 - (1 + M/k)**(-k) - P
def calculate_k(M, P):
    try:
        sol = scipy.optimize.root_scalar(f, args=(M, P), bracket=(0.001,2), method='bisect')
        return sol.root
    except Exception as err:
        # print(err)
        return np.nan

class Ov:
    """Class wrapping the core model providing functionality for parameter setting
    result postprocessing, plotting etc."""
    
    def __init__(self, parameters, ov_model = ov_model, checkpoint_directory = None, store_raw = False):
        
        # Two copies need because of iterations over dict(?)
        # @Todo fix this!
        parameters = copy.deepcopy(parameters)
        p = copy.deepcopy(parameters)
        
        # Transform some input parameters to parameters used by the model
        if 'log_beta_sx' in p:
            p['log_beta_sh'] = p['log_beta_sx']
            p['log_beta_sd'] = p['log_beta_sx']
            p['log_beta_sc'] = p['log_beta_sx']
        if 'beta_sx' in p:
            p['beta_sh'] = p['beta_sx']
            p['beta_sd'] = p['beta_sx']
            p['beta_sc'] = p['beta_sx']

        # Transform any parameter with a log-prefix 
        for key, val in p.items():
            if key[:4] == 'log_':
                parameters[key[4:]] = 10**val
            else:
                parameters[key] = val
        if 'beta_sx' in parameters:
            parameters['beta_sh'] = parameters['beta_sx']
            parameters['beta_sd'] = parameters['beta_sx']
            parameters['beta_sc'] = parameters['beta_sx']

        # Translate numbered parameters for transmission to humans to array. This interface allows to avoid
        # hierarchical parameter sets in fitting process, which would make things complicated
        if 'beta_hf_0' in parameters:
            parameters['beta_hf'] = [parameters['beta_hf_0']]
            
            i = 1
            while f"beta_hf_{i}_factor" in parameters:
                parameters['beta_hf'] += [parameters['beta_hf'][(i-1)] * parameters[f"beta_hf_{i}_factor"]]
                i += 1
            
            while f"beta_hf_{i}" in parameters:
                parameters['beta_hf'] += [parameters[f"beta_hf_{i}"]]
                i += 1

        self.parameters = parameters
        self.ov_model = ov_model
        self.checkpoint_directory = checkpoint_directory
        self.store_raw = store_raw

    def run(self, timer = False, initial_population_only=False):
        history_raw, humans_raw = self.run_core(timer=timer, initial_population_only=initial_population_only)
        
        history, humans = self.postprocess_results(history_raw, humans_raw)
        return history, humans

    def run_core(self, timer=False, initial_population_only=False):

        # Run core model
        with Timer(output=timer):
            history, humans = self.ov_model(self.parameters, checkpoint_directory=self.checkpoint_directory,
                                           initial_population_only=initial_population_only)

        return history, humans
    
    def postprocess_results(self, history, humans):
        if self.store_raw:
            self.history_raw, self.humans_raw = history, humans
        # Convert structured arrays from model output to dataframe
        history, humans = pd.DataFrame(history), pd.DataFrame(humans)
        
        # Calculate results that are functions of other variables
        history.index = history['t']/365
        history['worms_per_human'] = history['humans'] / self.parameters['N']
        history['worms_per_human_eligible'] = history['humans'] /  history['eating_eligible']
        history['prevalence_humans'] = (history['humans_positive'] /
                                        self.parameters['N'])
        history['prevalence_humans_eligible'] = (history['humans_positive'] /
                                                 history['eating_eligible'])
        history['worms_per_positive_human'] =  (history['humans'] /
                                     history['humans_positive'])
        history['worms_per_cat'] = history['cats'] / self.parameters['N_cats']
        history['worms_per_dog'] = history['dogs'] / self.parameters['N_dogs']
        history['prevalence_snails'] = history['snails'] / self.parameters['N_snails']
        history['prevalence_fish'] = history['fish'] / self.parameters['N_fish']
        
        history['humans_epg_none_prop'] = history['humans_epg_none']/history['eating_eligible']
        history['humans_epg_low_prop'] = history['humans_epg_low']/history['eating_eligible']
        history['humans_epg_medium_prop'] = history['humans_epg_medium']/history['eating_eligible']
        history['humans_epg_high_prop'] = history['humans_epg_high']/history['eating_eligible']
		
        history['k'] = history.apply(lambda row: calculate_k(row['worms_per_human'], row['prevalence_humans']), axis=1)

        # Store processed results
        self.history, self.humans = history, humans

        return (history, humans)
    
    def dask_submit(self, client):
        self.future = client.submit(self.run_core)
        if hasattr(self, "name"):
            self.future.model_name = self.name
            
        return self.future
    
    def dask_postprocess(self):
        self.postprocess_results(*self.future.result())

    def plot_timeseries_to_axes(self, axs):
        self.history[['worms_per_human', 'worms_per_cat', 'worms_per_dog']].plot(ax=axs[0])
        self.history[['prevalence_fish']].plot(ax=axs[1])
        self.history[['prevalence_snails']].plot(ax=axs[2])
        return

    def plot_timeseries_to_separate_axes(self, axs, set_title=True, zero_ylim=True, lb=None, ub=None, hlines=None):
        lb = 0 if lb == None else lb
        ub = self.history.t.iloc[-1] if ub == None else ub
        data = self.history[(self.history.t>=lb) & (self.history.t<=ub)].copy()
        data.index.names = ['years']
        data.prevalence_humans_eligible.plot(ax=axs[0])
        data.worms_per_human_eligible.plot(ax=axs[1])
        data.worms_per_cat.plot(ax=axs[2])
        data.worms_per_dog.plot(ax=axs[3])
        data.prevalence_fish.plot(ax=axs[4])
        data.prevalence_snails.plot(ax=axs[5])
        if hlines is not None:
            kws = {'c': 'r', 'ls': '--', 'alpha': .5}
            axs[0].axhline(hlines['prevalence_humans_eligible'], **kws)
            axs[1].axhline(hlines['worms_per_human_eligible'], **kws)
            axs[2].axhline(hlines['worms_per_cat'], **kws)
            axs[3].axhline(hlines['worms_per_dog'], **kws)
            axs[4].axhline(hlines['prevalence_fish'], **kws)
            axs[5].axhline(hlines['prevalence_snails'], **kws)

        if set_title:
            titles = ['prev humans', 'worms per h', 'worms per cat', 'worms per dog', 'prev fish', 'prev snails']
            for ax, title in zip(axs, titles):
                if(zero_ylim):
                    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
                ax.set_title(title)
        return

    def plot_worm_distribution(self, log = False, log_scale=True, **kwargs):
        worms = self.humans['worms'][self.humans['worms']>0]
        sns.displot(worms, kde = False, log = log, log_scale = log_scale, **kwargs)
        
    def plot_epg_distribution(self, log_scale = True, log = False):
            
        epg = self.parameters['worms_to_epg_transformation'](
            self.humans['worms'][self.humans['age']>=self.parameters['minimum_age_for_worm_infection']])
        epg = epg[epg>0]#.rename('epg')
        sns.displot(epg, log_scale = log_scale, log = log, kde = False)
        
    def plot_epg_distribution_data(self, log = True):
        # Helper function from web for legend placement in histplot
        def move_legend(ax, new_loc, **kws):
            old_legend = ax.legend_
            handles = old_legend.legendHandles
            labels = [t.get_text() for t in old_legend.get_texts()]
            title = old_legend.get_title().get_text()
            ax.legend(handles, labels, loc=new_loc, title=title, **kws)

        # Create dataframe with data and model output
        
        epg_data = pd.read_csv(os.path.join(os.path.dirname(__file__), "parameters", "IDRC.csv"))[['ovepg']].copy()
        epg_data.rename({'ovepg': 'epg'}, inplace=True, axis=1)
        epg_data_pos = epg_data[epg_data.epg>0].copy()
        epg_data_pos['source'] = 'data'
        epg_model_pos = self.humans[self.humans.epg>0][['epg']].copy()
        epg_model_pos['source'] = 'model'

        # Draw as many observations from model output as are in data until you like the plot KEKW
        idx_rnd = np.random.choice(len(epg_model_pos), size=len(epg_data_pos))
        epg_model_pos_rnd = epg_model_pos.iloc[idx_rnd]
        df = pd.concat([epg_data_pos, epg_model_pos_rnd]).melt('source', value_name='Eggs per gram')

        if log:
            plt.axvline(x=1000, c='r')
            plt.axvline(x=10000, c='r')
            ax = sns.histplot(data=df, x='Eggs per gram',
                              hue='source', log_scale=True,
                              bins=25, multiple="dodge", kde=True)
            ylim = ax.get_ylim()[1]
            ax.set_ylim(0, ylim + 10)
            plt.text(10**2, ax.get_ylim()[1]-10, 'light infection', c='r', ha='center')
            plt.text(10**3.5, ax.get_ylim()[1]-10, 'moderate', c='r', ha='center')
            plt.text(10**4.5, ax.get_ylim()[1]-10, 'heavy', c='r', ha='center')
            move_legend(ax, "upper left", bbox_to_anchor=(1.01, 1))
        else:
            plt.axvline(x=1000, c='r')
            plt.axvline(x=10000, c='r')
            ax = sns.histplot(data=df, x='Eggs per gram', hue='source', multiple="dodge", kde=True)
            ylim = ax.get_ylim()[1]
            ax.set_ylim(0, ylim + 10)
            plt.text(-1100, ax.get_ylim()[1]-200, 'light infection', c='r', ha='center', va='bottom', rotation = 90)
            plt.text(10**3.7, ax.get_ylim()[1]-150, 'moderate', c='r', ha='center', rotation=45)
            plt.text(10**4.5, ax.get_ylim()[1]-25, 'heavy infection', c='r', ha='center')
            move_legend(ax, "upper left", bbox_to_anchor=(1.01, 1))
        
    def plot_beta_distribution(self, log_scale = True, log = False):
        sns.histplot(self.humans.beta_hf[self.humans.beta_hf>0], log_scale = log_scale)

    def plot_histogram_to_axes(self, y, title = None, log_x = False, log_y = False):
        
        sns.displot(y, log_scale = log_x, log = log_y, kde=False)
        return
        ax = plt.gca()
        y_max = ax.get_ylim()[1]
        N_h = y.shape[0]
        y_ticks = [N_h * i/100 for i in range(0,101,2)]
        y_tick_labels = [f'{i}%' for i in range(0,101,2)]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)
        ax.set_ylim(0, y_max)
        ax.set_xlabel('')
        if title is not None:
            ax.set_title(title)

    def plot(self):
        fig, axs = plt.subplots(1, 3, figsize=(12, 3))
        self.plot_timeseries_to_axes(axs)
        return
        self.result[['worms_per_human', 'worms_per_cat', 'worms_per_dog']].plot()
        #self.history, self.humans =

    def plot_separate(self, **kwargs):
        fig, axs = plt.subplots(3, 2, figsize=(8, 9))
        self.plot_timeseries_to_separate_axes(axs.flatten(), **kwargs)
        plt.tight_layout()
        return
    
    def plot_humans(self):
        fig, axs = plt.subplots(1, 2, figsize=(8, 3))
        self.history['humans_variance'].plot(ax=axs[0], legend=True)
        self.history['humans_maximum'].plot(ax=axs[1], legend=True)
        
    def plot_humans_epg_variance(self):
        plt.plot(self.history['humans_epg_variance'])
        
    def plot_humans_epg(self, set_title=True, **kwargs):
        
        fig, axs = plt.subplots(3, 3, figsize=(12, 9))
        axs = axs.flatten()
        
        self.plot_humans_epg_to_axes(axs, **kwargs)
        if set_title:
            titles = ['zero epg', 'max epg', 'var worms',
                      'prev light infection', 'prev moderate infection', 'prev heavy infection',
                     'light epg mean', 'moderate epg mean', 'heavy epg mean']
            for ax, title in zip(axs, titles):
                ax.set_title(title)
        plt.tight_layout()

    def plot_humans_epg_to_axes(self, axs, lb = 0, ub = None, hlines=None):
		
        ub = self.history.t.iloc[-1] if ub == None else ub
        data = self.history[(self.history.t>=lb) & (self.history.t<=ub)].copy()
        data.index.names = ['years']
        data.humans_epg_none_prop.plot(ax=axs[0])
        data.humans_epg_maximum.plot(ax=axs[1])
        data.humans_eligible_variance.plot(ax=axs[2])
        data.humans_epg_low_prop.plot(ax=axs[3])
        data.humans_epg_medium_prop.plot(ax=axs[4])
        data.humans_epg_high_prop.plot(ax=axs[5])
        data.humans_epg_low_mean.plot(ax=axs[6])
        data.humans_epg_medium_mean.plot(ax=axs[7])
        data.humans_epg_high_mean.plot(ax=axs[8])

        for ax in axs:
            ylim = ax.get_ylim()
            ax.set_ylim(0, ylim[1]*1.1)
        
        if hlines is not None:
            kws = {'c': 'r', 'ls': '--', 'alpha': .5}
            axs[2].axhline(hlines['humans_eligible_variance'], **kws)
            axs[3].axhline(hlines['humans_epg_low_prop'], **kws)
            axs[4].axhline(hlines['humans_epg_medium_prop'], **kws)
            axs[5].axhline(hlines['humans_epg_high_prop'], **kws)
            axs[6].axhline(hlines['humans_epg_low_mean'], **kws)
            axs[7].axhline(hlines['humans_epg_medium_mean'], **kws)
            axs[8].axhline(hlines['humans_epg_high_mean'], **kws)
