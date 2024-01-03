import copy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from dask.distributed import as_completed

class Collection():
    
    def __init__(self):
        self.models = []
        pass
    
    def add_model(self, model):
        self.models += [model]
        
    def run_models(self):
        for model in self.models:
            model.run()
    
    def dask_submit_models(self, client, verbose = False, blocking = True):
        
        # Submit all the model runs to dask
        futures = [] 
        for model in self.models:
            futures += [model.dask_submit(client)]

        if blocking:
            # Wait until model runs completed
            for future in as_completed(futures):
                if verbose:
                    print(f'{future.model_name} is done!')
        
            # Postprocess all the model results from futures stored inside models
            self.dask_postprocess()
        else:
            self.futures = futures
            return self.futures

    def dask_postprocess(self):
        for model in self.models:
            model.dask_postprocess()

    def plot_models(self):
        for model in self.models:
            model.plot_separate()

    def plot_models_same_axes(self, yearly = False):
        
        plotting_vars = ['prevalence_humans', 'worms_per_positive_human', 'worms_per_dog',
                         'worms_per_cat', 'prevalence_fish',
                        'prevalence_snails', 'humans_epg_low_prop', 'humans_epg_medium_prop',
                        'humans_epg_high_prop']
        
        self._set_default_model_names() 
        df = self.create_full_result_df(yearly = yearly)
        df = df.loc[df['variable'].isin(plotting_vars)]
        
        g = sns.FacetGrid(df, col='variable', col_wrap = 3, hue='model_name', sharey=False,
                         col_order=plotting_vars)
        g.map(sns.lineplot, "t", "value")
        g.set(ylim=(0, None))
        g.add_legend()
        
            
    def plot_models_same_axes0(self):
        
        fig, axs = plt.subplots(2, 3, figsize=(12, 6))
        for model in self.models:
            model.plot_timeseries_to_separate_axes(axs.flatten())
    
    def plot_epg_same_axes(self):
        fig, axs = plt.subplots(2, 3, figsize=(12, 6))
        axs = axs.flatten()
        
        for model in self.models:
            model.plot_humans_epg_to_axes(axs.flatten())
            
    def _set_default_model_names(self):
        i = 0
        for model in self.models:
            if not hasattr(model, "model_name"):
                model.model_name = f"unnamed_{i}"
                i += 1
                
    def create_full_result_df(self, yearly = False):
        
        dfs = []
        for model in self.models:
            df = model.history.copy()
            if yearly:
                df = df[df.t % 365 == 0]
            df = pd.melt(df, id_vars = 't')
            df['model_name'] = model.model_name
            dfs += [df]
            
        df = pd.concat(dfs)
        df['t'] = df['t']/365
        return df