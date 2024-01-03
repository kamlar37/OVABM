import copy

import time
from types import SimpleNamespace
from contexttimer import Timer

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import scipy.integrate


class Ov:

    # Default parameters taken from intervention paper
    PARAMETERS_DEFAULT = {
        'n_h': 15705,
        'n_d': 8437,
        'n_c': 6098,
        'n_s': 31019,
        'n_f': 9701,

        'b_hf': 5.9785e-6,
        'b_df': 3.2337e-7,
        'b_cf': 2.9608e-6,
        'b_sh': 1.0210e-11,
        'b_sd': 2.8635e-11,
        'b_sc': 4.7734e-12,
        'b_fs': 1.2900e-5,

        'mu_ph': 1/4.8/365,
        'mu_pd': 1/2.2/365,
        'mu_pc': 1/1.5/365,
        'mu_s': 1/365,
        'mu_f': 1/1.5/365,

        'i_d': 0,
        'i_e': 0,

        'y0': [33, 3, 13, 0.003, 0.3],
    }

    CONFIG_DEFAULT = {
        'tf': 365 * 100,
        'integration_method': 'Radau',
        'upper_threshold': 1e4,
        'lower_threshold': 1e-6,
    }

    def __init__(self, parameters={}.copy(), config={}.copy(), timer=False):
        self.parameters = dict(self.PARAMETERS_DEFAULT, **parameters.copy())
        self.config = dict(self.CONFIG_DEFAULT, **config)
        self.result = None
        self.run(timer)

    def run(self, timer=False):
        c = SimpleNamespace(**self.config)
        p = SimpleNamespace(**self.parameters)

        with Timer(output=timer):
            self.resultobj = scipy.integrate.solve_ivp(
                lambda t,y: self._dydt(t, y, p),
                (0, c.tf), p.y0,
                method=c.integration_method
            )

        self.result = pd.DataFrame(
            self.resultobj.y.T,
            index=self.resultobj.t.T,
            columns=['h', 'd', 'c', 's', 'f']
        )

        return self.result.copy()

    def plot(self, **kwargs):
        self.result.plot(**kwargs)
        return

    def plot_timeseries_to_axes(self, ax1, ax2, ax3):
        self.result[['h', 'd', 'c']].plot(ax=ax1)
        self.result[['f']].plot(ax=ax2)
        self.result[['s']].plot(ax=ax3)
        return

    def _dydt(self, t, y, p):
            w_h, w_d, w_c, i_s, i_f = y

            dw_h = p.b_hf *  i_f * (1 - p.i_e) - p.mu_ph * w_h
            dw_d = p.b_df *  i_f - p.mu_pd * w_d
            dw_c = p.b_cf *  i_f - p.mu_pc * w_c
            di_s = (p.b_sh *  w_h * (1 - p.i_d)
                    + p.b_sd * w_d
                    + p.b_sc * w_c) * (1 - i_s) - p.mu_s * i_s
            di_f = p.b_fs * i_s * (1 - i_f) - p.mu_f * i_f

            return([dw_h, dw_d, dw_c, di_s, di_f])


class OvIntervention(Ov):

    def run(self, timer=False):
        parameters = self.parameters.copy()

        p = SimpleNamespace(**parameters)
        c = SimpleNamespace(**self.config)

        events = (
            [self._event_factory(c.upper_threshold, i) for i in range(5)]
            + [self._event_factory(c.lower_threshold, i) for i in range(5)]
        )

        try:
            parameters['schedule']['t_next'] = \
                parameters['schedule'].t.shift(-1, fill_value=c.tf)
        except KeyError:
            parameters['schedule'] = pd.DataFrame.from_dict({
                't': [0],
                't_next': [c.tf],
                'population': [np.array((1, 1, 1, 1, 1))],
                'parameters': [{},],})
        
        results_list_y = []
        results_list_t = []
        t0 = 0
        y0 = p.y0
        i = 0
        time0 = time.time()
        while i < len(parameters['schedule']):

            parameters.update(parameters['schedule'].iloc[i].parameters)
            p = SimpleNamespace(**parameters)

            t0 = p.schedule.iloc[i].t
            tf = p.schedule.iloc[i].t_next

            y0 = p.schedule.iloc[i].population * y0

            # not true if multiple schedule entries for one timepoint
            if tf>t0:
                result = scipy.integrate.solve_ivp(
                    lambda t,y: self._dydt(t, y, p),
                    (t0, tf), y0, method=c.integration_method, # rtol=1e-10,
                    events=events,
                    t_eval=np.arange(t0, tf+1, 1))

                results_list_y.append(result.y)
                results_list_t.append(result.t)

                y0 = result.y[:,-1]

            if result.status == 1: # means current integration stopped due to termination event
                break

            i = i+1

        if timer:
            print('took {:.3f} seconds'.format(time.time() - time0))

        self.result = pd.DataFrame(
            np.concatenate(results_list_y, axis=1).T,
            index=np.concatenate(results_list_t).T,
            columns=['worms_per_human', 'worms_per_dog', 'worms_per_cat', 'prevalence_snails', 'prevalence_fish']
        )
        
        self.result['t'] = np.concatenate(results_list_t).T/365

        self.result.event_termination = result.status == 1

        return copy.deepcopy(self.result)

        
    def _event_factory(self, threshold, i):
        def event_function(t, y):
            return y[i] - threshold
        event_function.terminal = True
        return event_function

def calculate_parameters(w_h, w_c, w_d, i_f, i_s, n_h, n_d, n_c, n_s, n_f, mu_ph, mu_pc, mu_pd, mu_s, mu_f, weights=[1, 1, 1], **kwargs):

    weights = [weight/weights[0] for weight in weights]

    b_hf = (mu_ph*w_h)/(i_f)

    b_df = (mu_pd*w_d)/(i_f)

    b_cf = (mu_pc*w_c)/(i_f)

    b_sh = -(i_s*mu_s)/((-1+i_s)*(w_d*weights[1] + w_h + w_c*weights[2]))

    b_sd = b_sh * weights[1]

    b_sc = b_sh * weights[2]

    b_fs = -(i_f*mu_f)/((-1+i_f)*i_s)

    lbda_s = (i_s*mu_s)/(1-i_s)

    return {'b_hf': b_hf, 'b_df': b_df, 'b_cf': b_cf,
            'b_sh': b_sh, 'b_sd': b_sd, 'b_sc': b_sc,
            'b_fs': b_fs,
            'lbda_s': lbda_s
    }
