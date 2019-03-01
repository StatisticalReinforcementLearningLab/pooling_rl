import pandas as pd
import numpy as np
import pickle

import os
import math

import random
from datetime import datetime
random.seed(datetime.now())
from sklearn import preprocessing

class feature_transformation:


        def __init__(self):


            with open('{}label_to_temperature_value_stats.pkl'.format(root),'rb') as f:
                self.weather_label_to_val = pickle.load(f)

            with open('{}loc_label_to_intervention_label_tref.pkl'.format(root),'rb') as f:
                self.loc_label_to_val = pickle.load(f)


            with open('{}dists_base_olog.pkl'.format(root),'rb') as f:
                dists = pickle.load(f)

            with open('{}key_matches_base_olog.pkl'.format(root),'rb') as f:
                matched = pickle.load(f)


            with open('{}dists_intervention_anti_sedentary_olog.pkl'.format(root),'rb') as f:
                dists_intervention_anti_sedentary = pickle.load(f)

            with open('{}key_matches_intervention_anti_sedentary_olog.pkl'.format(root),'rb') as f:
                matched_intervention_anti_sedentary = pickle.load(f)


            with open('{}dists_intervention_activity_suggestion_olog.pkl'.format(root),'rb') as f:
                dists_intervention_activity_suggestion = pickle.load(f)

            with open('{}key_matches_intervention_activity_suggestion_olog.pkl'.format(root),'rb') as f:
                matched_intervention_activity_suggestion = pickle.load(f)


            with open('{}initial_location_distributions_est_tref_3_1.pkl'.format(root),'rb') as f:
                loc_lookup_prior = pickle.load(f)

            with open('{}location_conditiononed_on_last_location_tref_3_1.pkl'.format(root),'rb') as f:
                loc_lookup_transition = pickle.load(f)

            with open('{}initial_temperature_distributions_est_tref_3_1.pkl'.format(root),'rb') as f:
                temperature_lookup_prior = pickle.load(f)

            with open('{}temperature_conditiononed_on_last_temperature_tref_3_1.pkl'.format(root),'rb') as f:
                temperature_lookup_transition = pickle.load(f)

                    
        ##difficult to write this generally
        ##this should not call standardize
        def history_to_matrix_for_gp(self,which_history = None):
            pass

        
        def get_history_decision_time_avail(self,exp,glob):
            to_return = {}
                            
            for userid,data in exp.population.items():
                to_return[userid]= {k:v for k,v in data.history.items() if k<glob.last_global_update_time and v['avail'] and v['decision_time']}
            
            return to_return
        
        ##this method assumes that the history is stored as a dictionary with keys,
        ##where the keys are states containing previous and weather
        def history_semi_continuous(self,history_dict,glob):
            probs ={}
            actions = {}
            users = {}
            baseline_data = {}
            responsivity_data = {}

            history_count = 0
            for user_id,history in history_dict.items():

                for hk,h in history.items():
                    base_line = [h[b] for b in glob.baseline_keys]
                    responsivity =[h[b] for b in glob.responsivity_keys]
                    ##transform weather before this
                    baseline_data[history_count]=base_line
                    responsivity[history_count]=responsivity
                    probs[history_count]=h['prob']
                    actions[history_count]=h['action']
                    users[history_count]=user_id
                    history_count = history_count+1
            if glob.standardize:
                baseline_data = self.standardize(baseline_data)
            return {'base':baseline_data,'resp':responsivity_data,'probs':probs,'actions':actions,'users':users}

        def standardize(self,data):
            
            key_lookup = [i for i in sorted(data.keys())]
            X = [data[i] for i in key_lookup]
            
            new_x = preprocessing.scale(np.array(X))
            return {i:new_x[i] for i in key_lookup}
