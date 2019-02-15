import pandas as pd
import sys
sys.path
sys.path.append('pooling_rl/models')
import numpy as np
import pickle
import random
import os
import math
import sim_functions_cleaner  as sf
import operator
import study
import time
import TS_personal_params_pooled as pp
import TS_global_params_pooled as gtp
from numpy.random import uniform

#sys.path.append('../simulation')
import TS_fancy_pooled
#import TS_fancy_pooled
import eta
import pooling_bandits as pb
from sklearn import preprocessing
import tensorflow as tf




def initialize_policy_params_TS(experiment,context_dimension):
    
    global_p =gtp.TS_global_params(10,context_dimension)
    personal_p = pp.TS_personal_params()
    global_p =gtp.TS_global_params(10,context_dimension)
    
    global_p.kdim = 11
    global_p.baseline_indices = [0,1,2,3,4,5,6,7,8]
    global_p.psi_indices = [7,8]
    global_p.user_id_index = 9
    global_p.user_day_index =10
    #print(type(personal_p))
    
    for person in experiment.population.keys():
        initial_context = [0 for i in range(context_dimension)]
        personal_p.mus0[person]= global_p.get_mu0(initial_context)
        personal_p.mus1[person]= global_p.get_mu1(initial_context)
        personal_p.mus2[person]= global_p.get_mu2(initial_context)
        
        personal_p.sigmas0[person]= global_p.get_asigma(len( personal_p.mus0[person]))
        personal_p.sigmas1[person]= global_p.get_asigma(len( personal_p.mus1[person]))
        personal_p.sigmas2[person]= global_p.get_asigma(len( personal_p.mus2[person]))
        
        
        
        personal_p.batch[person]=[[] for i in range(len(experiment.person_to_time[person]))]
        personal_p.batch_index[person]=0
        
        personal_p.etas[person]=eta.eta()
        
        personal_p.last_update[person]=experiment.person_to_time[person][0]
    
    
    return global_p ,personal_p


def get_history(write_dir,dt):
    to_return = {}
    for d in [f for f in os.listdir(write_dir) if f!='.DS_Store']:
        participant = {}
        for f in os.listdir('{}/{}'.format(write_dir,d)):
            if f!='.DS_Store':
                time = int(f.split('_')[1])
                if time <=dt:
                    with open('{}/{}/{}'.format(write_dir,d,f),'rb') as f:
                        ld = pickle.load(f)
                    participant[time]=ld
                    
        pid = d.split('_')[1]
        if len(participant)>0:
            to_return[int(pid)]=participant
    return to_return
def create_phi_new(history_dict,pi,global_params):
    #these things will be accessed by the global params
    indices = ['weather','location']
    g0 = ['location']
    f1=['ltps']
    
    ##returns phi and psi indices
    ##this could be a bit faster not appending all the time
    all_data = []
    steps=[]
    for user_id,history in history_dict.items():
        #history = d.history
        #history_keys = sorted(history)
        for hk,h in history.items():
            
            h = history[hk]
            if h['decision_time']:
                v = [1]
                v.extend([h[i] for i in indices])
                v.append(pi*1)
                v.extend([pi*h[i] for i in f1])
                action = h['action']
                if action<0:
                    action=0
                
                v.append((action-pi)*1)
                v.extend([(action-pi)*h[i] for i in f1])
                v.append(action)
                v.append(float(user_id))
                v.append(float(h['study_day']))
                all_data.append(v)
                steps.append(h['steps'])
    return all_data,steps
def make_history_new(write_directory,pi,glob):
    g = get_history(write_directory,glob.decision_times)
    ad = create_phi_new(g,pi,glob)
    if len(ad[0])==0:
        return [[],[]]
    
    new_x = preprocessing.scale(np.array(ad[0]))
    new_y = preprocessing.scale(np.array(ad[1]))
    y = np.array([[float(r)] for r in new_y])
    X = new_x
    return [X,y]


def new_kind_of_simulation(experiment,policy=None,personal_policy_params=None,global_policy_params=None):
    write_directory = '../../murphy_lab/lab/pooling/temp'

    for time in experiment.study_days:
           
            #if time> experiment.study_days[0]:
                #history  = pb.make_history(experiment)
            if time==experiment.last_update_day+pd.DateOffset(days=1):
                experiment.last_update_day=time
                ##global update
                #print(time)
                #print(experiment.last_update_day+pd.DateOffset(days=1))
                ##am i checking the current time (need to check the 
                #current time make sure i'm not using all of the history)
               
                #print(history)
                
                ##these lines
                history = make_history_new(write_directory,.6,global_policy_params)
                temp_params = TS_fancy_pooled.global_updates(history[0],history[1],global_policy_params,train_type = 'Static')
                #print(temp_params['cov'].shape)
                #global_policy_params.update_params(temp_params)
                #del temp_params
                ##update global params using these temp_params
                
                
                
            ##update global context
            ##global context shared across all participants
            tod = sf.get_time_of_day(time)
            dow = sf.get_day_of_week(time)
            if time==experiment.study_days[0]:
                
                weather = sf.get_weather_prior(tod,time.month)
            elif time.hour in experiment.weather_update_hours and time.minute==0:
                weather = sf.get_next_weather(str(tod),str(time.month),weather)
            ##location depends on person 
            
            for person in experiment.dates_to_people[time]:
                dt=False
                action = 0 
                prob=0
                #1
                ##for every active person update person specific aspects of their context
                participant = experiment.population[person]
                #update global context variables
                participant.set_tod(tod)
                participant.set_dow(dow)
                participant.set_wea(weather)
                
                
                availability = (uniform() < 0.8)
                participant.set_available(availability)
                
                if time == participant.times[0]:
                    #get first location 
                    location = sf.get_location_prior(str(participant.gid),str(dow),str(tod))
                    participant.set_inaction_duration(0)
                    participant.set_action_duration(0)
                    participant.set_duration(0)
                    participant.set_dosage(0)
                    #personal_policy_params.etas[participant.pid]
                    
                    
                if time <= participant.times[1]:
                    steps_last_time_period = 0  
                    
                    ##set first pre-treatment, yesterday step count, variation and dosage
                else:
                    #print(time)
                    steps_last_time_period = participant.steps
                
                 
                    #get var id
                    
                #if time.date() <= participant.times[0].date():
                    #steps_yesterday = 0    
                #else:
                    #steps_yesterday =  participant.find_yesterday_steps(time)
                    #steps_yesterday = sf.to_yid(steps_yesterday)
                steps_yesterday=0    
                if time.hour in experiment.location_update_hours:
                    location = sf.get_next_location(participant.gid,dow,tod,participant.get_loc())
                
                if time.date()>(participant.times[0]+pd.DateOffset(days=1)).date():
                  
                    if time.hour==0 and time.minute==0:
                        variation = 0
                        #participant.find_variation(time)
                else:
                    variation = 1 
                
                participant.set_loc(location)
                ##maybe faster to update instead of query?
                #participant.set_last_time_period_steps(steps_last_time_period)
                #participant.set_yesterday_steps(steps_yesterday)
                #participant.set_variation(variation)
                
                ##continue
                #2
                ##for every active person take an action according to current context, policy, and parameters
                
                
                ##for now:
                ##eval with empty array 
                if time in participant.decision_times and availability:
                    
                    dt=True
                    action=0
                    global_policy_params.decision_times =   global_policy_params.decision_times+1
                    if policy==None:
                        action = sf.get_action(policy)
                        
                        
                        
                    elif policy=='TS':
                        #some context slice
                            prob = TS.prob_cal_ts([int(tod),int(dow)],participant.dosage,\
                                              personal_policy_params.mus2[participant.pid],personal_policy_params.sigmas2[participant.pid],\
                                                 global_policy_params)
                            action = int(uniform() < prob)
                            
                        
                            
                    elif policy=='TS_fancy':
                        #previous
                        #Z, X, mu_beta, Sigma_beta, init,current_eta
                        
                        ##need to make eta part of the global policy params
                        prob = TS_fancy_pooled.prob_cal([location,weather,steps_last_time_period,variation,steps_yesterday],participant.dosage,\
                                              personal_policy_params.mus2[participant.pid],personal_policy_params.sigmas2[participant.pid],\
                                                 global_policy_params,personal_policy_params.etas[participant.pid])
                        action = int(uniform() < prob)
                        
                        
                        
                    ##is this the same as in the TS?
                    ##don't think so, but for now keep like this
                    ##no it isn't, i have to redo this
                    participant.update_dosage(action)
                    
                    context = [action,participant.gid,tod,dow,location,weather,sf.get_pretreatment(participant.steps),\
                              steps_yesterday,variation,sf.dosage_to_dosage_key(participant.dosage)]
                    
                    participant.steps_last_time_period = participant.steps
                    steps = sf.get_steps_action(context)
                    participant.steps = steps
                else:
                    participant.steps_last_time_period = participant.steps
                    steps = sf.get_steps_no_action(participant.gid,tod,dow,location,weather,participant.steps)
                    participant.steps = steps
                
                
                
                
                ##history:
                context_dict =  {'steps':steps,'action':action,'weather':weather,'location':location,\
                                'ltps':steps_last_time_period,'duration':participant.duration,\
                                'study_day':participant.current_day_counter,'decision_time':dt}
                #participant.history[time]=context_dict
                
            #3
            ##for every active person generate a step count given current context
            
            
            
            
            ##update at midnight (here we have ensured that no one has a ) experiment.update_hour
            
                if time in participant.decision_times and availability:
                    ##global_dt_counter
                    ##update the policy
                    #print(personal_policy_params.batch_index[participant.pid])
                   
                    my_directory = '{}/participant_{}'.format(write_directory,participant.pid)
                    if not os.path.exists(my_directory):
                        os.makedirs(my_directory)
                    with open('{}/day_{}'.format(my_directory,global_policy_params.decision_times),'wb') as f:
                        pickle.dump(context_dict,f)
                    participant.current_day_counter=participant.current_day_counter+1


if __name__=="__main__":
    experiment = study.study()
    glob,personal = initialize_policy_params_TS(experiment,11)
    new_kind_of_simulation(experiment,'TS_fancy',personal,glob)    

