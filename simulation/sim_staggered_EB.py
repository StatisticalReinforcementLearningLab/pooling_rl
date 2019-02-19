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




def initialize_policy_params_TS(experiment):
    
    global_p =gtp.TS_global_params(10)
    personal_p = pp.TS_personal_params()
    #global_p =gtp.TS_global_params(10,context_dimension)
    
    global_p.kdim = 11
    global_p.baseline_indices = [0,1,2,3,4,5,6]
    global_p.psi_indices = [4,6]
    global_p.user_id_index = 7
    global_p.user_day_index =8
    
    global_p.baseline_features = ['location','weather']
    global_p.psi_features = ['location']
    
    #print(type(personal_p))
    
    for person in experiment.population.keys():
        experiment.population[person].root = '../../regal/murphy_lab/pooling/distributions/'
        initial_context = [0 for i in range(global_p.theta_dim)]
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
    
    global_p.write_directory = '../../regal/murphy_lab/pooling/temp_EB'
    return global_p ,personal_p

def new_kind_of_simulation(experiment,policy=None,personal_policy_params=None,global_policy_params=None):
    #write_directory = '../../murphy_lab/lab/pooling/temp'
    experiment.last_update_day=experiment.study_days[0]
 
    for time in experiment.study_days:
        
        #if time> experiment.study_days[0]:
        #history  = pb.make_history(experiment)
        if time==experiment.last_update_day+pd.DateOffset(days=1):
            experiment.last_update_day=time
            if global_policy_params.decision_times>10:
                    
                    history =pb.make_history_new(uniform(),glob)
                    temp_params = TS_fancy_pooled.global_updates(history[0],history[1],global_policy_params,train_type = 'empirical_bayes')
                    global_policy_params.update_params(temp_params)
                    global_policy_params.history = history
                
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
                    
                    if time.hour==0 and time.minute==0:
                        participant.current_day_counter=participant.current_day_counter+1
                    
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
                if time in participant.decision_times:
                                        #print(personal_policy_params.batch_index[participant.pid])
                    
                    
                    ##if we have made no global updates

                            
                    
                        
                    
                    
                   
                    
                    
                    
                    dt=True
                    action=0
                    
                    
                    if global_policy_params.decision_times>10 and global_policy_params.history!=None:
                            if   not global_policy_params.updated_cov:
                                 global_policy_params.update_cov(global_policy_params.decision_times)   
                            #print( global_policy_params.decision_times)
                            history = global_policy_params.history                    ##update my mu2 and sigma2
                            temp = pb.calculate_posterior(global_policy_params,\
                                                  participant.pid,participant.current_day_counter,\
                                                  history[0], history[1] )
                    else:
                        temp = [personal_policy_params.mus2[participant.pid],personal_policy_params.sigmas2[participant.pid]]
                    mu_beta = temp[0]
                    Sigma_beta = temp[1]
                    personal_policy_params.update_mus(participant.pid,mu_beta,2)
                    personal_policy_params.update_sigmas(participant.pid,Sigma_beta,2)    
                    
                    
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
                    

                    
                    if availability:
                    

                   
                    
                    
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
                    context_dict =  {'steps':steps,'action':action,'weather':weather,'location':location,\
                                'ltps':steps_last_time_period,'duration':participant.duration,\
                                'study_day':participant.current_day_counter,'decision_time':dt,'time':time}
                
                    my_directory = '{}/participant_{}'.format(global_policy_params.write_directory,participant.pid)
                    if not os.path.exists(my_directory):
                        os.makedirs(my_directory)
                    with open('{}/day_{}'.format(my_directory,global_policy_params.decision_times),'wb') as f:
                        pickle.dump(context_dict,f)
                        
                        
                    global_policy_params.decision_times =   global_policy_params.decision_times+1
                
                    
                    
                else:
                        participant.steps_last_time_period = participant.steps
                        steps = sf.get_steps_no_action(participant.gid,tod,dow,location,weather,participant.steps)
                        participant.steps = steps     
                
                ##history:

                #participant.history[time]=context_dict
                
            #3
            ##for every active person generate a step count given current context
            
            
            
            
            ##update at midnight (here we have ensured that no one has a ) experiment.update_hour
            
                             

if __name__=="__main__":



    experiment = study.study( '../../regal/murphy_lab/pooling/distributions/')
    glob,personal = initialize_policy_params_TS(experiment)
    new_kind_of_simulation(experiment,'TS_fancy',personal,glob)
    print('finished')    

