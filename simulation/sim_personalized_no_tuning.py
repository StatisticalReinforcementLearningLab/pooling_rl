import pandas as pd
import sys
sys.path
sys.path.append('../../../home00/stomkins/pooling_rl/models')
sys.path.append('../../../home00/stomkins/pooling_rl/simulation')
#sys.path.append('../pooling_rl/simulation')
import numpy as np
import pickle
import random
import os
import math
import sim_functions_cleaner  as sf
import operator
import study
import time as time_module
import run_gpy
import TS_personal_params_pooled as pp
import TS_global_params_pooled as gtp
from numpy.random import uniform
import TS_fancy_pooled
import TS
##import TS_fancy_pooled

import pooling_bandits as pb
from sklearn import preprocessing
import tensorflow as tf
import feature_transformations
import simple_bandits

def initialize_policy_params_TS(experiment,update_period,standardize=False):
    #,'location_1','location_2','location_3'
    #'continuous_temp',
    global_p =gtp.TS_global_params(21,baseline_keys=['tod','dow','pretreatment','location'],psi_features=[0,5], responsivity_keys= ['tod','dow','pretreatment','location'])
    personal_p = pp.TS_personal_params()
    #global_p =gtp.TS_global_params(10,context_dimension)
    
    
    
    #global_p.mu_dimension = 64
    
    global_p.kdim =24
    #194
    global_p.baseline_indices = [i for i in range(24)]
    #[i for i in range(192)]
    #[0,1,2,3,4,5,6]
    global_p.psi_indices =[0,5]
    #[0,64]
    global_p.user_id_index =21
    
    global_p.psi_features =[0,5]
    #[0,64]
    
    global_p.update_period = update_period
    
    global_p.standardize = standardize
    #print(type(personal_p))
    
    for person in experiment.population.keys():
        experiment.population[person].root = '../../regal/murphy_lab/pooling/distributions/'
        initial_context = [0 for i in range(global_p.theta_dim)]
        personal_p.mus0[person]= global_p.get_mu0(initial_context)
        personal_p.mus1[person]= global_p.get_mu1(global_p.num_baseline_features)
        personal_p.mus2[person]= global_p.get_mu2(global_p.num_responsivity_features)
        
        personal_p.sigmas0[person]= global_p.get_asigma(len( personal_p.mus0[person]))
        personal_p.sigmas1[person]= global_p.get_asigma(global_p.num_baseline_features+1)
        personal_p.sigmas2[person]= global_p.get_asigma( global_p.num_responsivity_features+1)
        
        
        
        personal_p.batch[person]=[[] for i in range(len(experiment.person_to_time[person]))]
        personal_p.batch_index[person]=0
        
        #personal_p.etas[person]=eta.eta()
        
        personal_p.last_update[person]=experiment.person_to_time[person][0]


    return global_p ,personal_p

def get_optimal_reward(beta,states):
    return np.dot(beta,states)


def new_kind_of_simulation(experiment,policy=None,personal_policy_params=None,global_policy_params=None,sim=None,which_gen=None,tf = None):
    #write_directory = '../../murphy_lab/lab/pooling/temp'
    experiment.last_update_day=experiment.study_days[0]
    for time in experiment.study_days:
        
        #if time> experiment.study_days[0]:
        #history  = pb.make_history(experiment)
        if time==experiment.last_update_day+pd.DateOffset(days=global_policy_params.update_period):
            experiment.last_update_day=time
            #print('Global update', time,global_policy_params.decision_times, file=open('updates_{}_{}.txt'.format(len(experiment.population),global_policy_params.update_period), 'a'))
            if global_policy_params.decision_times>2:
                global_policy_params.last_global_update_time=time
    
        tod = sf.get_time_of_day(time)
        dow = sf.get_day_of_week(time)
        #if time==experiment.study_days[0]:
        # print('init weather')
        #   weather = tf.get_weather_prior(tod,time.month,seed=experiment.rando_gen)
        #  temperature = tf.continuous_temperature(weather)
        #elif time.hour in experiment.weather_update_hours and time.minute==0:
        #weather = tf.get_next_weather(str(tod),str(time.month),weather,seed=experiment.rando_gen)
        #temperature = tf.continuous_temperature(weather)
        ##location depends on person
        
        for person in experiment.dates_to_people[time]:
            dt=False
                action = 0
                prob=0
                #1
                ##for every active person update person specific aspects of their context
                participant = experiment.population[person]
                participant.set_tod(tod)
                participant.set_dow(dow)
                
                if time==participant.last_update_day+pd.DateOffset(days=global_policy_params.update_period):
                    
                    history = participant.history
                    
                    #print(participant.pid)
                    #print('updated')
                    #print(len(history))
                    #print(participant.last_update_day)
                    #print(history)
                    #return {participant.pid:history}
                    temp_hist = tf.get_history_decision_time_avail_single({participant.pid:history},time)
                    
                    temp_hist= tf.history_semi_continuous(temp_hist,global_policy_params)
                    #sf.get_data_for_txt_effect_u
                    
                    context,steps,probs,actions= tf.get_form_TS(temp_hist)
                    #sf.get_data_for_txt_effect_update(history,global_policy_params)
                    stepsmean = steps.mean()
                    global_policy_params.mu_theta[0]=stepsmean
                    #phi = get_phi(context,probs,actions,[i for i in range(len(context[0]))],[i for i in range(len(context[0]))])
                    
                    temp = TS.policy_update_ts_new( context,steps,probs,actions,global_policy_params.sigma,\
                                                   personal_policy_params.mus1[participant.pid],\
                                                   personal_policy_params.sigmas1[participant.pid],\
                                                   personal_policy_params.mus2[participant.pid],\
                                                   personal_policy_params.sigmas2[participant.pid],
                                                   
                                                   )
                    mu_beta = temp[0]
                    Sigma_beta = temp[1]
                    personal_policy_params.update_mus(participant.pid,mu_beta,2)
                    personal_policy_params.update_sigmas(participant.pid,Sigma_beta,2)
                    participant.last_update_day=time
                #history = None
                
                #update global context variables
                
                #participant.set_wea(weather)
                
                #random.seed(participant.pid)
                availability = (participant.rando_gen.uniform() < 0.8)
                participant.set_available(availability)
                
                if time == participant.times[0]:
                    #get first location
                    location = tf.get_location_prior(str(participant.gid),str(tod),str(dow),seed=participant.rando_gen)
                    participant.set_inaction_duration(0)
                    participant.set_action_duration(0)
                
                
                
                if time <= participant.times[0]:
                    steps_last_time_period = 0
        
                ##set first pre-treatment, yesterday step count, variation and dosage
                else:
                    
                    if time.hour==0 and time.minute==0:
                        participant.current_day_counter=participant.current_day_counter+1
                    
                    #print(time)
                    steps_last_time_period = participant.steps
                
                

                if time.hour in experiment.location_update_hours and time.minute==0:
                    location = tf.get_next_location(participant.gid,tod,dow,participant.get_loc(),seed=participant.rando_gen)
                
                    
                    
                    participant.set_loc(location)
                    
                    
                    prob = -1
                    add=None
                    optimal_action = -1
                    optimal_reward = -100
                    if time in participant.decision_times:
                    #print(personal_policy_params.batch_index[participant.pid])
                    
                    
                    ##if we have made no global updates
                    

                    
                    
                        dt=True
                        action=0
                    
                    
                    
                    
                        if policy==None:
                            action = sf.get_action(policy)
                    
                    

                        elif policy=='TS':
                        
                        #,int(location==1),int(location==2),int(location==3)
                        #temperature,
                        #sf.get_pretreatment()
                            z=np.array([1,tod,dow,sf.get_pretreatment(participant.steps),location])
                        
                        
                            prob = TS.prob_cal_ts(z,0,personal_policy_params.mus2[participant.pid],personal_policy_params.sigmas2[participant.pid],global_policy_params,seed=experiment.algo_rando_gen)
                        
                        #print('prob {}'.format(prob))
                        #random.seed(participant.pid)
                            action = int(experiment.algo_rando_gen.uniform() <prob)
                        #int(experiment.rando_gen.uniform() < prob)
                        
                        
                        
                        if availability:
                            
                            
                            
                            
                            
                            
                            
                            context = [action,participant.gid,tod,dow,sf.get_pretreatment(steps_last_time_period),location,\
                                       0,0,0]
                                
                                
                            steps = tf.get_steps_action(context,seed=participant.rando_gen)
                                       
                                       
                            add = action*sf.get_add_no_action(z,experiment.beta,participant.Z)
                                       
                                       
                            participant.steps =  steps+add
                                       
                            optimal_reward = get_optimal_reward(experiment.beta,z)
                            optimal_action = int(optimal_reward>=0)

                        else:
                        
                            steps = tf.get_steps_no_action(participant.gid,tod,dow,location,sf.get_pretreatment(steps_last_time_period),seed=participant.rando_gen)
                            participant.steps = steps
                    
                    
                    
                        global_policy_params.decision_times =   global_policy_params.decision_times+1
                                               
                                               
                                               
                    else:
                                                   #participant.steps_last_time_period = participant.steps
                        steps = tf.get_steps_no_action(participant.gid,tod,dow,location,sf.get_pretreatment(steps_last_time_period),seed=participant.rando_gen)
                        participant.steps = steps
                                                       
                                                       ##history:
                                                       
                                                       ##history:
                        context_dict =  {'steps':participant.steps,'add':add,'action':action,'weather':None,'location':location,'location_1':int(location==1),\
                                                           'ltps':steps_last_time_period,'location_2':int(location==2),'location_3':int(location==3),\
                                                               'study_day':participant.current_day_counter,\
                                                                   'temperature':None,'decision_time':dt,\
                                                                       'time':time,'avail':availability,'prob':prob,\
                                                                           'dow':dow,'tod':tod,\
                                                                               'pretreatment':sf.get_pretreatment(steps_last_time_period),\
                                                                                   'optimal_reward':optimal_reward,'optimal_action':optimal_action}
                        participant.history[time]=context_dict
                        if participant.pid==0 and time in participant.decision_times:
                            print(participant.rando_gen.rand())
                            print(context_dict)


def make_to_save(exp):
    to_save  = {}
    for pid,pdata in exp.population.items():
        for time,context in pdata.history.items():
            key = '{}-{}-{}'.format(pid,time,pdata.gid)
            to_save[key]=context
    return to_save


def get_regret(experiment):
    optimal_actions ={}
    rewards = {}
    
    for pid,person in experiment.population.items():
        for time,data in person.history.items():
            if data['decision_time'] and data['avail']:
                key = time
                if key not in optimal_actions:
                    optimal_actions[key]=[]
                if key not in rewards:
                    rewards[key]=[]
                if data['optimal_action']!=-1:
                    optimal_actions[key].append(int(data['action']==data['optimal_action']))
                    regret = int(data['action']!=data['optimal_action'])*(abs(data['optimal_reward']))
                    rewards[key].append(regret)
    return optimal_actions,rewards


if __name__=="__main__":
    
    ##parse command line arguments
    
    population = sys.argv[1]
    update_time = sys.argv[2]
    study_length = sys.argv[3]
    start_index = sys.argv[4]
    end_index = sys.argv[5]
    case =sys.argv[6]

    for i in range(int(start_index),int(end_index)):
            #for case in ['case_one','case_two','case_three']:
            root = 'pooling/distributions/'
            pop_size=population
            experiment = study.study('pooling/distributions/',pop_size,'short',which_gen=case,sim_number=int(start_index))
            glob,personal = initialize_policy_params_TS(experiment,int(update_time),standardize=False,root=root)
            ft = feature_transformations.feature_transformation('pooling/distributions/')
            
            hist = new_kind_of_simulation(experiment,'TS',personal,glob,i,case,ft)
            
            to_save = make_to_save(experiment)
            gids = make_to_groupids(experiment)
            actions,rewards = get_regret(experiment)

            filename = '{}/results/population_size_personalized_no_tuning_weighted_poolednewbigtest_{}_update_days_{}_{}_p_{}_{}_new_params_six_weeks.pkl'.format('pooling',pop_size,update_time,study_length,case,i)
#'likelis':glob.to_save_params,
            with open(filename,'wb') as f:
                pickle.dump({'gids':gids,'regrets':rewards,'actions':actions,'history':to_save},f)




#if global_policy_params.decision_times%100==0:
# my_directory = '{}/pop_size_{}_update_{}_study_length_{}/participant_{}'.format(global_policy_params.write_directory,participant.pid,experiment.study_length,len(experiment.population),global_policy_params.update_period)
#if not os.path.exists(my_directory):
#   os.makedirs(my_directory)
#with open('{}/history_{}.pkl'.format(my_directory,global_policy_params.decision_times),'wb') as f:
#   pickle.dump(participant.history,f)



