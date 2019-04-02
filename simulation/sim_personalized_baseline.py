import pandas as pd
import sys
sys.path
sys.path.append('pooling_rl/models')
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
import run_gpy_simple
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
    
    global_p =gtp.TS_global_params(21,baseline_keys=['tod','dow','pretreatment','location'],psi_features=[0,5], responsivity_keys= ['tod','dow','pretreatment','location'])
    personal_p = pp.TS_personal_params()
    
    
    global_p.standardize = standardize
    global_p.kdim =21
    
    global_p.baseline_indices = [i for i in range(len(global_p.baseline_keys))]
    
    global_p.psi_indices =[0,5]
    #[0,64]
    
    #192
    #global_p.user_day_index =19
    #193
    
    #global_p.baseline_features = [i for i in range(192)]
    global_p.psi_features =[0,5]
    #[0,64]
    
    global_p.update_period = update_period
    initial_context = [0 for i in range(global_p.theta_dim)]
    
    global_p.mus0= global_p.get_mu0(initial_context)
    #global_p.get_mu0(initial_context)
    global_p.mus1= global_p.get_mu1(global_p.num_baseline_features)
    global_p.mus2= global_p.get_mu2(global_p.num_responsivity_features)
    #np.array([.120,3.3,-.11])
    #global_p.get_mu2(global_p.num_responsivity_features)
    
    #global_p.sigmas0= global_p.get_asigma(len( personal_p.mus0[person]))
    global_p.sigmas1= global_p.get_asigma(global_p.num_baseline_features+1)
    global_p.sigmas2= global_p.get_asigma( global_p.num_responsivity_features+1)
    
    
    
    #print(type(personal_p))
    
    for person in experiment.population.keys():
        experiment.population[person].root = '../../murphy_lab/lab/pooling/distributions/'
        
        
        
        personal_p.batch[person]=[[] for i in range(len(experiment.person_to_time[person]))]
        personal_p.batch_index[person]=0
        
        #personal_p.etas[person]=eta.eta()
        
        personal_p.last_update[person]=experiment.person_to_time[person][0]


    return global_p ,personal_p

def get_optimal_reward(beta,states):
    return np.dot(beta,states)

def new_kind_of_simulation(experiment,policy=None,personal_policy_params=None,global_policy_params=None,feat_trans=None):
    #write_directory = '../../murphy_lab/lab/pooling/temp'
    experiment.last_update_day=experiment.study_days[0]
    additives = []
    for time in experiment.study_days:
        
        #if time> experiment.study_days[0]:
        #history  = pb.make_history(experiment)
        #if global_policy_params.decision_times> 900:
        #break



    
    
    
        #del history
        ##update global context
        ##global context shared across all participants
        tod = sf.get_time_of_day(time)
        dow = sf.get_day_of_week(time)
            #if time==experiment.study_days[0]:
            #print('init weather')
            #weather = feat_trans.get_weather_prior(tod,time.month)
            #elif time.hour in experiment.weather_update_hours and time.minute==0:
            #weather = feat_trans.get_next_weather(str(tod),str(time.month),weather)
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
        
        
        
                    print('Global update', time,global_policy_params.decision_times,time_module.strftime('%l:%M%p %Z on %b %d, %Y'),file=open('outs/updates_personalized_baseline_EB_{}_{}_six_weeks_only_pplus.txt'.format(len(experiment.population),global_policy_params.update_period), 'a'))
                #update global context variables

                #participant.set_wea(weather)
                
                
                availability = (participant.rando_gen.uniform() < 0.8)
                participant.set_available(availability)
                
                if time == participant.times[0]:
                    #get first location
                    location = feat_trans.get_location_prior(str(participant.gid),str(tod),str(dow),seed = participant.rando_gen)
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
                
                
                    
                    if time.hour in experiment.location_update_hours:
                        location = feat_trans.get_next_location(participant.gid,tod,dow,participant.get_loc(),seed = participant.rando_gen)
                    
                    
                    
                    participant.set_loc(location)
                        
                        
                        
                prob = -1
                add=None
                optimal_action = -1
                optimal_reward = -100
                if time in participant.decision_times:
                    
                    
                    
                    dt=True
                    action=0
                    
                    if policy==None:
                        action = sf.get_action(policy)

                    elif policy=='TS':
                        
                        
                        if 'pretreatment' in global_policy_params.baseline_keys:
                            to_call = sf.get_pretreatment(steps_last_time_period)
                        else:
                            to_call = steps_last_time_period
                    
                        ##want this to be a function
                        z=np.array([1,tod,dow,to_call,location])
                        prob = TS.prob_cal_ts(z,0,global_policy_params.mus2,global_policy_params.sigmas2,\
                                              global_policy_params)
                                              #if participant.pid==1:

                                              #print('prob _ {}'.format(prob))
                                              #print(type(prob))
                        action = int(experiment.algo_rando_gen.uniform() < prob)

                    if availability:

                        context = [action,participant.gid,tod,dow,sf.get_pretreatment(steps_last_time_period),location,\
                                   0,0,0]
                            
                                   #participant.steps_last_time_period = participant.steps
                                   #print(sf.get_pretreatment(participant.steps))
                                   
                        steps = feat_trans.get_steps_action(context,seed = participant.rando_gen)
                                   
                                   #add = sf.get_add_two(action,z,experiment.beta,participant.Z)
                        add = action*sf.get_add_no_action(z,experiment.beta,participant.Z)
                        additives.append([action,add,prob])
                        participant.steps = steps+add
                                   
                                   ##calculate optimal
                        optimal_reward = get_optimal_reward(experiment.beta,z)
                        optimal_action = int(optimal_reward>=0)

                    else:
    #participant.steps_last_time_period = participant.steps
                        steps = feat_trans.get_steps_no_action(participant.gid,tod,dow,location,sf.get_pretreatment(steps_last_time_period),seed = participant.rando_gen)
                        participant.steps = steps
        
        
        
                    global_policy_params.decision_times =   global_policy_params.decision_times+1
            
            
            
                else:
                #participant.steps_last_time_period = participant.steps
                    steps = feat_trans.get_steps_no_action(participant.gid,tod,dow,location,sf.get_pretreatment(steps_last_time_period),seed = participant.rando_gen)
                    participant.steps = steps
                
                ##history:
                context_dict =  {'steps':participant.steps,'add':add,'action':action,'location':location,'location_1':int(location==1),\
                    'ltps':steps_last_time_period,'location_2':int(location==2),'location_3':int(location==3),\
                        'study_day':participant.current_day_counter,\
                            'decision_time':dt,\
                                'time':time,'avail':availability,'prob':prob,\
                                    'dow':dow,'tod':tod,\
                                        'pretreatment':sf.get_pretreatment(steps_last_time_period),\
                                    'optimal_reward':optimal_reward,'optimal_action':optimal_action}
                participant.history[time]=context_dict


    return additives

def make_to_save(exp):
    to_save  = {}
    for pid,pdata in exp.population.items():
        for time,context in pdata.history.items():
            key = '{}-{}'.format(pid,time)
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
                    regret = int(data['action']!=data['optimal_action'])*(abs(data['optimal_reward']-data['steps']))
                    rewards[key].append(regret)
    return optimal_actions,rewards


def make_to_groupids(exp):
    to_save  = {}
    for pid,pdata in exp.population.items():
        gid  = pdata.gid
        key = 'participant-{}'.format(pid)
        to_save[key]=gid
    return to_save        


if __name__=="__main__":
    
    ##parse command line arguments
    
    population = sys.argv[1]
    update_time = sys.argv[2]
    study_length = sys.argv[3]
    start_index = sys.argv[4]
    end_index = sys.argv[5]
    case = sys.argv[6]

    for i in range(int(start_index),int(end_index)):
            root = 'pooling/distributions/'
            pop_size=32
            experiment = study.study(root,pop_size,'short',which_gen=case,sim_number=int(start_index))
            feat_trans = feature_transformations.feature_transformation(root)
            glob,personal = initialize_policy_params_TS(experiment,int(update_time),standardize=False)


            hist = new_kind_of_simulation(experiment,'TS',personal,glob,feat_trans=feat_trans)
            to_save = make_to_save(experiment)
            gids = make_to_groupids(experiment)
            actions,rewards = get_regret(experiment)
    
    #filename = '{}/results/{}/population_size_{}_update_days_{}_{}_static_sim_{}.pkl'.format('../../Downloads/pooling_results/batch/',case,pop_size,7,'short',i)
    # with open(filename,'wb') as f:
    #     pickle.dump(to_Save,f)
    

        
        
            filename = '{}/results/population_size_personalized_baseline_EB_{}_update_days_{}_{}_batch_{}_{}_new_params_six_weeks_only_pplus.pkl'.format('pooling',pop_size,update_time,study_length,case,i)
            with open(filename,'wb') as f:
                pickle.dump({'history':to_save,'gids':gids,'likelis':glob.to_save_params,'regrets':rewards,'actions':actions},f)
