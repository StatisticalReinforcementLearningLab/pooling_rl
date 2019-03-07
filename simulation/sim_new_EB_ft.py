import pandas as pd
import sys
sys.path
sys.path.append('pooling_rl/models')
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

def initialize_policy_params_TS(experiment,update_period,root,standardize=False):
    
    global_p =gtp.TS_global_params(21,baseline_keys=['pretreatment','weather','dow','tod'],psi_features=[0,5], responsivity_keys= ['pretreatment','weather','dow','tod'])
    personal_p = pp.TS_personal_params()
    
    
    global_p.standardize = standardize
    #global_p.kdim =21
    
    #global_p.baseline_indices =  [i for i in range(self.theta_dim)]
    
    global_p.psi_indices =[0,5]
    #[0,64]
    global_p.user_id_index =None
    #192
    #global_p.user_day_index =19
    #193
    
    #global_p.baseline_features = [i for i in range(192)]
    global_p.psi_features =[0,5]
    
    global_p.update_period = update_period
    
    #print(type(personal_p))
    
    for person in experiment.population.keys():
        experiment.population[person].root = root
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

def new_kind_of_simulation(experiment,policy=None,personal_policy_params=None,global_policy_params=None,sim_num=None,case=None,feat_trans=None):
    #write_directory = '../../murphy_lab/lab/pooling/temp'
    experiment.last_update_day=experiment.study_days[0]
    for time in experiment.study_days:
        
       
        #history  = pb.make_history(experiment)
        if time==experiment.last_update_day+pd.DateOffset(days=global_policy_params.update_period):
            experiment.last_update_day=time
            print('Global update', time,global_policy_params.decision_times,time_module.strftime('%l:%M%p %Z on %b %d, %Y'),file=open('../../murphy_lab/lab/pooling/{}_two/updates_safer_three_{}_{}.txt'.format(case,len(experiment.population),global_policy_params.update_period), 'a'))
            if global_policy_params.decision_times>200:
                glob.last_global_update_time=time
                
                ##hows this happening now?
                temp =feat_trans.get_history_decision_time_avail(experiment,time)
                    #print(history[1])
                t = feat_trans.history_semi_continuous(temp,global_policy_params)
                history = feat_trans.get_phi_from_history_lookups(t)
            
                ##CHANGE THIS
                try:
                    temp_params = run_gpy.run(history[0],history[1],history[2],global_policy_params)
                
                except Exception as e:
                    print(e)
                    print('was error')
                temp_params={'cov':global_policy_params.cov,'noise':global_policy_params.noise_term,\
                        'like':-100333
}
                #cov,X_dim,noise_term
                print(temp_params)
                inv_term = pb.get_inv_term(temp_params['cov'],history[0].shape[0],temp_params['noise'])
                #if to_save_params not None:
                global_policy_params.to_save_params[time]=temp_params['like']
                print('global_info', time,global_policy_params.decision_times,temp_params['noise'],time_module.strftime('%l:%M%p %Z on %b %d, %Y'),temp_params['like'],file=open('../../murphy_lab/lab/pooling/{}_two/updates_global_{}_{}_{}.txt'.format(case,len(experiment.population),global_policy_params.update_period,sim_num), 'a'))
                global_policy_params.update_params(temp_params)
                global_policy_params.inv_term=inv_term
                #print(temp_params)
                global_policy_params.history = history
                #del history
            ##update global context
            ##global context shared across all participants
        tod = sf.get_time_of_day(time)
        dow = sf.get_day_of_week(time)
        if time==experiment.study_days[0]:
            print('init weather')
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
                  
                    
                    
                if time <= participant.times[0]:
                    steps_last_time_period = 0  
                    
                    ##set first pre-treatment, yesterday step count, variation and dosage
                else:
                    
                    if time.hour==0 and time.minute==0:
                        participant.current_day_counter=participant.current_day_counter+1
                    
                    #print(time)
                    steps_last_time_period = participant.steps
                
                 

                if time.hour in experiment.location_update_hours:
                    location = sf.get_next_location(participant.gid,dow,tod,participant.get_loc())
                
    
                
                participant.set_loc(location)
                
                

                prob = -1
                if time in participant.decision_times:
        
                    
                    dt=True
                    action=0
                    
                    
                    if global_policy_params.decision_times>20 and global_policy_params.history!=None:
                     
                            history = global_policy_params.history
                            temp = pb.calculate_posterior_faster(global_policy_params,\
                                                  participant.pid,participant.current_day_counter,\
                                                  history[0], history[1],history[2] )
                    
                    #print(temp[0].shape)
                    else:
                        #print('here')
                        temp = [personal_policy_params.mus2[participant.pid],personal_policy_params.sigmas2[participant.pid]]
                    mu_beta = temp[0]
                    Sigma_beta = temp[1]
                    personal_policy_params.update_mus(participant.pid,mu_beta,2)
                    personal_policy_params.update_sigmas(participant.pid,Sigma_beta,2)    
                    
                    
                    if policy==None:
                        action = sf.get_action(policy)
                        
                        
                        
                    elif policy=='TS':
                      
                        z=np.array([1,sf.get_pretreatment(participant.steps),weather,dow,tod])
                        
                        prob = TS.prob_cal_ts(z,0,personal_policy_params.mus2[participant.pid],personal_policy_params.sigmas2[participant.pid],global_policy_params)
                        action = int(uniform() < prob)
                            
                    if availability:
                  
                        context = [action,participant.gid,tod,dow,weather,sf.get_pretreatment(participant.steps),location,\
                              0,0,0]
                    
                        #participant.steps_last_time_period = participant.steps
                        steps = sf.get_steps_action(context)
                        add = sf.get_add_two(action,z,experiment.beta[:-3],participant.Z)
                        participant.steps = steps+add
                        
                    else:
                        #participant.steps_last_time_period = participant.steps
                        steps = sf.get_steps_no_action(participant.gid,tod,dow,location,weather,participant.steps)
                        participant.steps = steps

                

                    global_policy_params.decision_times =   global_policy_params.decision_times+1
                
                    
                    
                else:
                    #participant.steps_last_time_period = participant.steps
                        steps = sf.get_steps_no_action(participant.gid,tod,dow,location,weather,participant.steps)
                        participant.steps = steps     
                
                ##history:
                context_dict =  {'steps':participant.steps,'action':action,'weather':weather,'location_1':int(location==1),\
                    'ltps':steps_last_time_period,'location_2':int(location==2),'location_3':int(location==3),\
                        'study_day':participant.current_day_counter,'decision_time':dt,'time':time,'avail':availability,'prob':prob,'dow':dow,'tod':tod,'pretreatment':sf.get_pretreatment(steps_last_time_period)}



# my_directory = '{}/pop_size_{}_update_{}_study_length_{}/participant_{}'.format(global_policy_params.write_directory,len(experiment.population),global_policy_params.update_period,experiment.study_length,participant.pid)
#  if not os.path.exists(my_directory):
# os.makedirs(my_directory)
                        # with open('{}/history_{}.pkl'.format(my_directory,global_policy_params.decision_times),'wb') as f:
#   pickle.dump(participant.history,f)


                participant.history[time]=context_dict
                    #if global_policy_params.decision_times%100==0:
                    
                    # to_save = make_to_save(experiment)
                    # gids = make_to_groupids(experiment)
                    
                    # filename = '{}/results/population_size_{}_update_days_{}_{}_EB_{}_{}_testing_{}_safer_f.pkl'.format('../../murphy_lab/lab/pooling',pop_size,update_time,study_length,case,sim_num,global_policy_params.decision_times)
                    # with open(filename,'wb') as f:
#pickle.dump({'history':to_save,'gids':gids,'likelis':glob.to_save_params},f)

def make_to_save(exp):
    to_save  = {}
    for pid,pdata in exp.population.items():
        for time,context in pdata.history.items():
            key = '{}-{}'.format(pid,time)
            to_save[key]=context
    return to_save


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
  

    for i in range(int(start_index),int(end_index)):
        for case in ['case_one','case_two','case_three']:
            root = '../../murphy_lab/lab/pooling/distributions/'
            pop_size=population
            experiment = study.study('../../murphy_lab/lab/pooling/distributions/',pop_size,'short',which_gen=case)
            glob,personal = initialize_policy_params_TS(experiment,7,standardize=True,root=root)
            ft = feature_transformations.feature_transformation('../../murphy_lab/lab/pooling/distributions/')
          
            hist = new_kind_of_simulation(experiment,'TS',personal,glob,i,case,ft)
            
            to_save = make_to_save(experiment)
            gids = make_to_groupids(experiment)
            
            filename = '{}/results/population_size_{}_update_days_{}_{}_EB_{}_{}_new_params.pkl'.format('../../murphy_lab/lab/pooling',pop_size,update_time,study_length,case,i)
            with open(filename,'wb') as f:
                pickle.dump({'history':to_save,'gids':gids,'likelis':glob.to_save_params},f)
