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
import time as time_module
import TS_personal_params_pooled as pp
import TS_global_params_pooled as gtp
from numpy.random import uniform

##sys.path.append('../simulation')
import TS_fancy_pooled
import TS
##import TS_fancy_pooled

import pooling_bandits as pb
from sklearn import preprocessing
import tensorflow as tf




def initialize_policy_params_TS(experiment,update_period):
    
    global_p =gtp.TS_global_params(20,baseline_features=['tod','dow','weather','pretreatment','location'],psi_features=[0,6], resp_features= ['tod','dow','weather','pretreatment','location'])
    personal_p = pp.TS_personal_params()
    #global_p =gtp.TS_global_params(10,context_dimension)
    
    
    
    #global_p.mu_dimension = 64

    global_p.kdim =20
    #194
    global_p.baseline_indices = [i for i in range(18)]
    #[i for i in range(192)]
    #[0,1,2,3,4,5,6]
    global_p.psi_indices =[0,6]
    #[0,64]
    global_p.user_id_index =18
    #192
    global_p.user_day_index =19
    #193
    
    #global_p.baseline_features = [i for i in range(192)]
    global_p.psi_features =[0,6]
    #[0,64]
    
    global_p.update_period = update_period
    
    
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
    
    global_p.write_directory = '../../murphy_lab/lab/pooling/EB'
    return global_p ,personal_p

def new_kind_of_simulation(experiment,policy=None,personal_policy_params=None,global_policy_params=None):
    #write_directory = '../../murphy_lab/lab/pooling/temp'
    experiment.last_update_day=experiment.study_days[0]
    for time in experiment.study_days:
        
        #if time> experiment.study_days[0]:
        #history  = pb.make_history(experiment)
        if time==experiment.last_update_day+pd.DateOffset(days=global_policy_params.update_period):
            experiment.last_update_day=time
            print('Global update', time,global_policy_params.decision_times,time_module.strftime('%l:%M%p %Z on %b %d, %Y'), file=open('updates_{}_{}.txt'.format(len(experiment.population),global_policy_params.update_period), 'a'))
            if global_policy_params.decision_times>200:
                glob.last_global_update_time=time
                history =pb.make_history_new(.1,glob,experiment)
                    #print(history[1])
                print(history[0].shape)
                print(history[1].shape)
                temp_params = pb.run(history[0],history[1],global_policy_params,gp_train_type = 'empirical_bayes')
          
                global_policy_params.update_params(temp_params)
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
                    #participant.set_duration(0)
                    #participant.set_dosage(0)
                    #personal_policy_params.etas[participant.pid]
                    
                    
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
                ##maybe faster to update instead of query?
                #participant.set_last_time_period_steps(steps_last_time_period)
                #participant.set_yesterday_steps(steps_yesterday)
                #participant.set_variation(variation)
                
                ##continue
                #2
                ##for every active person take an action according to current context, policy, and parameters
                
                

                prob = -1
                if time in participant.decision_times:
                                        #print(personal_policy_params.batch_index[participant.pid])
                    
                    
                    ##if we have made no global updates

                    
                    
                    
                    dt=True
                    action=0
                    
                    
                    if global_policy_params.decision_times>200 and global_policy_params.history!=None:
                        ##do i need this?
                        # if   not global_policy_params.updated_cov:
                        #     global_policy_params.update_cov(global_policy_params.decision_times)
                            #print( global_policy_params.decision_times)
                            history = global_policy_params.history                    ##update my mu2 and sigma2
                            ##change dimension of mu
                            temp = pb.calculate_posterior_faster(global_policy_params,\
                                                  participant.pid,participant.current_day_counter,\
                                                  history[0], history[1] )
                    
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
                        #some context slice
                        
                        ##both f_one and g_one
                        #one_hot_vector = pb.get_one_hot_encodings(global_policy_params,{'steps':steps,'weather':weather,'location':location,'ltps':participant.steps,'study_day':participant.current_day_counter,'decision_time':dt,'time':time,'avail':availability})
                        #z = np.zeros(global_policy_params.num_responsivity_features+1)
                        #old
                        #z[0]=1
                        #z[1:]=one_hot_vector
                        z=np.array([1,tod,dow,weather,sf.get_pretreatment(participant.steps),location])
                        
                        prob = TS.prob_cal_ts(z,0,personal_policy_params.mus2[participant.pid],personal_policy_params.sigmas2[participant.pid],global_policy_params)
                        action = int(uniform() < prob)
                            
                        

                        
                        
                        
                    ##is this the same as in the TS?
                    ##don't think so, but for now keep like this
                    ##no it isn't, i have to redo this
                    

                    
                    if availability:
                    

                   
                    
                    
                   
                    
                        context = [action,participant.gid,tod,dow,weather,sf.get_pretreatment(participant.steps),location,\
                              0,0,0]
                    
                        #participant.steps_last_time_period = participant.steps
                        steps = sf.get_steps_action(context)
                        participant.steps = steps
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
                context_dict =  {'steps':steps,'action':action,'weather':weather,'location':location,\
                    'ltps':steps_last_time_period,\
                        'study_day':participant.current_day_counter,'decision_time':dt,'time':time,'avail':availability,'prob':prob,'dow':dow,'tod':tod,'pretreatment':sf.get_pretreatment(steps_last_time_period)}
                participant.history[time]=context_dict


                if global_policy_params.decision_times%100==0:
                    my_directory = '{}/pop_size_{}_update_{}_study_length_{}/participant_{}'.format(global_policy_params.write_directory,participant.pid,experiment.study_length,len(experiment.population),global_policy_params.update_period)
                    if not os.path.exists(my_directory):
                        os.makedirs(my_directory)
                    with open('{}/history_{}.pkl'.format(my_directory,global_policy_params.decision_times),'wb') as f:
                        pickle.dump(participant.history,f)



            #3
            ##for every active person generate a step count given current context
            
            
            
            
            ##update at midnight (here we have ensured that no one has a ) experiment.update_hour
            

def make_to_save(exp):
    to_save  = {}
    for pid,pdata in exp.population.items():
        for time,context in pdata.history.items():
            key = '{}-{}'.format(pid,time)
            to_save[key]=context
    return to_save


if __name__=="__main__":

    ##parse command line arguments
    
    population = sys.argv[1]
    update_time = sys.argv[2]
    study_length = sys.argv[3]
    
    #print(str(sys.argv))
    #print(type(population))
    update_time = int(update_time)
    #print(update_time)
    #print(type(update_time))
    #print(population)
  
    #'../../regal/murphy_lab/pooling/distributions/'
    #
    #print()
    experiment = study.study('../../regal/murphy_lab/pooling/distributions/' ,population,study_length)
   
    glob,personal = initialize_policy_params_TS(experiment,update_time)
  
    new_kind_of_simulation(experiment,'TS',personal,glob)
    
    to_save = make_to_save(experiment)
    
    filename = '{}/results/population_size_{}_update_days_{}_{}_EB.pkl'.format('../../murphy_lab/lab/pooling',population,update_time,study_length)
    with open(filename,'wb') as f:
        pickle.dump(to_save,f)
    
    print('finished')    

