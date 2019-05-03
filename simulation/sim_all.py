import pandas as pd
import sys
#sys.path
#sys.path.append('../../../home00/stomkins/pooling_rl/models')
#sys.path.append('../../../home00/stomkins/pooling_rl/simulation')
import pandas as pd
sys.path
sys.path.append('../models')
sys.path.append('models')
import numpy as np
import pickle
import random
import os
import math
import run_gpytorchkernel
import run_gpytorchkernel_simple
import run_gpytorchkernel_noEB
import operator
import study
import time as time_module
import simple_bandits
import TS_personal_params_pooled as pp
import TS_global_params_pooled as gtp
from numpy.random import uniform
import run_gpy
#sys.path.append('../simulation')
import TS_fancy_pooled
import TS
#import TS_fancy_pooled
import eta
import pooling_bandits as pb
import warnings
warnings.simplefilter('ignore')
from sklearn import preprocessing
import tensorflow as tf
import gc
import feature_transformations as ft


def initialize_policy_params_TS(experiment,update_period,\
                                standardize=False,baseline_features=None,psi_features=None,\
                                responsivity_keys=None,algo_type=None):
    #,'location_1','location_2','location_3'
    #'continuous_temp',
    u_params=None
    if algo_type=='pooling_four':
        u_params =[0.1081,0.2735,0.0178,0.0655,0.9567,0.8582,0.9528,0.9531,0.8941,0.9495]
        u_params = [0.06451840481880405,0.01370354099136279,0.010851198582671185,0.015391588363759446,0.725637691598818,1.227441604469287,0.9107303216615658,0.9863941781916185,1.3498034928863532,0.5768819568022332]
#[0.3167,0.4156,0.1055,0.1776,1.0605,0.4591,1.0597,1.0604,0.7273,1.0590]
    
    global_p =gtp.TS_global_params(21,baseline_features=baseline_features,psi_features=psi_features, responsivity_keys= responsivity_keys,uparams = u_params)
    personal_p = pp.TS_personal_params()
    #global_p =gtp.TS_global_params(10,context_dimension)
    
    
    
    #global_p.mu_dimension = 64
    
    global_p.kdim =24
    #194
    global_p.baseline_indices = [i for i in range(3+ 3*len(baseline_features))]
    #[i for i in range(192)]
    #[0,1,2,3,4,5,6]
    global_p.psi_indices = [0] + [1+baseline_features.index(j) for j in psi_features] \
    + [len(baseline_features)+1] + [(2+len(baseline_features))+baseline_features.index(j) for j in psi_features]
    #[0,64]
    global_p.user_id_index =0
    
    global_p.psi_features =psi_features
    #[0,64]
    
    #print(global_p.psi_indices )
    
    global_p.update_period = update_period
    
    global_p.standardize = standardize
    
    
    
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
    
    #4.83
    global_p.mu2_knot = np.array([4.6]+[0 for i in range(global_p.num_responsivity_features)])
    global_p.mu1_knot = np.zeros(global_p.num_baseline_features+1)
    global_p.sigma1_knot = np.eye(global_p.num_baseline_features+1)
    global_p.sigma2_knot = np.eye(global_p.num_responsivity_features+1)
    #print(type(personal_p))
    
    for person in experiment.population.keys():
        #experiment.population[person].root = '../../regal/murphy_lab/pooling/distributions/'
        initial_context = [0 for i in range(global_p.theta_dim)]
        
        
        
        
        
        if algo_type!='batch':
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



def get_optimal_reward(beta,states,Z):
        if Z is None:
        
            return np.dot(beta,states)
        return np.dot(beta,states)+Z


def make_to_save(exp):
        to_save  = {}
        for pid,pdata in exp.population.items():
            for time,context in pdata.history.items():
            
                key = '{}-{}-{}'.format(pid,time,pdata.gid)
                to_save[key]={k:v for k,v in context.items() if k in ['steps','decision_time','avail','action']}
        return to_save

def new_kind_of_simulation(experiment,policy=None,personal_policy_params=None,global_policy_params=None,generative_functions=None,which_gen=None,feat_trans = None,algo_type = None,case=None,sim_num=None,train_type='None'):
    #write_directory = '../../murphy_lab/lab/pooling/temp'
    experiment.last_update_day=experiment.study_days[0]
    tod_check = set([])
    
    
    additives = []
    
    for time in experiment.study_days:

        if time==experiment.last_update_day+pd.DateOffset(days=global_policy_params.update_period):
            experiment.last_update_day=time

            if global_policy_params.decision_times>2:
                global_policy_params.last_global_update_time=time


                if algo_type=='batch' or algo_type=='pooling' or algo_type=='pooling_four':

                    temp_hist = feat_trans.get_history_decision_time_avail(experiment,time)
                    temp_hist= feat_trans.history_semi_continuous(temp_hist,global_policy_params)
                    context,steps,probs,actions= feat_trans.get_form_TS(temp_hist)
                    temp_data = feat_trans.get_phi_from_history_lookups(temp_hist)

                    steps = feat_trans.get_RT(temp_data[2],temp_data[0],global_policy_params.mu_theta,global_policy_params.theta_dim)
                    
                    
                    #####HERE GET NEW NOISE TERM
                    if train_type=='EB':
                        temp_params = run_gpytorchkernel.run(temp_data[0], temp_data[1],steps,global_policy_params)
                        global_policy_params.noise_term = temp_params['noise']
                    temp = TS.policy_update_ts_new( context,steps,probs,actions,global_policy_params.noise_term,\
                                                       global_policy_params.mu1_knot,\
                                                       global_policy_params.sigma1_knot,\
                                                       global_policy_params.mu2_knot,\
                                                       global_policy_params.sigma2_knot)

                    mu_beta = temp[0]
                    Sigma_beta = temp[1]
                    #print(mu_beta)
                    if algo_type=='batch':
                        global_policy_params.update_mus(None,mu_beta,2)
                        global_policy_params.update_sigmas(None,Sigma_beta,2)
                    
                    ##train the noise
                    
                    
                    else :
                        #global_posterior = mu_beta
                        #global_posterior_sigma = Sigma_beta
                        if train_type=='EB':
                            if algo_type=='pooling':
                            
                                try:
                              
                                    # print('steps {}'.format(steps.mean()))
                                    temp_params = run_gpytorchkernel.run(temp_data[0], temp_data[1],steps,global_policy_params)
                                    experiment.iters.append(temp_params['iters'])
                                  
                                    if temp_params['cov'] is not None:
                                        global_policy_params.update_params(temp_params)
                                        global_policy_params.lr = global_policy_params.lr/2
                                except Exception as e:
                               
                                    temp_params={'cov':global_policy_params.cov,\
                                    'noise':global_policy_params.noise_term,\
                                            'like':-100333,'sigma_u':global_policy_params.sigma_u}
                            else:
                                try:
                                
                                #print(baseline_features)
                                #temp_params = run_gpy.run(temp_data[0], temp_data[1],np.array([[i] for i in steps]),global_policy_params)
                                #print('steps {}'.format(steps.std()**2))
                                    temp_params = run_gpytorchkernel_larger.run(temp_data[0], temp_data[1],steps,global_policy_params)
                                
                                #print(temp_data[0].shape)
                                #print('temp params one {}'.format(temp_params))
                                    if temp_params['cov'] is not None:
                                        global_policy_params.update_params_more(temp_params)
                                except Exception as e:
                                    pass
                        ### don't update tuning parameters
                        ###do get everything else
                        else:
                            if algo_type=='pooling':
                                temp_params = run_gpytorchkernel_noEB.run(temp_data[0], temp_data[1],steps,global_policy_params)
                                global_policy_params.cov =temp_params['cov']
                        

#print('temp params {}'.format(temp_params))
                        inv_term = simple_bandits.get_inv_term(global_policy_params.cov,temp_data[0].shape[0],global_policy_params.noise_term)

                        global_policy_params.inv_term=inv_term
                                    #print(temp_params)
                        global_policy_params.history =temp_data
                            #[temp_data[0], temp_data[1],steps]
                        


        tod = feat_trans.get_time_of_day(time)
        dow = feat_trans.get_day_of_week(time)

        if time==experiment.study_days[0]:
                        #print('init weather')
            weather = feat_trans.get_weather_prior(tod,time.month,seed=experiment.weather_gen)
                            #temperature = tf.continuous_temperature(weather)
        elif time.hour in experiment.weather_update_hours and time.minute==0:
            weather = feat_trans.get_next_weather(str(tod),str(time.month),weather,seed=experiment.weather_gen)

        for person in experiment.dates_to_people[time]:
            participant = experiment.population[person]
            dt=int(time in participant.decision_times)
            action = 0
            prob=0


            if algo_type=='personalized'  and time==participant.last_update_day+pd.DateOffset(days=global_policy_params.update_period):
                #('made update')
                temp_hist = feat_trans.get_history_decision_time_avail_single({participant.pid:participant.history},time)
                temp_hist= feat_trans.history_semi_continuous(temp_hist,global_policy_params)
                context,steps,probs,actions= feat_trans.get_form_TS(temp_hist)
                temp_data = feat_trans.get_phi_from_history_lookups(temp_hist)
                steps = feat_trans.get_RT_o(steps,temp_data[0],global_policy_params.mu_theta,global_policy_params.theta_dim)
                
                
                #####HERE GET NEW NOISE TERM
                if train_type=='EB':
                    temp_params = run_gpytorchkernel.run(temp_data[0], temp_data[1],steps,global_policy_params)
                    global_policy_params.noise_term = temp_params['noise']
                
                temp = TS.policy_update_ts_new( context,steps,probs,actions,global_policy_params.noise_term,\
                                                           global_policy_params.mu1_knot,\
                                                           global_policy_params.sigma1_knot,\
                                                           global_policy_params.mu2_knot,\
                                                           global_policy_params.sigma2_knot,
                                                           #personal_policy_params.mus1[participant.pid],\
                                                           #personal_policy_params.sigmas1[participant.pid],\
                                                           #personal_policy_params.mus2[participant.pid],\
                                                           #personal_policy_params.sigmas2[participant.pid],
                                                           
                                                           )
                mu_beta = temp[0]
                Sigma_beta = temp[1]
                personal_policy_params.update_mus(participant.pid,mu_beta,2)
                personal_policy_params.update_sigmas(participant.pid,Sigma_beta,2)
                participant.last_update_day=time
            elif (algo_type=='pooling' or algo_type=='pooling_four')  and global_policy_params.decision_times>2 and global_policy_params.history is not None and  time==participant.last_update_day+pd.DateOffset(days=global_policy_params.update_period):
                history = global_policy_params.history
                #print(history)
                if algo_type=='pooling':
                    temp = simple_bandits.calculate_posterior_faster(global_policy_params,\
                                                         participant.pid,participant.current_day_counter,\
                                                         history[0], history[1],history[2] )
            
                                                         #global_posterior = mu_beta
                                            #global_posterior_sigma = Sigma_beta
                else:
                    temp = simple_bandits.calculate_posterior_current(global_policy_params,\
                                                                     participant.pid,participant.current_day_counter,\
                                                                     history[0], history[1],history[2] )
                mu_beta = temp[0]
                Sigma_beta = temp[1]
                    #if participant.pid==2:
                    # print(np.array(history[2]).std())
                    #print('after posterior')
                    #print(mu_beta)
                personal_policy_params.update_mus(participant.pid,mu_beta,2)
                personal_policy_params.update_sigmas(participant.pid,Sigma_beta,2)
                #``ere', time,global_policy_params.decision_times,'here here',file=open('pooling/{}/updates_global_newbigtest_{}_{}_{}six_weeks_only_onoise_herecurrent.txt'.format(case,len(experiment.population),global_policy_params.update_period,sim_num), 'a'))
                participant.last_update_day=time
            participant.set_tod(tod)
            participant.set_dow(dow)

            availability = (participant.rando_gen.uniform() < 0.8)
    
            participant.set_available(availability)
            if time == participant.times[0]:
                location = feat_trans.get_location_prior(str(participant.gid),str(tod),str(dow),seed = participant.rando_gen)
            elif time.hour in experiment.location_update_hours and time.minute==0 :
                location = feat_trans.get_next_location(participant.gid,tod,dow,participant.get_loc(),seed =participant.rando_gen)

            participant.set_loc(location)

            if time <= participant.times[0]:
                        steps_last_time_period = 0
            else:
                if time.hour==0 and time.minute==0:
                    participant.current_day_counter=participant.current_day_counter+1

                steps_last_time_period = participant.steps

            prob = .6
            add=None
            optimal_action = -1
            optimal_reward = -100
            if dt:
                if policy=='TS':
                    pretreatment = feat_trans.get_pretreatment(steps_last_time_period)
                    z = [1]
                    calc = [1,tod,dow,pretreatment,location]
                    if 'tod' in global_policy_params.baseline_features:
                        z.append(tod)
                    if 'dow' in global_policy_params.baseline_features:
                        z.append(dow)
                    if 'pretreatment' in global_policy_params.baseline_features:
                        z.append(pretreatment)
                    if 'location' in global_policy_params.baseline_features:
                        z.append(location)

                    if algo_type=='batch':
                        prob = TS.prob_cal_ts(z,0,global_policy_params.mus2,global_policy_params.sigmas2,global_policy_params,seed = experiment.algo_rando_gen)
                    elif algo_type=='personalized':
                        prob = TS.prob_cal_ts(z,0,personal_policy_params.mus2[participant.pid],personal_policy_params.sigmas2[participant.pid],global_policy_params,seed=experiment.algo_rando_gen)
                    elif algo_type=='pooling':
                                prob = TS.prob_cal_ts(z,0,personal_policy_params.mus2[participant.pid],personal_policy_params.sigmas2[participant.pid],global_policy_params,seed=experiment.algo_rando_gen)
                                    
                    action = int(experiment.algo_rando_gen.uniform() < prob)
                    if availability:
                        context = [action,participant.gid,tod,dow,weather,pretreatment,location,\
                                       0,0,0]
                        steps = feat_trans.get_steps_action(context,seed = participant.rando_gen)
                        add = action*(feat_trans.get_add_no_action(calc,participant.beta,participant.Z))
                        participant.steps = steps+add
                        optimal_reward = get_optimal_reward(participant.beta,calc,participant.Z)
                        optimal_action = int(optimal_reward>=0)
                    else:
                        steps = feat_trans.get_steps_no_action(participant.gid,tod,dow,location,\
                        pretreatment,weather,seed = participant.rando_gen)
                        participant.steps = steps

                    global_policy_params.decision_times =   global_policy_params.decision_times+1
                else:
                        steps = feat_trans.get_steps_no_action(participant.gid,tod,dow,location,\
                                                               pretreatment,weather,seed = participant.rando_gen)
                        participant.steps = steps
                context_dict =  {'steps':participant.steps,'add':add,'action':action,'location':location,'location_1':int(location==1),\
'ltps':steps_last_time_period,'location_2':int(location==2),'location_3':int(location==3),\
    'study_day':participant.current_day_counter,\
        'decision_time':dt,\
            'time':time,'avail':availability,'prob':prob,\
                'dow':dow,'tod':tod,'weather':weather,\
                    'pretreatment':feat_trans.get_pretreatment(steps_last_time_period),\
                        'optimal_reward':optimal_reward,'optimal_action':optimal_action,\
                            'mu2':global_policy_params.mus2,'gid':participant.gid}

                participant.history[time]=context_dict


def get_regret(experiment):
    optimal_actions ={}
    rewards = {}
    actions = {}
    for pid,person in experiment.population.items():
        for time,data in person.history.items():
            if data['decision_time'] and data['avail']:
                key = time
                if key not in optimal_actions:
                    optimal_actions[key]=[]
                if key not in rewards:
                    rewards[key]=[]
                if key not in actions:
                    actions[key]=[]
                if data['optimal_action']!=-1:
                    optimal_actions[key].append(int(data['action']==data['optimal_action']))
                    regret = int(data['action']!=data['optimal_action'])*(abs(data['optimal_reward']))
                    rewards[key].append(regret)
                    actions[key].append(data['action'])
    return optimal_actions,rewards

def make_to_groupids(exp):
    to_save  = {}
    for pid,pdata in exp.population.items():
        gid  = pdata.gid
        key = 'participant-{}'.format(pid)
        to_save[key]=gid
    return to_save

def run_many(algo_type,cases,sim_start,sim_end,update_time,dist_root,write_directory,train_type):
    for case in cases:
        #,'case_two','case_three'
        #case = 'case_one'
        
        #'dow','pretreatment',
        #'tod','dow',
        baseline = ['pretreatment','location']
        
        
        
        for u in [update_time]:
            
            all_actions = {}
            all_rewards = {}
            #'../../Downloads/distributions/'
            feat_trans = ft.feature_transformation(dist_root)
            
            for sim in range(sim_start,sim_end):
                pop_size=32
                experiment = study.study(dist_root,pop_size,'_short_unstaggered',which_gen=case,sim_number=sim)
                experiment.update_beta(set(baseline))
               
                psi = []
                if algo_type=='pooling_four':
                    psi = ['location']
                
                glob,personal = initialize_policy_params_TS(experiment,7,standardize=False,baseline_features=baseline,psi_features=psi,responsivity_keys=baseline,algo_type =algo_type)
                
                hist = new_kind_of_simulation(experiment,'TS',personal,glob,feat_trans=feat_trans,algo_type=algo_type,case=case,sim_num=sim,train_type=train_type)
                to_save = make_to_save(experiment)
                actions,rewards = get_regret(experiment)
                gids = make_to_groupids(experiment)
                
            
            
                filename = '{}{}/population_size_{}_update_days_{}_{}_static_sim_{}_424_newtuning.pkl'.format('{}{}/'.format(write_directory,algo_type),case,pop_size,u,'short',sim)
                with open(filename,'wb') as f:
                    pickle.dump({'gids':gids,'regrets':rewards,'actions':actions,'history':to_save,'pprams':personal,'gparams':glob},f)
        #filename = '{}/results/{}/population_size_{}_update_days_{}_{}_static_sim_regrets_actions_l_prelocb.pkl'.format('../../Downloads/pooling_results/{}/'.format(algo_type),case,pop_size,u,'short')
        #with open(filename,'wb') as f:
# pickle.dump({'actions':all_actions,'regrets':all_rewards},f)



