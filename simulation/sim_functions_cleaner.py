import pandas as pd
import numpy as np
import pickle

import os
import math

import random
from datetime import datetime
random.seed(datetime.now())
from scipy.stats import halfnorm


if '/Users/sabina/pooling_rl/' in os.getcwd():
    root ='../../Downloads/distributions/'
else:
    root = '../../regal/murphy_lab/pooling/distributions/'
#'../../../../Volumes/dav/HeartSteps/pooling_rl_shared_data/distributions/'
#root = '../../murphy_lab/lab/pooling/distributions/'

#with open('{}steps_both_groups_logs_dosage_estf_bbiit_swapped.pkl'.format(root),'rb') as f:
#    dists = pickle.load(f)


####
with open('{}label_to_temperature_value_stats.pkl'.format(root),'rb') as f:
    weather_label_to_val = pickle.load(f)

with open('{}loc_label_to_intervention_label_tref.pkl'.format(root),'rb') as f:
    loc_label_to_val = pickle.load(f)


#with open('{}conversion_pretreatment.pkl'.format(root),'rb') as f:
#    hour_pretreatment_label_to_val = pickle.load(f)


with open('{}dists_non_intervention.pkl'.format(root),'rb') as f:
    dists = pickle.load(f)

with open('{}key_matches_non_intervention.pkl'.format(root),'rb') as f:
    matched = pickle.load(f)


with open('{}dists_intervention_anti_sedentary.pkl'.format(root),'rb') as f:
    dists_intervention_anti_sedentary = pickle.load(f)

with open('{}key_matches_intervention_anti_sedentary.pkl'.format(root),'rb') as f:
    matched_intervention_anti_sedentary = pickle.load(f)


with open('{}dists_intervention_activity_suggestion.pkl'.format(root),'rb') as f:
    dists_intervention_activity_suggestion = pickle.load(f)

with open('{}key_matches_intervention_activity_suggestion.pkl'.format(root),'rb') as f:
    matched_intervention_activity_suggestion = pickle.load(f)


#with open('{}interventions_both_groups_estf.pkl'.format(root),'rb') as f:
#intervention_dists = pickle.load(f)

def get_location_prior(group_id,day_of_week,time_of_day):
    with open('{}initial_location_distributions_est_tref.pkl'.format(root),'rb') as f:
        loc_lookup = pickle.load(f)
    key = '{}-{}-{}'.format(group_id,day_of_week,time_of_day)
    
    ##make a bit smoother while loop instead 
    if key in loc_lookup:
        ps = loc_lookup[key]
    else:
        print('sdf')
        print(key)
                
    val = np.argmax(np.random.multinomial(1,ps))
    return val


def get_weather_prior(time_of_day,month):
    with open('{}initial_temperature_distributions_est.pkl'.format(root),'rb') as f:
        loc_lookup = pickle.load(f)
    key = '{}-{}'.format(time_of_day,month)
    
    ##make a bit smoother while loop instead 
    if key in loc_lookup:
        ps = loc_lookup[key]
    else:
        key =  '{}'.format(time_of_day)
        ps = loc_lookup[key]
        
                
    val = np.argmax(np.random.multinomial(1,ps))
    return val

def get_time_of_day(an_index):
    with open('{}hour_to_id.pkl'.format(root),'rb') as f:
        hour_lookup = pickle.load(f)
    return hour_lookup[an_index.hour]


def get_day_of_week(an_index):
    with open('{}day_to_id.pkl'.format(root),'rb') as f:
        hour_lookup = pickle.load(f)
    return hour_lookup[an_index.dayofweek]

def dosage_to_dosage_key(dosage):
        
        dosage = int(dosage)
        if dosage<=33:
            return 0 
        elif dosage>33 and dosage<=66:
            return 1
        return 2


def get_possible_keys(context):
    
    
    keys = []
    
    #keys.append('-'.join([str(i) for i in context]))
    for i in range(len(context)):
        stop = len(context)-i
        #for j in range(stop):
        #if stop>=1:
        key = '-'.join([str(context[j]) for j in range(stop)])
        
        keys.append(key)
    keys.append('{}-mean'.format(context[0]))
    return keys
      
    
    
def get_next_weather(tod,month,weather):
    #print('weather changed')
    with open('{}temperature_conditiononed_on_last_temperature_est.pkl'.format(root),'rb') as f:
        loc_dists =pickle.load(f)
    

    
    relevant_context = [tod,month,weather]
    
    context_key = '-'.join([str(c) for c in relevant_context])
    possible_keys = get_possible_keys(relevant_context)
    
    keys = [context_key]
    keys.extend(possible_keys)
 
    i=0
 
    while keys[i] not in loc_dists and i<len(keys):
        i=i+1
        

    dist = loc_dists[keys[i]]
    
    val = np.argmax(np.random.multinomial(1,dist))
    
    return val

def get_pretreatment(steps):
    #steps = math.log(steps+.5)
    return int(steps>math.log(.5))

def get_next_location(gid,dow,tod,loc):
    
    context = [gid,dow,tod,loc]
    with open('{}location_conditiononed_on_last_location_tref.pkl'.format(root),'rb') as f:
        loc_dists =pickle.load(f)
    
    context_key = '-'.join([str(c) for c in context])
    if context_key in loc_dists:
        dist = loc_dists[context_key]
    
        val = np.argmax(np.random.multinomial(1,dist))

    else:
        context_key = '-'.join([str(c) for c in context[:-1]])
        with open('{}initial_location_distributions_est_tref.pkl'.format(root),'rb') as f:
            loc_lookup = pickle.load(f)
        dist = loc_lookup[context_key]
        val = np.argmax(np.random.multinomial(1,dist))
        
    return val


def get_steps_no_action(gid,tod,dow,loc,wea,pre):
    
    keys = ['gid',str(gid),'tod',str(tod),'dow',str(dow),'wea',str(wea),'pre',str(get_pretreatment(pre)),'loc',str(loc)]
    
    new_key = '-'.join(keys)
    
    
    dist_key = matched[new_key]

    dist = dists[dist_key]
    scale=dist[1]
    if scale==0:
        scale=scale+.001
    
    ##CHANGE TO TRUNCATED
    #x = halfnorm.rvs(loc=dist[0],scale=dist[1])
    x=np.random.normal(loc=dist[0],scale=scale)
    #while(x<0):
    #x = np.random.normal(loc=dist[0],scale=dist[1])
    return x

def get_steps_action(context):
    ids = ['aint','gid','tod','dow','wea','pre','loc']
    context = [str(c) for c in context]
    new_key = []
    message_type=context[0]
    context[0]='1'
    
    for i in range(len(ids)):
        new_key.append(ids[i])
        new_key.append(context[i])
    new_key = '-'.join(new_key)


    if message_type==1:
        dist_key = matched_intervention_activity_suggestion[new_key]
        dist = dists_intervention_activity_suggestion[dist_key]
    else:
        dist_key = matched_intervention_anti_sedentary[new_key]
        dist = dists_intervention_anti_sedentary[dist_key]
    
    #x = halfnorm.rvs(loc=dist[0],scale=dist[1])
    scale=dist[1]
    if scale==0:
        scale=scale+.001
    x = np.random.normal(loc=dist[0],scale=scale)
    # while(x<0):
    #      x = np.random.normal(loc=dist[0],scale=dist[1])
    return x
    


def to_yid(steps):
    
    for i in range(len(yesterday_chunks)):
        if steps>=yesterday_chunks[i][0] and steps<yesterday_chunks[i][1]:
            return i
    


def get_raw_temperature(weather_key):
    
    dist = weather_label_to_val[weather_key]

    x = np.random.normal(loc=dist[0],scale=dist[1])
    
    
    ##wrong way to do this?
    while(x<0):
         x = np.random.normal(loc=dist[0],scale=dist[1])
    return x
    
def get_raw_location(loc_key):
    ##gid, location, tod, dow
    
    dist = loc_label_to_val[weather_key]

    val = np.argmax(np.random.multinomial(1,dist))
    
    return val

def transform_to_required(context):
    #'group_id','time_of_day','pretreatment','day_of_week','location','variation','yesterday','weather','dosage'
    weather_id = context[get_index('weather')]
    pretreatment_id = context[get_index('preatreatment')]
    location_id = context[get_index('location')]
    
    yesterday_id = context[get_index('yesterday_id')]
    
    ##output should be: [dosage,temperature,location,variation,pre-treatment steps, yesterday steps]
    
    variation = context[get_index('variation')]
    dosage = context[get_index('dosage_id')]
    
    raw_temperature = get_raw_temperature(weather_id)
    
    #loc_key = 
    
    
#will be algorithm, needs to communicate with algorithm
#will be algorithm, needs to communicate with algorithm
def get_action(action_algorithm):

    if action_algorithm==None:
        
        available = random.random()>.8
        
        if available:
            
        
            return int(random.random()>.4)
        return 0
    

def simulate_run(num_people,time_indices,decision_times,action_algorithm = None,group_ids=None):
    
    
    initial_contexts = get_initial_context(num_people,time_indices[0],group_ids)
    

    
    all_people = []
    
    states = []
    
    for n in range(num_people):
        #print('person')
        initial_context = initial_contexts[n]
        #print('group id')
        #print(initial_context[0])
        
        inaction_duration = 0 
        action_duration = 0 
    
        initial_steps = get_steps(initial_context,-1,0)

        current_steps = initial_steps
        hour_steps = 0 
        steps_last_half_hour =0
        action = -1 
        all_steps = []
    
        last_day = time_indices[0]
    
        new_day = False
    
        #for d in range(num_days):
    
    
        start_of_day = 0 
        end_of_day=0
        current_index=0
    
    
        first_week = time_indices[0].date()+pd.DateOffset(days=8)
    
        for i in time_indices:
            
            
            
            states.append(initial_context)
            if i.date()!=last_day.date():
            #print('trigger')
            #print(i.date())
            #print(last_day.date())
            #print('hi there')
                new_day=True
            
            ##durations
            if hour_steps>0:
                action_duration = action_duration+1
                inaction_duration=0
            else:
                inaction_duration = inaction_duration+1
                action_duration = 0 
            duration = action_duration
            if action_duration==0:
                duration = inaction_duration
            
            duration = int(duration>5)
            
            decision_time = bool(i in decision_times)
        #print(decision_time)
            if i!=time_indices[0]:
                hour_steps = current_steps+steps_last_half_hour
            #decision_time = bool(i in decision_times)
            
            ##need to modify this
            #my_context = get_context(initial_context,current_steps,i,decision_time)
                lsc = initial_context[get_index('yesterday')]
                variation = initial_context[get_index('variation')]
                if new_day:
                #lsc=0
                
                
                ##would love to break this out more cleanly 
                    if i<first_week:
                        variation = get_variation_pre_week(variation,all_steps,time_indices,last_day)
                    else:
                        #print('got variation')
                        variation = get_variation(all_steps,time_indices,last_day)
                        #return variation
                
                    lsc = get_new_lsc(all_steps[start_of_day:end_of_day])
                #variation = get_new_variation()
                
            
            ##action will be the last action

            #return my_context
                if i in decision_times:
                    #print('decision time')
                    action = get_action(my_context,action_algorithm)
                #print(action)
                else:
                    action = -1
                    
                my_context =get_context_revised(i,initial_context,hour_steps,decision_time,lsc,variation,action)
            ##redo get_steps
                next_steps = get_steps(my_context,action,duration) 
                all_steps.append(next_steps)
                initial_context = my_context
                steps_last_half_hour=current_steps
                current_steps = next_steps
                
            else:
                if i in decision_times:
                    #print('type two decision time')
                    action = get_action(initial_context,action_algorithm)
                else:
                    action = -1
                next_steps = get_steps(initial_context,action,duration) 
                all_steps.append(next_steps)
                current_steps = next_steps
            if new_day:
            
                start_of_day = current_index
                new_day=False
            last_day = i
            end_of_day = current_index  
            current_index = current_index+1
        
        all_people.append(all_steps)
  
    return all_people,states



def get_add_one(action,state_vector,beta):
    return action*np.dot(beta,state_vector)

def get_add_two(action,state_vector,beta,Z):
    if Z is None:
        Z=0
    return action*(np.dot(beta,state_vector)+Z)

def get_add_three(action,state_vector,sigma,beta):

    return action*(np.dot(beta,state_vector)+Z)

def get_additive(action,state_vector,beta,Z=None,which='case_one'):
    if which=='case_one':
        return get_add_one(action,state_vector,beta)
    elif which == 'case_two':
        return get_add_two(action,state_vector,Z,beta)
    elif which=='case_three':
        return get_add_three(action,state_vector,Z,beta)
    else:
        return 'wrong case'



def get_data_for_txt_effect_update_batch(exp,glob):
    all_data = []
    steps=[]
    probs = []
    actions = []
    
    ##might add pi to the user's history
    for user_id,data in exp.population.items():
        history_dict=data.history
        if len(history_dict)>0:
            for hk,h in history_dict.items():
                if h['avail'] and h['decision_time']:
                    pi = h['prob']
                    
                    v=[h[i] for i in glob.responsivity_features]
                    steps.append(h['steps'])
                    probs.append(pi)
                    actions.append(h['action'])
                    all_data.append(v)

    return np.array(all_data),np.array(steps),np.array(probs),np.array(actions)


def get_data_for_txt_effect_update(history_dict,glob):
    all_data = []
    steps=[]
    probs = []
    actions = []
    
    ##might add pi to the user's history
    #for user_id,history in history_dict.items():
    
    for hk,h in history_dict.items():
        if h['avail'] and h['decision_time']:
            pi = h['prob']
            
            v=[h[i] for i in glob.responsivity_features]
            steps.append(h['steps'])
            probs.append(pi)
            actions.append(h['action'])
            all_data.append(v)

    return np.array(all_data),np.array(steps),np.array(probs),np.array(actions)
