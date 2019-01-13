import pandas as pd
import numpy as np
import pickle
import random
import os
import math

root =  '../../../../Volumes/dav/HeartSteps/pooling_rl_shared_data/distributions/'


#with open('{}steps_both_groups_logs_dosage_estf_bbiit_swapped.pkl'.format(root),'rb') as f:
#    dists = pickle.load(f)


####
with open('{}label_to_temperature_value_stats.pkl'.format(root),'rb') as f:
    weather_label_to_val = pickle.load(f)


with open('{}conversion_pretreatment.pkl'.format(root),'rb') as f:
    hour_pretreatment_label_to_val = pickle.load(f)

    
with open('{}trial_dists.pkl'.format(root),'rb') as f:
    dists = pickle.load(f)
    
with open('{}key_matches.pkl'.format(root),'rb') as f:
    matched = pickle.load(f)
    
    
with open('{}interventions_both_groups_estf.pkl'.format(root),'rb') as f:
    intervention_dists = pickle.load(f)

def get_location_prior(group_id,day_of_week,time_of_day):
    with open('{}initial_location_distributions_est_hwo.pkl'.format(root),'rb') as f:
        loc_lookup = pickle.load(f)
    key = '{}-{}-{}'.format(group_id,day_of_week,time_of_day)
    
    ##make a bit smoother while loop instead 
    if key in loc_lookup:
        ps = loc_lookup[key]
    else:
        key =  '{}-{}'.format(group_id,day_of_week)
        if key  in loc_lookup:
            ps = loc_lookup[key]
        else:
            key =  '{}'.format(group_id)
            if key  in loc_lookup:
                ps = loc_lookup[key]
                
            else:
                ps = loc_lookup['mean']
                
    val = np.argmax(np.random.multinomial(10,ps))
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
        
                
    val = np.argmax(np.random.multinomial(10,ps))
    return val

def get_time_of_day(an_index):
    with open('{}hour_to_id.pkl'.format(root),'rb') as f:
        hour_lookup = pickle.load(f)
    return hour_lookup[str(an_index.hour)]


def get_day_of_week(an_index):
    with open('{}day_to_id.pkl'.format(root),'rb') as f:
        hour_lookup = pickle.load(f)
    return hour_lookup[an_index.dayofweek]

def get_index(key):
    #day_of_week,time_of_day,last_steps,last_steps_hour,location,varia,weather,dosage
    keys = ['group_id','time_of_day','pretreatment','day_of_week','location','variation','yesterday','weather','dosage']
   
    kl = {keys[i]:i for i in range(len(keys))}
    
    return kl[key]


def get_initial_context(num_people,first_index,group_ids=None):
    '''States:
    [group,day_of_week,time_of_day,location,weather,previous_step_count,dosage,]
    
    '''
    
    all_people = []
    for person in range(num_people):
        ##.95 is an approximation
        if group_ids==None:
            print('hello')
            group_id = int(random.random()>4.0/36)+1
        else:
            group_id=group_ids[person]
        #group_id = 2
        day_of_week = get_day_of_week(first_index)
        time_of_day = get_time_of_day(first_index)
        first_location = get_location_prior(group_id,day_of_week,time_of_day)
        weather = get_weather_prior(time_of_day,first_index.month)
        #weather = 0 
        dosage = 1
        variation = 1
        pretreatment = 0 
        ##yesterday's step count could be drawn from distribution
        ysc = 0
        #day_of_week,time_of_day,dosage,location,last_steps,last_steps_hour,varia
        #'group_id','day_of_week','time_of_day','weather','location','dosage','yesterday','pretreatment','variation'#'group_id','day_of_week','time_of_day','yesterday','weather','location','dosage','pretreatment','variation'
        #'group_id','day_of_week','time_of_day','yesterday','pretreatment','location','variation','weather','dosage'
        all_people.append([group_id,time_of_day,pretreatment,day_of_week,\
                           first_location,variation,ysc,weather,dosage])
      
    return all_people


def get_initial_steps(contexts):
    
    return [steps_given_context(person_context) for person_context in contexts]

    
    
def get_time(current_time):
    
    #needs to be a time delta
    
    return current_time+1


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


def get_other_key(context):
    return ['-'.join([str(c) for c in context[:3]])]

def get_dosage_simple(dosage):
    dosage = int(dosage)
    if dosage<=33:
        return 0 
    elif dosage>33 and dosage<=66:
        return 1
    return 2
    

def context_to_key(context,duration):
    #'group_id','time_of_day','pretreatment','day_of_week','location','variation','yesterday','weather','dosage'
    ##this could be a mapping
    #'var',context[5]
    #'yst',context[6],,'wea',context[7],'loc',context[4]
    #,'pre',context[2],'var',context[5]
    #,'tod',context[1],'dow',context[3]
    context = [str(c) for c in context]
 #,'pre',str(duration),'wea',context[7]
    selected_context = ['gid',context[0],'tod',context[1],'dow',context[3],'loc',context[4],'pre',context[2],'wea',context[7]]
    
    return '-'.join(selected_context)

def get_steps_no_action(context,duration):
    
    
    if context[2]==1:
        if duration==0:
            prekey = 0
        else:
            prekey=1
    else:
        if duration==1:
            prekey=3
        else:
            prekey=2
    
    new_key = context_to_key(context,prekey)
    if str(new_key[-1])!=str(context[7]):
        print('oops')
        print(new_key[-1])
        print(context[7])
              
    #print(new_key[-1])
    dist_key = matched[new_key]
    #if len(dist_key)>17:
        #print(dist_key)

    dist = dists[dist_key]
    #new_context = modify_context_no_dosage(context)
    
    #context_key = '-'.join([str(c) for c in new_context[:7]])
    #possible_keys = get_possible_keys(new_context[:7])
    #get_possible_keys(new_context)
    #get_other_key(context)
    #get_possible_keys(new_context)
    #keys = [context_key]
    #keys = []
    #keys.extend(possible_keys)
    
    
    
    #keys = possible_keys
    #print(keys)
    #keys = [k+'-{}'.format(action) for k in keys]
    #print(keys)
    #i=0
    #while keys[i] not in dists:
        #print(i)
        #i=i+1
    #print(keys[i])
    #print(keys[-1])
    #dist = dists[keys[i]]
    
    #dist = dists['{}-mean'.format(context[0])]
    
    #if i==len(keys)-1:
    #    print(keys)
    #    print(new_context)
     #   print( context)
    
    x = np.random.normal(loc=dist[0],scale=dist[1])
    while(x<0):
         x = np.random.normal(loc=dist[0],scale=dist[1])
    return x



def get_steps_action(context,action):
    #nkey = '{}-{}'.format(action,context)
    print('here')
   # print(context)
    #print(action)
    
    this_context = [action]
    this_context.extend(context)
    possible_keys = get_possible_keys(this_context)
    
    context_key = '-'.join([str(c) for c in this_context])
    
    keys = [context_key]
    keys.extend(possible_keys)
    #print(keys)
    #keys = [k+'-{}'.format(action) for k in keys]
    #print(keys)
    i=0
    while keys[i] not in intervention_dists:
        #print(i)
        i=i+1
    #print(keys[i])
    #print(keys[-1])
    dist = intervention_dists[keys[i]]
    
    
    x = np.random.normal(loc=dist[0],scale=dist[1])
    while(x<0):
         x = np.random.normal(loc=dist[0],scale=dist[1])
    return x
    
def modify_context_no_dosage(context):
    
    lkeys = ['group_id','time_of_day','pretreatment','day_of_week','location','variation','yesterday','weather','dosage'  ]
    
    kl = {i:lkeys[i] for i in range(len(lkeys))}
    
    ##dosage at end
    new_context = [context[i] for i in range(len(lkeys)) if kl[i]!='dosage' ]
    
    return new_context


def get_steps(context,action,duration):
    
    
    
    if action==-1:
        return get_steps_no_action(context,duration)

    return get_steps_action(context,action)


def get_next_location(context):
    
    with open('{}location_conditiononed_on_last_location_hwo.pkl'.format(root),'rb') as f:
        loc_dists =pickle.load(f)
    
    #relevant_context = [context[get_index('group_id')],context[get_index('day_of_week')],context[get_index('time_of_day')],context[get_index('location')]]
    
    context_key = '-'.join([str(c) for c in context])
    possible_keys = get_possible_keys(context)
    
    keys = [context_key]
    keys.extend(possible_keys)
    #print(possible_keys)
    i=0
    #print(keys[-1])
    while  i<len(keys) and keys[i] not in loc_dists:
        i=i+1
    if i==len(keys):
        context_key = '-'.join([str(c) for c in context[:-1]])
        with open('{}initial_location_distributions_est_hwo.pkl'.format(root),'rb') as f:
            loc_lookup = pickle.load(f)
        dist = loc_lookup[context_key]
        val = np.argmax(np.random.multinomial(10,dist))
        
    else:
        dist = loc_dists[keys[i]]
    
        val = np.argmax(np.random.multinomial(10,dist))
    
    return val
            
                
    
    
def get_next_weather(context,month):
    #print('weather changed')
    with open('{}temperature_conditiononed_on_last_temperature_est.pkl'.format(root),'rb') as f:
        loc_dists =pickle.load(f)
    

    
    relevant_context = [context[get_index('time_of_day')],str(month),context[get_index('weather')]]
    
    context_key = '-'.join([str(c) for c in relevant_context])
    possible_keys = get_possible_keys(relevant_context)
    
    keys = [context_key]
    keys.extend(possible_keys)
    #print(keys)
    i=0
    #print(keys[-1])
    while keys[i] not in loc_dists and i<len(keys):
        i=i+1
    dist = loc_dists[keys[i]]
    
    val = np.argmax(np.random.multinomial(1,dist))
    
    return val

def get_pretreatment(steps):
    steps = math.log(steps+.5)
    #chunks =  [[0, 117.],[ 117.,330.],[330.,759.8],[759.8,100000000]]
    chunks =  [[-.7, 3.23867845],[ 3.23867845,4.95229972],[4.95229972,5.95713187],[5.95713187,100000000]]
    
    #for i in range(len(chunks)):
    #    if steps>=chunks[i][0] and steps<chunks[i][1]:
    #        return i
        
    return int(steps>math.log(.5))
        
##what do I need here
def get_new_dosage(current_dosage,action):
    if action==1:
        current_dosage = current_dosage+2
    else:
        current_dosage=current_dosage-1
    if current_dosage>100:
        current_dosage=100
    if current_dosage<1:
        current_dosage=1 
    return int(current_dosage)

def get_context_revised(current_index,current_context,current_steps,decision_time,ysc,variation,last_action):
        
    day_of_week = get_day_of_week(current_index)
    time_of_day = get_time_of_day(current_index)
    
    
    #new_ysc = get_new_lsc(current_steps)
    
    
    if decision_time:
        location = get_next_location([current_context[get_index('group_id')],day_of_week,time_of_day,current_context[get_index('location')]])
    
        dosage = get_new_dosage(current_context[get_index('dosage')],last_action)
        
        weather = get_next_weather(current_context,current_index.month)
        
        pretreatment_new = get_pretreatment(current_steps)
        
        
    else:
        location = current_context[get_index('location')]
        dosage = current_context[get_index('dosage')]
        weather = current_context[get_index('weather')]
        pretreatment_new = get_pretreatment(current_steps)
        
        #'group_id','day_of_week','time_of_day','weather','location','dosage','yesterday','pretreatment','variation
    #'group_id','day_of_week','time_of_day','yesterday','weather','location','dosage','pretreatment','variation'
    #'group_id','day_of_week','time_of_day','yesterday','pretreatment','location','variation','weather','dosage'
    return [current_context[0],time_of_day,  pretreatment_new,day_of_week,\
    location,variation,ysc,    weather,dosage,  ]

def to_yid(steps):
    with open('{}yesterday_step_ids.pkl'.format(root),'rb') as f:
         yesterday_chunks  = pickle.load(f)

    
    for i in range(len(yesterday_chunks)):
        if steps>=yesterday_chunks[i][0] and steps<yesterday_chunks[i][1]:
            return i
    
def get_new_lsc(step_slice):
    ##should threshold (is thresholded elsewhere?)
    #print('hi there')
    s =sum(step_slice)**.5
    return to_yid(s)
    if s<0:
        return 0
    if s>203:
        return 203
    return to_yid(s)




def get_variation_pre_week(variation,all_steps,time_indices,i):
    
    two_days = time_indices[0].date()+pd.DateOffset(days=1)
    is_set = False
    #first_index_last_day=-1
    if i>two_days:
        #print(i)
        for j in range(len(time_indices)):
            if time_indices[j].date()==i.date():
                if not is_set:
                    #print('set')
                    #print(j)
                    first_index_last_day = j
                    #print(first_index_last_day)
                    is_set = True
            if time_indices[j]== i:
                last_index_last_day = j
    
        pre_steps = all_steps[:first_index_last_day]
        post_steps = all_steps[first_index_last_day:last_index_last_day]
        
        #print(pre_steps)
        #print(post_steps)
        
        return int(np.array(pre_steps).std()>np.array(post_steps).std())
        
    else:
        return variation
    
def get_day_indices(indices):
    
    days = set([])
    day_indices = []
    
    to_return = {}
    
    lookup = {}
    
    j=0
    for i in indices:
        if i.date() not in to_return:
            to_return[i.date()]=[]
            #days.add(i.date())
            day_indices.append(i)
            
        to_return[i.date()].append(i)
        lookup[i]=j
        j=j+1
    return day_indices,to_return,lookup
    

def get_variation(all_steps,time_indices,i):
    
    #two_days = time_indices[0].date()+pd.DateOffset(days=1)
    is_set = False
    
    day_indices,dd,lookup = get_day_indices(time_indices)
    
    one_week_ago = i.date()-pd.DateOffset(days=7)
    #print(i)
    #print(one_week_ago)
    
    #print(all_steps)
    
    ##same length?
    #first_index,middle_index,last_index = get_indices(time_indices,one_week_ago,)
    
    last_index = dd[one_week_ago.date()][0]
    last = all_steps[lookup[last_index]:lookup[dd[i.date()][0]]][:-1]
    
    #print(dd[i.date()][0])
    c = all_steps[lookup[dd[i.date()][0]]:lookup[dd[i.date()][-1]]]
            #return c
  
        
        #print(pre_steps)
        #print(post_steps)
        
    return int(np.array(last).std()>np.array(c).std())



def transform_to_required(context):
    #'group_id','time_of_day','pretreatment','day_of_week','location','variation','yesterday','weather','dosage'
    weather_id = context[get_index('weather')]
    pretreatment_id = context[get_index('preatreatment')]
    #location_id = context[get_index('location')]
    #variation_id = context[get_index('variation')]
    
    yesterday_id = context[get_index('yesterday_id')]
    

    
#will be algorithm, needs to communicate with algorithm
#will be algorithm, needs to communicate with algorithm
def get_action(initial_context,steps,action_algorithm):
    return -1
    if action_algorithm==None:
        
        available = random.random()>.8
        
        if available:
            
        
            return int(random.random()>.4)
        return 0
    elif action_algorithm=='TS':
        algo_input = get_input(action_algorithm,context)

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
                        variation = get_variation(all_steps,time_indices,last_day)
                
                
                    lsc = get_new_lsc(all_steps[start_of_day:end_of_day])
                #variation = get_new_variation()
                
            
            ##action will be the last action
                my_context = get_context_revised(i,initial_context,hour_steps,decision_time,lsc,variation,action)
            #return my_context
                if i in decision_times:
                    #print('decision time')
                    action = get_action(my_context,hour_steps,action_algorithm)
                #print(action)
                else:
                    action = -1
            ##redo get_steps
                next_steps = get_steps(my_context,action,duration) 
                all_steps.append(next_steps)
                initial_context = my_context
                steps_last_half_hour=current_steps
                current_steps = next_steps
                
            else:
                if i in decision_times:
                    #print('type two decision time')
                    action = get_action(initial_context,hour_steps,action_algorithm)
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


