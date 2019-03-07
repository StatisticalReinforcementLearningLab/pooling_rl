import pandas as pd
import numpy as np
import pickle

import os
import math
from numpy.random import uniform
import random
from datetime import datetime
random.seed(datetime.now())
from scipy.stats import halfnorm
import sim_functions_cleaner as sf
import feature_transformations as tf


def new_kind_of_simulation(experiment,policy=None,tf = None):
    #write_directory = '../../murphy_lab/lab/pooling/temp'
    experiment.last_update_day=experiment.study_days[0]
    for time in experiment.study_days:
   
    
        
    
        tod = sf.get_time_of_day(time)
        dow = sf.get_day_of_week(time)
        if time==experiment.study_days[0]:
            print('init weather')
            weather = tf.get_weather_prior(tod,time.month)
            temperature = tf.continuous_temperature(weather)
        elif time.hour in experiment.weather_update_hours and time.minute==0:
            weather = tf.get_next_weather(str(tod),str(time.month),weather)
            temperature = tf.continuous_temperature(weather)
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
                    location = tf.get_location_prior(str(participant.gid),str(tod),str(dow))
                    participant.set_loc(location)
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
                    location = tf.get_next_location(participant.gid,tod,dow,participant.get_loc())
                
                    participant.set_loc(location)
                    
                    
                    prob = -1
            if time in participant.decision_times:
                    #print(personal_policy_params.batch_index[participant.pid])
                    
                    
                    ##if we have made no global updates
                    
                    
                    
                    
                       dt=True
                       action=-1
                    
                    
                    
                    
                    #if policy==None:
                    #action = sf.get_action(policy)
                    
                    
                    
                  
                        
                        
                       
                        
            steps = tf.get_steps_no_action(participant.gid,tod,dow,location,weather,sf.get_pretreatment(steps_last_time_period))
            participant.steps = steps
                    
                    
                    
                                    
                                 
                                            
                       
                                            
            context_dict =  {'steps':steps,'action':action,'continuous_temp':temperature,'weather':weather,'location_1':int(location==1),\
                                                'ltps':steps_last_time_period,'location_2':int(location==2),'location_3':int(location==3),\
                                                    'study_day':participant.current_day_counter,'decision_time':dt,'time':time,\
                                                        'avail':availability,'temperature':temperature,'prob':prob,'dow':dow,'tod':tod,'pretreatment':sf.get_pretreatment(steps_last_time_period)}
            participant.history[time]=context_dict







