import pickle
import participant
import random

class study:
    
    
    
    '''
    A population is an object which tracks participants. 
    It also tracks the total number of active days. 
    Also which participants are involved at which times. 
    '''
    
    def __init__(self,root,population_size,study_length):
        #root =  '../../../../Volumes/dav/HeartSteps/pooling_rl_shared_data/processed/'
        self.root =root
            #'../../murphy_lab/lab/pooling/distributions/'
        
        with open('{}person_to_time_indices_pop_{}_{}.pkl'.format(root,population_size,study_length),'rb') as f:
            pse=pickle.load(f)
        with open('{}person_to_decision_times_pop_{}_{}.pkl'.format(root,population_size,study_length),'rb') as f:
            dts=pickle.load(f)
        with open('{}time_to_active_participants_pop_{}_{}.pkl'.format(root,population_size,study_length),'rb') as f:
            dates_to_people=pickle.load(f)
        with open('{}all_ordered_times_{}.pkl'.format(root,study_length),'rb') as f:
            study_days=pickle.load(f)
        
        self.person_to_time = pse 
        
        self.person_to_decision_times = dts 
        
        self.dates_to_people = dates_to_people 
        
        
        self.study_days = study_days 
        
        self.population = {}
    
        self.init_population()
        
        self.history = {}
        
        self.weather_update_hours = [6,10,12,16,20]
        
        self.location_update_hours = set([9,12,16,18,20])
        
        self.update_hour = 0
        self.update_minute = 30
        self.last_update_day = study_days[0]
        
    def get_gid(self):
        #4.0/36
         return int(random.random()>.5)+1
    
    def init_population(self):
         
        for k,v in self.person_to_time.items():
            
            ##Get GID 
            
            gid = self.get_gid()
            
            person = participant.participant(pid=k,gid=gid,times=v,decision_times = self.person_to_decision_times[k])
            
            self.population[k]=person
