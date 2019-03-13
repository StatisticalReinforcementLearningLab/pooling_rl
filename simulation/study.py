import pickle
import participant
import random
import numpy as np

class study:
    
    
    
    '''
    A population is an object which tracks participants. 
    It also tracks the total number of active days. 
    Also which participants are involved at which times. 
    '''
    
    def __init__(self,root,population_size,study_length,which_gen='case_one'):
        #root =  '../../../../Volumes/dav/HeartSteps/pooling_rl_shared_data/processed/'
        self.root =root
            #'../../murphy_lab/lab/pooling/distributions/'
        
        with open('{}person_to_time_indices_pop_{}_{}_unstaggered.pkl'.format(root,population_size,study_length),'rb') as f:
            pse=pickle.load(f)
        with open('{}person_to_decision_times_pop_{}_{}_unstaggered.pkl'.format(root,population_size,study_length),'rb') as f:
            dts=pickle.load(f)
        with open('{}time_to_active_participants_pop_{}_{}_unstaggered.pkl'.format(root,population_size,study_length),'rb') as f:
            dates_to_people=pickle.load(f)
        with open('{}all_ordered_times_{}_unstaggered.pkl'.format(root,study_length),'rb') as f:
            study_days=pickle.load(f)
        
        self.person_to_time = pse 
        
        self.person_to_decision_times = dts 
        
        self.dates_to_people = dates_to_people 
        
        
        self.study_days = study_days 
        
        self.population = {}
    
    
        
        self.history = {}
        
        self.weather_update_hours = [9,12,15,18,20]
        
        self.location_update_hours = set([9,12,16,18,20])
        
        self.update_hour = 0
        self.update_minute = 30
        self.last_update_day = study_days[0]
        self.study_length=study_length
        self.Z_one = -0.0773
        self.Z_two =0.647
            #1.5265399999999998
        #intercept,tod,dow,weather,previous steps,loc 1,loc 2,loc 3
        #took location out
        #,0.03189,0.03189,   0.14462
        #took weather out
        #,0.23429
        #dow
        #, -0.28816
        #tod
        #0.03533
        #-0.38
        #tod = -.138
        #intercept,pre,wea,dow,tod
#, -0.06405887,  0.09447067,  0.16481901, -0.20736237,
#0.01136896
#weighted
#self.beta =np.array([.13, -0.06405887,0.09447067,  0.16481901,-0.20736237,0.01136896])
        self.beta = np.array([0.0977,0.08576116, 0.11150488,0.18145672, -0.21754123,0.25537478])
        #self.beta =np.array([-.75,.27,.14,-.04])
        #old
            #np.array([-0.88722  ,1.99952,0.23429])
        self.sigma =0.191575
            #0.6304924999999999
    
        self.init_population(which_gen)
            
    def get_gid(self):
        #4.0/36
        #return 2
         return int(random.random()>=.5)+1
    
    def init_population(self,which_gen):
         
        for k,v in self.person_to_time.items():
            
            ##Get GID 
            
            gid = self.get_gid()
            
            Z=None
            if which_gen=='case_two':
                Z = self.Z_one
                if gid==2:
                    Z=self.Z_two
            if which_gen=='case_three':
                #**2
                Z=np.random.normal(loc=0,scale=self.sigma)

            
            
            person = participant.participant(pid=k,gid=gid,times=v,decision_times = self.person_to_decision_times[k],Z=Z)
            
            self.population[k]=person
