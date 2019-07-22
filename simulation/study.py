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
    
    def __init__(self,root,population_size,study_length,which_gen='case_one',sim_number = None):
        #root =  '../../../../Volumes/dav/HeartSteps/pooling_rl_shared_data/processed/'
        self.root =root
            #'../../murphy_lab/lab/pooling/distributions/'
        self.study_seed = sim_number+30000000
        self.sim_number = sim_number
        self.algo_rando_gen = np.random.RandomState(seed=8000000)
        self.weather_gen = np.random.RandomState(seed=9000000)
        
        #self.rando_gen = np.random.RandomState(seed=sim_number+2000)
        with open('{}person_to_time_indices_pop_{}{}.pkl'.format(root,population_size,study_length),'rb') as f:
            pse=pickle.load(f)
        with open('{}person_to_decision_times_pop_{}{}.pkl'.format(root,population_size,study_length),'rb') as f:
            dts=pickle.load(f)
        with open('{}time_to_active_participants_pop_{}{}.pkl'.format(root,population_size,study_length),'rb') as f:
            dates_to_people=pickle.load(f)
        with open('{}all_ordered_times{}.pkl'.format(root,study_length),'rb') as f:
            study_days=pickle.load(f)
        
        self.person_to_time = pse 
        
        self.person_to_decision_times = dts 
        
        self.dates_to_people = dates_to_people 
        
        self.iters = []
        
        self.study_days = study_days 
        
        self.population = {}
    
        self.algo_seed = 2000
        
        self.history = {}
        
        self.weather_update_hours = [9,12,15,18,20]
        
        self.location_update_hours = set([9,12,16,18,20])
        
        self.update_hour = 0
        self.update_minute = 30
        self.last_update_day = study_days[0]
        self.study_length=study_length
        self.Z_one =0.0
            #-0.10736186999999998
        self.Z_two =-0.35
            #0.10736186999999998
            #0.5922135199999999
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
#0.0977, 0.0858,   0.111,  .181,  -0.217, 0.255

        #full states only
        #self.beta = np.array([0.12161856941717532, -0.08655013,  0.09176659,  0.18781728, -0.29102962, 0.07076406])
          #independent only
           #np.array([0.0977, 0.0858,0.111,0.181,-0.217,0.255])
           #  real weather 0.24941052, real location 0.22540878
           
           
        ##real current agreed upon beta
        #self.beta =np.array([ 0.05,  0.25,  0.25,  0.25, -0.3 ])
        
        
        
            #np.array([ 0.05,  0.25, -0.3 ,  0.25,  0.25, -0.3 ])
            #np.array([0.05,  0.25,  0.25,  0.25, -0.3])
#np.array([0.13747917218640332, -0.08988142,   0.11982505, -0.16109622, 0.10403158])
            #np.array([0.0912183, -0.11717383,  0.15143829, -0.1007128,  0.3115448])
            #np.array([ 0.05, -0.1 ,  0.1 , -0.15,  0.2 ])
        #these are redone without weather and then always good to send....
            #np.array([0.07912183,  0.11717383,  0.12143829, -0.07507128,  0.25115448])
        #np.array([0.04772972,  0.0696645 ,  0.21549613, 0.24941052 ,-0.22475609 , 0.22540878])
        #np.array([0.05,  -0.1,  0.05,  0.2	 , -0.3,  0.6 ])
            #np.array([0.25,  -0.1 ,  0.25, -0.1 ,-0.1 ,0.25])
            #np.array([0.04772972,  0.0696645 ,  0.21549613, 0.24941052 ,-0.22475609 , 0.22540878])
        #np.array([0.14225149398166387 , -0.11665832,  0.11808621,  0.15601783, -0.18131626,0.01280371])


            
            #np.array([0.12161856941717532, -0.08655013,  0.09176659,  0.18781728, -0.29102962, 0.07076406])
        #np.array([0.14225149398166387 , -0.11665832,  0.11808621,  0.15601783, -0.18131626,0.01280371])

        #np.array([0.12161856941717532, -0.08655013,  0.09176659,  0.18781728, -0.29102962, 0.07076406])
        #np.array([0.0977, 0.0858,0.111,0.181,-0.217,0.255])
        #hybrid
        ##original from table
        #np.array([0.14225149398166387 , -0.11665832,  0.11808621,  0.15601783, -0.18131626,
        #0.01280371])
        #self.beta =np.array([-.75,.27,.14,-.04])
        #old
            #np.array([-0.88722  ,1.99952,0.23429])
        self.sigma =.33
            #.45
            #.38
            #.325**.5
            #0.325
            #.325**.5
            #0.125
            #0.12244165000000001
            #0.6304924999999999
    
        self.init_population(which_gen,True)
            
    def get_gid(self):
        #4.0/36
        #return 2
         return int(random.random()>=.5)+1
    
    def update_beta(self,features):
        #self.beta =np.array([ 0.05,  0.25,  0.25,  0.25, -0.3 ])
        #0.05,  0.3 ,  0.3 , -0.35
        self.beta =np.array([   0.05,  0.25,  0.25,  -0.3, 0.25])
        
        potential_features = ['intercept','tod','dow','pretreatment','location']
        new = np.array([self.beta[0]]+[self.beta[i] for i in range(len(self.beta)) if potential_features[i] in features])
        #self.beta = beta
        self.regret_beta = new
    
    
    
    
    def init_population(self,which_gen,location):
         
        for k,v in self.person_to_time.items():
            
            ##Get GID 
            
            #gid = self.get_gid()
            


            person_seed = k+self.sim_number*1000
            rg=np.random.RandomState(seed=person_seed)
            
            gid = int(rg.uniform()>=.5)+1
            #print('init')
            
            Z=None
            if which_gen=='case_two':
                Z = self.Z_one
                if gid==2:
                    Z=self.Z_two
            if which_gen=='case_three':
                                        #**2
                                        #
                Z=rg.normal(loc=0,scale=self.sigma)
            
            
            #print(k)
            #print(self.sim_number)
            ##this_beta = [i for i in [  0.05,  0.25,  0.25,  0.25, -0.3]]
            ##if gid==1:
            
                ##this_beta[-1]=-1*this_beta[-1]
                #this_beta[2]=this_beta[2]+Z/2
                #this_beta[3]=this_beta[3]+Z/2
            this_beta = [i for i in [  0.05,  0.25,  0.25,  -0.3, 0.25]]
            if location:
                if which_gen=='case_two':
                    offset = .25
                    if gid==2:
                        offset = offset*-1
                    this_beta[-1]=this_beta[-1]+offset
                if which_gen=='case_three':
                                                    
                    l=rg.normal(loc=-0.3,scale=0.25)
                    this_beta[-1]=this_beta[-1]+l
            
            person = participant.participant(pid=k,gid=gid,times=v,decision_times = self.person_to_decision_times[k],Z=Z,rg=rg,beta=np.array(this_beta))
            #print(person.rando_gen.uniform())
#print(person.beta)
#print(Z)
            self.population[k]=person



