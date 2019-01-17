import numpy as np
import pandas as pd
import math

class participant:
    
    
    
    '''
    A population is an object which tracks participants. 
    It also tracks the total number of active days. 
    Also which participants are involved at which times. 
    '''
    
    def __init__(self,pid=None,times=None,decision_times=None,gid=None):
        self.root =  '../../../../Volumes/dav/HeartSteps/pooling_rl_shared_data/processed/'
        
        self.pid = pid
        
        self.times =  times
        
        self.decision_times = decision_times
        
        self.gid = gid 
        
        self.current_day = None
        
        self.first_week = None
        
        self.history = {}
        
        self.tod = None
        self.dow = None
        self.weather = None
        self.location = None
        
        ##intervention related states
        self.ltps = None
        self.pds = None
        self.variation = None

    def set_current_day(self):
        pass
    
    def get_current_day(self):
        pass

        
    def set_tod(self,tod):
        self.tod=tod
    
    def get_tod(self):
        return self.tod
        
            
    def set_dow(self,dow):
        self.dow = dow
    
    def get_dow(self):
        return self.dow
        
    def set_wea(self,wea):
        self.weather = wea
    
    def get_wea(self):
        return self.weather
            
    def set_loc(self,location):
        self.location = location
    
    def get_loc(self):
        return self.location
    
    def set_last_time_period_steps(self,steps):
        self.ltps = steps
        
    def set_yesterday_steps(self,steps):
        self.pds = steps
        
    def set_variation(self,var):
        self.variation = var
    
    
    def get_pretreatment(self,steps):
        steps = math.log(steps+.5)
        return int(steps>math.log(.5))
        
    def find_last_time_period_steps(self,timestamp):
        ##looking at the last hour
        #pass
        thirty_minutes_ago = timestamp-pd.Timedelta(minutes=30)
        one_hour_ago = timestamp-pd.Timedelta(minutes=60)
        
        steps_thirty = self.history[thirty_minutes_ago]['steps']
        steps_hour = self.history[one_hour_ago]['steps']
        return self.get_pretreatment(steps_thirty+steps_hour)
        
    def find_yesterday_steps(self,time):
        yesterday = time-pd.DateOffset(days=1)
        
        y_steps = [self.history[t]['steps'] for t in self.history.keys() if t.date()==\
                  yesterday]
        
        y_steps = sum(y_steps)**.5
        return y_steps
    
    
    def get_day_steps_raw(self,day):
        
        
        y_steps = [self.history[t]['steps'] for t in self.history.keys() if t.date()==\
                  day.date()]
        return y_steps
    
    def steps_range(self,time_one,time_two):
        
        y_steps = [self.history[t]['steps'] for t in self.history.keys() if t>=time_one and t<time_two]
        return y_steps
    
    def get_day_slices(self,start,end):
        days = pd.date_range(start = start,end =end,freq='D')
        return days
    
    
    def find_variation(self,time):
         
        if time<self.times[0].date()+pd.DateOffset(days=8):
            start = self.times[0]
        else:
            start = time-pd.DateOffset(days=8)

        end = time-pd.DateOffset(days=2)        
        days = self.get_day_slices(start,end)
        stds = [np.array(self.get_day_steps_raw(d)).std() for d in days]
        median = np.median(np.array(stds))
        
        yesterday = time-pd.DateOffset(days=1)
        yesterday_steps = np.array(self.get_day_steps_raw(yesterday)).std()
        
        #print(days)
        #print(start)
        #print(end)
        #print(time)
        #print(yesterday)
        #print(len(stds))
        #print(len(self.get_day_steps_raw(yesterday)))
        
        return int(yesterday_steps>median)
        
        
         