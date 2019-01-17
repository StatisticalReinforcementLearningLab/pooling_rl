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
        
    
    
         