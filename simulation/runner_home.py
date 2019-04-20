import pandas as pd
import sys
sys.path
sys.path.append('../../../home00/stomkins/pooling_rl/models')
sys.path.append('../../../home00/stomkins/pooling_rl/simulation')
#sys.path.append('../pooling_rl/simulation')
import sim_all





if __name__=="__main__":
    
    ##parse command line arguments
    
    population = sys.argv[1]
    update_time = sys.argv[2]
    study_length = sys.argv[3]
    start_index = sys.argv[4]
    end_index = sys.argv[5]
    case =sys.argv[6]
    root = '../Downloads/distributions/'
    write_directory = '../Downloads/results_4_18/'
    sim_all.run_many('pooling',[case],int(start_index),int(end_index),int(update_time),root,write_directory)

