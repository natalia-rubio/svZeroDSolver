import numpy as np
from scipy.interpolate import interp1d

class Inflow():
    ''' Handles inflow and inflow files'''
    def __init__(self, inflow_file, inverse = False, smooth = True, n_points = 1000):
        if inflow_file is not None:
            self.inflow = np.loadtxt(inflow_file)
        else:
            self.inflow = None
            return
        
        self.check_valid_flow()
        
        if inverse:
            self.inverse_flow()
        if smooth:
            self.smooth_flow(n_points)
        
        self.compute_vals()

        
    def compute_vals(self):
        self.t = self.inflow[:, 0]
        self.Q = self.inflow[:, 1]
        # cycle length
        self.tc = (self.t[-1] - self.t[0])
        
        self.mean_inflow = np.trapz(self.Q, self.t) / self.tc
        self.max_inflow = self.Q.max()
        self.min_inflow = self.Q.min()
    
    def write_flow(self, flowpath):
        np.savetxt(flowpath, self.inflow)
    
    def inverse_flow(self):
        '''inverse the pos and negative flow values'''
        self.inflow[:, 1] *= -1

    def check_valid_flow(self):
        if self.inflow[0, 1] - self.inflow[-1, 1] != 0:
            time_diff = self.inflow[1, 0] - self.inflow[0,0]
            self.inflow = np.append(self.inflow, np.array([[time_diff + self.inflow[-1, 0], self.inflow[0, 1]]]), axis = 0)
            
            
    def smooth_flow(self, n_points):

        f = interp1d(self.inflow[:, 0], self.inflow[:, 1], kind = 'cubic')
        x = np.linspace(self.inflow[0, 0], self.inflow[-1, 0], n_points)
        y = f(x)
        
        self.inflow = np.array(list(zip(x, y)))
          
    def plot_flow(self, output_file):
        
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            print(e, ': aborting plot')
        
        ## save inflow graph
        fig,ax = plt.subplots(1,1 )
        ax.plot(self.t, self.Q)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('flow (ml/s)')
        fig.savefig(output_file)