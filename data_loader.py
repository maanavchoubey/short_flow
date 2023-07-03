import numpy as np
import time 
import os
import torch 
from torch.utils.data.dataset import Dataset 
from torch.utils.data import DataLoader
from config import PREVIOUS_FRAMES

class LoadDataShort(Dataset):
    def __init__(self, seeds, t):
        self.seeds = torch.from_numpy(seeds) if isinstance(seeds, np.ndarray) else seeds
        self.t = torch.FloatTensor([t])
        
    def __len__(self):
        return 1  # We have only one sample in the dataset
    
    def __getitem__(self, idx):
        return self.seeds, self.t
    
class LoadData(Dataset):
    '''
        This if for small datasets
    '''
    def stats(self):
        print("x range({},{}) y range({},{}) t range ({}. {})"
              .format(self.x_min,self.x_max,self.y_min,self.y_max, self.t_min, self.t_max))

    def __init__(self, data_dir):
        self.data_dir = data_dir
        try:
            start_ = 0
            if data_dir == "./data/500_sobol.npy" or data_dir== "./data/5000_sobol.npy" or data_dir == "./data/500_short.npy" or data_dir== "./data/5000_short.npy":
                end_ = 62
            elif data_dir == "./data/doublegyre_test513.npy" or data_dir == "./data/doublegyre_train513.npy" or data_dir == "./data/doublegyre_static.npy" or data_dir == "./data/doublegyre_static_test.npy":
                end_ = 514
            #end_ = 62
            # print("Loading: 100-200")
            # data = np.load(self.data_dir)[200:300, :, 0:2]
            data = np.load(self.data_dir)[start_:end_, :, 0:2]
            # data = np.load(self.data_dir)
            print("Start and end time", start_,end_)
        except:
            data = np.load(self.data_dir)
        print(data.shape)
        self.data = []
        x_min = np.min(data[:, :, 0])
        x_max = np.max(data[:, :, 0])
        y_min = np.min(data[:, :, 1])
        y_max = np.max(data[:, :, 1])
        print("x range({},{}) y range({},{})".format(x_min,x_max,y_min,y_max))
        t_min = 0
        t_max = data.shape[0] - 1

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max 
        self.t_min = t_min
        self.t_max = t_max
        self.stats()

        ## scale to [0, 1]
        minval = 0
        maxval = 1
        data[:, :, 0] = (data[:, :, 0] - x_min) / (x_max - x_min) * (maxval - minval) + minval
        data[:, :, 1] = (data[:, :, 1] - y_min) / (y_max - y_min) * (maxval - minval) + minval
        
        for j in range(data.shape[1]):
            trajectories = data[:, j, :]
            num_fm = data.shape[0] - 1
            # num_fm = 10
            for i in range(0, num_fm):
                end = trajectories[i+1, :]
                t = (i - t_min) / (t_max - t_min) * (maxval - minval) +  minval ## start time       
                seed = trajectories[0, :]
                self.data.append({
                        "start": torch.FloatTensor(seed),
                        "end": torch.FloatTensor(end),
                        "time": torch.FloatTensor([t])
                        })
                
        ''' Use this for separate X & Y

        for j in range(data.shape[1]):
            trajectories = data[:, j, :]
            num_fm = data.shape[0] - 1
            # num_fm = 10
            for i in range(0, num_fm):
                end = trajectories[i+1, :]
                t = (i - t_min) / (t_max - t_min) * (maxval - minval) +  minval ## start time       
                seed_x = trajectories[0, 0]
                seed_y = trajectories[0, 1]
                self.data.append({
                        "start_x": torch.FloatTensor(seed_x),
                        "start_y": torch.FloatTensor(seed_y),
                        "end": torch.FloatTensor(end),
                        "time": torch.FloatTensor([t])
                        })
                        
                        '''
    def __len__(self):
        # print("total data size", len(self.data))
        return len(self.data)
    
    def __getitem__(self,index):
        np.random.seed(seed = int(time.time() + index))
        data = self.data[index]
        start = data["start"]
        '''
        Uncomment for MLP_Short
        start_x = data["start_x"]
        start_y = data["start_y"]
        '''
        end = data["end"]
        t = data["time"]
        #return start_x, start_y, end, t
        return start, end, t

class LoadData_StartEnd(LoadData):
    '''
        Description: This function allows the user to generate a subset of 
        simulation data by specifying the starting and ending indices. 
        It's useful when the model is not able to predict trajectories beyond 
        a certain length, and the user wants to limit the data to a specific range of timesteps.
        @Params
            - data_dir: directory of the data
            - start: starting index
            - end: ending index
        For example: In the dataset total trajector is 0-1000. The model can not predict long trajectories.
        In that case give start indexs as 0 and end as 100. To generate data for first 100 timesteps
    '''
    def __init__(self, data_dir,start,end, seed_points=None):
        self.data_dir = data_dir
        try:
            data = np.load(self.data_dir)
            data = data[start:end,:, 0:2]
            print("checking",start,end)
            if seed_points is not None:
                print("Addeding manual seed points, This is for eval")
                data[0,:,:] = seed_points
        except Exception as e:
            print(e)
            data = np.load(self.data_dir)
        print(data.shape)
        self.data = []
        x_min = np.min(data[:, :, 0])
        x_max = np.max(data[:, :, 0])
        y_min = np.min(data[:, :, 1])
        y_max = np.max(data[:, :, 1])
        x_min = 0
        x_max = 1
        y_min = 0
        y_max = 1

        t_min = 0
        t_max = data.shape[0] - 1

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max 
        self.t_min = t_min
        self.t_max = t_max
        self.stats()

        ## scale to [0, 1]
        minval = 0
        maxval = 1
        data[:, :, 0] = (data[:, :, 0] - x_min) / (x_max - x_min) * (maxval - minval) + minval
        data[:, :, 1] = (data[:, :, 1] - y_min) / (y_max - y_min) * (maxval - minval) + minval

        
        for j in range(data.shape[1]):
            trajectories = data[:, j, :]
            num_fm = data.shape[0] - 1
            # num_fm = 10
            for i in range(0, num_fm):
                end = trajectories[i+1, :]
                t = (i - t_min) / (t_max - t_min) * (maxval - minval) +  minval ## start time       
                seed = trajectories[0, :]
                self.data.append({
                        "start": torch.FloatTensor(seed),
                        "end": torch.FloatTensor(end),
                        "time": torch.FloatTensor([t])
                        })
    def __len__(self):
        # print("total data size", len(self.data))
        return len(self.data)
    
    def __getitem__(self,index):
        np.random.seed(seed = int(time.time() + index))
        data = self.data[index]
        start = data["start"]
        end = data["end"]
        t = data["time"]
        return start, end, t

class LoadData_StartEnd_Skip(LoadData):
    '''
        Description: This function allows the user to generate a subset of 
        simulation data by specifying the starting and ending indices. 
        It's useful when the model is not able to predict trajectories beyond 
        a certain length, and the user wants to limit the data to a specific range of timesteps.
        This function skips timestep as well (0,100,5)
        @Params
            - data_dir: directory of the data
            - start: starting index
            - end: ending index
        For example: In the dataset total trajector is 0-1000. The model can not predict long trajectories.
        In that case give start indexs as 0 and end as 100. To generate data for first 100 timesteps
        while skipping N skipping.
    '''
    def __init__(self, data_dir,start,end, seed_points=None):
        self.data_dir = data_dir
        try:
            data = np.load(self.data_dir)
            data = data[start:end,:, 0:2]
            print("checking",start,end)
            if seed_points is not None:
                print("Addeding manual seed points, This is for eval")
                data[0,:,:] = seed_points
        except Exception as e:
            print(e)
            data = np.load(self.data_dir)
        print(data.shape)
        self.data = []
        end_ = end

        x_min = 0
        x_max = 1
        y_min = 0
        y_max = 1
        t_min = 0
        t_max = data.shape[0] - 1

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max 
        self.t_min = t_min
        self.t_max = t_max
        self.stats()

        ## scale to [0, 1]
        minval = 0
        maxval = 1
        data[:, :, 0] = (data[:, :, 0] - x_min) / (x_max - x_min) * (maxval - minval) + minval
        data[:, :, 1] = (data[:, :, 1] - y_min) / (y_max - y_min) * (maxval - minval) + minval

        for j in range(data.shape[1]):
            trajectories = data[:, j, :]
            num_fm = data.shape[0] -1
            # num_fm = 10
            for i in range(0, num_fm, 5):
                if start== 0 and  end_ == 1000 and i + 5 >= 1000:
                    continue
                if start==750 and i + 5 >= 250:
                    continue
                # if j ==0:
                    # print(i)
                end = trajectories[i+5, :]
                t = (i - t_min) / (t_max - t_min) * (maxval - minval) +  minval ## start time       
                seed = trajectories[0, :]
                self.data.append({
                        "start": torch.FloatTensor(seed),
                        "end": torch.FloatTensor(end),
                        "time": torch.FloatTensor([t])
                        })
    def __len__(self):
        # print("total data size", len(self.data))
        return len(self.data)
    
    def __getitem__(self,index):
        np.random.seed(seed = int(time.time() + index))
        data = self.data[index]
        start = data["start"]
        end = data["end"]
        t = data["time"]
        return start, end, t

class LoadData_StartEndLong(Dataset):
    '''
        In this data loader we can set start and end
        but the seed points will always be x0 and y0 even if start and end are 100 200
        Use this only long method
    '''
    def __init__(self, data_dir,start,end, seed_points=None):
        self.data_dir = data_dir
        try:
            # print("Loading: 100-200")
            # data = np.load(self.data_dir)[200:300, :, 0:2]
            data = np.load(self.data_dir)
            if seed_points is not None:
                print("Addeding manual seed points, This is for eval")
                data[0,:,:] = seed_points

            print("checking",start,end)
        except Exception as e:
            print(e)
            data = np.load(self.data_dir)
        print(data.shape)
        self.data = []
        x_min = 0
        x_max = 1
        y_min = 0
        y_max = 1
        # x_min = np.min(data[start:end, :, 0])
        # x_max = np.max(data[start:end, :, 0])
        # y_min = np.min(data[start:end, :, 1])
        # y_max = np.max(data[start:end, :, 1])
        t_min = 0
        t_max = end - start -1
        ## scale to [0, 1]
        minval = 0
        maxval = 1
        data[:, :, 0] = (data[:, :, 0] - x_min) / (x_max - x_min) * (maxval - minval) + minval
        data[:, :, 1] = (data[:, :, 1] - y_min) / (y_max - y_min) * (maxval - minval) + minval
        data_seed = data[0:10,:,0:2]
        data = data[start:end,:, 0:2]
        
        for j in range(data.shape[1]):
            trajectories = data[:, j, :]
            trajectories_seed = data_seed[:, j, :]
            num_fm = data.shape[0] - 1
            # num_fm = 10
            for i in range(0, num_fm):
                end = trajectories[i+1, :]
                t = (i - t_min) / (t_max - t_min) * (maxval - minval) +  minval ## start time       
                seed = trajectories_seed[0, :]
                self.data.append({
                        "start": torch.FloatTensor(seed),
                        "end": torch.FloatTensor(end),
                        "time": torch.FloatTensor([t])
                        })
    def __len__(self):
        # print("total data size", len(self.data))
        return len(self.data)
    
    def __getitem__(self,index):
        np.random.seed(seed = int(time.time() + index))
        data = self.data[index]
        start = data["start"]
        end = data["end"]
        t = data["time"]
        return start, end, t

class LoadData_StartEnd_MultiTask(LoadData):
    '''
        Description: This Dataloader is used to generate dataset for multi task learning.
        @Params
            - data_dir: directory of the data
            - start: starting index
            - end: ending index
            - seed_points: starting point default none
            - rey: adding renolds number
    '''
    def __init__(self, data_dir,start,end, seed_points=None, rey=0.10):
        self.data_dir = data_dir
        try:
            # print("Loading: 100-200")
            # data = np.load(self.data_dir)[200:300, :, 0:2]
            data = np.load(self.data_dir)
            data = data[start:end,:, 0:2]
            print("checking",start,end)
            if seed_points is not None:
                print("Addeding manual seed points, This is for eval")
                data[0,:,:] = seed_points
        except Exception as e:
            print(e)
            data = np.load(self.data_dir)
        print(data.shape)
        self.data = []
        x_min = np.min(data[:, :, 0])
        x_max = np.max(data[:, :, 0])
        y_min = np.min(data[:, :, 1])
        y_max = np.max(data[:, :, 1])
        x_min = 0
        x_max = 1
        y_min = 0
        y_max = 1

        t_min = 0
        t_max = data.shape[0] - 1

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max 
        self.t_min = t_min
        self.t_max = t_max
        self.stats()

        ## scale to [0, 1]
        minval = 0
        maxval = 1
        data[:, :, 0] = (data[:, :, 0] - x_min) / (x_max - x_min) * (maxval - minval) + minval
        data[:, :, 1] = (data[:, :, 1] - y_min) / (y_max - y_min) * (maxval - minval) + minval

        
        for j in range(data.shape[1]):
            trajectories = data[:, j, :]
            num_fm = data.shape[0] - 1
            # num_fm = 10
            for i in range(0, num_fm):
                end = trajectories[i+1, :]
                t = (i - t_min) / (t_max - t_min) * (maxval - minval) +  minval ## start time       
                seed = trajectories[0, :]
                self.data.append({
                        "start": torch.FloatTensor(seed),
                        "end": torch.FloatTensor(end),
                        "time": torch.FloatTensor([t, rey])
                        })
    def __len__(self):
        # print("total data size", len(self.data))
        return len(self.data)
    
    def __getitem__(self,index):
        np.random.seed(seed = int(time.time() + index))
        data = self.data[index]
        start = data["start"]
        end = data["end"]
        t = data["time"]
        return start, end, t
    
class LoadPointsData2(Dataset):
    '''
        This is for large datasets. Use data generated by preprocessing.py
        Data.shape = (N,5) -> (X0,Y0,Xn,Yn,n)
        X0: X positing at 0th timestep
        Y0: Y position at -th timestep
        Xn: X position at nth timestep
        Yn: Y position at nth timestep
        n: timestep
        @Params:
            - data_dir: data directory
            - dim: number of dimenions
        
    '''
    def __init__(self, data_dir, dim):
        self.data_dir = data_dir
        self.dim = dim
        print(self.data_dir)
        data = np.load(self.data_dir)

        print("data shape", data.shape)
        num_samples = data.shape[0]
        self.length = data.shape[1]
        self.data = data

    def __len__(self):
        # print("total data size", len(self.data))
        return len(self.data)

    def __getitem__(self,index):
        np.random.seed(seed = int(time.time() + index))
        data = self.data[index]
        start = torch.FloatTensor(data[0:self.dim])
        end = torch.FloatTensor(data[self.dim : self.dim+ self.dim])
        t = torch.FloatTensor([data[self.length-1]])

        return start, end, t

class LoadPointsData3(Dataset):
    '''
        This is for large datasets for multitask learning. Use data generated by preprocessing.py
        Data.shape = (N,5) -> (X0,Y0,Xn,Yn,n, RN)
        X0: X positing at 0th timestep
        Y0: Y position at -th timestep
        Xn: X position at nth timestep
        Yn: Y position at nth timestep
        n: timestep
        Rn: Reynolds Number
        @Params:
            - data_dir: data directory
            - dim: number of dimenions
        
    '''
    def __init__(self, data_dir, dim):
        self.data_dir = data_dir
        self.dim = dim
        print(self.data_dir)
        data = np.load(self.data_dir)

        print("data shape", data.shape)
        num_samples = data.shape[0]
        self.length = data.shape[1]
        self.data = data

    def __len__(self):
        # print("total data size", len(self.data))
        return len(self.data)

    def __getitem__(self,index):
        np.random.seed(seed = int(time.time() + index))
        data = self.data[index]
        start = torch.FloatTensor(data[0:self.dim])
        end = torch.FloatTensor(data[self.dim : self.dim+ self.dim])
        t = torch.FloatTensor(data[self.length-2:])

        return start, end, t
    

class LoadDataLSTM(Dataset):
    def __init__(self, data_dir, length=PREVIOUS_FRAMES):
        self.data_dir = data_dir
        self.length = length
        data = np.load(self.data_dir)[:, :, 0:2]
        # print(self.data_dir)
        print("data shape: ", data.shape)
        self.data = []
        x_min = np.min(data[:, :, 0])
        x_max = np.max(data[:, :, 0])
        y_min = np.min(data[:, :, 1])
        y_max = np.max(data[:, :, 1])
        t_min = 0
        t_max = data.shape[0] - 1
        ## scale to [0, 1]
        minval = 0
        maxval = 1
        data[:, :, 0] = (data[:, :, 0] - x_min) / (x_max - x_min) * (maxval - minval) + minval
        data[:, :, 1] = (data[:, :, 1] - y_min) / (y_max - y_min) * (maxval - minval) + minval
        
        for j in range(data.shape[1]):
            trajectories = data[:, j, :]
            num_fm = data.shape[0] - 1
            
            for i in range(0, num_fm - self.length):
                end = trajectories[i+self.length, :]
                t_list = []
                for t in range(self.length):
                    t_list.append(i + t)
                t = np.array(t_list)
                t = (t - t_min) / (t_max - t_min) * (maxval - minval) +  minval ## start time      
                t = np.reshape(t, (self.length, 1)) 
                starts = trajectories[i:i+self.length, :]
                # starts = trajectories[0:self.length, :]
                input = np.concatenate((starts, t), axis = 1)
                # print(input.shape, input)
                self.data.append({
                        "input": torch.FloatTensor(input),
                        "end": torch.FloatTensor(end),
                        })
    def __len__(self):
        # print("total data size", len(self.data))
        return len(self.data)
    
    def __getitem__(self,index):
        np.random.seed(seed = int(time.time() + index))
        data = self.data[index]
        input = data["input"]
        end = data["end"]

        return input, end

class LoadDataLSTM_FUTURE(Dataset):
    def __init__(self, data_dir, length=PREVIOUS_FRAMES, ahead=2):
        self.data_dir = data_dir
        self.length = length
        data = np.load(self.data_dir)[:, :, 0:2]
        print("Loading Future", data.shape, "Future Points: ",ahead)
        self.data = []
        # x_min = np.min(data[:, :, 0])
        x_min = np.min(data[:, :, 0])
        x_max = np.max(data[:, :, 0])
        y_min = np.min(data[:, :, 1])
        y_max = np.max(data[:, :, 1])
        t_min = 0
        t_max = data.shape[0] - 1
        ## scale to [0, 1]
        minval = 0
        maxval = 1
        data[:, :, 0] = (data[:, :, 0] - x_min) / (x_max - x_min) * (maxval - minval) + minval
        data[:, :, 1] = (data[:, :, 1] - y_min) / (y_max - y_min) * (maxval - minval) + minval
        
        for j in range(data.shape[1]):
            trajectories = data[:, j, :]
            num_fm = data.shape[0] - 1
            for i in range(0, num_fm - self.length - ahead +1):
                end_i = i+self.length
                end = trajectories[end_i:end_i+ahead, :]
                t_list = []
                for t in range(self.length):
                    t_list.append(i + t)
                t = np.array(t_list)
                t = (t - t_min) / (t_max - t_min) * (maxval - minval) +  minval ## start time      
                t = np.reshape(t, (self.length, 1)) 
                starts = trajectories[i:i+self.length, :]
                # starts = trajectories[0:self.length, :]
                input = np.concatenate((starts, t), axis = 1)
                # print(input.shape, input)
                self.data.append({
                        "input": torch.FloatTensor(input),
                        "end": torch.FloatTensor(end.reshape(2*ahead)),
                        })
    def __len__(self):
        # print("total data size", len(self.data))
        return len(self.data)
    
    def __getitem__(self,index):
        np.random.seed(seed = int(time.time() + index))
        data = self.data[index]
        input = data["input"]
        end = data["end"]

        return input, end


class LoadDataLSTM_EVAL(Dataset):
    def __init__(self, start_points, t_start, t_end,bounds):
        # timestamp += PREVIOUS_FRAMES
        self.data = []
        x_min = bounds.x_min
        x_max = bounds.x_max
        y_min = bounds.y_min
        y_max = bounds.y_max
        minval = 0
        maxval = 1
        start_points[:, :, 0] = (start_points[:, :, 0] - x_min) / (x_max - x_min) * (maxval - minval) + minval
        start_points[:, :, 1] = (start_points[:, :, 1] - y_min) / (y_max - y_min) * (maxval - minval) + minval
        # print(start_points.shape)
        t_min = bounds.t_min
        t_max = bounds.t_max
        
        # print("Checking",start, end)
            # print(trajectories.shape)
            # end = trajectories[i+self.length, :]
        t_list = []
            # start = t-self.length
            # end = t
        for tt in range(t_start, t_end):
            t_list.append(tt)
        tt = np.array(t_list)
        tt = (tt - t_min) / (t_max - t_min) * (maxval - minval) +  minval ## start time      
        tt = np.reshape(tt, (t_end - t_start, 1)) 
            # # starts = trajectories[0:self.length, :]
        for i in range(start_points.shape[1]):
            cur_start_points = start_points[:, i, :]
            inputs = np.concatenate((cur_start_points, tt), axis = 1)
            # print(input.shape, input)
            self.data.append({
                "input": torch.FloatTensor(inputs),
                # "end": torch.FloatTensor(end),
            })
    def __getitem__(self,index):
        np.random.seed(seed = int(time.time() + index))
        data = self.data[index]
        inputs = data["input"]
        return inputs


    def __len__(self):
        # print("total data size", len(self.data))
        return len(self.data)


if __name__ == "__main__":
    # dataset = LoadDataLSTM_FUTURE("./data/500_sobol.npy",length=5, ahead=5)
    dataset = LoadData_StartEnd_Skip("./data/ense/ense7400_random_1k.npy",start=0,end=251)
    # x = np.zeros((10,5,6))
    # dataset = LoadDataLSTM_EVAL(x,length=5,t=7)
    # dataloader_train = DataLoader(dataset, batch_size=500, shuffle=False, num_workers=2, drop_last=False)
    # counter = 0 
    # for i,d in enumerate(dataloader_train):
        # counter +=1
        # print(d[0].shape)
        # print(d[1].shape)
        # print("Done")
        # break
    # print("Exit", counter * 500)
    # count = 0  
    