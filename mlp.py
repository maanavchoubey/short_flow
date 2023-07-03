import torch
from torch import nn

import torch
import torch.nn as nn

class MLP_Short(nn.Module):
    def __init__(self, dropout=0.4):
        super(MLP, self).__init__()
        self.basic_x = nn.Sequential(
            nn.Linear(1, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        self.basic_y = nn.Sequential(
            nn.Linear(1, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        self.basic2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.basic3 = nn.Sequential(
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        self.basic4 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.basic_fc1 = nn.Sequential(
            nn.Linear(1, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
        )
        self.basic_fc2 = nn.Sequential(
            nn.Linear(16, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
        )
        self.basic_fc3 = nn.Sequential(
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        self.basic_fc4 = nn.Sequential(
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.basic_fc5 = nn.Sequential(
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        self.basic_fc6 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.basic_output1 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
        )
        self.basic_output2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
        )
        self.basic_output3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        self.basic_output4 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.basic_output5 = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(64, 2),
            nn.ReLU(),
        )

        layer_list = [
            self.basic_x,
            self.basic_y,
            self.basic2,
            self.basic3,
            self.basic4,
            self.fc1,
            self.basic_fc1,
            self.basic_fc2,
            self.basic_fc3,
            self.basic_fc4,
            self.basic_fc5,
            self.basic_fc6,
            self.fc2,
            self.basic_output1,
            self.basic_output2,
            self.basic_output3,
            self.basic_output4,
            self.basic_output5,
            self.fc3,
        ]

        self.module_list = nn.ModuleList(layer_list)

    def forward(self, x, y, timestamp):
        encoder_x = self.basic_x(x)
        encoder_y = self.basic_y(y)
        encoder = torch.cat((encoder_x, encoder_y), dim=1)
        encoder = self.basic2(encoder)
        encoder = self.basic3(encoder)
        encoder = self.basic4(encoder)
        encoder = self.fc1(encoder)

        tencoder = self.basic_fc1(timestamp)
        tencoder = self.basic_fc2(tencoder)
        tencoder = self.basic_fc3(tencoder)
        tencoder = self.basic_fc4(tencoder)
        tencoder = self.basic_fc5(tencoder)
        tencoder = self.basic_fc6(tencoder)
        tencoder = self.fc2(tencoder)

        concat = torch.cat((encoder, tencoder), dim=1)
        decode = self.basic_output1(concat)
        decode = self.basic_output2(decode)
        decode = self.basic_output3(decode)
        decode = self.basic_output4(decode)
        decode = self.basic_output5(decode)

        return self.fc3(decode)


class MLP(nn.Module):
    def __init__(self, dropout = 0.4):
        super(MLP, self).__init__()
        self.basic1 = nn.Sequential(         
            nn.Linear(2, 64),
            nn.LayerNorm(64),                              
            nn.ReLU(),                      
        )
        self.basic2 = nn.Sequential(         
            nn.Linear(64, 128),
            nn.LayerNorm(128),                              
            nn.ReLU(),                      
        )

        self.basic3 = nn.Sequential(         
            nn.Linear(128, 256),
            nn.LayerNorm(256),                              
            nn.ReLU(),                      
        )

        self.basic4 = nn.Sequential(         
            nn.Linear(256, 512),
            nn.LayerNorm(512),                              
            nn.ReLU(),                      
        )

        self.fc1 = nn.Sequential(         
            nn.Linear(512, 512),                            
            nn.ReLU(),                      
        )


        self.basic_fc1 = nn.Sequential(         
            nn.Linear(1, 16),
            nn.LayerNorm(16),                              
            nn.ReLU(),                      
        )

        self.basic_fc2 = nn.Sequential(         
            nn.Linear(16, 32),
            nn.LayerNorm(32),                              
            nn.ReLU(),                      
        )

        self.basic_fc3 = nn.Sequential(         
            nn.Linear(32, 64),
            nn.LayerNorm(64),                              
            nn.ReLU(),                      
        )

        self.basic_fc4 = nn.Sequential(         
            nn.Linear(64, 128),
            nn.LayerNorm(128),                             
            nn.ReLU(),                      
        )

        self.basic_fc5 = nn.Sequential(         
            nn.Linear(128,256),
            nn.LayerNorm(256),                              
            nn.ReLU(),                      
        )

        self.basic_fc6 = nn.Sequential(         
            nn.Linear(256, 512),
            nn.LayerNorm(512),                              
            nn.ReLU(),                      
        )

        self.fc2 = nn.Sequential(         
            nn.Linear(512, 512),                            
            nn.ReLU(),                      
        )

        self.basic_output1 = nn.Sequential(         
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),                              
            nn.ReLU(),                      
        )

        self.basic_output2 = nn.Sequential(         
            nn.Linear(1024, 512),
            nn.LayerNorm(512),                              
            nn.ReLU(),                      
        )

        self.basic_output3 = nn.Sequential(         
            nn.Linear(512, 256),
            nn.LayerNorm(256),                              
            nn.ReLU(),                      
        )

        self.basic_output4 = nn.Sequential(         
            nn.Linear(256, 128),
            nn.LayerNorm(128),                              
            nn.ReLU(),                      
        )

        self.basic_output5 = nn.Sequential(         
            nn.Linear(128,64),
            nn.LayerNorm(64),                              
            nn.ReLU(),                      
        )

        self.fc3 = nn.Sequential(         
            nn.Linear(64, 2),                            
            nn.ReLU(),                      
        )

        layer_list = [
           self.basic1,
           self.basic2,
           self.basic3,
           self.basic4,
           self.fc1, 
           self.basic_fc1,
           self.basic_fc2,
           self.basic_fc3,
           self.basic_fc4,
           self.basic_fc5,
           self.basic_fc6,
           self.fc2,
           self.basic_output1,
           self.basic_output2,
           self.basic_output3,
           self.basic_output4,
           self.basic_output5,
           self.fc3
        ]

        self.module_list = nn.ModuleList(layer_list)

        # self.freeze_timestamp()
    
    def freeze_timestamp(self):
        print("Freezing timestamp")
        li = [
           self.basic_fc1,
           self.basic_fc2,
           self.basic_fc3,
           self.basic_fc4,
           self.basic_fc5,
           self.basic_fc6,
           self.fc2, 
        ]
        for each in li:
            for param in each.parameters():
                param.requires_grad = False
                # print(param.require_grads)

    def forward(self, start0, timestamp):
        encoder = self.basic1(start0)
        encoder = self.basic2(encoder)
        encoder = self.basic3(encoder)
        encoder = self.basic4(encoder)
        encoder = self.fc1(encoder)

        tencoder = self.basic_fc1(timestamp)
        # print("ten", tencoder.shape)
        tencoder = self.basic_fc2(tencoder)
        # print("fc2", tencoder.shape)
        tencoder = self.basic_fc3(tencoder)
        # print("fc3", tencoder.shape)
        tencoder = self.basic_fc4(tencoder)
        # print("here 1", tencoder.shape)
        tencoder = self.basic_fc5(tencoder)
        tencoder = self.basic_fc6(tencoder)
        tencoder = self.fc2(tencoder)

        concat = torch.cat((encoder, tencoder), dim=1)
        # print("Checking shape: ", concat.shape)
        decode = self.basic_output1(concat)
        decode = self.basic_output2(decode)
        decode = self.basic_output3(decode)
        decode = self.basic_output4(decode)
        decode = self.basic_output5(decode)
        
        return self.fc3(decode)

class MLP_Attention(nn.Module):
    def __init__(self, dropout = 0.4):
        super(MLP_Attention, self).__init__()

        net = 100
        self.basic1 = nn.Sequential(         
            nn.Linear(2, net),
            nn.LayerNorm(net),                              
            nn.ReLU(),                      
        )
        self.basic2 = nn.Sequential(         
            nn.Linear(net, net),
            nn.LayerNorm(net),                              
            nn.ReLU(),                      
        )

        self.basic3 = nn.Sequential(         
            nn.Linear(net, net),
            nn.LayerNorm(net),                              
            nn.ReLU(),                      
        )

        self.basic4 = nn.Sequential(         
            nn.Linear(net, net),
            nn.LayerNorm(net),                              
            nn.ReLU(),                      
        )
        self.basic5 = nn.Sequential(         
            nn.Linear(net, net),
            nn.LayerNorm(net),                              
            nn.ReLU(),                      
        )
        self.basic6 = nn.Sequential(         
            nn.Linear(net, net),
            nn.LayerNorm(net),                              
            nn.ReLU(),                      
        )
        self.basic7 = nn.Sequential(         
            nn.Linear(net, net),
            nn.LayerNorm(net),                              
            nn.ReLU(),                      
        )
        self.basic8 = nn.Sequential(         
            nn.Linear(net, net),
            nn.LayerNorm(net),                              
            nn.ReLU(),                      
        )

        self.fc1 = nn.Sequential(         
            nn.Linear(net, net),                            
            nn.ReLU(),                      
        )


        self.basic_fc1 = nn.Sequential(         
            nn.Linear(1, net),
            nn.LayerNorm(net),                              
            nn.ReLU(),                      
        )

        self.basic_fc2 = nn.Sequential(         
            nn.Linear(net, net),
            nn.LayerNorm(net),                              
            nn.ReLU(),                      
        )

        self.basic_fc3 = nn.Sequential(         
            nn.Linear(net, net),
            nn.LayerNorm(net),                              
            nn.ReLU(),                      
        )

        self.basic_fc4 = nn.Sequential(         
            nn.Linear(net, net),
            nn.LayerNorm(net),                             
            nn.ReLU(),                      
        )

        self.basic_fc5 = nn.Sequential(         
            nn.Linear(net,net),
            nn.LayerNorm(net),                              
            nn.ReLU(),                      
        )

        self.basic_fc6 = nn.Sequential(         
            nn.Linear(net, net),
            nn.LayerNorm(net),                              
            nn.ReLU(),                      
        )

        self.basic_fc7 = nn.Sequential(         
            nn.Linear(net, net),
            nn.LayerNorm(net),                              
            nn.ReLU(),                      
        )

        self.basic_fc8 = nn.Sequential(         
            nn.Linear(net, net),
            nn.LayerNorm(net),                              
            nn.ReLU(),                      
        )
        

        self.fc2 = nn.Sequential(         
            nn.Linear(net, net),                            
            nn.ReLU(),                      
        )

        self.basic_output1 = nn.Sequential(         
            nn.Linear(net*2, 512),
            nn.LayerNorm(512),                              
            nn.ReLU(),                      
        )

        self.basic_output2 = nn.Sequential(         
            nn.Linear(512, 512),
            nn.LayerNorm(512),                              
            nn.ReLU(),                      
        )

        self.basic_output3 = nn.Sequential(         
            nn.Linear(512, 256),
            nn.LayerNorm(256),                              
            nn.ReLU(),                      
        )

        self.basic_output4 = nn.Sequential(         
            nn.Linear(256, 128),
            nn.LayerNorm(128),                              
            nn.ReLU(),                      
        )

        self.basic_output5 = nn.Sequential(         
            nn.Linear(128,64),
            nn.LayerNorm(64),                              
            nn.ReLU(),                      
        )

        self.fc3 = nn.Sequential(         
            nn.Linear(64, 2),                            
            nn.ReLU(),                      
        )

        self.attention = nn.MultiheadAttention(512,4, batch_first=True)

        layer_list = [
           self.basic1,
           self.basic2,
           self.basic3,
           self.basic4,
           self.fc1, 
           self.basic_fc1,
           self.basic_fc2,
           self.basic_fc3,
           self.basic_fc4,
           self.basic_fc5,
           self.basic_fc6,
           self.fc2,
           self.basic_output1,
           self.basic_output2,
           self.basic_output3,
           self.basic_output4,
           self.basic_output5,
           self.fc3
        ]

        # self.module_list = nn.ModuleList(layer_list)

    def forward(self, start0, timestamp):
        encoder = self.basic1(start0)
        s1 = encoder
        encoder = self.basic2(encoder)
        encoder = self.basic3(encoder) + s1
        s2 = encoder
        encoder = self.basic4(encoder)
        encoder = self.basic5(encoder) + s2
        s3 = encoder
        encoder = self.basic6(encoder)
        encoder = self.basic7(encoder) + s3
        encoder = self.basic8(encoder)
        encoder = self.fc1(encoder)

        tencoder = self.basic_fc1(timestamp)
        ts1 = tencoder
        # print("ten", tencoder.shape)
        tencoder = self.basic_fc2(tencoder)
        # print("fc2", tencoder.shape)
        tencoder = self.basic_fc3(tencoder) + ts1
        ts2 = tencoder
        # print("fc3", tencoder.shape)
        tencoder = self.basic_fc4(tencoder)
        # print("here 1", tencoder.shape)
        tencoder = self.basic_fc5(tencoder) + ts2
        ts3 = tencoder
        tencoder = self.basic_fc6(tencoder)
        tencoder = self.basic_fc7(tencoder) + ts3
        tencoder = self.basic_fc8(tencoder)
        tencoder = self.fc2(tencoder)

        # en_attention, _ = self.attention(encoder,encoder,encoder)
        # en_attention = torch.add(en_attention, encoder)
        # encoder = torch.add(encoder, tencoder)
        # tencoder = self.fc2(tencoder)

        # print(encoder)
        # print(encoder)
        concat = torch.cat((encoder, tencoder), dim=1)
        # print("Checking shape: ", concat.shape)
        decode = self.basic_output1(concat)
        decode = self.basic_output2(decode)
        decode = self.basic_output3(decode)
        decode = self.basic_output4(decode)
        decode = self.basic_output5(decode) 
        
        return self.fc3(decode) 

if __name__ == "__main__":
    start = torch.rand(100, 2)
    time = torch.rand(100, 1)
    # model = MLP_Attention()
    model = MLP()
    model.freeze_timestamp()
    for name,pram in model.named_parameters():
        print(name,pram.requires_grad)
    end = model(start, time)
    print("results shape: ", end.shape)