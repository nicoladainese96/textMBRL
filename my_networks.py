# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

verbose = False
vprint = print if verbose else lambda *args, **kwargs: None


class FixedDynamicsValueNet(nn.Module):
    def __init__(self, 
                 gym_env,
                 emb_dim=10,
                ):
        super().__init__()
        self.embedding = nn.Embedding(len(gym_env.vocab), emb_dim)
        
        name_shape = gym_env.observation_space['name']
        inv_shape = gym_env.observation_space['inv']
        n_channels = (name_shape[2]*name_shape[3]+inv_shape[0])*emb_dim
        
        self.conv_net = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )

        self.maxpool = nn.MaxPool2d(gym_env.observation_space['name'][1])

        self.value_mlp = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )
        
    def forward(self, frame):
        device = next(self.parameters()).device
        x = self.embedding(frame['name'].to(device))
        s = x.shape
        B, W, H = s[:3]
        x = x.reshape(*s[:3],-1).permute(0, 3, 1, 2)
        inv = self.embedding(frame['inv'].to(device))
        inv = inv.reshape(B,-1,1,1)
        inv = inv.expand(B,-1,W,H)
        z = torch.cat([x,inv], axis=1)
        z_conv = self.conv_net(z)
        z_flat = self.maxpool(z_conv).view(B,-1)
        v = self.value_mlp(z_flat)
        return v

class FixedDynamicsValueNet_v1(nn.Module):
    """
    W.r.t. FixedDynamicsValueNet introduces separate embeddings for name and inv
    """
    def __init__(self, 
                 gym_env,
                 emb_dim=10,
                ):
        super().__init__()
        self.name_embedding = nn.Embedding(len(gym_env.vocab), emb_dim)
        self.inv_embedding = nn.Embedding(len(gym_env.vocab), emb_dim)
        
        name_shape = gym_env.observation_space['name']
        inv_shape = gym_env.observation_space['inv']
        n_channels = (name_shape[2]*name_shape[3]+inv_shape[0])*emb_dim
        
        self.conv_net = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )

        self.maxpool = nn.MaxPool2d(gym_env.observation_space['name'][1])

        self.value_mlp = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )
        
    def forward(self, frame):
        device = next(self.parameters()).device
        x = self.name_embedding(frame['name'].to(device))
        s = x.shape
        B, W, H = s[:3]
        x = x.reshape(*s[:3],-1).permute(0, 3, 1, 2)
        inv = self.inv_embedding(frame['inv'].to(device))
        inv = inv.reshape(B,-1,1,1)
        inv = inv.expand(B,-1,W,H)
        z = torch.cat([x,inv], axis=1)
        z_conv = self.conv_net(z)
        z_flat = self.maxpool(z_conv).view(B,-1)
        v = self.value_mlp(z_flat)
        return v

class FixedDynamicsPVNet(nn.Module):
    """
    Double head for value and policy
    """
    def __init__(self, 
                 gym_env,
                 emb_dim=10,
                ):
        super().__init__()
        self.name_embedding = nn.Embedding(len(gym_env.vocab), emb_dim)
        self.inv_embedding = nn.Embedding(len(gym_env.vocab), emb_dim)
        
        name_shape = gym_env.observation_space['name']
        inv_shape = gym_env.observation_space['inv']
        n_channels = (name_shape[2]*name_shape[3]+inv_shape[0])*emb_dim
        
        self.conv_net = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )

        self.maxpool = nn.MaxPool2d(gym_env.observation_space['name'][1])

        self.value_mlp = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )
        
        self.policy_mlp = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,len(gym_env.action_space))
        )
        
    def forward(self, frame):
        device = next(self.parameters()).device
        x = self.name_embedding(frame['name'].to(device))
        s = x.shape
        B, W, H = s[:3]
        x = x.reshape(*s[:3],-1).permute(0, 3, 1, 2)
        
        inv = self.inv_embedding(frame['inv'].to(device))
        inv = inv.reshape(B,-1,1,1)
        inv = inv.expand(B,-1,W,H)
        
        z = torch.cat([x,inv], axis=1)
        z_conv = self.conv_net(z)
        z_flat = self.maxpool(z_conv).view(B,-1)
        
        v = self.value_mlp(z_flat)
        logits = self.policy_mlp(z_flat)
        #print("logits: ", logits)
        action_mask = 1-frame['valid'].to(device)
        #print("action_mask: ", action_mask)
        probs = F.softmax(logits.masked_fill((action_mask).bool(), float('-inf')), dim=-1) 
        #print("probs: ", probs)
        return v, probs

class ResidualConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        x -> conv -> batch norm -> relu -> conv -> batch norm -> relu -> + -> relu -> out
          |                                                              |
          -------------------------------->-------------------------------
          
        Args:
          in_channels (int):  Number of input channels.
          out_channels (int): Number of output channels.
          stride (int):       Controls the stride.
        """
        super(ResidualConv, self).__init__()
        # YOUR CODE HERE
        
        # RESIDUAL PART
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # SKIP CONNECTION PART
        if stride==1 and in_channels==out_channels:
            self.skip_net = None
        else:
            self.skip_net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        
    def forward(self, x):
        # YOUR CODE HERE
        
        h1 = F.relu(self.bn1(self.conv1(x)))
        res = F.relu(self.bn2(self.conv2(h1)))
        
        if self.skip_net is None:
            skip_x = x
        else:
            skip_x = self.skip_net(x)
        
        out = F.relu(res+skip_x)
        return out

class FixedDynamicsValueNet_v2(nn.Module):
    def __init__(self, 
                 gym_env,
                 emb_dim=10,
                ):
        super().__init__()
        self.name_embedding = nn.Embedding(len(gym_env.vocab), emb_dim)
        self.inv_embedding = nn.Embedding(len(gym_env.vocab), emb_dim)
        
        name_shape = gym_env.observation_space['name']
        inv_shape = gym_env.observation_space['inv']
        n_channels = (name_shape[2]*name_shape[3]+inv_shape[0])*emb_dim
        
        self.conv_net = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=1),
            ResidualConv(64,64),
            ResidualConv(64,64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1)
        )

        self.maxpool = nn.MaxPool2d(gym_env.observation_space['name'][1])

        self.value_mlp = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )
        
    def forward(self, frame):
        device = next(self.parameters()).device
        x = self.name_embedding(frame['name'].to(device))
        s = x.shape
        B, W, H = s[:3]
        x = x.reshape(*s[:3],-1).permute(0, 3, 1, 2)
        inv = self.inv_embedding(frame['inv'].to(device))
        inv = inv.reshape(B,-1,1,1)
        inv = inv.expand(B,-1,W,H)
        z = torch.cat([x,inv], axis=1)
        z_conv = self.conv_net(z)
        z_flat = self.maxpool(z_conv).view(B,-1)
        v = self.value_mlp(z_flat)
        return v

class FixedDynamicsValueNet_v3(nn.Module):
    def __init__(self, 
                 gym_env,
                 emb_dim=10,
                 conv_channels=64,
                 conv_layers=2,
                 residual_layers=2,
                 linear_features_in=128,
                 linear_feature_hidden=128
                ):
        """
        Use 2 1x1 convolutions before the residual layers.
        """
        super().__init__()
        self.name_embedding = nn.Embedding(len(gym_env.vocab), emb_dim)
        self.inv_embedding = nn.Embedding(len(gym_env.vocab), emb_dim)
        
        name_shape = gym_env.observation_space['name']
        inv_shape = gym_env.observation_space['inv']
        n_channels = (name_shape[2]*name_shape[3]+inv_shape[0])*emb_dim
        
        self.conv_net_layers = nn.ModuleList([
            nn.Conv2d(n_channels, conv_channels, kernel_size=1), # 1x1 conv to mix inv and name channels
            nn.ReLU()
        ])
        
        for i in range(conv_layers-1):
            self.conv_net_layers.append(nn.Conv2d(conv_channels, conv_channels, kernel_size=1))
            self.conv_net_layers.append(nn.ReLU())
        for i in range(residual_layers):
            self.conv_net_layers.append(ResidualConv(conv_channels, conv_channels))
        self.conv_net_layers.append(nn.Conv2d(conv_channels, linear_features_in, kernel_size=3, padding=1))

        self.maxpool = nn.MaxPool2d(gym_env.observation_space['name'][1])

        self.value_mlp = nn.Sequential(
            nn.Linear(linear_features_in, linear_feature_hidden),
            nn.ReLU(),
            nn.Linear(linear_feature_hidden,1)
        )
    
    def forward(self, frame):
        device = next(self.parameters()).device
        
        # Embed name (grid representation) and inv (inventory)
        x = self.name_embedding(frame['name'].to(device))
        inv = self.inv_embedding(frame['inv'].to(device))
        
        # Reshape both to (B,C,W,H) tensors to be concatenated together along C axis
        s = x.shape
        B, W, H = s[:3]
        x = x.reshape(*s[:3],-1).permute(0, 3, 1, 2) # (B, C1, W, H)
        inv = inv.reshape(B,-1,1,1)
        inv = inv.expand(B,-1,W,H) # (B, C2, W, H)
        z = torch.cat([x,inv], axis=1)
        
        # Process in convolutional layers
        for layer in self.conv_net_layers:
            z = layer(z)
        # Summarize spatial dimensions with maxpool along (W,H) -> out_channels = in_features to MLP
        z_flat = self.maxpool(z).view(B,-1)
        # Refine features with 1 hidden linear layer and then predict logits for all support values 
        # (like if they were classes) 
        v = self.value_mlp(z_flat)
        return v


class FixedDynamicsPVNet_v2(nn.Module):
    def __init__(self, 
                 gym_env,
                 emb_dim=10,
                ):
        super().__init__()
        self.name_embedding = nn.Embedding(len(gym_env.vocab), emb_dim)
        self.inv_embedding = nn.Embedding(len(gym_env.vocab), emb_dim)
        
        name_shape = gym_env.observation_space['name']
        inv_shape = gym_env.observation_space['inv']
        n_channels = (name_shape[2]*name_shape[3]+inv_shape[0])*emb_dim
        
        self.conv_net = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=1),
            ResidualConv(64,64),
            ResidualConv(64,64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1)
        )

        self.maxpool = nn.MaxPool2d(gym_env.observation_space['name'][1])

        self.value_mlp = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )
        
        self.policy_mlp = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,len(gym_env.action_space))
        )
        
    def forward(self, frame):
        device = next(self.parameters()).device
        x = self.name_embedding(frame['name'].to(device))
        s = x.shape
        B, W, H = s[:3]
        x = x.reshape(*s[:3],-1).permute(0, 3, 1, 2)
        
        inv = self.inv_embedding(frame['inv'].to(device))
        inv = inv.reshape(B,-1,1,1)
        inv = inv.expand(B,-1,W,H)
        
        z = torch.cat([x,inv], axis=1)
        z_conv = self.conv_net(z)
        z_flat = self.maxpool(z_conv).view(B,-1)
        
        v = self.value_mlp(z_flat)
        logits = self.policy_mlp(z_flat)
        #print("logits: ", logits)
        action_mask = 1-frame['valid'].to(device)
        #print("action_mask: ", action_mask)
        probs = F.softmax(logits.masked_fill((action_mask).bool(), float('-inf')), dim=-1) 
        #print("probs: ", probs)
        return v, probs

class ValueMLP(nn.Module):
    def __init__(self, gym_env, emb_dim=10):
        super().__init__()
        self.name_embedding = nn.Embedding(len(gym_env.vocab), emb_dim)
        self.inv_embedding = nn.Embedding(len(gym_env.vocab), emb_dim)
        
        name_shape = gym_env.observation_space['name']
        inv_shape = gym_env.observation_space['inv']
        n_features = (name_shape[0]*name_shape[1]*name_shape[2]*name_shape[3]+inv_shape[0])*emb_dim
        print("n_features: ", n_features)
        
        self.value_mlp = nn.Sequential(
            nn.Linear(n_features,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )
        
    def forward(self, frame):
        device = next(self.parameters()).device
        x = self.name_embedding(frame['name'].to(device))
        s = x.shape
        B, W, H = s[:3]
        x = x.reshape(B,-1)
        inv = self.inv_embedding(frame['inv'].to(device))
        inv = inv.reshape(B,-1)
        z = torch.cat([x,inv], axis=1)
        v = self.value_mlp(z)
        return v

# Discrete support
class DiscreteSupportValueNet(nn.Module):
    def __init__(self, 
                 gym_env,
                 emb_dim=10,
                 support_size=2,
                ):
        super().__init__()
        self.embedding = nn.Embedding(len(gym_env.vocab), emb_dim)
        self.support_size = support_size
        
        name_shape = gym_env.observation_space['name']
        inv_shape = gym_env.observation_space['inv']
        n_channels = (name_shape[2]*name_shape[3]+inv_shape[0])*emb_dim
        
        self.conv_net = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )

        self.maxpool = nn.MaxPool2d(gym_env.observation_space['name'][1])

        self.value_mlp = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128, support_size*2+1)
        )
        
    def forward(self, frame):
        v_logits = self.logits(frame)
        v = support_to_scalar_v1(v_logits, self.support_size)
        return v
    
    def logits(self, frame):
        device = next(self.parameters()).device
        x = self.embedding(frame['name'].to(device))
        s = x.shape
        B, W, H = s[:3]
        x = x.reshape(*s[:3],-1).permute(0, 3, 1, 2)
        inv = self.embedding(frame['inv'].to(device))
        inv = inv.reshape(B,-1,1,1)
        inv = inv.expand(B,-1,W,H)
        z = torch.cat([x,inv], axis=1)
        z_conv = self.conv_net(z)
        z_flat = self.maxpool(z_conv).view(B,-1)
        v_logits = self.value_mlp(z_flat)
        return v_logits

class DiscreteSupportValueNet_v1(nn.Module):
    def __init__(self, 
                 gym_env,
                 emb_dim=10,
                 support_size=10
                ):
        super().__init__()
        self.name_embedding = nn.Embedding(len(gym_env.vocab), emb_dim)
        self.inv_embedding = nn.Embedding(len(gym_env.vocab), emb_dim)
        self.support_size = support_size
        
        name_shape = gym_env.observation_space['name']
        inv_shape = gym_env.observation_space['inv']
        n_channels = (name_shape[2]*name_shape[3]+inv_shape[0])*emb_dim
        
        self.conv_net = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=1),
            ResidualConv(64,64),
            ResidualConv(64,64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1)
        )

        self.maxpool = nn.MaxPool2d(gym_env.observation_space['name'][1])

        self.value_mlp = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,support_size*2+1)
        )
        
    def forward(self, frame):
        v_logits = self.logits(frame)
        v = support_to_scalar_v1(v_logits, self.support_size)
        return v
    
    def logits(self, frame):
        device = next(self.parameters()).device
        x = self.name_embedding(frame['name'].to(device))
        s = x.shape
        B, W, H = s[:3]
        x = x.reshape(*s[:3],-1).permute(0, 3, 1, 2)
        inv = self.inv_embedding(frame['inv'].to(device))
        inv = inv.reshape(B,-1,1,1)
        inv = inv.expand(B,-1,W,H)
        z = torch.cat([x,inv], axis=1)
        z_conv = self.conv_net(z)
        z_flat = self.maxpool(z_conv).view(B,-1)
        v_logits = self.value_mlp(z_flat)
        return v_logits

class DiscreteSupportValueNet_v2(nn.Module):
    def __init__(self, 
                 gym_env,
                 emb_dim=10,
                 support_size=10,
                 conv_channels=64,
                 residual_layers=2,
                 linear_features_in=128,
                 linear_feature_hidden=128
                ):
        super().__init__()
        self.name_embedding = nn.Embedding(len(gym_env.vocab), emb_dim)
        self.inv_embedding = nn.Embedding(len(gym_env.vocab), emb_dim)
        self.support_size = support_size
        
        name_shape = gym_env.observation_space['name']
        inv_shape = gym_env.observation_space['inv']
        n_channels = (name_shape[2]*name_shape[3]+inv_shape[0])*emb_dim
        
        self.conv_net_layers = nn.ModuleList([
            nn.Conv2d(n_channels, conv_channels, kernel_size=1), # 1x1 conv to mix inv and name channels
            nn.ReLU()
        ])
        for i in range(residual_layers):
            self.conv_net_layers.append(ResidualConv(conv_channels, conv_channels))
        self.conv_net_layers.append(nn.Conv2d(conv_channels, linear_features_in, kernel_size=3, padding=1))

        self.maxpool = nn.MaxPool2d(gym_env.observation_space['name'][1])

        self.value_mlp = nn.Sequential(
            nn.Linear(linear_features_in, linear_feature_hidden),
            nn.ReLU(),
            nn.Linear(linear_feature_hidden,support_size*2+1)
        )
        
    def forward(self, frame):
        v_logits = self.logits(frame)
        v = support_to_scalar_v1(v_logits, self.support_size)
        return v
    
    def logits(self, frame):
        device = next(self.parameters()).device
        
        # Embed name (grid representation) and inv (inventory)
        x = self.name_embedding(frame['name'].to(device))
        inv = self.inv_embedding(frame['inv'].to(device))
        
        # Reshape both to (B,C,W,H) tensors to be concatenated together along C axis
        s = x.shape
        B, W, H = s[:3]
        x = x.reshape(*s[:3],-1).permute(0, 3, 1, 2) # (B, C1, W, H)
        inv = inv.reshape(B,-1,1,1)
        inv = inv.expand(B,-1,W,H) # (B, C2, W, H)
        z = torch.cat([x,inv], axis=1)
        
        # Process in convolutional layers
        for layer in self.conv_net_layers:
            z = layer(z)
        # Summarize spatial dimensions with maxpool along (W,H) -> out_channels = in_features to MLP
        z_flat = self.maxpool(z).view(B,-1)
        # Refine features with 1 hidden linear layer and then predict logits for all support values 
        # (like if they were classes) 
        v_logits = self.value_mlp(z_flat)
        return v_logits


class DiscreteSupportValueNet_v3(nn.Module):
    def __init__(self, 
                 gym_env,
                 emb_dim=10,
                 support_size=10,
                 conv_channels=64,
                 conv_layers=2,
                 residual_layers=2,
                 linear_features_in=128,
                 linear_feature_hidden=128
                ):
        """
        Use 2 1x1 convolutions before the residual layers.
        """
        super().__init__()
        self.name_embedding = nn.Embedding(len(gym_env.vocab), emb_dim)
        self.inv_embedding = nn.Embedding(len(gym_env.vocab), emb_dim)
        self.support_size = support_size
        
        name_shape = gym_env.observation_space['name']
        inv_shape = gym_env.observation_space['inv']
        n_channels = (name_shape[2]*name_shape[3]+inv_shape[0])*emb_dim
        
        self.conv_net_layers = nn.ModuleList([
            nn.Conv2d(n_channels, conv_channels, kernel_size=1), # 1x1 conv to mix inv and name channels
            nn.ReLU()
        ])
        
        for i in range(conv_layers-1):
            self.conv_net_layers.append(nn.Conv2d(conv_channels, conv_channels, kernel_size=1))
            self.conv_net_layers.append(nn.ReLU())
        for i in range(residual_layers):
            self.conv_net_layers.append(ResidualConv(conv_channels, conv_channels))
        self.conv_net_layers.append(nn.Conv2d(conv_channels, linear_features_in, kernel_size=3, padding=1))

        self.maxpool = nn.MaxPool2d(gym_env.observation_space['name'][1])

        self.value_mlp = nn.Sequential(
            nn.Linear(linear_features_in, linear_feature_hidden),
            nn.ReLU(),
            nn.Linear(linear_feature_hidden,support_size*2+1)
        )
        
    def forward(self, frame):
        v_logits = self.logits(frame)
        v = support_to_scalar_v1(v_logits, self.support_size)
        return v
    
    def logits(self, frame):
        device = next(self.parameters()).device
        
        # Embed name (grid representation) and inv (inventory)
        x = self.name_embedding(frame['name'].to(device))
        inv = self.inv_embedding(frame['inv'].to(device))
        
        # Reshape both to (B,C,W,H) tensors to be concatenated together along C axis
        s = x.shape
        B, W, H = s[:3]
        x = x.reshape(*s[:3],-1).permute(0, 3, 1, 2) # (B, C1, W, H)
        inv = inv.reshape(B,-1,1,1)
        inv = inv.expand(B,-1,W,H) # (B, C2, W, H)
        z = torch.cat([x,inv], axis=1)
        
        # Process in convolutional layers
        for layer in self.conv_net_layers:
            z = layer(z)
        # Summarize spatial dimensions with maxpool along (W,H) -> out_channels = in_features to MLP
        z_flat = self.maxpool(z).view(B,-1)
        # Refine features with 1 hidden linear layer and then predict logits for all support values 
        # (like if they were classes) 
        v_logits = self.value_mlp(z_flat)
        return v_logits


class DiscreteSupportPVNet(nn.Module):
    def __init__(self, 
                 gym_env,
                 emb_dim=10,
                 support_size=10,
                 conv_channels=64,
                 residual_layers=2,
                 linear_features_in=128,
                 linear_feature_hidden=128
                ):
        super().__init__()
        self.name_embedding = nn.Embedding(len(gym_env.vocab), emb_dim)
        self.inv_embedding = nn.Embedding(len(gym_env.vocab), emb_dim)
        self.support_size = support_size
        
        name_shape = gym_env.observation_space['name']
        inv_shape = gym_env.observation_space['inv']
        n_channels = (name_shape[2]*name_shape[3]+inv_shape[0])*emb_dim
        
        self.conv_net_layers = nn.ModuleList([
            nn.Conv2d(n_channels, conv_channels, kernel_size=1), # 1x1 conv to mix inv and name channels
            nn.ReLU()
        ])
        for i in range(residual_layers):
            self.conv_net_layers.append(ResidualConv(conv_channels, conv_channels))
        self.conv_net_layers.append(nn.Conv2d(conv_channels, linear_features_in, kernel_size=3, padding=1))

        self.maxpool = nn.MaxPool2d(gym_env.observation_space['name'][1])

        self.value_mlp = nn.Sequential(
            nn.Linear(linear_features_in, linear_feature_hidden),
            nn.ReLU(),
            nn.Linear(linear_feature_hidden,support_size*2+1)
        )
        
        self.policy_mlp = nn.Sequential(
            nn.Linear(linear_features_in,linear_feature_hidden),
            nn.ReLU(),
            nn.Linear(linear_feature_hidden,len(gym_env.action_space))
        )
        
    def forward(self, frame, return_v_logits=False):
        z_flat = self.encode(frame)
        
        ### Value head ###
        v_logits = self.logits(frame, z_flat) # just apply value MLP
        v = support_to_scalar_v1(v_logits, self.support_size) # softmax done inside here!
        
        ### Policy head ###
        device = next(self.parameters()).device
        logits = self.policy_mlp(z_flat)
        action_mask = 1-frame['valid'].to(device)
        probs = F.softmax(logits.masked_fill((action_mask).bool(), float('-inf')), dim=-1) 
        
        if return_v_logits:
            return v_logits, probs
        else:
            return v, probs
    
    def encode(self, frame):
        device = next(self.parameters()).device
        
        # Embed name (grid representation) and inv (inventory)
        x = self.name_embedding(frame['name'].to(device))
        inv = self.inv_embedding(frame['inv'].to(device))
        
        # Reshape both to (B,C,W,H) tensors to be concatenated together along C axis
        s = x.shape
        B, W, H = s[:3]
        x = x.reshape(*s[:3],-1).permute(0, 3, 1, 2) # (B, C1, W, H)
        inv = inv.reshape(B,-1,1,1)
        inv = inv.expand(B,-1,W,H) # (B, C2, W, H)
        z = torch.cat([x,inv], axis=1)
        
        # Process in convolutional layers
        for layer in self.conv_net_layers:
            z = layer(z)
        # Summarize spatial dimensions with maxpool along (W,H) -> out_channels = in_features to MLP
        z_flat = self.maxpool(z).view(B,-1)
        return z_flat
    
    def logits(self, frame, z_flat=None):
        """ Auxiliary funciton"""
        if z_flat is None:
            z_flat = self.encode(frame)
        # Refine features with 1 hidden linear layer and then predict logits for all support values 
        # (like if they were classes) 
        v_logits = self.value_mlp(z_flat)
        return v_logits
    
    
class DiscreteSupportPVNet_v3(nn.Module):
    def __init__(self, 
                 gym_env,
                 emb_dim=10,
                 support_size=10,
                 conv_channels=64,
                 conv_layers=2,
                 residual_layers=2,
                 linear_features_in=128,
                 linear_feature_hidden=128
                ):
        """
        Use 2 1x1 convolutions before the residual layers.
        """
        super().__init__()
        self.name_embedding = nn.Embedding(len(gym_env.vocab), emb_dim)
        self.inv_embedding = nn.Embedding(len(gym_env.vocab), emb_dim)
        self.support_size = support_size
        
        name_shape = gym_env.observation_space['name']
        inv_shape = gym_env.observation_space['inv']
        n_channels = (name_shape[2]*name_shape[3]+inv_shape[0])*emb_dim
        
        self.conv_net_layers = nn.ModuleList([
            nn.Conv2d(n_channels, conv_channels, kernel_size=1), # 1x1 conv to mix inv and name channels
            nn.ReLU()
        ])
        
        for i in range(conv_layers-1):
            self.conv_net_layers.append(nn.Conv2d(conv_channels, conv_channels, kernel_size=1))
            self.conv_net_layers.append(nn.ReLU())
        for i in range(residual_layers):
            self.conv_net_layers.append(ResidualConv(conv_channels, conv_channels))
        self.conv_net_layers.append(nn.Conv2d(conv_channels, linear_features_in, kernel_size=3, padding=1))

        self.maxpool = nn.MaxPool2d(gym_env.observation_space['name'][1])

        self.value_mlp = nn.Sequential(
            nn.Linear(linear_features_in, linear_feature_hidden),
            nn.ReLU(),
            nn.Linear(linear_feature_hidden,support_size*2+1)
        )
        self.policy_mlp = nn.Sequential(
            nn.Linear(linear_features_in,linear_feature_hidden),
            nn.ReLU(),
            nn.Linear(linear_feature_hidden,len(gym_env.action_space))
        )
        
    def forward(self, frame, return_v_logits=False):
        z_flat = self.encode(frame)
        
        ### Value head ###
        v_logits = self.logits(frame, z_flat) # just apply value MLP
        v = support_to_scalar_v1(v_logits, self.support_size) # softmax done inside here!
        
        ### Policy head ###
        device = next(self.parameters()).device
        logits = self.policy_mlp(z_flat)
        action_mask = 1-frame['valid'].to(device)
        probs = F.softmax(logits.masked_fill((action_mask).bool(), float('-inf')), dim=-1) 
        
        if return_v_logits:
            return v_logits, probs
        else:
            return v, probs
    
    def encode(self, frame):
        device = next(self.parameters()).device
        
        # Embed name (grid representation) and inv (inventory)
        x = self.name_embedding(frame['name'].to(device))
        inv = self.inv_embedding(frame['inv'].to(device))
        
        # Reshape both to (B,C,W,H) tensors to be concatenated together along C axis
        s = x.shape
        B, W, H = s[:3]
        x = x.reshape(*s[:3],-1).permute(0, 3, 1, 2) # (B, C1, W, H)
        inv = inv.reshape(B,-1,1,1)
        inv = inv.expand(B,-1,W,H) # (B, C2, W, H)
        z = torch.cat([x,inv], axis=1)
        
        # Process in convolutional layers
        for layer in self.conv_net_layers:
            z = layer(z)
        # Summarize spatial dimensions with maxpool along (W,H) -> out_channels = in_features to MLP
        z_flat = self.maxpool(z).view(B,-1)
        return z_flat
    
    def logits(self, frame, z_flat=None):
        """ Auxiliary funciton"""
        if z_flat is None:
            z_flat = self.encode(frame)
        # Refine features with 1 hidden linear layer and then predict logits for all support values 
        # (like if they were classes) 
        v_logits = self.value_mlp(z_flat)
        return v_logits


def support_to_scalar_v1(probabilities, support_size, probs_in_input=False):
    """
    logits: tensor of shape (batch_size, 2*support_size+1)
    support_size: int
    """
    if not probs_in_input:
        probabilities = torch.softmax(probabilities, dim=1)
    support = (
        torch.tensor([x for x in range(-support_size, support_size + 1)]) # equivalent of multiplying by support_size
        .expand(probabilities.shape)
        .float()
        .to(device=probabilities.device)
    )
    #print("support values: ", support/support_size)
    x = torch.sum(support * probabilities, dim=1, keepdim=False)/support_size # divide to get result in [-1,1]

    return x

def scalar_to_support_v1(x, support_size):
    """
    x: tensor of shape (batch_size, 1) with all elements in [-1,1]
    support_size: int
    """
    b = x.shape[0]
    x = x*support_size
    x = torch.clamp(x, -support_size, support_size)
    floor = x.floor()
    prob = x - floor
    logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1).to(x.device)
    logits.scatter_(
        2, (floor + support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1)
    )
    indexes = floor + support_size + 1
    prob = prob.masked_fill_(2 * support_size < indexes, 0.0)
    indexes = indexes.masked_fill_(2 * support_size < indexes, 0.0)
    logits.scatter_(2, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
    return logits.view(b,-1)

def support_to_scalar(logits, support_size):
    """
    Transform a categorical representation to a scalar
    See paper appendix Network Architecture
    """
    # Decode to a scalar
    probabilities = torch.softmax(logits, dim=1)
    support = (
        torch.tensor([x for x in range(-support_size, support_size + 1)])
        .expand(probabilities.shape)
        .float()
        .to(device=probabilities.device)
    )
    x = torch.sum(support * probabilities, dim=1, keepdim=True)

    # Invert the scaling (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (
        ((torch.sqrt(1 + 4 * 0.001 * (torch.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))
        ** 2
        - 1
    )
    return x


def scalar_to_support(x, support_size):
    """
    Transform a scalar to a categorical representation with (2 * support_size + 1) categories
    See paper appendix Network Architecture
    """
    # Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x

    # Encode on a vector
    x = torch.clamp(x, -support_size, support_size)
    floor = x.floor()
    prob = x - floor
    logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1).to(x.device)
    logits.scatter_(
        2, (floor + support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1)
    )
    indexes = floor + support_size + 1
    prob = prob.masked_fill_(2 * support_size < indexes, 0.0)
    indexes = indexes.masked_fill_(2 * support_size < indexes, 0.0)
    logits.scatter_(2, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
    return logits


# ### Relational modules

class PositionalEncoding(nn.Module):
    """
    Adds two extra channels to the feature dimension, indicating the spatial 
    position (x and y) of each cell in the feature map using evenly spaced values
    between −1 and 1. Then projects the feature dimension to n_features through a 
    linear layer.
    """
    def __init__(self, n_kernels, n_features):
        super(PositionalEncoding, self).__init__()
        self.projection = nn.Linear(n_kernels + 2, n_features)

    def forward(self, x):
        """
        Accepts an input of shape (batch_size, linear_size, linear_size, n_kernels)
        Returns a tensor of shape (linear_size**2, batch_size, n_features)
        """
        x = self.add_encoding2D(x)
        if debug:
            print("x.shape (After encoding): ", x.shape)
        x = x.view(x.shape[0], x.shape[1],-1)
        if debug:
            print("x.shape (Before transposing and projection): ", x.shape)
        x = self.projection(x.transpose(2,1))
        x = x.transpose(1,0)
        
        if debug:
            print("x.shape (PositionalEncoding): ", x.shape)
        return x
    
    @staticmethod
    def add_encoding2D(x):
        x_ax = x.shape[-2]
        y_ax = x.shape[-1]
        
        x_lin = torch.linspace(-1,1,x_ax)
        xx = x_lin.repeat(x.shape[0],y_ax,1).view(-1, 1, y_ax, x_ax).transpose(3,2)
        
        y_lin = torch.linspace(-1,1,y_ax).view(-1,1)
        yy = y_lin.repeat(x.shape[0],1,x_ax).view(-1, 1, y_ax, x_ax).transpose(3,2)
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    
        x = torch.cat((x,xx.to(device),yy.to(device)), axis=1)
        return x


class PositionwiseFeedForward(nn.Module):
    """
    Applies 2 linear layers with ReLU and dropout layers
    only after the first layer.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class AttentionBlock(nn.Module):
    def __init__(self, n_features, n_heads, n_hidden=64, dropout=0.1):
        """
        Args:
          n_features: Number of input and output features. (d_model)
          n_heads: Number of attention heads in the Multi-Head Attention.
          n_hidden: Number of hidden units in the Feedforward (MLP) block. (d_k)
          dropout: Dropout rate after the first layer of the MLP and the two skip connections.
        """
        super(AttentionBlock, self).__init__()
        self.norm = nn.LayerNorm(n_features)
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.MultiheadAttention(n_features, n_heads, dropout)
        self.ff = PositionwiseFeedForward(n_features, n_hidden, dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
          x of shape (n_pixels**2, batch_size, n_features): Input sequences.
          mask of shape (batch_size, max_seq_length): Boolean tensor indicating which elements of the input
              sequences should be ignored.
        
        Returns:
          z of shape (max_seq_length, batch_size, n_features): Encoded input sequence.

        Note: All intermediate signals should be of shape (n_pixels**2, batch_size, n_features).
        """

        attn_output, attn_output_weights = self.attn(x,x,x, key_padding_mask=mask) # MHA step
        x_norm = self.dropout(self.norm(attn_output + x)) # add and norm
        z = self.ff(x_norm) # FF step
        return self.dropout(self.norm(z)) # add and norm


class RelationalModule(nn.Module):
    """Implements the relational module from paper Relational Deep Reinforcement Learning"""
    def __init__(self, n_kernels=24, n_features=256, n_heads=4, n_attn_modules=2, n_hidden=64, dropout=0):
        """
        Parameters
        ----------
        n_kernels: int (default 24)
            Number of features extracted for each pixel
        n_features: int (default 256)
            Number of linearly projected features after positional encoding.
            This is the number of features used during the Multi-Headed Attention
            (MHA) blocks
        n_heads: int (default 4)
            Number of heades in each MHA block
        n_attn_modules: int (default 2)
            Number of MHA blocks
        """
        super(RelationalModule, self).__init__()
        
        enc_layer = AttentionBlock(n_features, n_heads, n_hidden=n_hidden, dropout=dropout)
        
        #encoder_layers = clones(enc_layer, n_attn_modules)
        encoder_layers = nn.ModuleList([enc_layer for _ in range(n_attn_modules)])
        self.net = nn.Sequential(
            PositionalEncoding(n_kernels, n_features),
            *encoder_layers)
        
        #if debug:
        #    print(self.net)
        
    def forward(self, x):
        """Expects an input of shape (batch_size, n_pixels, n_kernels)"""
        x = self.net(x)
        if debug:
            print("x.shape (RelationalModule): ", x.shape)
        return x


class FeaturewiseMaxPool(nn.Module):
    """Applies max pooling along a given axis of a tensor"""
    def __init__(self, pixel_axis):
        super(FeaturewiseMaxPool, self).__init__()
        self.max_along_axis = pixel_axis
        
    def forward(self, x):
        x, _ = torch.max(x, axis=self.max_along_axis)
        if debug:
            print("x.shape (FeaturewiseMaxPool): ", x.shape)
        return x


class ResidualLayer(nn.Module):
    """
    Implements residual layer. Use LayerNorm and ReLU activation before applying the layers.
    """
    def __init__(self, n_features, n_hidden):
        super(ResidualLayer, self).__init__()
        self.norm = nn.LayerNorm(n_features)
        self.w1 = nn.Linear(n_features, n_hidden)
        self.w2 = nn.Linear(n_hidden, n_features)

    def forward(self, x):
        out = F.relu(self.w1(self.norm(x)))
        out = self.w2(out)
        return out + x
