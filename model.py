import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embeds = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        
        self.fc1 = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        captions = captions[:, :-1] 
        captions = self.embeds(captions)

        features = features.view(features.shape[0], 1, -1)
        feat = torch.cat((features, captions), dim=1)

        lstm_out, _ = self.lstm(feat)
        out = self.fc1(lstm_out)

        return out

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        texts = []
        count = 0
        while count < max_len:
            if count != 0:   
                inputs = self.embeds(inputs)
                
            inputs = inputs.view(inputs.shape[0], 1, -1)   
            lstm_out, states = self.lstm(inputs, states)
            output = self.fc1(lstm_out)
            output = output.view(1, -1)
            _, top_class = torch.max(output, 1)   

            texts.append(top_class.item())
            if top_class == 1:
                break
            
            inputs = top_class
            count += 1
        return texts
