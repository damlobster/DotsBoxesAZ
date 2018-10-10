import torch
import torch.nn as nn
import torch.nn.functional as F
 
N_CH = 256

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.conv0 = nn.Conv2d(3, N_CH, 3, padding=1)
        self.bn0 = nn.BatchNorm2d(N_CH)
        self.conv1 = nn.Conv2d(N_CH, N_CH, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(N_CH)
        self.conv2 = nn.Conv2d(N_CH, N_CH, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(N_CH)
        self.conv3 = nn.Conv2d(N_CH, N_CH, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(N_CH)
        self.conv4 = nn.Conv2d(N_CH, N_CH, 3, padding=0)
        self.bn4 = nn.BatchNorm2d(N_CH)

        self.fc0 = nn.Linear(1024, 512)
        self.bn_fc0 = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(512, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        
        self.value_fc = nn.Linear(256, 1)

        self.policy_fc = nn.Linear(256, 32)

    def forward(self, x):
        x = self.bn0(F.relu(self.conv0(x)))
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = x.view(x.size()[0], -1)
        x = self.bn_fc0(F.relu(self.fc0(x)))
        x = self.bn_fc1(F.relu(self.fc1(x)))

        value = torch.tanh(self.value_fc(x))

        policy = F.log_softmax(self.policy_fc(x), dim=1)

        return policy, value

    def save_parameters(self, filename):
        print("Model saved to:", filename)
        torch.save(self.state_dict(), filename)

    def load_parameters(self, filename):
        print("Model loaded from:", filename)
        self.load_state_dict(torch.load(filename))
