import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F
 
N_CH = 256

class SymmetriesGenerator(nn.Module):
    def __init__(self, rotations=True):
        super(SymmetriesGenerator, self).__init__()
        self.rotations = rotations

    @torch.no_grad()
    def forward(self, boards, policies, values):
        def _symmerty(t, dims):
            h = torch.flip(t[:, 0, :, :-1], dims)
            h = torch.cat((h, t[:, 0, :, -1].unsqueeze(2)), 2)
            
            v = torch.flip(t[:, 1, :-1, :], dims)
            v = torch.cat((v, t[:, 1, -1, :].unsqueeze(1)), 1)

            if t.size()[1] == 3:
                s = t[:, 2, :, :]
                result = torch.cat((h.unsqueeze(1),v.unsqueeze(1), s.unsqueeze(1)), 1)
            else:
                result = torch.cat((h.unsqueeze(1),v.unsqueeze(1)), 1)
            return result

        def _rotate(t):
            h = t[:, 1, :-1, :].transpose(1,2)
            h = torch.cat((h, t[:, 1, -1, :].unsqueeze(2)), 2)
            v = t[:, 0, :, :-1].transpose(1,2)
            v = torch.cat((v, t[:, 0, :, -1].unsqueeze(1)), 1)
            if t.size()[1] == 3:
                s = t[:, 2, :, :]
                return torch.cat((h.unsqueeze(1),v.unsqueeze(1), s.unsqueeze(1)), 1)
            else:
                return torch.cat((h.unsqueeze(1),v.unsqueeze(1)), 1)

        policies = policies.reshape(-1, 2, *boards.size()[-2:])
        results = [(boards, policies, values)]
        results.append((_symmerty(boards, [1]), _symmerty(policies, [1]), values))
        results.append((_symmerty(boards, [2]), _symmerty(policies, [2]), values))
        results.append((_symmerty(boards, [1, 2]), _symmerty(policies, [1, 2]), values))

        if self.rotations:
            results += list((_rotate(f), _rotate(p), v) for f, p, v in results)
        

        batch_size = boards.size()[0]
        boards, policies, values = zip(*results)

        boards = torch.cat(boards)
        policies = torch.cat(policies, 0)
        policies = policies.reshape(policies.size()[0], -1)
        values = torch.cat(values)

        perm = torch.randperm(boards.size()[0], device=values.device)
        boards = boards[perm].split(batch_size)
        policies = policies[perm].split(batch_size)
        values = values[perm].split(batch_size)
        return boards, policies, values


class SimpleNN(nn.Module):
    def __init__(self, params=None):
        super(SimpleNN, self).__init__()
        self.params = params
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

    def save_parameters(self, generation):
        filename = self.params.nn.chkpts_filename
        fn = filename.format(generation)
        logger.info("Model saved to: %s", fn)
        torch.save(self.state_dict(), fn)

    def load_parameters(self, generation, to_device=None):
        filename = self.params.nn.chkpts_filename
        fn = filename.format(generation)
        logger.info("Model loaded from: %s", fn)
        self.load_state_dict(torch.load(fn, map_location='cpu'))
        self.to(to_device)