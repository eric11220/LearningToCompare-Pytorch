import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from repnet import repnet_deep, Bottleneck

from cnnencoder import CNNEncoder

class ResnetConv(nn.Module):
    def __init__(self, orig_model):
        super(ResnetConv, self).__init__()
        self.features = nn.Sequential(
            # stop at conv4
            *list(orig_model.children())[:-2]
        )
    def forward(self, x):
        x = self.features(x)
        return x

class Compare(nn.Module):
    def __init__(self, n_way, k_shot, resume=None, hidden_size=8, input_size=64):
        super(Compare, self).__init__()

        self.n_way = n_way
        self.k_shot = k_shot

        # Load pre-trained model

        if resume is not None:
            model = resnet(num_classes=80, depth=20)
            model = torch.nn.DataParallel(model).cuda()
            checkpoint = torch.load(resume)
            model.load_state_dict(checkpoint['state_dict'])

            model = ResnetConv(model.module)
            self.repnet = model
        else:
            self.repnet = CNNEncoder().cuda()

        # we need to know the feature dim, so here is a forwarding.
        repnet_sz = self.repnet(Variable(torch.rand(2, 3, 32, 32)).cuda()).size()
        self.c = repnet_sz[1]
        self.d = repnet_sz[2]

        # this is the input channels of layer1&layer2
        self.inplanes = 2 * self.c
        assert repnet_sz[2] == repnet_sz[3]
        print('repnet sz:', repnet_sz)

        # after relational module
        self.layer1 = nn.Sequential(
                        nn.Conv2d(128,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))

        self.fc = nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size,1),
            nn.Sigmoid()
        )

    def forward(self, support_x, support_y, query_x, query_y, train=True):
        batchsz, setsz, c_, h, w = support_x.size()
        querysz = query_x.size(1)
        c, d = self.c, self.d

        support_xf = self.repnet(support_x.view(batchsz * setsz, c_, h, w)).view(batchsz, setsz, c, d, d)
        query_xf = self.repnet(query_x.view(batchsz * querysz, c_, h, w)).view(batchsz, querysz, c, d, d)

        support_xf = support_xf.unsqueeze(1).expand(-1, querysz, -1, -1, -1, -1)
        query_xf = query_xf.unsqueeze(2).expand(-1, -1, setsz, -1, -1, -1)

        comb = torch.cat([support_xf, query_xf], dim=3)
        comb = comb.view(batchsz * querysz * setsz, 2 * c, d, d)
        comb = self.layer1(comb)
        comb = self.layer2(comb)
        score = self.fc(comb.view(batchsz * querysz * setsz, -1)).view(batchsz, querysz, setsz, 1).squeeze(3)

        support_yf = support_y.unsqueeze(1).expand(batchsz, querysz, setsz)
        query_yf = query_y.unsqueeze(2).expand(batchsz, querysz, setsz)
        label = torch.eq(support_yf, query_yf).float()

        # score: [b, querysz, setsz]
        # label: [b, querysz, setsz]
        if train:
            loss = torch.pow(label - score, 2).sum() / batchsz
            return loss

        else:
            # [b, querysz, setsz]
            rn_score_np = score.cpu().data.numpy()
            pred = []
            # [b, setsz]
            support_y_np = support_y.cpu().data.numpy()
            for i, batch in enumerate(rn_score_np):
                for j, query in enumerate(batch):
                    # query: [setsz]
                    sim = []  # [n_way]
                    for way in range(self.n_way):
                        sim.append(np.sum(query[way * self.k_shot: (way + 1) * self.k_shot]))
                    idx = np.array(sim).argmax()
                    pred.append(support_y_np[i, idx * self.k_shot])
            # pred: [b, querysz]
            pred = Variable(torch.from_numpy(np.array(pred).reshape((batchsz, querysz)))).cuda()

            correct = torch.eq(pred, query_y).sum()
            return pred, correct
