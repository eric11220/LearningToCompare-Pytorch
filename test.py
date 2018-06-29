import argparse
import torch, os
import numpy as np
from torch import optim
from torch.autograd import Variable
from MiniImagenet import MiniImagenet
from compare import Compare
from torchvision.transforms import transforms
from PIL import Image
from utils import make_imgs

batch_size = 10
n_way = 20
k_shot = 5
image_dir = "../mini-imagenet/images"
test_img_dir = "../mini-imagenet//test"
novel_class = ["class_00", "class_23", "class_32", "class_48", "class_57", \
               "class_60", "class_66", "class_71", "class_91", "class_93", \
               "class_10", "class_30", "class_35", "class_54", "class_59", \
               "class_64", "class_69", "class_82", "class_92", "class_95"]

def parse_inputs():
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt', help="Saved checkpoint")
    parser.add_argument('--mode', help="Evaluate or testing", default="eval")
    return parser.parse_args()

if __name__ == '__main__':
    from MiniImagenet import MiniImagenet
    from torch.utils.data import DataLoader
    from torchvision.utils import make_grid
    from tensorboardX import SummaryWriter
    from datetime import datetime

    args = parse_inputs()

    # Multi-GPU support
    print('To run on single GPU, change device_ids=[0] and downsize batch size! \nmkdir ckpt if not exists!')
    net = torch.nn.DataParallel(Compare(n_way, k_shot), device_ids=[0]).cuda()

    if os.path.exists(args.ckpt):
        print('load checkpoint ...', args.ckpt)
        net.load_state_dict(torch.load(args.ckpt))
    net.eval()

    transform = transforms.Compose([ lambda x: Image.open(x).convert('RGB'),
                                               transforms.ToTensor() ])

    # Need to load all novel support images first
    support_cnt = {}
    support_x, support_y = [], []
    with open("../mini-imagenet/train.csv", "r") as inf:
        for line in inf:
            path, cls = line.strip().split(",")
            path = os.path.join(image_dir, path)
            if cls not in novel_class:
                continue

            if support_cnt.get(cls, 0) >= k_shot:
                continue
            support_cnt[cls] = 1 if support_cnt.get(cls, 0) == 0 \
                                 else support_cnt[cls] + 1

            support_x.append(path)
            support_y.append(int(cls.split('_')[1]))

    support_x = np.asarray(support_x)
    support_y = np.asarray(support_y, dtype=np.uint8)

    ind = np.argsort(support_y)
    support_x = support_x[ind]
    support_y = support_y[ind]

    support_x = [transform(path) for path in support_x]
    support_x = torch.stack(support_x)
    support_y = torch.from_numpy(support_y)

    support_x = Variable(support_x).cuda()
    support_y = Variable(support_y).cuda()

    n_support, n_channel, h, w = support_x.size()
    support_x = support_x.expand(1, n_support, n_channel, h, w)
    support_y = support_y.expand(1, n_support)

    if args.mode == "eval":
        paths, labels = [], []
        with open("../mini-imagenet/val.csv", 'r') as inf:
            for line in inf:
                name, cls = line.strip().split(',')
                paths.append(os.path.join("../mini-imagenet/images", name))
                labels.append(int(cls.split('_')[1]))
        labels = np.asarray(labels, dtype=np.uint8)
    else:
        img_ids = np.asarray([path.split('.')[0] for path in sorted(os.listdir(test_img_dir))], dtype=np.uint32)
        paths = [os.path.join(test_img_dir, path) for path in sorted(os.listdir(test_img_dir))]

    preds = None
    start, total_correct = 0, 0
    while start < len(paths):
        end = start + batch_size
        query_x = torch.stack([transform(path) for path in paths[start:end]])
        query_x = query_x.expand(1, len(query_x), n_channel, h, w)

        if args.mode == "eval":
            query_y = torch.from_numpy(labels[start:end]).cuda()
            query_y = query_y.expand(1, len(query_y))
        else:
            query_y = support_y[:, :len(query_x)]

        start += batch_size
        pred, correct = net(support_x, support_y, query_x, query_y, False)
        pred = pred[0].data.cpu().numpy()
        preds = pred if preds is None else np.concatenate((preds, pred))

        correct = correct.sum() # multi-gpu support
        total_correct += correct.item()

    if args.mode != "eval":
        ind = np.argsort(img_ids)
        img_ids = img_ids[ind]
        preds = preds[ind]

        with open("pred.csv", "w") as outf:
            outf.write("image_id,predicted_label\n")
            for img_id, pred in zip(img_ids, preds):
                outf.write("%s,%d\n" % (img_id, pred))

    print('<<<<>>>>accuracy:', total_correct / len(paths))
