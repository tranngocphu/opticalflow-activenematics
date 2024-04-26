import sys
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(abspath(__file__))))

import argparse
import torch
import imageio.v2 as imageio
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from RAFT.core.raft import RAFT
from RAFT.core.utils.utils import InputPadder


def get_velocity(image1, image2, args):

    image1 = Image.open(image1).convert('RGB')
    image2 = Image.open(image2).convert('RGB')

    image1 = transforms.ToTensor()(image1).unsqueeze(0)
    image2 = transforms.ToTensor()(image2).unsqueeze(0)

    padder = InputPadder(image1.shape, 8)
    image1, image2 = padder.pad(image1, image2)

    with torch.no_grad():
        _, flow_forward = model(image1, image2, iters=args.iter, test_mode=True)
    
    flow_fw = padder.unpad(flow_forward).squeeze(0).permute(1, 2, 0).cpu().numpy()
   
    return flow_fw



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    # define model params
    parser.add_argument('--model', help="restore checkpoint", default='./models/weights.pth')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--save_location', help="save the results in local or oss")
    parser.add_argument('--save_path', help=" local path to save the result")
    parser.add_argument('--iter', type=int, default=24)
    parser.add_argument('--alpha', default=0.75)
    parser.add_argument('--splatting', default='max')    
    args = parser.parse_args()

    
    # load model
    torch.cuda.empty_cache()    
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))    
    model.cuda()
    model.eval()
    
    # compute a single velocity field from a pair of images
    image1 = "./example/data/img_000000000_Red_000.tif"
    image2 = "./example/data/img_000000001_Red_000.tif"    
    flows = get_velocity(image1, image2, args)
    vx, vy = flows[:,:,0], flows[:,:,1]

    # visualize flow field overlay on first image
    fig, ax = plt.subplots()
    im1 = plt.imread(image1)
    ax.imshow(im1, cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    n = 20  # grid size for ploting velocity vectors
    x, y = np.meshgrid(np.arange(0, vx.shape[1], n), np.arange(0, vx.shape[0], n))
    ax.quiver(x, y, vx[::n, ::n], vy[::n, ::n], color="red", scale_units="xy", angles="xy", scale=0.5, width=0.001)
    fig.savefig('./example/velocity_plot.png', dpi=150)