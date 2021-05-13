import sys

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from datasets import ImageDataset
from models import Generator

input_nc = 3
output_nc = 3
dataroot = 'data/horse2zebra/'
batch_size = 1
image_size = 256
n_cpu = 8
cuda = True
generator_A2B = 'output/netG_A2B.pth'
generator_B2A = 'output/netG_B2A.pth'

netG_A2B = Generator(input_nc, output_nc)
netG_B2A = Generator(output_nc, input_nc)

if cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()

# Load state dicts
netG_A2B.load_state_dict(torch.load(generator_A2B))
netG_B2A.load_state_dict(torch.load(generator_B2A))

netG_A2B.eval()
netG_B2A.eval()

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
input_A = Tensor(batch_size, input_nc, image_size, image_size)
input_B = Tensor(batch_size, output_nc, image_size, image_size)

transforms_ = [transforms.ToTensor(),
               transforms.Normalize((0.5,), (0.5,))]
dataloader = DataLoader(
    ImageDataset(dataroot, transforms_=transforms_, mode='test'),
    batch_size=batch_size, shuffle=False, num_workers=n_cpu)

for i, batch in enumerate(dataloader):
    # Set model input
    real_A = Variable(input_A.copy_(batch['A']))
    real_B = Variable(input_B.copy_(batch['B']))

    # Generate output
    fake_B = 0.5 * (netG_A2B(real_A).data + 1.0)
    fake_A = 0.5 * (netG_B2A(real_B).data + 1.0)

    # Save image files
    save_image(fake_A, 'output/A/%04d.png' % (i + 1))
    save_image(fake_B, 'output/B/%04d.png' % (i + 1))

    sys.stdout.write(
        '\rGenerated images %04d of %04d' % (i + 1, len(dataloader)))

sys.stdout.write('\n')
