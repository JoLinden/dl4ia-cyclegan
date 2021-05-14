import sys

import matplotlib.pyplot as plt
import numpy as np

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

criterion_cycle = torch.nn.L1Loss()

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

best_cycle_loss = np.inf
best_cycle_image = {}
loss_cycle_ABA_list = []
loss_cycle_BAB_list = []

for i, batch in enumerate(dataloader):
    # Set model input
    real_A = Variable(input_A.copy_(batch['A']))
    real_B = Variable(input_B.copy_(batch['B']))

    # Generate output
    fake_B_raw = netG_A2B(real_A).data
    fake_A_raw = netG_B2A(real_B).data
    fake_B = 0.5 * (fake_B_raw + 1.0)
    fake_A = 0.5 * (fake_A_raw + 1.0)

    # Save image files
    save_image(fake_A, 'output/A/%04d.png' % (i + 1))
    save_image(fake_B, 'output/B/%04d.png' % (i + 1))

    recovered_A = 0.5 * (netG_B2A(fake_B_raw) + 1.0)
    recovered_B = 0.5 * (netG_A2B(fake_A_raw) + 1.0)

    save_image(recovered_A, f'output/recovered/A/{(i+1):04d}.png')
    save_image(recovered_B, f'output/recovered/B/{(i+1):04d}.png')

    loss_cycle_ABA = criterion_cycle(recovered_A, real_A)
    loss_cycle_BAB = criterion_cycle(recovered_B, real_B)

    loss_cycle_ABA_list.append(loss_cycle_ABA.item())
    loss_cycle_BAB_list.append(loss_cycle_BAB.item())

    if loss_cycle_ABA < best_cycle_loss and loss_cycle_ABA < loss_cycle_BAB:
        best_cycle_loss = loss_cycle_ABA
        best_cycle_image = {
            'real': 0.5 * (real_A.cpu() + 1.0).squeeze(0).permute(1, 2, 0).numpy(),
            'fake': fake_B.cpu().squeeze(0).permute(1, 2, 0).numpy(),
            'recovered': recovered_A.cpu().squeeze(0).permute(1, 2, 0).detach().numpy()
        }
    elif loss_cycle_BAB < best_cycle_loss:
        best_cycle_loss = loss_cycle_BAB
        best_cycle_image = {
            'real': 0.5 * (real_B.cpu() + 1.0).squeeze(0).permute(1, 2, 0).numpy(),
            'fake': fake_A.cpu().squeeze(0).permute(1, 2, 0).numpy(),
            'recovered': recovered_B.cpu().squeeze(0).permute(1, 2, 0).detach().numpy()
        }
    sys.stdout.write(
        '\rGenerated images %04d of %04d' % (i + 1, len(dataloader)))

sys.stdout.write('\n')

real = best_cycle_image['real']
fake = best_cycle_image['fake']
recovered = best_cycle_image['recovered']

plt.tight_layout()
fig, ax = plt.subplots(1, 3, sharey='all')
ax[0].imshow(real)
ax[0].axis('off')
ax[0].set_title('Real')
ax[1].imshow(fake)
ax[1].axis('off')
ax[1].set_title('Fake')
ax[2].imshow(recovered)
ax[2].axis('off')
ax[2].set_title('Recovered')
plt.savefig('figures/best_cycle.png', bbox_inches='tight', pad_inches=0,
            dpi=300)
plt.show()

print(f'Best cycle loss: {best_cycle_loss}')
print(f'Average ABA loss: {np.mean(loss_cycle_ABA_list)}')
print(f'Average BAB loss: {np.mean(loss_cycle_BAB_list)}')
