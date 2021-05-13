import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import ImageDataset
from models import Discriminator, Generator
from utils import LambdaLR, ReplayBuffer

input_nc = 3
output_nc = 3
starting_epoch = 0
n_epochs = 50
dataroot = 'data/horse2zebra/'
batch_size = 1
image_size = 256
n_cpu = 8
learning_rate = 0.002
cuda = True
decay_epoch = 40

netG_A2B = Generator(input_nc, output_nc)
netG_B2A = Generator(output_nc, input_nc)
netD_A = Discriminator(input_nc)
netD_B = Discriminator(output_nc)

if cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()

criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=learning_rate, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=learning_rate, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(n_epochs, starting_epoch, decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(n_epochs, starting_epoch, decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(n_epochs, starting_epoch, decay_epoch).step)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
input_A = Tensor(batch_size, input_nc, image_size, image_size)
input_B = Tensor(batch_size, output_nc, image_size, image_size)
target_real = Variable(Tensor(batch_size).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(batch_size).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

transforms_ = [ transforms.Resize(int(image_size*1.12), Image.BICUBIC),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)) ]
dataloader = DataLoader(ImageDataset(dataroot, transforms_=transforms_, unaligned=True),
                        batch_size=batch_size, shuffle=True, num_workers=n_cpu)

losses = {'D_A': [],
          'G_A': [],
          'G_B': [],
          'D_B': [],
          'identity_A': [],
          'identity_B': [],
          'cycle_ABA': [],
          'cycle_BAB': []}

for epoch in range(starting_epoch, n_epochs):
    start = time.time()
    for i, batch in enumerate(dataloader):
        if i % 10 == 0:
            print(f'Epoch {epoch}, Batch {i}')

        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B) * 5.0
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A) * 5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()

        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward()

        optimizer_D_B.step()

    losses['G_A'].append(loss_GAN_A2B)
    losses['D_A'].append(loss_D_A)
    losses['G_B'].append(loss_GAN_B2A)
    losses['D_B'].append(loss_D_B)
    losses['identity_A'].append(loss_identity_A)
    losses['identity_B'].append(loss_identity_B)
    losses['cycle_ABA'].append(loss_cycle_ABA)
    losses['cycle_BAB'].append(loss_cycle_BAB)

        ###################################

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Save models checkpoints
    torch.save(netG_A2B.state_dict(), 'output/netG_A2B.pth')
    torch.save(netG_B2A.state_dict(), 'output/netG_B2A.pth')
    torch.save(netD_A.state_dict(), 'output/netD_A.pth')
    torch.save(netD_B.state_dict(), 'output/netD_B.pth')

    print(f'Epoch finished in {(time.time() - start):.2f} seconds\n')

epochs_axis = np.linspace(1, n_epochs, len(losses['G_A']))
plt.figure()
plt.plot(epochs_axis, losses['G_A'], label='G_A')
plt.plot(epochs_axis, losses['D_A'], label='D_A')
plt.plot(epochs_axis, losses['G_B'], label='G_B')
plt.plot(epochs_axis, losses['D_B'], label='D_B')
plt.plot(epochs_axis, losses['identity_A'], label='identity_A')
plt.plot(epochs_axis, losses['identity_B'], label='identity_B')
plt.plot(epochs_axis, losses['cycle_ABA'], label='cycle_ABA')
plt.plot(epochs_axis, losses['cycle_BAB'], label='cycle_BAB')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('figures/losses.png')
