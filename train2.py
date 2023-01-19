
from model import generator, discriminator
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
#import cuda
from model import _netlocalD,_netG
import utils
# device = torch.device("cuda")
# torch.cuda.init()
# # print(torch.cuda.is_available())
epochs=100
Batch_Size=64
lr=0.0002
beta1=0.5
over=4
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot',  default='trainingset/train', help='path to dataset')
opt = parser.parse_args()
try:
    os.makedirs("result/train/cropped")
    os.makedirs("result/train/real")
    os.makedirs("result/train/recon")
    os.makedirs("model")
except OSError:
    pass

transform = transforms.Compose([transforms.Resize(128),
                                    transforms.CenterCrop(128),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = dset.ImageFolder(root=opt.dataroot, transform=transform )
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=Batch_Size,
                                         shuffle=True, num_workers=2)

# ngpu = int(opt.ngpu)

wtl2 = 0.999

# custom weights initialization called on netG and netDcc
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


resume_epoch=0
netG = generator.generator()
netG.apply(weights_init)


netD = discriminator.discriminator()
netD.apply(weights_init)

criterion = nn.BCELoss()
criterionMSE = nn.MSELoss()

input_real = torch.FloatTensor(Batch_Size, 3, 128, 128)
input_cropped = torch.FloatTensor(Batch_Size, 3, 128, 128)
label = torch.FloatTensor(Batch_Size)
real_label = 1
fake_label = 0

real_center = torch.FloatTensor(Batch_Size, 3, 64,64)


netD.cuda()
netG.cuda()
criterion.cuda()
criterionMSE.cuda()
input_real, input_cropped,label = input_real.cuda(),input_cropped.cuda(), label.cuda()
real_center = real_center.cuda()


input_real = Variable(input_real)
input_cropped = Variable(input_cropped)
label = Variable(label)


real_center = Variable(real_center)

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

for epoch in range(resume_epoch,epochs):
    for i, data in enumerate(dataloader, 0):
        real_cpu, _ = data
        real_center_cpu = real_cpu[:,:,int(128/4):int(128/4)+int(128/2),int(128/4):int(128/4)+int(128/2)]
        batch_size = real_cpu.size(0)
        with torch.no_grad():
            input_real.resize_(real_cpu.size()).copy_(real_cpu)
            input_cropped.resize_(real_cpu.size()).copy_(real_cpu)
            real_center.resize_(real_center_cpu.size()).copy_(real_center_cpu)
            input_cropped[:,0,int(128/4+over):int(128/4+128/2-over),int(128/4+over):int(128/4+128/2-over)] = 2*117.0/255.0 - 1.0
            input_cropped[:,1,int(128/4+over):int(128/4+128/2-over),int(128/4+over):int(128/4+128/2-over)] = 2*104.0/255.0 - 1.0
            input_cropped[:,2,int(128/4+over):int(128/4+128/2-over),int(128/4+over):int(128/4+128/2-over)] = 2*123.0/255.0 - 1.0

        #start the discriminator by training with real data---
        netD.zero_grad()
        with torch.no_grad():
            label.resize_(batch_size).fill_(real_label)

        output = netD(real_center)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.data.mean()

        # train the discriminator with fake data---
        fake = netG(input_cropped)
        label.data.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()


        #train the generator now---
        netG.zero_grad()
        label.data.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG_D = criterion(output, label)

        wtl2Matrix = real_center.clone()
        wtl2Matrix.data.fill_(wtl2*10)
        wtl2Matrix.data[:,:,int(over):int(128/2 - over),int(over):int(128/2 - over)] = wtl2

        errG_l2 = (fake-real_center).pow(2)
        errG_l2 = errG_l2 * wtl2Matrix
        errG_l2 = errG_l2.mean()

        errG = (1-wtl2) * errG_D + wtl2 * errG_l2

        errG.backward()

        D_G_z2 = output.data.mean()
        optimizerG.step()

        print('[%d / %d][%d / %d] Loss_D: %.4f Loss_G: %.4f / %.4f l_D(x): %.4f l_D(G(z)): %.4f'
              % (epoch, epochs, i, len(dataloader),
                 errD.data, errG_D.data,errG_l2.data, D_x,D_G_z1, ))

        if i % 100 == 0:

            vutils.save_image(real_cpu,
                    'result/train/real/real_samples_epoch_%03d.png' % (epoch))
            vutils.save_image(input_cropped.data,
                    'result/train/cropped/cropped_samples_epoch_%03d.png' % (epoch))
            recon_image = input_cropped.clone()
            recon_image.data[:,:,int(128/4):int(128/4+128/2),int(128/4):int(128/4+128/2)] = fake.data
            vutils.save_image(recon_image.data,
                    'result/train/recon/recon_center_samples_epoch_%03d.png' % (epoch))
