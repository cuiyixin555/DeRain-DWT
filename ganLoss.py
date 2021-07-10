import torch
import torch.nn as nn
import torch.autograd as autograd

LAMBDA = 0.1
BATCH_SIZE = 12

# def calc_gradient_penalty(netD, real_data, fake_data):
#     alpha = torch.rand(BATCH_SIZE, 1)
#     alpha = alpha.expand(real_data.size())
#     alpha = alpha.cuda()
#
#     interpolates = alpha * real_data + ((1 - alpha) * fake_data)
#
#     interpolates = interpolates.cuda()
#     interpolates = autograd.Variable(interpolates, requires_grad=True)
#
#     disc_interpolates = netD(interpolates)
#
#     gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
#                               grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
#                               create_graph=True, retain_graph=True, only_inputs=True)[0]
#
#     gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
#
#     return gradient_penalty

def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA):
    MSGGan = False
    if  MSGGan:
        alpha = torch.rand(1, 1)
        alpha = alpha.cuda()  # cuda() #gpu) #if use_cuda else alpha

        interpolates = [alpha * rd + ((1 - alpha) * fd) for rd, fd in zip(real_data, fake_data)]
        interpolates = [i.cuda() for i in interpolates]
        interpolates = [torch.autograd.Variable(i, requires_grad=True) for i in interpolates]

        disc_interpolates = netD(interpolates)
    else:
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda()  # cuda() #gpu) #if use_cuda else alpha

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.cuda()#.cuda()
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),#.cuda(), #if use_cuda else torch.ones(
                                  #disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    #LAMBDA = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

    return gradient_penalty