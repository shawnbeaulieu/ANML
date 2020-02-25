import logging
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger("experiment")


def batchnorm(input, weight=None, bias=None, running_mean=None, running_var=None, training=True, eps=1e-5, momentum=0.1):
    ''' momentum = 1 restricts stats to the current mini-batch '''
    # This hack only works when momentum is 1 and avoids needing to track running stats
    # by substuting dummy variables
    running_mean = torch.zeros(np.prod(np.array(input.data.size()[1])))
    running_var = torch.ones(np.prod(np.array(input.data.size()[1])))
    return F.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)

def maxpool(input, kernel_size, stride=None):
    return F.max_pool2d(input, kernel_size, stride)

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

class Learner(nn.Module):
    """

    """

    def __init__(self, config):
        """

        :param config: network config file, type:list of (string, list)
        :param imgc: 1 or 3
        :param imgsz:  28 or 84
        """
        super(Learner, self).__init__()

        self.config = config
        self.neuromodulation = True

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()

        for i, (name, param) in enumerate(self.config):
            if 'conv' in name:
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'convt2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1])))

            elif 'linear' in name or 'nm_to_' in name or 'fc' in name:

                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'cat':
                pass
            elif name is 'cat_start':
                pass
            elif name is "rep":
                pass
            elif 'bn' in name:
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])


            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError

    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name is 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)' \
                      % (param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)' \
                      % (param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'linear':
                tmp = 'linear:(in:%d, out:%d)' % (param[1], param[0])
                info += tmp + '\n'

            elif name is 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)' % (param[0])
                info += tmp + '\n'

            elif name is 'cat':
                tmp = 'cat'
                info += tmp + "\n"
            elif name is 'cat_start':
                tmp = 'cat_start'
                info += tmp + "\n"

            elif name is 'rep':
                tmp = 'rep'
                info += tmp + "\n"


            elif name is 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)' % (param[0], param[1], param[2])
                info += tmp + '\n'
            elif name is 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)' % (param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info

    def forward(self, x, nm_input, vars=None, bn_training=True, feature=False):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """
        cat_var = False
        cat_list = []

        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0
        
        if True:
            # =========== NEUROMODULATORY NETWORK ===========

            #'conv1_nm'
            #'bn1_nm'
            #'conv2_nm'
            #'bn2_nm'
            #'conv3_nm'
            #'bn3_nm'
            #'nm_to_conv1'
            #'nm_to_conv2'
            #'nm_to_conv3'

          
            # Query the neuromodulatory network:
            #print(x.size(0))
            #for i in range(x.size(0)):
            
            data = x#[i].view(1,3,28,28)

            #if x.size(0) > 1:
            data_ = x#nm_input#.view(1,3,28,28)

            w,b = vars[0], vars[1]
            data_ = conv2d(data_, w, b)
            w,b = vars[2], vars[3]
            running_mean, running_var = self.vars_bn[0], self.vars_bn[1]
            data_ = F.batch_norm(data_, running_mean, running_var, weight=w, bias=b, training=True)
                #data_ = batchnorm(data_, w, b, momentum=1)
            data_ = F.relu(data_)
            data_ = maxpool(data_, kernel_size=2, stride=2)

            w,b = vars[4], vars[5]
            data_ = conv2d(data_, w, b)
            w,b = vars[6], vars[7]
            running_mean, running_var = self.vars_bn[2], self.vars_bn[3]
            data_ = F.batch_norm(data_, running_mean, running_var, weight=w, bias=b, training=True)

                #data_ = batchnorm(data_, w, b, momentum=1)
            data_ = F.relu(data_)
            data_ = maxpool(data_, kernel_size=2, stride=2)

            w,b = vars[8], vars[9]
            data_ = conv2d(data_, w, b)
            w,b, = vars[10], vars[11]
            running_mean, running_var = self.vars_bn[4], self.vars_bn[5]
            data_ = F.batch_norm(data_, running_mean, running_var, weight=w, bias=b, training=True)
                #data_ = batchnorm(data_, w, b, momentum=1)
            data_ = F.relu(data_)
            data_ = maxpool(data_, kernel_size=2, stride=2)

                #data_ = data_.view(data_.size(0), 64)
            
            w,b = vars[12], vars[13]
            conv1_mask = F.sigmoid(F.linear(data_.view(data_.size(0), 64), w, b)).view(64,3,3,3)

            w,b = vars[14], vars[15]
            conv2_mask = F.sigmoid(F.linear(data_.view(data_.size(0), 64), w, b)).view(64,64,3,3)

            w,b = vars[16], vars[17]
            conv3_mask = F.sigmoid(F.linear(data_.view(data_.size(0), 64), w, b)).view(64,64,3,3)

            w,b = vars[18], vars[19]
            fc_mask = F.sigmoid(F.linear(data_.view(data_.size(0), 64), w,b)).view(1000, 64)

                # =========== PREDICTION NETWORK ===========

                #'conv1'
                #'bn1'
                #'conv2'
                #'bn2'
                #'conv3'
                #'bn3'
                #'fc'


                # Present original data (x) to the prediction network:

            w,b = vars[20], vars[21]
           
            #if x.size(0) > 1:
            w = w*conv1_mask # Modulate the first set of convolutional weights
            
            data = conv2d(data, w, b)
            w,b = vars[22], vars[23]
            running_mean, running_var = self.vars_bn[6], self.vars_bn[7]
            data = F.batch_norm(data, running_mean, running_var, weight=w, bias=b, training=True)

            #data = batchnorm(data, w, b, momentum=1)
            data = F.relu(data)
            data = maxpool(data, kernel_size=2, stride=2)

            w,b = vars[24], vars[25]
          
            #if x.size(0) > 1:
            w = w*conv2_mask # Modulate the second set of convolutional weights
            
            data = conv2d(data, w, b)
            w,b = vars[26], vars[27]
            running_mean, running_var = self.vars_bn[8], self.vars_bn[9]
            data = F.batch_norm(data, running_mean, running_var, weight=w, bias=b, training=True)
               
            #data = batchnorm(data, w, b, momentum=1)
            data = F.relu(data)
            data = maxpool(data, kernel_size=2, stride=2)
            w,b = vars[28], vars[29]
            
            #if x.size(0) > 1:
            w = w*conv3_mask

            data = conv2d(data, w, b)
            w,b, = vars[30], vars[31]
            running_mean, running_var = self.vars_bn[10], self.vars_bn[11]
            data = F.batch_norm(data, running_mean, running_var, weight=w, bias=b, training=True)
                
            #data = batchnorm(data, w, b, momentum=1)
            data = F.relu(data)
            data = maxpool(data, kernel_size=2, stride=2)

            #data = data.view(data.size(0), 64)
            w,b = vars[32], vars[33]
            #w = w*fc_mask
            data = F.linear(data.view(data.size(0), 64), w, b)
            #data = F.relu(data)                

            #w,b = vars[32], vars[33]
            #data = F.linear(data, w, b)

                #try:
                #    prediction = torch.cat([prediction, data], dim=0)
                #except:
                #    prediction = data


        else:

            for name, param in self.config:
                # assert(name == "conv2d")
                if name == 'conv2d':
                    w, b = vars[idx], vars[idx + 1]
                    x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                    idx += 2
                    # print(name, param, '\tout:', x.shape)
                elif name == 'convt2d':
                    w, b = vars[idx], vars[idx + 1]
                    x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                    idx += 2
                elif name == 'linear':

                    w, b = vars[idx], vars[idx + 1]
                    x = F.linear(x, w, b)
                    if cat_var:
                        cat_list.append(x)
                    idx += 2

                elif name == 'rep':
                    # print(x.shape)
                    if feature:
                        return x
                elif name == "cat_start":
                    cat_var = True
                    cat_list = []

                elif name == "cat":
                    cat_var = False
                    x = torch.cat(cat_list, dim=1)

                elif name == 'bn':
                    if not self.neuromodulation:
                        w, b = vars[idx], vars[idx + 1]
                        running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
                        x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                        idx += 2
                        bn_idx += 2
                elif name == 'flatten':
                    # print(x.shape)

                    x = x.view(x.size(0), -1)

                elif name == 'reshape':
                    # [b, 8] => [b, 2, 2, 2]
                    x = x.view(x.size(0), *param)
                elif name == 'relu':
                    x = F.relu(x, inplace=param[0])
                elif name == 'leakyrelu':
                    x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
                elif name == 'tanh':
                    x = F.tanh(x)
                elif name == 'sigmoid':
                    x = torch.sigmoid(x)
                elif name == 'upsample':
                    x = F.upsample_nearest(x, scale_factor=param[0])
                elif name == 'max_pool2d':
                    x = F.max_pool2d(x, param[0], param[1], param[2])
                elif name == 'avg_pool2d':
                    x = F.avg_pool2d(x, param[0], param[1], param[2])

                else:
                    raise NotImplementedError

        # make sure variable is used properly
        #assert idx == len(vars)
        #assert bn_idx == len(self.vars_bn)
        return data#prediction

    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars
