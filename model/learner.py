import logging
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F



logger = logging.getLogger("experiment")







def batchnorm(input, weight=None, bias=None, running_mean=None, running_var=None, training=True, eps=1e-5, momentum=0.1):
    ''' momentum = 1 restricts stats to the current mini-batch：momentum = 1限制了对当前小批量的统计'''




    running_mean = torch.zeros(np.prod(np.array(input.data.size()[1])))
    running_var = torch.ones(np.prod(np.array(input.data.size()[1])))
    return F.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)

def maxpool(input, kernel_size, stride=None):
    return F.max_pool2d(input, kernel_size, stride)

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

def convt2d(input, weight, bias=None, stride=2, padding=0, output_padding=0, groups=1):
    return F.conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups)




class Learner(nn.Module):

    def __init__(self, config, neuromodulation=None, autoencoder=None, oml=None):
        """
        :param config: network config file, type:list of (string, list)
        :param imgc: 1 or 3
        :param imgsz:  28 or 84
        https://blog.csdn.net/yaoxunji/article/details/107673937
        """


        super(Learner, self).__init__()

        self.config = config
        self.Neuromodulation = neuromodulation
        self.Autoencoder = autoencoder
        self.Oml = oml

        self.vars = nn.ParameterList()

        self.vars_bn = nn.ParameterList()

        self.vars_auto_bn = nn.ParameterList()


        for i, (name, param) in enumerate(self.config):
            if name == 'conv2d' or (name.startswith('conv') and not name.startswith('convt')):

                w = nn.Parameter(torch.ones(*param[:4]))

                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)

                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name == 'convt2d':

                w = nn.Parameter(torch.ones(*param[:4]))

                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)

                self.vars.append(nn.Parameter(torch.zeros(param[1])))

            elif name == 'linear':

                w = nn.Parameter(torch.ones(*param))

                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)

                self.vars.append(nn.Parameter(torch.zeros(param[0])))


            elif 'nm_to' in name or name == 'fc':


                w = nn.Parameter(torch.ones(*param))

                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)

                







                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name == 'cat':
                pass
            elif name == 'cat_start':
                pass
            elif name == "rep":
                pass
            elif 'bn' in name:

                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)

                self.vars.append(nn.Parameter(torch.zeros(param[0])))


                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])













            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError

    def parse_config(self, config, vars_list):

        for i, info_dict in enumerate(config):

            if info_dict["name"] == 'conv2d':
                w, b = oml.nn.conv2d(info_dict["config"], info_dict["adaptation"], info_dict["meta"])
                vars_list.append(w)
                vars_list.append(b)

            elif info_dict["name"] == 'linear':
                param_config = info_dict["config"]
                w, b = oml.nn.linear(param_config["out"], param_config["in"], info_dict["adaptation"],
                                     info_dict["meta"])

                vars_list.append(w)
                vars_list.append(b)

            elif info_dict["name"] in ['tanh', 'rep', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                                       'flatten', 'reshape', 'leakyrelu', 'sigmoid', 'rotate']:
                continue
            else:
                print(info_dict["name"])
                raise NotImplementedError
        return vars_list

    def add_rotation(self):
        self.rotate = nn.Parameter(torch.ones(2304,2304))
        torch.nn.init.uniform_(self.rotate)
        self.rotate_inverse = nn.Parameter(torch.inverse(self.rotate))



        logger.info("Inverse computed")


    def reset_vars(self):
        """
        Reset all adaptation parameters to random values. Bias terms are set to zero and other terms to default values of kaiming_normal_
       将所有适应性参数重置为随机值。偏置项被设置为零，其他项被设置为kaiming_normal_的默认值
        :return:
        """
        for var in self.vars:
            if var.adaptation is True:
                if len(var.shape) > 1:
                    torch.nn.init.kaiming_normal_(var)
                else:
                    torch.nn.init.zeros_(var)


    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name == 'conv2d' or (name.startswith('conv') and not name.startswith('convt')):
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)' \
                      % (param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name == 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)' \
                      % (param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name == 'linear':
                tmp = 'linear:(in:%d, out:%d)' % (param[1], param[0])
                info += tmp + '\n'

            elif name == 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)' % (param[0])
                info += tmp + '\n'

            elif name == 'cat':
                tmp = 'cat'
                info += tmp + "\n"
            elif name == 'cat_start':
                tmp = 'cat_start'
                info += tmp + "\n"

            elif name == 'rep':
                tmp = 'rep'
                info += tmp + "\n"


            elif name == 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)' % (param[0], param[1], param[2])
                info += tmp + '\n'
            elif name == 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)' % (param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info


    def forward(self, x, vars=None, bn_training=True, feature=False):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        这个函数可以通过微调调用，但是，在微调中，我们不希望更新running_mean / running_var。
        bn的weights/bias被更新，它已经被fast_weights分开。
        实际上，为了不更新running_mean/running_var，我们需要设置update_bn_statistics=False
        但是权重/偏差将被更新，并且不是脏初始参数通过fast_weiths
        :参数x: [b, 1,28,28]  网络输入
        : param var:
        :param bn_training:设置为False为不更新
        :收益:x，损失，可能性，KLD
        """


        cat_var = False
        cat_list = []

        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0
        if self.Neuromodulation:










          

            for i in range(x.size(0)):
                data = x[i].view(1,3,28,28)
                nm_data = x[i].view(1,3,28,28)




                w,b = vars[0], vars[1]
                nm_data = conv2d(nm_data, w, b)


                w,b = vars[2], vars[3]
                running_mean, running_var = self.vars_bn[0], self.vars_bn[1]
                nm_data = F.batch_norm(nm_data, running_mean, running_var, weight=w, bias=b, training=True)

                nm_data = F.relu(nm_data)
                nm_data = maxpool(nm_data, kernel_size=2, stride=2)


                w,b = vars[4], vars[5]
                nm_data = conv2d(nm_data, w, b)
                w,b = vars[6], vars[7]
                running_mean, running_var = self.vars_bn[2], self.vars_bn[3]
                nm_data = F.batch_norm(nm_data, running_mean, running_var, weight=w, bias=b, training=True)

                nm_data = F.relu(nm_data)
                nm_data = maxpool(nm_data, kernel_size=2, stride=2)

                w,b = vars[8], vars[9]
                nm_data = conv2d(nm_data, w, b)
                w,b = vars[10], vars[11]
                running_mean, running_var = self.vars_bn[4], self.vars_bn[5]
                nm_data = F.batch_norm(nm_data, running_mean, running_var, weight=w, bias=b, training=True)
                nm_data = F.relu(nm_data)



                nm_data = nm_data.view(nm_data.size(0), 1008)



                w,b = vars[12], vars[13]
                fc_mask = F.sigmoid(F.linear(nm_data, w, b)).view(nm_data.size(0), 2304)













                w,b = vars[14], vars[15]
            
                data = conv2d(data, w, b)

                w,b = vars[16], vars[17]
                running_mean, running_var = self.vars_bn[6], self.vars_bn[7]
                data = F.batch_norm(data, running_mean, running_var, weight=w, bias=b, training=True)
                data = F.relu(data)
                data = maxpool(data, kernel_size=2, stride=2)


                w,b = vars[18], vars[19]


                data = conv2d(data, w, b, stride=1)
                w,b = vars[20], vars[21]
                running_mean, running_var = self.vars_bn[8], self.vars_bn[9]
                data = F.batch_norm(data, running_mean, running_var, weight=w, bias=b, training=True)
                data = F.relu(data)
                data = maxpool(data, kernel_size=2, stride=2)


                w,b = vars[22], vars[23]

                data = conv2d(data, w, b, stride=1)
                w,b, = vars[24], vars[25]
                running_mean, running_var = self.vars_bn[10], self.vars_bn[11]
                data = F.batch_norm(data, running_mean, running_var, weight=w, bias=b, training=True)
                data = F.relu(data)


                data = data.view(data.size(0), 2304)
                data = data*fc_mask


                w,b = vars[26], vars[27]
                data = F.linear(data, w, b)

                try:
                    prediction = torch.cat([prediction, data], dim=0)
                except:
                    prediction = data

        elif self.Autoencoder:



            for i in range(x.size(0)):
                auto_data = x[i].view(1, 3, 28, 28)







                w, b = vars[0], vars[1]





                auto_data = conv2d(auto_data, w, b)
                w, b = vars[2], vars[3]



                running_mean, running_var = self.vars_bn[0], self.vars_bn[1]
                auto_data = F.batch_norm(auto_data, running_mean, running_var, weight=w, bias=b, training=True)

                auto_data = F.relu(auto_data)
                auto_data = maxpool(auto_data, kernel_size=2, stride=2)

                w, b = vars[4], vars[5]



                auto_data = conv2d(auto_data, w, b)
                w, b = vars[6], vars[7]



                running_mean, running_var = self.vars_bn[2], self.vars_bn[3]
                auto_data = F.batch_norm(auto_data, running_mean, running_var, weight=w, bias=b, training=True)

                auto_data = F.relu(auto_data)
                auto_data = maxpool(auto_data, kernel_size=2, stride=2)
                hidden_data = auto_data










                w, b = vars[8], vars[9]

                auto_data = convt2d(auto_data, w, b)
                auto_data = F.relu(auto_data)

                w, b = vars[10], vars[11]

                auto_data = convt2d(auto_data, w, b)
                auto_data = F.tanh(auto_data)









                try:
                    auto_img = torch.cat([auto_img, auto_data], dim=0)
                except:
                    auto_img = auto_data










































                w, b = vars[12], vars[13]



                hidden_data = hidden_data.view(hidden_data.size(0), -1)
                if feature:
                    return hidden_data


                hidden_data = F.linear(hidden_data, w, b)

                hidden_data = F.relu(hidden_data)

                hidden_list = hidden_data[0].tolist()


                w, b = vars[14], vars[15]
                hidden_data = F.linear(hidden_data, w, b)







                try:
                    auto_prediction = torch.cat([auto_prediction, hidden_data], dim=0)
                except:
                    auto_prediction = hidden_data




        else:

            for name, param in self.config:

                if name == 'conv2d':
                    w, b = vars[idx], vars[idx + 1]
                    x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                    idx += 2

                elif name == 'tconvt2d':
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

                    if feature:
                        return x
                elif name == "cat_start":
                    cat_var = True
                    cat_list = []

                elif name == "cat":
                    cat_var = False
                    x = torch.cat(cat_list, dim=1)

                elif name == 'bn':
                    w, b = vars[idx], vars[idx + 1]
                    running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
                    x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                    idx += 2
                    bn_idx += 2
                elif name == 'flatten':



                    x = x.view(x.size(0), -1)

                elif name == 'reshape':

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

        if self.Neuromodulation:
            return(prediction)
        elif self.Autoencoder:
            return(auto_img, auto_prediction)
        else:
            return (x)



    def zero_grad(self, vars=None):
        """
        torch.nn.Parameter直接默认requires_grad=True，因此需要此函数来设置
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
