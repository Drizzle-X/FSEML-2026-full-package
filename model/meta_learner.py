import logging
import copy
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

import matplotlib.pyplot as plt

import model.learner as Learner
from model.meta_learner_base import BaseContinualMetaLearner

logger = logging.getLogger("experiment")


class MetaLearingClassification(BaseContinualMetaLearner):
    """
    MetaLearingClassification Learner
    实例化模型，定义优化器和损失函数
    """

    def __init__(self, args, config, treatment):

        super(MetaLearingClassification, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.update_step = args.update_step

        if args.treatment == "Neuromodulation":
            self.neuromodulation = True
            self.autoencoder = None
            self.oml = None
        elif args.treatment == "AE":
            self.autoencoder = True
            self.neuromodulation = None
            self.oml = None
        elif args.treatment == "OML":
            self.oml = True
            self.autoencoder = None
            self.neuromodulation = None
        else:
            self.neuromodulation = None
            self.autoencoder = None
            self.oml = None








        self.net = Learner.Learner(config, self.neuromodulation, self.autoencoder, self.oml)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.meta_iteration = 0




    def reset_classifer(self, class_to_reset):
        bias = self.net.parameters()[-1]
        weight = self.net.parameters()[-2]

        torch.nn.init.kaiming_normal_(weight[class_to_reset].unsqueeze(0))


    def reset_layer(self, layer_to_reset):
        if layer_to_reset % 2 == 0:
            weight = self.net.parameters()[layer_to_reset]
            torch.nn.init.kaiming_normal_(weight)
        else:
            bias = self.net.parameters()[layer_to_reset]
            bias.data = torch.ones(bias.data.size())

    def inner_update(self, x, fast_weights, y, bn_training):
        lemda = 0.7
        if self.autoencoder:
            recons_img, logits = self.net(x, fast_weights, bn_training=bn_training)
            Loss_auto = torch.nn.MSELoss()










            loss_auto = Loss_auto(recons_img, x)

            loss_pre = F.cross_entropy(logits, y)



            loss = (1-lemda)*0.001*loss_auto + lemda*loss_pre
            loss = loss.to(torch.float32)


        else:
            logits = self.net(x, fast_weights, bn_training=bn_training)
            loss = F.cross_entropy(logits, y)

        if fast_weights is None:
            fast_weights = self.net.parameters()

        grad = torch.autograd.grad(loss, fast_weights, allow_unused=False)





        """
        # 梯度更新，\theta = \theta - lr*grad

        # 1.zip(grad, fast_weights)将grad和fast_weights中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
        # 2.map()将列表中的元组(grad, fast_weights)[0],(grad, fast_weights)[1],(grad, fast_weights)[2]依次通过前面的lambda函数：
        # 3.if p[1].learn   #当p[1].learn参数（来自utils.utils.freeze_layers为TRUE，则层未冻结，进行梯度更新）
        #    p[1] - self.update_lr * p[0]
        # else p[1]   #冻结层不更新
        # 注意这里，在inner loop中我们的神经调节网络是不进行更新的，因此要保证只有预测网络可以进行学习更新
        """
        fast_weights = list(
            map(lambda p: p[1] - self.update_lr * p[0] if p[1].learn else p[1], zip(grad, fast_weights)))

        for params_old, params_new in zip(self.net.parameters(), fast_weights):
            params_new.learn = params_old.learn

        return fast_weights

    def meta_loss(self, x, fast_weights, y, bn_training):
        lemda = 0.7
        if self.autoencoder:
            recons_img, logits = self.net(x, fast_weights, bn_training=bn_training)
            Loss_auto = torch.nn.MSELoss()
            loss_auto = Loss_auto(recons_img, x)


            loss_pre = F.cross_entropy(logits, y)





            loss = (1 - lemda) * 0.001 * loss_auto + lemda * loss_pre
            loss = loss.to(torch.float32)

            return loss, logits, loss_auto, loss_pre





        else:
            logits = self.net(x, fast_weights, bn_training=bn_training)
            loss_q = F.cross_entropy(logits, y)
            return loss_q, logits

    def eval_accuracy(self, logits, y):
        pred_q = F.softmax(logits, dim=1).argmax(dim=1)
        correct = torch.eq(pred_q, y).sum().item()

        return correct

    def forward(self, x_traj, y_traj, x_rand, y_rand):
        """
        :param x_traj:   Input data of sampled trajectory
        :param y_traj:   Ground truth of the sampled trajectory
        :param x_rand:   Input data of the random batch of data
        :param y_rand:   Ground truth of the random batch of data
        :return:
        """

        x_traj, y_traj, x_rand, y_rand = self.maybe_apply_label_patch_augmentation(
            x_traj,
            y_traj,
            x_rand,
            y_rand,
        )

        fast_weights = self.inner_update(x_traj[0], None, y_traj[0], False)
        for k in range(1, self.update_step):

            fast_weights = self.inner_update(x_traj[k], fast_weights, y_traj[k], False)

        meta_loss, logits, loss_auto, loss_pre = self.meta_loss(x_rand[0], fast_weights, y_rand[0], False)


        with torch.no_grad():
            pred_q = F.softmax(logits, dim=1).argmax(dim=1)
            classification_accuracy = torch.eq(pred_q, y_rand[0]).sum().item()




        self.net.zero_grad()


        NM_reset = False
    
        if NM_reset:

            layers_to_reset = list(range(14, 28))
            grads = torch.autograd.grad(meta_loss, self.net.get_adaptation_parameters(fast_weights), create_graph=True)
        
            for idx in range(len(self.net.parameters())):
                if idx in layers_to_reset:
                    self.net.parameters()[idx].grad = None
                else:
                    self.net.parameters()[idx].grad = grads[idx]
        else:
            meta_loss.backward()

        self.optimizer.step()
        
        classification_accuracy /= len(x_rand[0])
        self.meta_iteration += 1

        return classification_accuracy, meta_loss, loss_auto, loss_pre


class AEMetaLearner(MetaLearingClassification):
    branch_name = "AE"

    def __init__(self, args, config):
        super().__init__(args, config, "AE")


class OMLMetaLearner(MetaLearingClassification):
    branch_name = "OML"

    def __init__(self, args, config):
        super().__init__(args, config, "OML")


class NeuromodulationMetaLearner(MetaLearingClassification):
    branch_name = "Neuromodulation"

    def __init__(self, args, config):
        super().__init__(args, config, "Neuromodulation")






















































































def main():
    pass


if __name__ == '__main__':
    main()
