import logging
import copy
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

import model.learner as Learner

logger = logging.getLogger("experiment")


class MetaLearingClassification(nn.Module):
    """
    MetaLearingClassification Learner
    """

    def __init__(self, args, config):

        super(MetaLearingClassification, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.update_step = args.update_step

        self.net = Learner.Learner(config)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.meta_lr)

    def shuffle_labels(self,targets):
        targets = (targets+2)%len(targets) # offset
        # targets = (len(targets)-1-targets)%len(targets) # flip
        return targets

    def reset_classifer(self, class_to_reset):
        bias = self.net.parameters()[-1]
        weight = self.net.parameters()[-2]
        torch.nn.init.kaiming_normal_(weight[class_to_reset].unsqueeze(0))

    def reset_layer(self, layer_to_reset):
        bias = self.net.parameters()[-1]
        bn_layers = [20,21,24,25,28,29]
        if layer_to_reset in bn_layers:
            if layer_to_reset % 2 == 0:
                p = self.net.parameters()[layer_to_reset]
                self.net.parameters()[layer_to_reset].data = nn.Parameter(torch.ones_like(p))
            else:
                p = self.net.parameters()[layer_to_reset]
                self.net.parameters()[layer_to_reset].data = nn.Parameter(torch.zeros_like(p))
            running_mean = nn.Parameter(torch.zeros_like(p), requires_grad=False)
            running_var = nn.Parameter(torch.ones_like(p), requires_grad=False)
            self.net.vars_bn[self.bn_reset_counter] = running_mean
            self.net.vars_bn[self.bn_reset_counter + 1] = running_var
            self.bn_reset_counter += 2

        else:
            if layer_to_reset % 2 == 0:
                weight = self.net.parameters()[layer_to_reset]#-2]
                torch.nn.init.kaiming_normal_(weight)
            else:
                bias = self.net.parameters()[layer_to_reset]
                bias.data = torch.ones(bias.data.size())

    def sample_training_data(self, iterators, it2, steps=2, reset=True):

        # Sample data for inner and meta updates

        x_traj = []
        y_traj = []
        x_rand = []
        y_rand = []

        counter = 0

        class_cur = 0
        class_to_reset = 0
        for it1 in iterators:
            for img, data in it1:
                class_to_reset = data[0].item()
                if reset:
                    next
                    # Resetting weights corresponding to classes in the inner updates; this prevents
                    # the learner from memorizing the data (which would kill the gradients due to inner updates)
                    #for class_ in range(1000):
                    #self.reset_classifer(class_to_reset)
                    #self.bn_reset_counter = 0
                    #layers_to_reset = [18,19,20,21,22,23,24,25,26,27,28,29]
                    #for layer in layers_to_reset:
                    #    self.reset_layer(layer)

                #self.net.cuda()
                counter += 1
                x_traj.append(img)
                y_traj.append(data)
                if counter % int(steps / len(iterators)) == 0:
                    class_cur += 1
                    break

        # To handle a corner case; nothing interesting happening here
        if len(x_traj) < steps:
            it1 = iterators[-1]
            for img, data in it1:
                counter += 1
                x_traj.append(img)
                y_traj.append(data)
                if counter % int(steps % len(iterators)) == 0:
                    break

        # Sampling the random batch of data
        counter = 0
        for img, data in it2:
            if counter == 1:
                break
            x_rand.append(img)
            y_rand.append(data)
            counter += 1

        class_cur = 0
        counter = 0
        x_rand_temp = []
        y_rand_temp = []
        for it1 in iterators:
            for img, data in it1:
                counter += 1
                x_rand_temp.append(img)
                y_rand_temp.append(data)
                if counter % int(steps / len(iterators)) == 0:
                    class_cur += 1
                    break

        y_rand_temp = torch.cat(y_rand_temp).unsqueeze(0)
        x_rand_temp = torch.cat(x_rand_temp).unsqueeze(0)
        x_traj, y_traj, x_rand, y_rand = torch.stack(x_traj), torch.stack(y_traj), torch.stack(x_rand), torch.stack(
            y_rand)

        x_rand = torch.cat([x_rand, x_rand_temp], 1)
        y_rand = torch.cat([y_rand, y_rand_temp], 1)

        return x_traj, y_traj, x_rand, y_rand

    def inner_update(self, x, fast_weights, y, bn_training, nm_update):

        logits = self.net(x, fast_weights, bn_training=bn_training)
        loss = F.cross_entropy(logits, y)
        if fast_weights is None:
            fast_weights = self.net.parameters()

        grad = torch.autograd.grad(loss, fast_weights)

        nm_layers = list(range(12, 20))
        
        if not nm_update:
            for idx in nm_layers:
                fast_weights[idx].learn = False

        fast_weights = list(
            map(lambda p: p[1] - self.update_lr * p[0] if p[1].learn else p[1], zip(grad, fast_weights)))

        for params_old, params_new in zip(self.net.parameters(), fast_weights):
            params_new.learn = params_old.learn

        return fast_weights

    def meta_loss(self, x, fast_weights, y, bn_training):

        logits = self.net(x, fast_weights, bn_training=bn_training)
        loss_q = F.cross_entropy(logits, y)
        return loss_q, logits

    def eval_accuracy(self, logits, y):
        pred_q = F.softmax(logits, dim=1).argmax(dim=1)
        correct = torch.eq(pred_q, y).sum().item()
        return correct

    def Prepare_Data(self, x_traj, y_traj, num_tasks):

        training_data = {str(t):None for t in range(num_tasks)}
        training_targets = {str(t):None for t in range(num_tasks)}

        validation_data = {str(t):None for t in range(num_tasks)}
        validation_targets = {str(t):None for t in range(num_tasks)}

        train_counter = 0

        for task in training_data.keys():

            training_data[task] = x_traj[train_counter:train_counter+15]
            training_targets[task] = y_traj[train_counter:train_counter+15]

            validation_data[task] = x_traj[train_counter+15:train_counter+20]
            validation_targets[task] = y_traj[train_counter+15:train_counter+20]

            train_counter += 20 # add every pack of 20 instances to the respective dictionary entry

        return(training_data, training_targets, validation_data, validation_targets)

    def forward(self, x_traj, y_traj, x_rand, y_rand, num_tasks):
        """

        :param x_traj:   Input data of sampled trajectory
        :param y_traj:   Ground truth of the sampled trajectory
        :param x_rand:   Input data of the random batch of data
        :param y_rand:   Ground truth of the random batch of data
        :return:
        """

        fast_weights = None
        updates_per_task = 15
        layers_to_reset = list(range(20,34))

        #self.net.cuda()

        training_data, training_targets, validation_data, validation_targets = self.Prepare_Data(x_traj, y_traj, num_tasks)
        print(len(x_traj))

        tasks_for_training_nm = ['0']
        tasks_for_testing_nm = ['1', '2']

        for task in tasks_for_training_nm:
            
            for k in range(updates_per_task):

                omniglot_image = training_data[task][k]
                image_label = training_targets[task][k]

                fast_weights = self.inner_update(omniglot_image, fast_weights, image_label, True, True)

        for task in tasks_for_testing_nm:

            for k in range(updates_per_task):

                omniglot_image = training_data[task][k]
                image_label = training_targets[task][k]

                fast_weights = self.inner_update(omniglot_image, fast_weights, image_label, True, False)

        total_meta_loss = 0.0
        total_accuracy = 0.0

        for task in validation_data.keys():

            for k in range(len(validation_data[task])):
                meta_loss, logits = self.meta_loss(validation_data[task][k], fast_weights, validation_targets[task][k], False)
                total_meta_loss += meta_loss
                with torch.no_grad():
                    pred_q = F.softmax(logits, dim=1).argmax(dim=1)
                    total_accuracy += torch.eq(pred_q, validation_targets[task][k]).sum().item()

        total_accuracy /= (num_tasks*5)

            #meta_loss, logits = self.meta_loss(x_rand[0], fast_weights, y_rand[0], False)
            #total_meta_loss = meta_loss

            # Computing accuracy on the meta and traj set for understanding the learning
            
            #with torch.no_grad():

            #    pred_q = F.softmax(logits, dim=1).argmax(dim=1)
            #    classification_accuracy = torch.eq(pred_q, y_rand[0]).sum().item()  # convert to numpy

        # Taking the meta gradient step

        self.net.zero_grad()
        #meta_loss = total_meta_loss
        #meta_loss.backward()
       
        grads = torch.autograd.grad(total_meta_loss, self.net.parameters())
        
        for idx in range(len(self.net.parameters())):
          
            if idx in layers_to_reset:
                self.net.parameters()[idx].grad = None
            
            else:
                self.net.parameters()[idx].grad = grads[idx]

        self.optimizer.step()

        return total_accuracy, total_meta_loss


class MetaLearnerRegression(nn.Module):
    """
    MetaLearingClassification Learner
    """

    def __init__(self, args, config):
        """

        :param args:
        """
        super(MetaLearnerRegression, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.update_step = args.update_step

        self.net = Learner.Learner(config)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.meta_optim = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [1500, 2500, 3500], 0.1)

    def forward(self, x_traj, y_traj, x_rand, y_rand):

        losses_q = [0 for _ in range(len(x_traj) + 1)]

        for i in range(1):
            logits = self.net(x_traj[0], vars=None, bn_training=False)
            logits_select = []
            for no, val in enumerate(y_traj[0, :, 1].long()):
                logits_select.append(logits[no, val])
            logits = torch.stack(logits_select).unsqueeze(1)
            loss = F.mse_loss(logits, y_traj[0, :, 0].unsqueeze(1))
            grad = torch.autograd.grad(loss, self.net.parameters())

            fast_weights = list(
                map(lambda p: p[1] - self.update_lr * p[0] if p[1].learn else p[1], zip(grad, self.net.parameters())))
            for params_old, params_new in zip(self.net.parameters(), fast_weights):
                params_new.learn = params_old.learn

            with torch.no_grad():

                logits = self.net(x_rand[0], vars=None, bn_training=False)

                logits_select = []
                for no, val in enumerate(y_rand[0, :, 1].long()):
                    logits_select.append(logits[no, val])
                logits = torch.stack(logits_select).unsqueeze(1)
                loss_q = F.mse_loss(logits, y_rand[0, :, 0].unsqueeze(1))
                losses_q[0] += loss_q

            for k in range(1, len(x_traj)):
                logits = self.net(x_traj[k], fast_weights, bn_training=False)

                logits_select = []
                for no, val in enumerate(y_traj[k, :, 1].long()):
                    logits_select.append(logits[no, val])
                logits = torch.stack(logits_select).unsqueeze(1)

                loss = F.mse_loss(logits, y_traj[k, :, 0].unsqueeze(1))
                grad = torch.autograd.grad(loss, fast_weights)
                fast_weights = list(
                    map(lambda p: p[1] - self.update_lr * p[0] if p[1].learn else p[1], zip(grad, fast_weights)))

                for params_old, params_new in zip(self.net.parameters(), fast_weights):
                    params_new.learn = params_old.learn

                logits_q = self.net(x_rand[0, 0:int((k + 1) * len(x_rand[0]) / len(x_traj)), :], fast_weights,
                                    bn_training=False)

                logits_select = []
                for no, val in enumerate(y_rand[0, 0:int((k + 1) * len(x_rand[0]) / len(x_traj)), 1].long()):
                    logits_select.append(logits_q[no, val])
                logits = torch.stack(logits_select).unsqueeze(1)
                loss_q = F.mse_loss(logits, y_rand[0, 0:int((k + 1) * len(x_rand[0]) / len(x_traj)), 0].unsqueeze(1))

                losses_q[k + 1] += loss_q

        self.optimizer.zero_grad()

        loss_q = losses_q[k + 1]
        loss_q.backward()
        self.optimizer.step()

        return losses_q


def main():
    pass


if __name__ == '__main__':
    main()
