import numpy as np
import torch
from torch import nn


class BaseContinualMetaLearner(nn.Module):
    """
    Shared interface for continual meta learners.

    Each branch keeps its own model/loss behavior, but common data sampling
    and utility helpers live here so training/evaluation can treat every
    learner uniformly.
    """

    branch_name = "base"

    def add_patch_to_images(self, images, task_num):
        box_size = 8
        if task_num == 1:
            try:
                images[:, :, : box_size + 1, : box_size + 1] = torch.min(images)
            except Exception:
                images[:, : box_size + 1, : box_size + 1] = torch.min(images)
        elif task_num == 2:
            images[:, :, -(box_size + 1) :, -(box_size + 1) :] = torch.min(images)
        elif task_num == 3:
            images[:, :, : box_size + 1, -(box_size + 1) :] = torch.min(images)
        elif task_num == 4:
            images[:, :, -(box_size + 1) :, : box_size + 1] = torch.min(images)
        return images

    def shuffle_labels(self, targets, batch=False):
        if batch:
            new_target = (targets[0] + 2) % 1000
            for t in range(len(targets)):
                targets[t] = new_target
            return targets

        return (targets + 2) % 1000

    def sample_training_data(self, iterators, it2, steps=2, reset=True):
        x_traj = []
        y_traj = []
        x_rand = []
        y_rand = []

        counter = 0
        for it1 in iterators:
            for img, data in it1:
                class_to_reset = data[0].item()
                if reset:
                    self.reset_classifer(class_to_reset)

                counter += 1
                x_traj.append(img)
                y_traj.append(data)
                if counter % int(steps / len(iterators)) == 0:
                    break

        if len(x_traj) < steps:
            it1 = iterators[-1]
            for img, data in it1:
                counter += 1
                x_traj.append(img)
                y_traj.append(data)
                if counter % int(steps % len(iterators)) == 0:
                    break

        counter = 0
        for img, data in it2:
            if counter == 1:
                break
            x_rand.append(img)
            y_rand.append(data)
            counter += 1

        x_traj = torch.stack(x_traj)
        y_traj = torch.stack(y_traj)
        x_rand = torch.stack(x_rand)
        y_rand = torch.stack(y_rand)

        return x_traj, y_traj, x_rand, y_rand

    def enable_black_square_augmentation(self):
        return False

    def maybe_apply_label_patch_augmentation(self, x_traj, y_traj, x_rand, y_rand):
        if not self.enable_black_square_augmentation():
            return x_traj, y_traj, x_rand, y_rand

        x_traj_bs = self.add_patch_to_images(x_traj.clone(), task_num=1)
        y_traj_bs = self.shuffle_labels(y_traj.clone(), batch=True)
        del x_traj_bs, y_traj_bs

        for i in range(len(x_rand[0])):
            coin_flip = np.random.randn()
            if coin_flip > 0:
                x_rand[0][i] = self.add_patch_to_images(x_rand[0][i], task_num=1)
                y_rand[0][i] = self.shuffle_labels(y_rand[0][i], batch=False)

        return x_traj, y_traj, x_rand, y_rand
