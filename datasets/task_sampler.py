import copy
import logging

import numpy as np
import torch

logger = logging.getLogger("experiment")


class SamplerFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_sampler(dataset, tasks, trainset, testset=None, capacity=None, num_workers=0):
        if "omni" in dataset:
            return OmniglotSampler(tasks, trainset, testset, num_workers=num_workers)
        elif "Sin" == dataset:
            return SineSampler(tasks, capacity=capacity)
        elif "SinBaseline" in dataset:

            return SineBaselineSampler(tasks, capacity=capacity)


class SineSampler:

    def __init__(self, tasks, capacity):
        self.capacity = capacity
        self.tasks = tasks
        self.task_sampler = SampleSine(capacity)
        self.task_sampler.add_complete_iteraetor(self.tasks)
        self.sample_batch = self.task_sampler.sample_batch
        self.sample_trajectory = self.task_sampler.sample_trajectory

    def get_complete_iterator(self):
        return self.task_sampler.complete_iterator

    def sample_random(self):
        return self.task_sampler.get([np.random.choice(self.tasks)])

    def filter_upto(self, task):
        return self.task_sampler.filter_upto(task)

    def sample_task(self, t):
        return self.task_sampler.get(t)

    def sample_tasks(self, t):
        return self.task_sampler.get_task_trainset(t)


class SineBaselineSampler:

    def __init__(self, tasks, capacity):
        self.capacity = capacity
        self.tasks = tasks
        self.task_sampler = SampleSineBaseline(capacity)
        self.task_sampler.add_complete_iteraetor(self.tasks)
        self.sample_batch = self.task_sampler.sample_batch
        self.sample_trajectory = self.task_sampler.sample_trajectory

    def get_complete_iterator(self):
        return self.task_sampler.complete_iterator

    def sample_random(self):
        return self.task_sampler.get([np.random.choice(self.tasks)])

    def filter_upto(self, task):
        return self.task_sampler.filter_upto(task)

    def sample_task(self, t):
        return self.task_sampler.get(t)

    def sample_tasks(self, t):
        return self.task_sampler.get_task_trainset(t)


class SampleSineBaseline:

    def __init__(self, capacity):
        self.task_iterators = []
        self.iterators = {}
        self.capacity = capacity

    def add_complete_iteraetor(self, tasks):
        pass

    def add_task_iterator(self, task):

        amplitude = (np.random.rand() + 0.02) * (5)
        phase = np.random.rand() * np.pi
        decay = np.random.rand() * 0.4
        frequency = np.random.rand() * 2 + 1.0

        self.iterators[task] = {'id': task, 'phase': phase, 'amplitude': amplitude, 'decay': decay,
                                'frequency': frequency}

        logger.info("Task %d has been added to the list with phase %f and amp %f", task, phase, amplitude)

        return self.iterators[task]

    def get(self, tasks):
        for task in tasks:
            if task in self.iterators:
                return self.iterators[task]
            else:
                return self.add_task_iterator(task)

    def sample_batch(self, task, task_id, samples=10):

        x_samples = np.random.rand(samples) * 10 - 5

        x = np.zeros((samples, 11))
        x[:, 10] = x_samples

        x[:, task_id % 10] = 1

        targets = np.zeros((len(x_samples), 2))
        targets[:, 0] = task['amplitude'] * np.sin(x_samples + task['phase'])

        targets[:, 1] = int(float(task_id) / 10)

        return torch.tensor(x).float(), torch.tensor(targets).float()


    def sample_trajectory(self, task, len, test=False):
        xs = []
        ys = []

        for t in range(0, len):
            x = float(t) / 20

            y = task['amplitude'] * np.e ** (-x * task['decay']) * np.sin(
                2 * np.pi * x / task['frequency'] + task['phase'])
            xs.append(x)
            ys.append(y)

        return torch.tensor(xs).float(), torch.tensor(ys).float()


class SampleSine:


    def __init__(self, capacity):
        self.task_iterators = []
        self.iterators = {}
        self.capacity = capacity

    def add_complete_iteraetor(self, tasks):
        pass

    def add_task_iterator(self, task):

        amplitude = (np.random.rand() + 0.02) * (5)
        phase = np.random.rand() * np.pi
        decay = np.random.rand() * 0.4
        frequency = np.random.rand() * 2 + 1.0

        self.iterators[task] = {'id': task, 'phase': phase, 'amplitude': amplitude, 'decay': decay,
                                'frequency': frequency}




        return self.iterators[task]



    def get(self, tasks):
        for task in tasks:
            if task in self.iterators:
                return self.iterators[task]
            else:
                return self.add_task_iterator(task)

    def sample_batch(self, task, task_id, samples=10):

        x_samples = np.random.rand(samples) * 10 - 5
        x = np.zeros((len(x_samples), 2))

        x = np.zeros((samples, self.capacity))
        x[:, self.capacity - 1] = x_samples
        assert (task_id <= self.capacity - 1)
        x[:, task_id] = 1

        targets = np.zeros((len(x_samples), 2))
        targets[:, 0] = task['amplitude'] * np.sin(x_samples + task['phase'])
        targets[:, 1] = 0
        return torch.tensor(x).float(), torch.tensor(targets).float()

    def sample_trajectory(self, task, len, test=False):
        xs = []
        ys = []

        for t in range(0, len):
            x = float(t) / 20

            y = task['amplitude'] * np.e ** (-x * task['decay']) * np.sin(
                2 * np.pi * x / task['frequency'] + task['phase'])
            xs.append(x)
            ys.append(y)

        return torch.tensor(xs).float(), torch.tensor(ys).float()


class OmniglotSampler:

    def __init__(self, tasks, trainset, testset, num_workers=0):
        self.tasks = tasks
        self.task_sampler = SampleOmni(trainset, testset, num_workers=num_workers)
        self.task_sampler.add_complete_iteraetor(self.tasks)

    def get_complete_iterator(self):
        return self.task_sampler.complete_iterator

    def get_another_complete_iterator(self):
        return self.task_sampler.another_complete_iterator

    def sample_random(self):
        return self.task_sampler.get([np.random.choice(self.tasks)])

    def filter_upto(self, task):
        return self.task_sampler.filter_upto(task)

    def sample_task(self, t, train=True):
        return self.task_sampler.get(t, train)

    def sample_tasks(self, t, train=False):

        dataset = self.task_sampler.get_task_trainset(t, train)
        train_iterator = torch.utils.data.DataLoader(dataset,
                                                     batch_size=1,
                                                     shuffle=True, num_workers=self.task_sampler.num_workers)
        return train_iterator


class SampleOmni:

    def __init__(self, trainset, testset, num_workers=0):
        self.task_iterators = []
        self.trainset = trainset
        self.testset = testset
        self.iterators = {}
        self.test_iterators = {}
        self.num_workers = num_workers

    def add_complete_iteraetor(self, tasks):
        dataset = self.get_task_trainset(tasks, True)

        train_iterator = torch.utils.data.DataLoader(dataset,
                                                     batch_size=64,
                                                     shuffle=True, num_workers=self.num_workers)
        self.complete_iterator = train_iterator
        logger.info("Len of complete iterator = %d", len(self.complete_iterator) * 64)

        train_iterator2 = torch.utils.data.DataLoader(dataset,
                                                      batch_size=1,
                                                      shuffle=True, num_workers=self.num_workers)

        self.another_complete_iterator = train_iterator2

    def add_task_iterator(self, task, train):
        dataset = self.get_task_trainset([task], train)

        train_iterator = torch.utils.data.DataLoader(dataset,
                                                     batch_size=1,
                                                     shuffle=True, num_workers=self.num_workers)
        if train:
            self.iterators[task] = train_iterator
        else:
            self.test_iterators[task] = train_iterator
        print("Task %d has been added to the list" % task)
        return train_iterator

    def get(self, tasks, train):
        if train:
            for task in tasks:
                if task in self.iterators:
                    return self.iterators[task]
                else:
                    return self.add_task_iterator(task, True)
        else:
            for task in tasks:
                if task in self.test_iterators:
                    return self.test_iterators[task]
                else:
                    return self.add_task_iterator(task, False)

    def get_task_trainset(self, task, train):

        if train:
            trainset = copy.deepcopy(self.trainset)
        else:
            trainset = copy.deepcopy(self.testset)
        class_labels = np.array([x[1] for x in trainset._flat_character_images])

        indices = np.zeros_like(class_labels)

        for a in task:
            indices = indices + (class_labels == a).astype(int)

        indices = np.nonzero(indices)

        trainset._flat_character_images = [trainset._flat_character_images[i] for i in indices[0]]
        trainset.data = [trainset.data[i] for i in indices[0]]
        trainset.targets = [trainset.targets[i] for i in indices[0]]



        '''
        # print(trainset._flat_character_images) #[('0504_01.png', 550), ('0504_02.png', 550), ('0504_03.png', 550), ('0504_04.png', 550)...]
        # print("训练数据=", trainset.data) #['0504_01.png', '0504_02.png'...]
        # print("训练标签=", trainset.targets)#[550, 550, 550...]
        
        trainset所包含的属性:
        ['__abstractmethods__', '__add__', '__args__', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__extra__', '__format__', '__ge__', '__getattribute__', '__getit
        em__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__len__', '__lt__', '__module__', '__ne__', '__new__', '__next_in_mro__', '__orig_bases__', '__origin__', '__pa
        rameters__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__slotnames__', '__slots__', '__str__', '__subclasshook__', '__tree_hash__', '__weakref__', '_abc_
        cache', '_abc_generic_negative_cache', '_abc_generic_negative_cache_version', '_abc_registry', '_alphabets', '_cache_data', '_character_images', '_characters', '_check_integrity', '_fl
        at_character_images', '_get_target_folder', '_gorg', 'background', 'data', 'data2', 'download', 'download_url_prefix', 'folder', 'images_cached', 'new_flat', 'root', 'target_folder', '
        target_transform', 'targets', 'targets2', 'transform', 'zips_md5']
        '''
        return trainset

    def get_task_testset(self, task):

        trainset = copy.deepcopy(self.testset)
        class_labels = np.array([x[1] for x in trainset._flat_character_images])
        indices = np.zeros_like(class_labels)
        for a in task:
            indices = indices + (class_labels == a).astype(int)
        indices = np.nonzero(indices)
        trainset._flat_character_images = [trainset._flat_character_images[i] for i in indices[0]]
        trainset.data = [trainset.data[i] for i in indices[0]]
        trainset.targets = [trainset.targets[i] for i in indices[0]]

        return trainset

    def filter_upto(self, task):

        trainset = copy.deepcopy(self.trainset)
        trainset.data = trainset.data[trainset.data['target'] <= task]

        return trainset
