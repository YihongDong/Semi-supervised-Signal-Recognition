import numpy as np
import pickle
import random

class DataSet(object):
    def __init__(self, path, labeled_num = 40, rate = 0.5, num_seen_class=8, seed=2019):
        self.path = path
        self.labeled_num = labeled_num
        self.num_seen_class = num_seen_class
        self.seed = seed

        self.loadDataSet(path, self.labeled_num)
        self.splitData(rate,num_seen_class, seed)

    # public methods
    def getTrainAndTest(self):
        return self.X_labeled, self.Y_labeled, self.X_unlabeled, self.Y_unlabeled, self.X_test, self.Y_test

    def loadDataSet(self, path, labeled_num):
        Xd = pickle.load(open(path, 'rb'), encoding='iso-8859-1')

        # snrs(20) = -20 -> 18  mods(11) = ['8PSK', 'AM-DSB', ...]
        self.snrs, self.mods = map(
            lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])

        self.X = []
        self.lbl = []

        print(self.mods)
        print(self.snrs)

        self.class_count=[0]
        count=0

        for mod in self.mods:
            for snr in self.snrs:
                # select samples of certain SNR
                if not snr == 8:
                    continue

                single_data=Xd[(mod, snr)][:]
                self.X.append(single_data)
                count+=single_data.shape[0]

                print('Shape:{}'.format(single_data.shape[0]))
                for i in range(single_data.shape[0]):
                    self.lbl.append((mod, snr))

            self.class_count.append(count)

        self.X = np.vstack(self.X)
        print(self.X.shape)

    def splitData(self, rate=0.8, num_seen_class = 8, seed=2007):
        np.random.seed(seed)
        
        train_class=list(range(0,len(self.mods)))

        self.labeled_idx=[]
        self.test_idx=[]
        self.unlabeled_idx=[]

        for certain_class in train_class:
            class_idx=list(range(self.class_count[certain_class],self.class_count[certain_class+1]))

            class_train_idx=list(np.random.choice(class_idx, size=int(len(class_idx)*rate), replace=False))
            class_labeled_idx=list(np.random.choice(class_train_idx, size=self.labeled_num, replace=False))
            class_test_idx=list(set(class_idx)-set(class_train_idx))

            self.unlabeled_idx+=class_train_idx
            self.labeled_idx+=class_labeled_idx
            self.test_idx+=class_test_idx

        random.shuffle(self.unlabeled_idx)
        random.shuffle(self.labeled_idx)
        random.shuffle(self.test_idx)

        self.X_unlabeled=self.X[self.unlabeled_idx]
        self.X_labeled=self.X[self.labeled_idx]
        self.X_test = self.X[self.test_idx]

        self.Y_unlabeled = list(map(lambda x: train_class.index(self.mods.index(self.lbl[x][0])), self.unlabeled_idx))
        self.Y_labeled = list(map(lambda x: train_class.index(self.mods.index(self.lbl[x][0])), self.labeled_idx))
        self.Y_test = list(map(lambda x: train_class.index(self.mods.index(self.lbl[x][0])), self.test_idx))
