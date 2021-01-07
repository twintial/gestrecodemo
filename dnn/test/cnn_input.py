import numpy as np

from dnn.test import Utils

import os
import re
import time

class Data_Control:
    def __init__(self, filepath):
        self.files = os.listdir(filepath)
        alldata, self.Xlen, self.alllabel, self.flen, self.allindex, self.alluser = self.loadfile(filepath)
        alldata, _, self.length = self.cnn_padding(alldata, self.Xlen,self.flen )
        trainindex, testindex = self.indexsplit(alldata.shape[0], True)
        print(len(testindex))
        self.traindata = alldata[trainindex]
        self.trainlabel = self.alllabel[trainindex]
        self.trainuser = self.alluser[trainindex]
        self.testdata = alldata[testindex]
        self.testlabel = self.alllabel[testindex]
        self.testuser = self.alluser[testindex]


    def indexsplit(self,indexlength,israndom):
        if israndom is True:
            randomind = list(range(indexlength))
            np.random.shuffle(randomind)
            trainindex = randomind[:int(len(randomind) * 0.8)]
            testindex = list(filter(lambda j: j not in trainindex, list(randomind)))
        else:
            trainindex = []
            testindex = []
            for i in range(indexlength):
                if self.allindex[i] < 6000 and self.len_rate[i] >= 0:
                    if (self.alluser[i] == 2) and self.len_rate[i] == 0:
                        testindex.append(i)
                    elif self.alluser[i] >= 0:
                            trainindex.append(i)
            print(len(trainindex))
            np.random.shuffle(trainindex)
            labelindex = np.array(self.alllabel)
            test_ind = labelindex[testindex]
            test_ind = sorted(range(len(test_ind)), key=lambda k: test_ind[k])
            testindex = np.array(testindex)
            testindex = testindex[test_ind]
            testindex = testindex.tolist()
        return trainindex, testindex

    def loadfile(self,filepath):
        raw_data = []
        raw_data_len = []
        raw_label = []
        raw_index = []
        raw_user = []
        starttime = time.time()
        lasttime = time.time()
        kk = 0
        for file in self.files:
            pattern = re.compile(r'\d+')
            res = re.findall(pattern, file)
            if (len(res) == 3 and int(res[1]) >= 0):
                filename = filepath+file
                data = np.load(filename)
                sample = data['datapre']
                sample = sample.astype(np.float32)
                sample1 = sample[:, 0:8]
                samplediff1 = np.diff(sample1,axis=0)*5
                samplepad = np.array([0]*8)
                samplediff1 = np.vstack((samplediff1, samplepad))
                sample1 = np.hstack((sample1, samplediff1))
                sample2 = sample[:, 8:16]
                samplediff2 = np.diff(sample2, axis=0)*5
                samplepad = np.array([0]*8)
                samplediff2 = np.vstack((samplediff2, samplepad))
                sample2 = np.hstack((sample2, samplediff2))
                sample = sample1
                featurelen = sample.shape[1]
                raw_data.append(sample)
                raw_data_len.append(sample.shape[0])
                raw_label.append(int(res[0]))
                raw_index.append(int(res[1]))
                raw_user.append(int(res[2]))
                kk = kk+1
                if kk % 1000 == 0:
                    nowtime = time.time()
                    print("%d, %0fs" % (kk, nowtime-starttime))
        raw_data = np.array(raw_data)
        raw_label = np.array(raw_label)
        raw_index = np.array(raw_index)
        raw_user = np.array(raw_user)
        return raw_data, raw_data_len, raw_label,featurelen, raw_index, raw_user


    def cnn_padding(self, data, slen,flen):
        raw_data = data
        lengths = slen
        median_length = int(np.median(lengths))
        num_samples = len(lengths)
        padding_data = np.zeros([num_samples, median_length, flen])
        for idx, sample in enumerate(raw_data):
            temp = np.zeros([flen, median_length])
            sample = np.transpose(sample)
            if slen[idx] < median_length:
                len_diff = median_length - slen[idx]
                len_diff1 = len_diff//2
                len_diff2 = len_diff-len_diff1
                for xidx, x in enumerate(sample):
                    aa = [x[0]]*len_diff1
                    bb = [x[-1]]*len_diff2
                    cc = x.tolist()
                    temp[xidx,:] = [x[0]]*len_diff1+x.tolist()+[x[-1]]*len_diff2
            if slen[idx] > median_length:
                len_diff = slen[idx] - median_length
                temp = sample[:, len_diff:slen[idx]]
            if slen[idx] == median_length:
                temp = sample
            padding_data[idx, :, :] = np.array(temp).transpose()
        padding_data = np.array(padding_data)
        return padding_data, np.array(slen), median_length
