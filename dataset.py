#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Aravind Manoharan
aravindmanoharan05@gmail.com
https://github.com/aravindmanoharan
'''

import pickle
import pandas as pd
from random import shuffle
from PIL import Image, ImageSequence
import numpy as np
import cv2
import glob

class Dataset():
    def __init__(self,data):
        print ("Preparing dataset")
        self.dataset_mapping = pd.read_csv('dataset_mapping.csv')
        self.doc2vec = pickle.load(open('dataset_docvecs.pkl', 'rb'))
        self.farthest_pairs = pickle.load(open('farthest_pairs.p', 'rb'))
        self.filenames = list(set(self.dataset_mapping['FILENAME'].tolist()))
        self.num_files = len(glob.glob('data/*.gif'))
        self.train_num = int(0.8 * self.num_files)
        self.test_num = self.num_files - self.train_num
        
        self.train_n = 0
        self.test_n = 0
        self.curr_train_epoch = 0
        self.curr_test_epoch = 0
        self.batch_size = data["batch_size"]
        self.num_frames = data["num_frames"]
        self.img_resize = data["img_resize"]
        self.train_batches = [(x, x + self.batch_size) for x in range(0, self.train_num, self.batch_size)]
        self.test_batches = [(x, x + self.batch_size) for x in range(self.train_num, self.num_files, self.batch_size)]

    def shuffle_train(self):
        shuffle(self.train_batches)
        
    def shuffle_test(self):
        shuffle(self.test_batches)
        
    def load_train(self):
        
        gif_data = np.zeros((self.batch_size,self.num_frames,self.img_resize,self.img_resize))
        gif_embedding = np.zeros((self.batch_size, self.doc2vec.shape[1]))
        gif_far_embedding = np.zeros((self.batch_size, self.doc2vec.shape[1]))
        start, end = self.train_batches[self.train_n]
        for i,n in enumerate(range(start, end)):
            gif = Image.open('data/' + str(n+1) + '.gif')
            frames = [cv2.resize(np.array(frame.copy()),(self.img_resize,self.img_resize)) for frame in ImageSequence.Iterator(gif)]
            far_index = self.dataset_mapping.loc[self.dataset_mapping['FILENAME'] == (n+1)].index[0]
            gif_data[i] = self.max_frames(frames)
            gif_embedding[i] = self.doc2vec[n]
            gif_far_embedding[i] = self.doc2vec[self.farthest_pairs[far_index + 1]]
            
        self.train_n += 1
        if self.train_n > len(self.train_batches) - 1:
            self.curr_train_epoch += 1
            self.train_n = 0
            self.shuffle_train()
            
        return (gif_data, gif_embedding, gif_far_embedding)
    
    def load_test(self):
        
        test_gif_data = np.zeros((self.batch_size,self.num_frames,self.img_resize,self.img_resize))
        test_gif_embedding = np.zeros((self.batch_size, self.doc2vec.shape[1]))
        test_gif_far_embedding = np.zeros((self.batch_size, self.doc2vec.shape[1]))
        test_start, test_end = self.test_batches[self.test_n]
        for j,n in enumerate(range(test_start, test_end)):
            gif = Image.open('data/' + str(n+1) + '.gif')
            frames = [cv2.resize(np.array(frame.copy()),(self.img_resize,self.img_resize)) for frame in ImageSequence.Iterator(gif)]
            far_index = self.dataset_mapping.loc[self.dataset_mapping['FILENAME'] == (n+1)].index[0]
            test_gif_data[j] = self.max_frames(frames)
            test_gif_embedding[j] = self.doc2vec[n]
            test_gif_far_embedding[j] = self.doc2vec[self.farthest_pairs[far_index + 1]]

        self.test_n += 1
        if self.test_n > len(self.test_batches) - 1:
            self.curr_test_epoch += 1
            self.test_n = 0
            self.shuffle_test()
            
        return (test_gif_data, test_gif_embedding, test_gif_far_embedding)
    
    def max_frames(self,frames):
        if len(frames) == self.num_frames:
            return np.array(frames)
        
        if len(frames) < self.num_frames:
            while len(frames) < self.num_frames:
                frames.append(np.zeros((self.img_resize,self.img_resize)))
            return np.array(frames)
        
        if len(frames) > self.num_frames: 
            mid_frame = int(len(frames) / 2)
            mid_num_frame = int(self.num_frames/ 2)
            return np.array(frames[mid_frame-mid_num_frame:mid_frame+mid_num_frame])
        
if __name__ == '__main__':
    data = {"batch_size": 2, "num_frames": 32, "img_resize": 256}
    dataset = Dataset(data)
    a, b, c = dataset.load_train()
    d, e, f = dataset.load_test()
    g, _, _ = dataset.load_train()
    n = dataset.train_n
    m = dataset.train_batches
    o = dataset.test_batches


