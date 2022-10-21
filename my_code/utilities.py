import torch
import numpy as np

import random
import cv2
import os


class ImageIterator:
    def __init__(self, device, batch_size, mode='train', fixed_size=(512,512)):
        self.device = device
        self.mode = mode
        self.batch_size = batch_size
        self.image_names = [] # (healthy, disease)
        
        self.load_iamge_names()
        self.give_healthy = True
        
        # used for reshaping images into the named size
        self.fixed_size = fixed_size
        self.order = []

    def load_iamge_names(self):
        image_names = [[],[]]
        for root, directories, files in os.walk('.\\cassava\\' + self.mode):
            
            if root.endswith('healthy'):
                index = 1
            else:
                index = 0
            
            image_names[index] = image_names[index] + [root + '\\' + file for file in files]
        
        self.image_names.append(np.array(image_names[0]))
        self.image_names.append(np.array(image_names[1]))
    
    def set_give_healthy(self, state=True):
        self.give_healthy = state
    
    
    def __iter__(self):
        return self
    
    def __next__(self):
        
        if len(self.order) < self.batch_size:
            self.order = np.array(range(len(self.image_names[self.give_healthy])))
            np.random.shuffle(self.order)
        
        # take the next batch_size images
        selections_numbers = self.order[0:self.batch_size]
        self.order = self.order[self.batch_size:]
        
        arr = self.image_names[self.give_healthy]
        
        selections = arr[selections_numbers]
        images = torch.zeros(self.batch_size, 3, self.fixed_size[0], self.fixed_size[1])
        for i, selection in enumerate(selections):
            im = cv2.imread(selection)
            im = cv2.resize(im, self.fixed_size)
            im = im/255
            im = torch.from_numpy(im)
            im = im.swapaxes(0, 2)
            images[i] = im
        images.to(self.device)
        return images