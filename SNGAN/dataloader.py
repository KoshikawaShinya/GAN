import cv2
import numpy as np
import glob


class DataLoader:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name


    def load_batch(self, batch_size=1):

        img_pathes = glob.glob('imgs/%s/train/*'%(self.dataset_name))

        for i in range(len(img_pathes) // batch_size):
            img_batch = []
            idx = np.random.choice(img_pathes, batch_size, replace=False)
            
            for img in idx:
                img = cv2.imread(img, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_batch.append(img)
            
            img_batch = np.array(img_batch) / 127.5 - 1.0
            yield img_batch


