import torch
from torch.utils.data import Dataset, DataLoader
import imgaug.augmenters as iaa
import numpy as np
import pandas as pd
import os
import video_transforms as transforms 
import cv2
import imageio
import json
import skvideo.io

"""
Custom class to handle the reading of input data, selecting the number of frames and applying respective data transforms.
"""

class MyDataset(Dataset):
    def __init__(self,dataset_path,
        clip_length=8,
        clip_size=256,
        val_clip_length=None,
        val_clip_size=None,
        train_interval=2,
        val_interval=2,
        input_frame = 64,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],mode='train',end_size=[None,256,256],target_n_frames=40):
        self.target_n_frames = target_n_frames
        self.end_size = end_size
        self.dataset_path = dataset_path
        self.input_frame = input_frame
        self.df = pd.read_csv(dataset_path)
        # if mode=='train' else pd.read_csv(dataset_path)
        # self.df = self.df[self.df['Action Label']=='hitting baseball_kid']
        self.classes = sorted(self.df["Action Label"].unique()) # List of unique class names
        self.class_to_idx = {j: i for i, j in enumerate(self.classes)}
        
        # with open(json_className, 'w') as fp:
        #     json.dump(self.class_to_idx, fp)

        sometimes_aug = lambda aug: iaa.Sometimes(0.4, aug)
        sometimes_seq = lambda aug: iaa.Sometimes(0.9, aug)


        if mode == 'train':
            self.video_transform = transforms.Compose(
                            transforms=iaa.Sequential([
                                iaa.Resize({"shorter-side": 384, "longer-side":"keep-aspect-ratio"}),
                                iaa.CropToFixedSize(width=384, height=384, position='center'),
                                iaa.CropToFixedSize(width=clip_size, height=clip_size, position='uniform'),
                                sometimes_seq(iaa.Sequential([
                                    sometimes_aug(iaa.GaussianBlur(sigma=[0.1,0.2,0.3])),
                                    sometimes_aug(iaa.Add((-5, 15), per_channel=True)),
                                    sometimes_aug(iaa.AverageBlur(k=(1,2))),
                                    sometimes_aug(iaa.Multiply((0.8, 1.2))),
                                    sometimes_aug(iaa.GammaContrast((0.85,1.15),per_channel=True)),
                                    sometimes_aug(iaa.AddToHueAndSaturation((-16, 16), per_channel=True)),
                                    sometimes_aug(iaa.LinearContrast((0.85, 1.115))),
                                    sometimes_aug(
                                        iaa.OneOf([
                                            iaa.PerspectiveTransform(scale=(0.02, 0.05), keep_size=True),
                                            iaa.Rotate(rotate=(-10,10)),
                                        ])
                                    )
                                ])),
                                iaa.Fliplr(0.5)
                            ]),
                            normalise=[mean,std]
                        )
        else:
     
            self.video_transform=transforms.Compose(
                                        transforms=iaa.Sequential([
                                            iaa.Resize({"shorter-side": 294, "longer-side":"keep-aspect-ratio"}),
                                            iaa.CropToFixedSize(width=294, height=294, position='center'),
                                            iaa.CropToFixedSize(width=256, height=256, position='center')
                                        ]),
                                        normalise=[mean,std])


    def __len__(self):
        return self.df.shape[0]


    def __getitem__(self,index):
        row = self.df.iloc[index]
        label = row['Action Label']
        label = self.class_to_idx[label] # Convert it to int
        # path = row['Video Path'].split("/")
        # path =path[1]+"/"+path[2]+"/"+path[3]  #dont panic, i'll clean this up by creating new file where row is Kinetic-kids-Processed/
        # print("path is: ",path)

        path = os.path.join('data/',row['Video Path'])
        path_pt = path.split('.')[0] +f"_{self.target_n_frames}_" +'.npy'

        if not os.path.isfile(path_pt):
            vid = skvideo.io.vread(path)
        
            # frame_count = vid.count_frames()
            #fps =  vid.get_meta_data()['fps']
            # time = frame_count/fps
            frames = []
            for i, im in enumerate(vid):
                frames.append(im)
            pass
            frame_div = len(frames)//self.target_n_frames
            frames = frames[::frame_div]
            if len(frames) < self.target_n_frames:
                    difference = self.target_n_frames - len(frames)
                    frames.extend(frames[-difference:])

            frames = frames[:self.target_n_frames]
            
            video = np.stack(frames)
            np.save(path_pt,video)
        

        video = np.load(path_pt)
        video = self.video_transform(video,self.end_size)

        return video,label








