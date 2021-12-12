# Video-Based-Sports-Activity-Recognition-for-Children
The codebase for my masters thesis and ASPA paper
## ABSTRACT
Over the years, deep learning models have been able to record state-of-the- art (SOTA) performance on the task of activity recognition. The results of this can be seen in applications such as video surveillance, medical diag- nosis, robotics for human behavior characterization, and like in this study, recognition of human activities from videos. One of the factors that have contributed to the benchmark performance of these models is the availabil- ity of large-scale datasets. However, we have observed that these datasets are largely skewed towards adults. That is, they contain more videos of adults than kids. Out of 5014 videos from an adult-specific dataset, only 1109 videos contained kids performing an action. Since there exist visual differences in how an adult performs an activity as opposed to a child, in this study, we test if current SOTA deep learning models have some sys- temic biases in decoding the activity being performed by an adult or a kid. To do this, we create kid-specific and adult-specific datasets. Using a SOTA deep learning model trained on the different datasets, we test for the generalization ability of the deep learning model. Our results indicate that, while SOTA deep learning models can be used to classify kid activities, the kid-specific dataset is more complex to generalize to than the adult-specific dataset. The study also shows that the features learned from training on a kid-specific dataset alone can be used to classify adult activities while the reverse is not the case.

Full thesis can be found here: 

## Code Base
1. slowfastnet.py: This file contains the implementation of the slowfast architecture used in this study (we use ResNet50). You can request the pre-trained weights by sending an email to lizfeyisayo@gmail.com. 
2. train_model.py: Code to train all the models described in this work. It calls the SlowFast-ResNet50 from slowfast.py. Also, it performs data transformations using the custom MyDataset class defined in dataset.py. The training function is called from TrainTestCode.py.
3. config-default.yaml: Training configurations used as well as parameters set on wandb.ai
4. Evaluation.py: Runs the test set on each of the models using the saved checkpoints. Checkpoints can be found here: 
5. DownloadData: contains the urls to dowload the data described.
