# carpartssegment
car parts semantic segmentation with noisy labels
Main Features  
A clear and easy to navigate structure,  
A json config file with a lot of possibilities for parameter tuning,  
Supports various models, losses, Lr schedulers, data augmentations and datasets,
So, what's available ?



Models  
the model we used is pspnet, with resnet as backbone, whick has been trained in voc dataset. you can download 
the pretrained resnet-50 and resnet-101 from :https://drive.google.com/file/d/1am4GccZeiBPePjyoCi67--s2b4YapnNa/view?usp=share_link  
https://drive.google.com/file/d/18vgu4uty-oJeKh0PXNBqOumJlip0Ns75/view?usp=sharing  
(PSPNet) Pyramid Scene Parsing Network [Paper]



Datasets  
our dataset has been adapted to ade20k format, which is locate in carpartsegdata fold. you should unzip the train.zip and dev.zip firstly.
ADE20K: For ADE20K, simply download the images and their annotations for training and validation from sceneparsing.csail.mit.edu, and for the rest visit the website.



Losses  
In addition to the Cross-Entorpy loss, there is also  

Dice-Loss, which measures of overlap between two samples and can be more reflective of the training objective (maximizing the mIoU), but is highly non-convexe and can be hard to optimize.
CE Dice loss, the sum of the Dice loss and CE, CE gives smooth optimization while Dice loss is a good indicator of the quality of the segmentation results.
Focal Loss, an alternative version of the CE, used to avoid class imbalance where the confident predictions are scaled down.
Lovasz Softmax lends it self as a good alternative to the Dice loss, where we can directly optimization for the mean intersection-over-union based on the convex Lovász extension of submodular losses (for more details, check the paper: The Lovász-Softmax loss).
Learning rate schedulers
Poly learning rate, where the learning rate is scaled down linearly from the starting value down to zero during training. Considered as the go to scheduler for semantic segmentaion (see Figure below).
One Cycle learning rate, for a learning rate LR, we start from LR / 10 up to LR for 30% of the training time, and we scale down to LR / 25 for remaining time, the scaling is done in a cos annealing fashion (see Figure bellow), the momentum is also modified but in the opposite manner starting from 0.95 down to 0.85 and up to 0.95, for more detail see the paper: Super-Convergence.
our region aware loss, 



Data augmentation  
All of the data augmentations are implemented using OpenCV in \base\base_dataset.py, which are: rotation (between -10 and 10 degrees), random croping between 0.5 and 2 of the selected crop_size, random h-flip and blurring



Training  
To train a model, first download the dataset to be used to train the model, then choose the desired architecture, add the correct path to the dataset and set the desired hyperparameters (the config file is detailed below), then simply run:

python train.py --config config.json
The training will automatically be run on the GPUs (if more that one is detected and multipple GPUs were selected in the config file, torch.nn.DataParalled is used for multi-gpu training), if not the CPU is used. The log files will be saved in saved\runs and the .pth chekpoints in saved\, to monitor the training using tensorboard, please run:



Inference  
For inference, we need a PyTorch trained model, the images we'd like to segment and the config used in training (to load the correct model and other parameters),
see our inference example in regtest101.sh  
python simpseg.py --config config.json --model best_model.pth --images images_folder  
The predictions will be saved as .png images using the default palette in the passed fodler name, if not, outputs\ is used, for Pacal VOC the default palette is:


Here are the parameters availble for inference:  

--output       The folder where the results will be saved (default: outputs).
--extension    The extension of the images to segment (default: jpg).
--images       Folder containing the images to segment.
--model        Path to the trained model.
--mode         Mode to be used, choose either `multiscale` or `sliding` for inference (multiscale is the default behaviour).
--config       The config file used for training the model.
Trained Model:

Model	Backbone	PascalVoc val mIoU	PascalVoc test mIoU	Pretrained Model
