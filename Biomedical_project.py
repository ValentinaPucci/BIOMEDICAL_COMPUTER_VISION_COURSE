import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os
import shutil
from scipy.ndimage import zoom
import skimage.transform

import torch
import torch.nn
from torch import nn
from sklearn.metrics import f1_score
import copy
from tqdm import tqdm
from skimage import io, color, transform
from torchvision import transforms, utils, models
from torch.utils.data import Dataset, DataLoader
import glob
import csv




        
#------------------------PREPROCESSING CODE---------------------------------------------------------------------------------------   

def preprocessing(case):

#----------------------------------------- loading img e segm
    string_case = str(case)
    img = nib.load("/Users/valentinapucci/BIOMEDICAL_COMPUTER_VISION/kits19/data/case_"+string_case.zfill(5)+"/imaging.nii.gz")
    segm = nib.load("/Users/valentinapucci/BIOMEDICAL_COMPUTER_VISION/kits19/data/case_"+string_case.zfill(5)+"/segmentation.nii.gz")

    mri_img = img.get_fdata()
    mri_segm = segm.get_fdata()

#------------------------------------------- resizing img
    print('initial img shape: ', mri_img.shape)
    target_shape_img = [256, 256, 256]
    isotr_img = skimage.transform.resize(mri_img, target_shape_img, order=3, cval=0, clip=True, preserve_range=False)
    isotr_img_shape = isotr_img.shape
    factors = (
          target_shape_img[0]/isotr_img_shape[0],
          target_shape_img[1]/isotr_img_shape[1], 
          target_shape_img[2]/isotr_img_shape[2]
          )
    
    isotr_reshaped_img = zoom (isotr_img, factors, order=3, mode= 'nearest')
    reshaped_img_shape = isotr_reshaped_img.shape
    print ('Final img shape: ', reshaped_img_shape)
    print (' ')

#------------------------------------------------ resizing segm
    print('initial segm shape: ', mri_segm.shape)
    target_shape_segm = [256, 256, 256]
    isotr_segm = skimage.transform.resize(mri_segm, target_shape_segm, order=0, cval=0, clip=True, preserve_range=False)
    isotr_segm_shape = isotr_segm.shape
    factors = (
          target_shape_segm[0]/isotr_segm_shape[0],
          target_shape_segm[1]/isotr_segm_shape[1], 
          target_shape_segm[2]/isotr_segm_shape[2]
          )
    
    isotr_reshaped_segm = zoom (isotr_segm, factors, order=3, mode= 'nearest')
    reshaped_segm_shape = isotr_reshaped_segm.shape
    print ('Final segm shape: ', reshaped_segm_shape)
    print (' ')
     
#---------------------------------------------- saving good images 
    img_norm = (isotr_reshaped_img/255).astype(np.float16)
    segm_norm = (isotr_reshaped_segm/255).astype(np.float16)

    for i in range (50, 200):
        j+=1
        string_j = str(j)
        path_save_img = "/Users/valentinapucci/BIOMEDICAL_COMPUTER_VISION/2d/good_images/"+string_j+".npy"
        path_save_segm = "/Users/valentinapucci/BIOMEDICAL_COMPUTER_VISION/2d/good_segmentations/"+string_j+".npy"
        np.save(path_save_img, img_norm[i,:,:])
        np.save(path_save_segm, segm_norm[i,:,:])
        
    #----------------------------------creation of the new folders for splitting the dataset
    os.makedirs('good_images/images_train', exist_ok=True)
    os.makedirs('good_segmentations/segm_train', exist_ok=True)    
    os.makedirs('good_images/images_val', exist_ok=True)
    os.makedirs('good_segmentations/segm_val', exist_ok=True)      

    
    #----------------------------------------------splitting dataset in 2 different folders
    for i in range (0, 23500):
        string_i = str(i)
        shutil.move("/Users/valentinapucci/BIOMEDICAL_COMPUTER_VISION/2d/good_images/"+string_i+".npy", "/Users/valentinapucci/BIOMEDICAL_COMPUTER_VISION/2d/good_images/images_train")
        shutil.move("/Users/valentinapucci/BIOMEDICAL_COMPUTER_VISION/2d/good_segmentations/"+string_i+".npy", "/Users/valentinapucci/BIOMEDICAL_COMPUTER_VISION/2d/good_segmentations/segm_train")

    for i in range (23500, 31350):
        string_i = str(i)
        shutil.move("/Users/valentinapucci/BIOMEDICAL_COMPUTER_VISION/2d/good_images/"+string_i+".npy", "/Users/valentinapucci/BIOMEDICAL_COMPUTER_VISION/2d/good_images/images_val")
        shutil.move("/Users/valentinapucci/BIOMEDICAL_COMPUTER_VISION/2d/good_segmentations/"+string_i+".npy", "/Users/valentinapucci/BIOMEDICAL_COMPUTER_VISION/2d/good_segmentations/segm_val")

        
        
        
#------------------------NEURAL NETWORK CODE---------------------------------------------------------------------------------------     
        
#-----------------------------------Helper function to controll overlap
def image_viewer(output_classes, image, mask):
    x=mask[0].numpy()
    x*=output_classes # The function need the classes to be integer 
    io.imshow(color.label2rgb(x, image[0].numpy(), bg_label=0)) # Set bkg transparent and shows only other classes on top of the input png  
    
    
       
#----------------------------------------------training dataset
class MriDataset(Dataset): 
    
    def __init__(self, image_dir, mask_dir, output_classes, transform = None):
        self.image_dir = image_dir #  Getting image folder
        self.mask_dir = mask_dir  # Getting mask forlder
        self.transform = transform # If there are transformtion to apply
        self.output_classes = output_classes
        
    def __len__(self):
        return len(glob.glob(os.path.join(self.image_dir,'*.npy'))) # number of images found in the folder
        
    def __getitem__(self,idx):
        img_name = os.path.join(self.image_dir,'%d.npy'%idx)
        mask_name = os.path.join(self.mask_dir,'%d.npy'%idx)
        
        img = np.load(img_name).astype(np.uint8)
        mask = np.load(mask_name).astype(np.uint8)
        tmp = np.ndarray((mask.shape[0],mask.shape[1],self.output_classes),dtype=np.uint8) 
        
        for i in range(self.output_classes):
            tmp[:,:,i]=mask 

        sample = {'image': img, 'mask': tmp} # matched image and mask
        
        if self.transform:
            sample=self.transform(sample) # Eventual transformation to be made on the input data
        
        return sample



#----------------------------------------------validation dataset
class MriDataset2(Dataset):  
    
    def __init__(self, image_dir, mask_dir, output_classes, transform = None):
        self.image_dir = image_dir #  Getting image folder
        self.mask_dir = mask_dir  # Getting mask forlder
        self.transform = transform # If there are transformation to apply
        self.output_classes = output_classes
        
    def __len__(self):
        return len(glob.glob(os.path.join(self.image_dir,'*.npy'))) # number of images found in the folder
    
    def __getitem__(self,idx):
        
        idx = idx + 23500   #validation images and masks start from number 23500
        img_name = os.path.join(self.image_dir,'%d.npy'%idx)
        mask_name = os.path.join(self.mask_dir,'%d.npy'%idx)
        
        img = np.load(img_name).astype(np.uint8)
        mask = np.load(mask_name).astype(np.uint8)
        tmp = np.ndarray((mask.shape[0],mask.shape[1],self.output_classes),dtype=np.uint8) 
        
        for i in range(self.output_classes):
            tmp[:,:,i]=mask 

        sample = {'image': img, 'mask': tmp} # matched image and mask
        
        if self.transform:
            sample=self.transform(sample) # Eventual transformation to be made on the input data
        
        return sample
     
    
    
#-----------------------Resize function that applies same transformation to both image and mask coherently
class Resize(object):   
    def __init__(self, out_dim):
        assert isinstance(out_dim, (int, tuple)) # Accepting both one int or a tuple for scale H, W
        self.out_dim = out_dim      
    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        H, W = img.shape[:2]
        
        if isinstance(self.out_dim, int):
            if H > W:
                H_out, W_out = self.out_dim * H/ W, self.out_dim
            else:
                H_out, W_out = self.out_dim, self.out_dim * W/ H
        else:
            H_out, W_out = self.out_dim
        
        H_out, W_out = int(H_out), int(W_out) 
        img = transform.resize(img, (H_out, W_out))
        mask = transform.resize(mask, (H_out, W_out))
        return {'image': img, 'mask': mask}
    
    
    
    
#-------------------------function that transform input data to tensor of shape Nc, H, W    
class ToTensor(object):
    
    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        #if mask.ndim == 3:
        mask = mask.transpose((2, 0, 1))
        if img.ndim == 3:
            img = img.transpose((2, 0, 1))
        return {'image': torch.from_numpy(img).type(torch.FloatTensor),   #crea i tensori
                'mask': torch.from_numpy(mask).type(torch.FloatTensor)}
        
        
        
        
#-----------------Normalization of the input to have image and mask in [0 1]
class Normalize(object):
 
    def __call__(self, sample):
        img, mask = sample['image'], sample['mask']
        return {'image': mask/(mask.shape[2]-1),
                'mask': mask/(mask.shape[2]-1)}
        
        
        
        
#-----------------------------Freezing model to not retrain everything given the low dimensionality of the dataset
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
            
    
             
#------------------Load of pretrained model and change of last layer to match the number of classes to be predicted
def createModel(output_classes):
    my_model = models.segmentation.fcn_resnet101(pretrained=True)
    set_parameter_requires_grad(my_model, feature_extracting= True)
    my_model.classifier[4] = nn.Conv2d(512, output_classes, kernel_size=(1, 1), stride=(1, 1))
    my_model.aux_classifier[4] = nn.Conv2d(256, output_classes, kernel_size=(1, 1), stride=(1, 1))
    my_model.train() 
    return my_model
   
   
   
   
#---------------------------------Function for model training 
def train_model(model, criterion, dataloader, dataloader1, optimizer, metrics, bpath, num_epochs):
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Logger
    fieldnames = ['epoch', 'Train_loss', 'Val_loss'] + \
        [f'Train_{m}' for m in metrics.keys()] + \
        [f'Val_{m}' for m in metrics.keys()]
    with open(os.path.join(bpath, 'log2.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
 
    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
    
        batchsummary = {a: [0] for a in fieldnames}
        
 
        for phase in ['Train', 'Val']:
            if phase == 'Val':
                model.eval()   # Set model to evaluate mode
                dataloaders=dataloader1 #Select dataset for validation
              
            else:
                model.train()  # Set model to training mode
                dataloaders=dataloader # Select dataset for training
                    
            # Iterate over data.
            for sample in tqdm(iter(dataloaders)):
                inputs = sample['image'].to(device)
                masks = sample['mask'].to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()
 
                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    loss = criterion(outputs['out'], masks)
                    y_pred = outputs['out'].data.cpu().numpy().ravel()
                    y_true = masks.data.cpu().numpy().ravel()
                    for name, metric in metrics.items():
                        if name == 'f1_score':
                            batchsummary[f'{phase}_{name}'].append(
                                metric(y_true > 0, y_pred > 0, average='weighted'))
                        
                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()
            batchsummary['epoch'] = epoch
            epoch_loss = loss
            batchsummary[f'{phase}_loss'] = epoch_loss.item()
            print('{} Loss: {:.4f}'.format(phase, loss))
        for field in fieldnames[3:]:
            batchsummary[field] = np.mean(batchsummary[field])
        print(batchsummary)
        with open(os.path.join(bpath, 'log2.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)
            # deep copy the model
            
            if phase == 'Val' and loss < best_loss:
                best_loss = loss
                best_model_wts = copy.deepcopy(model.state_dict())
 
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model




#----------------------------Helper function for model semantic segmentation output
def decode_segmap(image, nc=3):
    label_colors = np.array([(0, 0, 0), (0, 255, 0), (0, 0, 255)]) # 0=background # 1=kidney, 2=tumor
               
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    
    
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb


#--------------------------------main code of the Neural Network
def My_neural_network():
    output_classes = 3
    #dataset and loader
    transformed_dataset_train = MriDataset(image_dir = '/Users/valentinapucci/BIOMEDICAL_COMPUTER_VISION/2d/good_images/images_train/', mask_dir = '/Users/valentinapucci/BIOMEDICAL_COMPUTER_VISION/2d/good_segmentations/segm_train/', 
                                           output_classes = output_classes,
                                           transform = transforms.Compose([ Normalize(), ToTensor()]))
    
    transformed_dataset_val = MriDataset2(image_dir = '/Users/valentinapucci/BIOMEDICAL_COMPUTER_VISION/2d/good_images/images_val/', mask_dir = '/Users/valentinapucci/BIOMEDICAL_COMPUTER_VISION/2d/good_segmentations/segm_val/', 
                                          output_classes = output_classes,
                                           transform = transforms.Compose([ Normalize(), ToTensor()]))
   
    
    # Visual evaluation of correct alignement between image and mask
    plt.figure()
    plt.show
    
    
    for i in range(len(transformed_dataset_train)):
        sample = transformed_dataset_train[i]

        print(i, sample['image'].shape, sample['mask'].shape)
        
        image_viewer(output_classes, **sample)
    
    dataloader = DataLoader(transformed_dataset_train, batch_size = 100, shuffle = True)
    dataloader1 = DataLoader(transformed_dataset_val, batch_size = 100, shuffle = False)

        
    # Model creation and criterion, optimizer and metric
    my_model = createModel(output_classes)
    criterion = torch.nn.MSELoss(reduction='mean')    
    optimizer = torch.optim.Adam(my_model.parameters(), lr=0.001)
    metrics = {'f1_score': f1_score}
    bpath = '/Users/valentinapucci/BIOMEDICAL_COMPUTER_VISION/2d'
    my_model_trained = train_model(my_model, criterion, dataloader, dataloader1, optimizer, metrics, bpath, num_epochs=1)
    

    
    # Getting first batch of the training data to run the model and see its performance
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
          sample_batched['mask'].size())  
        break
        
        
    # Visualization of the model output on one example image from training 
    out = my_model_trained(sample_batched['image'])['out']
    om = torch.argmax(out[0],dim=0).numpy()  #dim (int) – the dimension to reduce. 
    rgb = decode_segmap(om)
    plt.imshow(rgb); 


    PATH = f'/Users/valentinapucci/BIOMEDICAL_COMPUTER_VISION/2d/saved_models/model.py'
    PATH1 = f'/Users/valentinapucci/BIOMEDICAL_COMPUTER_VISION/2d/saved_models/model.txt'
    PATH2 = f'/Users/valentinapucci/BIOMEDICAL_COMPUTER_VISION/2d/saved_models/model.pth'
    torch.save(my_model_trained.state_dict(), PATH)
    torch.save(my_model_trained.state_dict(), PATH1)
    torch.save(my_model_trained.state_dict(), PATH2)
    
    
    
#-----------------------------------------------MAIN()------------------------------------------------------------------------------------

for case in range (0, 209):
    #if (case != 19 & case != 72 & case != 96 & case != 182):   
    print("  ")
    print('case: ', case)
    preprocessing(case)

My_neural_network()

'''
for i in range (120,255):
    string_slice = str(i)
    path_img = "/Users/valentinapucci/BIOMEDICAL_COMPUTER_VISION/2d/good_images/images_train/"+string_slice+".npy"
    path_segm = "/Users/valentinapucci/BIOMEDICAL_COMPUTER_VISION/2d/good_segmentations/segm_train/"+string_slice+".npy"
            
    img_slice = np.load(path_img)
    segm_slice = np.load(path_segm)
            
    fig = plt.figure(figsize=(8,8))
    fig.add_subplot(1,2,1)
    plt.imshow(img_slice, cmap='gray')
    fig = plt.figure(figsize=(8,8))
    fig.add_subplot(1,2,2)
    plt.imshow(segm_slice, cmap='gray')
'''


