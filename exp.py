#!python/bin

import tensorflow as tf
tf.random.set_seed(1)
import random
random.seed(1)
from pickle import FALSE
import os
import sys
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow.keras.backend as K
from tf_keras_vis.saliency import Saliency
from sklearn.metrics import confusion_matrix, accuracy_score

from split_w_stratification import *

from sklearn.metrics import roc_curve, auc

############################

covariatesFile = 'metadata.csv'
imageDirectory = 'path to directory with the images' 

###
#   PARAMETERS
###

def pmts(modelname, first_train, model_loaded, epochs, split, lr, ld, aug, extensions, use_age_sex, regu, batchsize):
    return {
    'covName': 'group_dummy',
    'metadata1': 'sex_dummy',
    'metadata2': 'age_znorm',
    'modelname': modelname,
    'aug': aug,
    'batch_size': batchsize,
    'num_epochs': epochs,
    'learning_rate': lr,
    'learning_decay': ld,
    'regu': float(regu) if regu != 'False' else 'False',
    'first_train': first_train,
    'num_metadata': 2,
    'age_and_sex' : use_age_sex,
    'ext_data': extensions.split(','),
    'model_loaded': model_loaded,
    'explainability': False,
    'ran_split': split,
    'imagex': 160,
    'imagey': 192,
    'imagez': 160
    }

#
parameters = pmts(sys.argv[1], #pmts(modelname (str),
                  sys.argv[2], #first_train (str)
                  sys.argv[3], #model_loaded (str)
                  int(sys.argv[4]), #epochs (int)
                  int(sys.argv[5]), #split (int)
                  float(sys.argv[6]), #lr (float)
                  float(sys.argv[7]), #ld (float)
                  sys.argv[8], #aug (str)
                  sys.argv[9], #extensions (str)
                  sys.argv[10], #use_age_sex (str)
                  sys.argv[11], #regu (str-float)
                  int(sys.argv[12])) #batchsize (int)

# define working directory
working_dir = 'path to where the models are saved'
loading_dir = 'path to where the models are loaded from'

# Get all images index
cov = pd.read_csv(covariatesFile)

# lets create binned age based on the mean age
conditions_age = [(cov['age'] <= cov['age'].mean()),
                  (cov['age'] > cov['age'].mean())
                  ]

v_namse = ['younger', 'older']

# create new column 'binned_age' to stratify with
cov['binned_age'] = np.select(conditions_age, v_namse)
IDs_list = cov['Subject'].to_numpy()
IDs = IDs_list

####
#crete a weight for the diferent studies
studies = cov.study.unique()
w_of_studies = {}
w_of_pd_studies = {}
w_of_hc_studies = {}

for i in studies:
    ith_study = cov[cov.study == i]
    ith_study_pd = cov.loc[(cov['study'] == i) & (cov['group_dummy'] == 1)]
    ith_study_hc = cov.loc[(cov['study'] == i) & (cov['group_dummy'] == 0)]
    w_of_studies[i] = 1 - len(ith_study)/len(cov)
    w_of_pd_studies[str(i)+'_PD'] = 1 - len(ith_study_pd)/len(cov[cov['group_dummy'] == 1])
    w_of_hc_studies[str(i)+'_HC'] = 1 - len(ith_study_hc)/len(cov[cov['group_dummy'] == 0])

#add weights to cov
cov['sample_weights'] = cov[['study','group_dummy']].apply(lambda x : w_of_pd_studies[x[0]+'_PD'] if (x[1] == 1) else w_of_hc_studies[x[0]+'_HC'], axis=1)

# Extract covariates (creates an array with patient ID in column 0, covariate (e.g. PD or HC) in column 1, and other covariates to adjust for in next columns. to pass to data generator)
covs = pd.concat([cov.loc[0:, "Subject"], cov.loc[0:, parameters['covName']], cov.loc[0:, parameters['metadata1']],
                  cov.loc[0:, parameters['metadata2']], cov.loc[0:,"sample_weights"], cov.loc[0:, "age"], 
                  cov.loc[0:, "study"], cov.loc[0:, "group"]],
                 axis=1)
covs = covs.to_numpy()

###
#  EXPLAINABILITY
###

model = tf.keras.models.load_model(working_dir+'/'+parameters['model_loaded']+'/'+parameters['model_loaded']+'.hdf5',compile=False)
# read the predictions to extract the true positives and true negatives
test_pred = pd.read_csv(working_dir + parameters['modelname'] + '/' + 'test' + '_split_predictions.csv')
valid_pred = pd.read_csv(working_dir + parameters['modelname'] + '/' + 'valid' + '_split_predictions.csv')

#save test predictions
#get the predictions in a bolean form
y_pred_bool = (np.array(test_pred['predictions'])>=0.5)
#make them ints
y_pred = y_pred_bool.astype(int)
#get the true labels
y_true = np.array(test_pred['group_dummy'])

TP_idx = np.where((y_pred ==1) & (y_true==1)) #true positives
TN_idx = np.where((y_pred ==0) & (y_true==0)) #true negatives
FP_idx = np.where((y_pred == 1) & (y_true == 0)) #false positive
FN_idx = np.where((y_pred == 0) & (y_true == 1)) #false negative

# get the ids for the true positive predictions for the false positive predictions
test_ids = test_pred.iloc[:,1].to_numpy()

TP_ids = test_ids[TP_idx]
TN_ids = test_ids[TN_idx]
FP_ids = test_ids[FP_idx]
FN_ids = test_ids[FN_idx]

# conf_matrix = confusion_matrix(y_true=y_true,y_pred=y_pred,labels=np.array([0,1]))
conf_matrix = confusion_matrix(y_true=y_true,y_pred=y_pred)

fpr, tpr, thresholds = roc_curve(y_true,test_pred.predictions,pos_label=1)
roc_auc = auc(fpr,tpr)
plt.figure()
lw = 2
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    lw=lw,
    label="ROC curve (area = %0.2f)" % roc_auc,
)
plt.plot([0, 1], [0, 1], color="red", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic")
plt.legend(loc="lower right")
plt.savefig(working_dir + parameters['modelname'] + '/' + parameters['modelname'] + '_roc_curve.png')


# preload the data to run explainability
def load_image_data(data_directory, ids_to_load, img_dim, n_samples, extentions):
    '''
    inputs: 
        data_directory: directory containing images 
        ids_to_load: np array of ids to load
        img_dim: tuple of x,y,z image dimensions
        n_samples: int
        extensions: list (e.g., ['ad','fa','md','rd','jac','aff'])
    outputs: 
        data: tensor containing image data for each patient
    '''
    (xdim, ydim, zdim) = img_dim
    patient_list = ids_to_load[:n_samples]
    data = np.zeros((len(patient_list), xdim, ydim, zdim, len(extentions)+2)) #placeholder tensor for image data
    data_cov = np.zeros((len(patient_list), 2)) #placeholder tensor for covaritates data
    imgs={} #declare dictionary to save metric maps

    for i,ID in enumerate(patient_list): 
        imgs={} #declare dictionary to save metric maps
        for n, ext in enumerate(extentions):
            if n == 0:
                #read metric
                imgs[str(ext)] = sitk.ReadImage(data_directory +str(ext)+'/'+ str(ID) + '__' + str(ext) + '.nii.gz')
                imgs[str(ext)] = sitk.GetArrayFromImage(imgs[str(ext)])
                #save it to Xn
                Xn = np.float32(imgs[str(ext)].reshape(xdim, ydim, zdim, 1))
            else:
                #read metric
                imgs[str(ext)] = sitk.ReadImage(data_directory +str(ext)+'/'+ str(ID) + '__' + str(ext) + '.nii.gz')
                imgs[str(ext)] = sitk.GetArrayFromImage(imgs[str(ext)])
                #save it to Xn
                Xn = np.concatenate((Xn,np.float32(imgs[str(ext)].reshape(xdim, ydim, zdim, 1))), axis=3)

        #read the t1w derived images
        #declare path
        jac_path = data_directory +'/jacobians/'+ str(ID) + '.nii.gz'
        t1w_path = data_directory +'/affine/'+ str(ID) + '.nii.gz'
        #read image
        jac_img = sitk.ReadImage(jac_path,sitk.sitkFloat32)
        t1w_img = sitk.ReadImage(t1w_path,sitk.sitkFloat32)
        #get array out of the image
        array_jac = sitk.GetArrayFromImage(jac_img)     
        array_t1w = sitk.GetArrayFromImage(t1w_img)
        #get the age and sex
        data_cov[i,:] = covs[np.where(covs[:, 0] == ID)][0, 2:4]

        #store 4D image
        data[i,] = np.concatenate((Xn,np.float32(array_jac.reshape(xdim, ydim, zdim, 1)),np.float32(array_t1w.reshape(xdim, ydim, zdim, 1))), axis=3)

    return [data,data_cov]

def model_modifier_function(cloned_model):
    cloned_model.layers[-1].activation = tf.keras.activations.linear
    
#create Saliency object.
saliency = Saliency(model,
                    model_modifier=model_modifier_function,
                    clone=True)

atlas = sitk.ReadImage('pd25_atlas.nii.gz')

def generate_saliency_maps(saliency_object,score_function,arrays,IDs,save_dir,save_dir_qc,extentions):
    '''
    saliency_object = object from library tf-keras-vis
    score_funct = score function according to the docs of the library
    arrays = list of 3D loaded arrays that will be used to calculate the saliency maps
    IDs = list of strings with the ids
    save_dir = directory that will be created with the respective predicitons
    '''
    for i, ID in enumerate(IDs):
        #smoothgrad
        saliency_map = saliency_object(score_function,
                            [arrays[0][i,:,:,:,:],arrays[1][i,:]],
                            smooth_samples=25, # The number of calculating gradients iterations.
                            smooth_noise=0.005,# noise spread level.
                            keepdims=True) 

        salm={} #declare dictionary to save met
        sal_ag_sex=np.zeros(2) #declare the variable to save covariates
        for n, ext in enumerate(extentions):
            #save images in a dictionary
            salm['salm_'+str(ext)] = saliency_map[0][0,:,:,:,n].numpy().reshape(parameters['imagex'],parameters['imagey'],parameters['imagez'])
            #save an image as a thumbnail
            plt.figure()
            plt.imshow(salm['salm_'+str(ext)][:,:,72])
            plt.title(ID+'_'+str(ext))
            plt.savefig(save_dir_qc+ID+'_'+str(ext)+'.png')  
            #convert to a SITK image to save it
            salm['salm_'+str(ext)] = sitk.GetImageFromArray(salm['salm_'+str(ext)])
            salm['salm_'+str(ext)].CopyInformation(atlas)
            sitk.WriteImage(salm['salm_'+str(ext)], save_dir + ID + '_' + str(ext) + '.nii.gz')
        
        #save covariates in a dictionary
        ith_sal_ag_sex = saliency_map[1].numpy().reshape(arrays[1][1,:].shape)
        #create the figure
        fig = plt.figure(figsize=[10,10])
        margins = {
            "left"   :     .1,
            "bottom" :     .3,
            "right"  :     .9,
            "top"    :     .9
        }
        fig.subplots_adjust(**margins)
        ax = fig.add_subplot(111)   
        ax.bar(['Sex','Age'],ith_sal_ag_sex,align = 'center')
        plt.xticks(rotation=270)
        ax.set_title(f"{ID}_age_sex")
        save_cov = working_dir + '/' + parameters['modelname'] + '/' + parameters['modelname'] + '_cov_/'
        if not os.path.exists(save_cov):
            os.makedirs(save_cov)
        fig.savefig(f"{save_cov}{ID}_age_sex.png")
        sal_ag_sex += ith_sal_ag_sex
    return sal_ag_sex


a = 'TP'
#directory to save the true positives
save_dir = working_dir + '/' + parameters['modelname'] + '/' + parameters['modelname'] + f'_{a}_sal/'
save_dir_qc = working_dir +'/'+parameters['modelname']+'/'+parameters['modelname']+f'_{a}_sal_qc/'
try:
    os.makedirs(save_dir)
except:
    print(save_dir + ' already exists')
try:
    os.makedirs(save_dir_qc)
except:
    print(save_dir_qc + ' already exists')

#load the images to use for the saliency map computation
array = load_image_data(imageDirectory, TP_ids, (parameters['imagex'],parameters['imagey'],parameters['imagez']), len(TP_ids), parameters['ext_data'])

def score_function(output):  # output shape is (batch_size, 1)
    return output[:, 0]  # for positive case

#generate_saliency_maps(saliency,score_function,array,TP_ids,save_dir)
saliency_cov_total = generate_saliency_maps(saliency,score_function,array,TP_ids,save_dir,save_dir_qc,['ad','fa','md','rd','jacobian','affine'])

#get the average of the covariates
ave_saliency_cov = saliency_cov_total / len(TP_ids)

#where to save the covariates explanations
save_dir = working_dir + '/' + parameters['modelname'] + '/'

#save the average maps
for ext in ['ad','fa','md','rd','jacobian','affine']:
    for count, ID in enumerate(TP_ids):
        #read the explanation for a particular imaging ext
        img = sitk.ReadImage(f"{save_dir}{ID}_{ext}.nii.gz")
        img_a = sitk.GetArrayFromImage(img)
        if count == 0:
            summap = img_a
        else:
            summap += img_a
        print(f'array:{ext}, read image:{count}, ID: {ID}')
            
    #divide by number of maps to average
    summap=summap/len(TP_ids)
    #get average summap image
    average=sitk.GetImageFromArray(summap)
    #copy info to keep the orientation info
    average.CopyInformation(atlas)
    sitk.WriteImage(average,f"{save_dir}_{ext}_avg.nii.gz")
