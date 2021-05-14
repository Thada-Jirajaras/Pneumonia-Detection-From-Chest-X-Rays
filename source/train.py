from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet import ResNet50 
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import Sequence
import sklearn.model_selection as skl
from glob import glob
import pandas as pd
import argparse
import os


def _create_dataset(input_file, output_trainfile, output_valfile):
    
    ## Below is some helper code to read all of your full image filepaths into a dataframe for easier manipulation
    all_xray_df = pd.read_csv(input_file)
    all_image_paths = {os.path.basename(x): x for x in 
                       glob(os.path.join('/data','images*', '*', '*.png'))}
    print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
    all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
    all_xray_df.sample(3)
    
    
    ## Here you may want to create some extra columns in your table with binary indicators of certain diseases 
    ## rather than working directly with the 'Finding Labels' column
    all_xray_df['Finding Labels'] =  all_xray_df['Finding Labels'].str.split('|')
    all_labels = all_xray_df['Finding Labels'].explode().unique()
    all_xray_df['Finding Labels'] = all_xray_df['Finding Labels'].map(set)
    for label in all_labels:
        all_xray_df[label] = all_xray_df['Finding Labels'].map(lambda x: 1 if label in x else 0)
    
    ## Here we can create a new column called 'pneumonia_class' that will allow us to look at 
    ## images with or without pneumonia for binary classification
    all_xray_df['pneumonia_class']  = all_xray_df['Pneumonia'].map(lambda x: 'Pneumonia' if x == 1 else 'Non-pneumonia') 
    
    ## split your original dataframe into two sets 
    ## that can be used for training and testing your model
    train_data, val_data = create_splits(all_xray_df,  stratify_on = 'pneumonia_class')
    train_data[['path', 'Pneumonia', 'pneumonia_class']].to_csv(output_trainfile, index = False)
    val_data[['path', 'Pneumonia', 'pneumonia_class']].to_csv(output_valfile, index = False)
    


def create_splits(dataset, stratify_on, random_state = 9):
    train_data, val_data = skl.train_test_split(dataset, 
                                   test_size = 0.2, 
                                   stratify = dataset[stratify_on],
                                   random_state = random_state)
    return train_data, val_data

def my_image_augmentation(horizontal_flip = True,
                         height_shift_range = 0.1,
                         width_shift_range = 0.1,
                         rotation_range = 5,
                         shear_range = 0.05,
                         zoom_range = 0.05):
    my_idg = ImageDataGenerator(rescale = 1. / 255.0,
                              horizontal_flip = horizontal_flip, 
                              vertical_flip = False, 
                              height_shift_range= height_shift_range, 
                              width_shift_range=width_shift_range, 
                              rotation_range=rotation_range, 
                              shear_range = shear_range,
                              zoom_range=zoom_range)
    return my_idg


def make_val_gen(val_df, imgpath_col, class_col, 
                 IMG_SIZE = (224, 224), batch_size = 32):
    
    my_val_idg = ImageDataGenerator(rescale = 1. / 255.0)    
    val_gen = my_val_idg.flow_from_dataframe(dataframe = val_df, 
                                            directory=None, 
                                             x_col = imgpath_col,
                                             y_col = class_col,
                                             class_mode = 'binary',
                                             target_size = IMG_SIZE, 
                                             batch_size = batch_size)  
    return val_gen

class TrainGen(Sequence):
    
    def __init__(self, train_data):
        
        # prepare image generator
        self.train_data = train_data.copy()
        self.batch_size = 32
        self.IMG_SIZE = (224, 224)
        self.imgpath_col = 'path'
        self.class_col = 'pneumonia_class'
        horizontal_flip = True,
        height_shift_range = 0.1
        width_shift_range = 0.1
        rotation_range = 5
        shear_range = 0.05
        zoom_range = 0.05
        self.my_idg = ImageDataGenerator(rescale = 1. / 255.0,
                              horizontal_flip = horizontal_flip, 
                              vertical_flip = False, 
                              height_shift_range= height_shift_range, 
                              width_shift_range=width_shift_range, 
                              rotation_range=rotation_range, 
                              shear_range = shear_range,
                              zoom_range=zoom_range)
        
        # Init the flow_from_dataframe generator for first iteration
        self.on_epoch_end()
    
    def on_epoch_end(self):
         
        # Update the flow_from_dataframe generator for first iteration    
        pneumonia_data = self.train_data[self.train_data.Pneumonia == 1]
        pneumonia_count = self.train_data[self.train_data.Pneumonia == 1].shape[0]
        sample_size = pneumonia_count + (self.batch_size - (2*pneumonia_count)%self.batch_size)
        non_pneumonia_data = self.train_data[self.train_data.Pneumonia == 0].sample(sample_size)
        sampled_train_data = pd.concat([pneumonia_data, non_pneumonia_data])
        self.n = sampled_train_data.shape[0]
        self.train_gen = self.my_idg.flow_from_dataframe(dataframe = sampled_train_data, 
                                directory=None, 
                                 x_col = self.imgpath_col,
                                 y_col = self.class_col,
                                 shuffle=True,    
                                 class_mode = 'binary',
                                 target_size = self.IMG_SIZE, 
                                 batch_size = self.batch_size)
    
    def __getitem__(self, index):
        trainX, trainY  = self.train_gen.next()
        return trainX, trainY
    
    def __len__(self):
        return self.n // self.batch_size





def load_pretrained_model(lay_of_interest = 'block5_pool'):
    
    # model = VGG16(include_top=True, weights='imagenet')
    # transfer_layer = model.get_layer(lay_of_interest)
    # vgg_model = Model(inputs = model.input, outputs = transfer_layer.output)
    
    # Todo
    model = VGG16(include_top=True, weights='imagenet')
    transfer_layer = model.get_layer(lay_of_interest)
    vgg_model = Model(inputs = model.input, outputs = transfer_layer.output)
    return vgg_model

def build_my_model(vgg_model):
    
    # my_model = Sequential()
    # ....add your pre-trained model, and then whatever additional layers you think you might
    # want for fine-tuning (Flatteen, Dense, Dropout, etc.)
    
    # if you want to compile your model within this function, consider which layers of your pre-trained model, 
    # you want to freeze before you compile 
    
    # also make sure you set your optimizer, loss function, and metrics to monitor
    
    # Todo
    for layer in vgg_model.layers[0:17]:
        layer.trainable = False
        
    my_model = Sequential()

    # Add the convolutional part of the VGG16 model from above.
    my_model.add(vgg_model)

    # Flatten the output of the VGG16 model because it is from a
    # convolutional layer.
    my_model.add(Flatten())

    # Add a dense (aka. fully-connected) layer.
    # This is for combining features that the VGG16 model has
    # recognized in the image.
    my_model.add(Dense(1, activation='sigmoid'))    
     
        
    ## STAND-OUT Suggestion: choose another output layer besides just the last classification layer of your modele
    ## to output class activation maps to aid in clinical interpretation of your model's results    
    
    return my_model

def build_callbacks_list(weight_path,
                         metric = 'val_loss',
                         mode = 'min'):
    checkpoint = ModelCheckpoint(weight_path, 
                                  monitor = metric, 
                                  verbose= 1 , 
                                  save_best_only = True, 
                                  mode = mode, 
                                  save_weights_only = True)

    early = EarlyStopping(monitor= metric, 
                           mode = mode, 
                           patience = 10)

    return([checkpoint, early])

def train(model, callbacks_list, train_gen, validation_data, epochs):
    optimizer = Adam(lr=1e-4)
    loss = 'binary_crossentropy'
    metrics = ['binary_accuracy']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Todo
    history = model.fit_generator(generator = train_gen, 
                               validation_data = validation_data,
                               epochs = epochs, 
                               callbacks = callbacks_list)
    return(history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='args')
    parser.add_argument('--someflag', type=int, default = 1)
    args = parser.parse_args()