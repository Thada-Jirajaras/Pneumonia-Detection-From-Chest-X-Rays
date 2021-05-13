from keras.preprocessing.image import ImageDataGenerator
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50 
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
import sklearn.model_selection as skl
import argparse

def create_splits(dataset, stratify_on ):
    train_data, val_data = skl.train_test_split(dataset, 
                                   test_size = 0.2, 
                                   stratify = dataset[stratify_on])
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


def make_train_gen(my_train_idg, train_df, imgpath_col, class_col, 
                   IMG_SIZE = (224, 224), batch_size = 128):
    train_gen = my_train_idg.flow_from_dataframe(dataframe = train_df, 
                                            directory=None, 
                                             x_col = imgpath_col,
                                             y_col = class_col,
                                             class_mode = 'binary',
                                             target_size = IMG_SIZE, 
                                             batch_size = batch_size)
    return train_gen


def make_val_gen(my_val_idg, val_df, imgpath_col, class_col, 
                 IMG_SIZE = (224, 224), batch_size = 128):
    val_gen = my_val_idg.flow_from_dataframe(dataframe = val_df, 
                                            directory=None, 
                                             x_col = imgpath_col,
                                             y_col = class_col,
                                             class_mode = 'binary',
                                             target_size = IMG_SIZE, 
                                             batch_size = batch_size)  
    return val_gen


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
    history = model.fit_generator(train_gen, 
                               validation_data = validation_data,
                               epochs = epochs, 
                               callbacks = callbacks_list)
    return(history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='args')
    parser.add_argument('--someflag', type=int, default = 1)
    args = parser.parse_args()