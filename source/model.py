from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.applications.resnet import ResNet50 
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model


def load_pretrained_VGG16(lay_of_interest = 'block5_pool', 
                          trainable_after_layer = 17):
    
    # Todo
    model = VGG16(include_top=True, weights='imagenet')
    transfer_layer = model.get_layer(lay_of_interest)
    vgg_model = Model(inputs = model.input, outputs = transfer_layer.output)
    for layer in vgg_model.layers[0:trainable_after_layer]:
        layer.trainable = False
        
    return vgg_model


def load_pretrained_ResNet50(lay_of_interest = 'avg_pool', 
                          trainable_after_layer = 17):
    
    # Todo
    model = ResNet50(include_top = True, weights='imagenet')     
    transfer_layer = model.get_layer(lay_of_interest)
    resnet50_model = Model(inputs = model.input, outputs = transfer_layer.output)
    for layer in resnet50_model.layers[0:trainable_after_layer]:
        layer.trainable = False
                  
    return resnet50_model

def build_my_model(model, flatten_transfer_layer = True):
    my_model = Sequential()
    my_model.add(model)
    if flatten_transfer_layer:
        my_model.add(Flatten())
    my_model.add(Dense(1, activation='sigmoid'))    
     
        
    ## STAND-OUT Suggestion: choose another output layer besides just the last classification layer of your modele
    ## to output class activation maps to aid in clinical interpretation of your model's results    
    
    return my_model
