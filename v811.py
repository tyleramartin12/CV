from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Activation, Flatten,Input, Convolution2D, MaxPooling2D
from keras.engine.topology import Layer
from keras.utils import np_utils
from keras.datasets import mnist
from keras import backend as K
from keras.models import load_model
from keras.callbacks import ModelCheckpoint


import os
import matplotlib
import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import zipfile
#import requests, StringIO
from sklearn import preprocessing


BATCH_SIZE = 20
NUM_CLASSES = 200
NUM_IMAGES_PER_CLASS = 500
NUM_IMAGES = NUM_CLASSES * NUM_IMAGES_PER_CLASS
#/homes/tam66/Documents/mini/ex
TRAINING_IMAGES_DIR = './tiny/train/'
TRAINING_IMAGES_DIR = 'tiny-imagenet-200/train/'
TRAIN_SIZE = NUM_IMAGES

NUM_VAL_IMAGES = 10000
VAL_IMAGES_DIR = './tiny/val/'
VAL_IMAGES_DIR = 'tiny-imagenet-200/val/'

IMAGE_SIZE = 64
NUM_CHANNELS = 3
IMAGE_ARR_SIZE = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS
IMAGES_URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'

def download_images(url):
    #if (os.path.isdir(TRAINING_IMAGES_DIR)):
    #    print ('Images already downloaded...')
    #    return
    #r = requests.get(url, stream=True)
    #print ('Downloading ' + url )
    #zip_ref = zipfile.ZipFile(StringIO.StringIO(r.content))
    zip_ref.extractall('/tiny.zip')
    zip_ref.close()
def load_training_images(image_dir, batch_size=500,max_classes=200):
    classes = 0

    image_index = 0
    
    images = np.ndarray(shape=(NUM_IMAGES, IMAGE_ARR_SIZE))
    names = []
    labels = []                       
    
    # Loop through all the types directories
    for type in os.listdir(image_dir):
        print(type)
        if os.path.isdir(image_dir + type + '/images/'):
            type_images = os.listdir(image_dir + type + '/images/')
            # Loop through all the images of a type directory
            batch_index = 0;
            #print ("Loading Class ", type)
            for image in type_images:
                image_file = os.path.join(image_dir, type + '/images/', image)

                # reading the images as they are; no normalization, no color editing
                image_data = mpimg.imread(image_file) 
                #print ('Loaded Image', image_file, image_data.shape)
                if (image_data.shape == (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)):
                    images[image_index, :] = image_data.flatten()

                    labels.append(type)
                    names.append(image)
                    
                    image_index += 1
                    batch_index += 1
                if (batch_index >= batch_size):
                    break;
            classes += 1
            if (classes >= max_classes):
                    break;
                    
    return (images, np.asarray(labels), np.asarray(names))
def get_label_from_name(data, name):
    for idx, row in data.iterrows():       
        if (row['File'] == name):
            return row['Class']
        
    return None
def load_validation_images(testdir, d, batch_size=NUM_VAL_IMAGES,max_classes=200):
    labels = []
    names = []
    image_index = 0
    
    images = np.ndarray(shape=(len(d), IMAGE_ARR_SIZE))
    val_images = os.listdir(testdir + '/images/')
           
    # Loop through all the images of a val directory
    batch_index = 0;
    
    
    for image in val_images:
        #if batch_index % 50 ==0:
            #print(batch_index)
        image_file = os.path.join(testdir, 'images/', image)
        #print (testdir, image_file)

        # reading the images as they are; no normalization, no color editing
        image_data = mpimg.imread(image_file) 
        if str(image) in d:
            if (image_data.shape == (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)):
                images[image_index, :] = image_data.flatten()
                image_index += 1
                labels.append(d[str(image)])
                #labels.append(get_label_from_name(validation_data, image))
                names.append(image)
                batch_index += 1
                
            #if (batch_index >= batch_size):
            #    break;
            
    images = images[:len(names)]
    print ("Loaded Validation images ", image_index)
    return (images, np.asarray(labels), np.asarray(names))
def plot_object(data):
    plt.figure(figsize=(1,1))
    image = data.reshape(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    plt.imshow(image, cmap = matplotlib.cm.binary,
               interpolation="nearest")
    plt.axis("off")
    plt.show()
def plot_objects(instances, images_per_row=10, **options):
    size = IMAGE_SIZE
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size,NUM_CHANNELS) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        if (row == len(instances)/images_per_row):
            break
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, **options)
    plt.axis("off")
    plt.show()  
def get_next_batch(batchsize=50):
    for cursor in range(0, len(training_images), batchsize):
        batch = []
        batch.append(training_images[cursor:cursor+batchsize])
        batch.append(training_labels_encoded[cursor:cursor+batchsize])       
        yield batch
def get_next_labels(batchsize=50):
    for cursor in range(0, len(training_images), batchsize):
        yield training_labels_encoded[cursor:cursor+batchsize]     
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
def load_data():
    num_batch_images_train = 300
    num_batch_images_test = 9800
    max_classes = 10
    training_images, training_labels, training_files = load_training_images(TRAINING_IMAGES_DIR,batch_size=num_batch_images_train,max_classes=max_classes)

    shuffle_index = np.random.permutation(len(training_labels))
    training_images = training_images[shuffle_index]
    training_labels = training_labels[shuffle_index]
    training_files  = training_files[shuffle_index]

    le = preprocessing.LabelEncoder()
    training_le = le.fit(training_labels)
    training_labels_encoded = training_le.transform(training_labels)
    
    val_data = pd.read_csv(VAL_IMAGES_DIR + 'val_annotations.txt', sep='\t', header=None, names=['File', 'Class', 'X', 'Y', 'H', 'W'])
    len_x = len(val_data)
    d = {}
    for i in range(0,len_x):
        a = val_data.iloc[i]['File']
        b = val_data.iloc[i]['Class']
        d[a]=b
    d_1 = {}
    for t in training_labels:
        for a,b in d.items():
            if b == t:
                d_1[a] = b
    d = d_1
    val_images, val_labels, val_files = load_validation_images(VAL_IMAGES_DIR, d, batch_size=num_batch_images_test,max_classes=max_classes)
    val_labels_encoded = training_le.transform(val_labels)
    
    return ((training_images,training_labels_encoded),(val_images,val_labels_encoded))
def buildCNNBasedDigitRecognitionModel():
    #n_inputs = 64*64*3
    #n_outputs = 200
    #X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
    #X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
    #y = tf.placeholder(tf.int32, shape=[None], name="y")

    model = Sequential ()

    #LAYER 
    model.add(Convolution2D(64, 3, 1, activation='relu',input_shape=(64,64,3)))

    #LAYER 
    #model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 1, activation='relu'))
    model.add(Convolution2D(64, 3, 1, activation='relu'))
    model.add(MaxPooling2D(pool_size=2,strides=2))
    model.add(Dropout(0.25))

    #LAYER 
    model.add(Convolution2D(128, 3, 1, activation='relu'))
    model.add(Convolution2D(128, 3, 1, activation='relu'))
    model.add(MaxPooling2D(pool_size=2,strides=2))
    model.add(Dropout(0.25))

    #LAYER
    model.add(Convolution2D(256, 3, 1, activation='relu'))
    model.add(Convolution2D(256, 3, 1, activation='relu'))
    model.add(MaxPooling2D(pool_size=2,strides=2))
    model.add(Dropout(0.25))

    #LAYER
    model.add(Convolution2D(512, 3, 1, activation='relu'))
    #model.add(Convolution2D(512, 3, 1, activation='relu'))
    model.add(MaxPooling2D(pool_size=2,strides=4))
    model.add(Dropout(0.25))

    model.add(Flatten())

    #LAYER
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    #LAYER
    model.add(Dense(200, activation='softmax'))
    '''
    model.add(Convolution2D(32, 3, 3, activation='relu',input_shape=(64,64,3)))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(200, activation='softmax'))
    '''
    return model

def load_and_prep():
    (train_images, train_labels),(test_images, test_labels) = load_data()

    train_images = train_images.reshape (train_images.shape[0], 64, 64,3).astype('float32')/255.0
    test_images = test_images.reshape (test_images.shape[0],  64, 64,3).astype('float32')/255.0
    train_labels = np_utils.to_categorical(train_labels, 200)
    test_labels = np_utils.to_categorical(test_labels, 200)
    return (train_images, train_labels),(test_images, test_labels)

def trainModels(train_images,train_labels,models):
    for model_name in models:
        model=models[model_name]
        print('Training model: ' +str(model_name))
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        filepath = 'mod.hdf5'
        callback = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
        callbacks = [callback]
        EPOCHS = 20
        model.fit(train_images, train_labels, batch_size=20, nb_epoch=EPOCHS, verbose=1,callbacks=callbacks)

def evaluateModels(test_images, test_labels, models):
    for model_name in models:
        model = models[model_name]
        print('-------------------------\nEvaluating model: '+ str(model_name))
        score = model.evaluate(test_images, test_labels, verbose=0)
        print('Test loss: ' + str(score[0]))
        print('Test accuracy: ' + str(score[1]))

def defineDigitRecognitionModels():
    models={}
    #models['simple_fully_connected'] = buildSimpleDigitRecognitionModel()
    #models['custom_fully_connected'] = buildCustomDigitRecognitionModel()
    models['convolution_based'] = buildCNNBasedDigitRecognitionModel()
    return models

def performDigitRecognitionExperiments():
    #download_images(IMAGES_URL)
    print('gathering train/test sets')
    (train_images, train_labels), (test_images, test_labels) = load_and_prep()

    x = int(len(test_labels)/2)
    print('----x-----')
    print(x) #6200
    val_images = test_images[x:]
    test_images = test_images[:x]

    val_labels = test_labels[x:]
    test_labels = test_labels[:x]

    print('defining model')
    models = defineDigitRecognitionModels()
    print('training')
    trainModels(train_images, train_labels, models)
    print('TEST')
    evaluateModels(test_images, test_labels, models)
    print('VAL')
    evaluateModels(val_images, val_labels, models)

def main():
    # Commands used to force Keras and Tensorflow to use CPU as GPU resources are very limited. Do not run your model training  and testing on GPUs on MSALT machines
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.logging.set_verbosity(tf.logging.ERROR)

    #with tf.device('/gpu:0'):
    with tf.device('/device:GPU:0'):
        performDigitRecognitionExperiments()

if __name__ == "__main__":
    main()