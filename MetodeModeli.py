import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import time

def time_print(sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60
    print("{} h {} min {:.3f} sec".format(int(hours),int(mins),sec))   


def make_model(odabrani_model, in_shape, train_baseline, br_klasa=6, lr=1e-4):

    baseline_model = odabrani_model(weights='imagenet', include_top=False, pooling='avg', input_shape=in_shape)
    baseline_model.trainable = train_baseline

    ### ARHITEKTURA MODELA
    inputs = tf.keras.Input(shape = in_shape)
    if train_baseline:
        x = baseline_model(inputs) 
    else:
        x = baseline_model(inputs, training=False)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    outputs = tf.keras.layers.Dense(br_klasa, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(loss='categorical_crossentropy', 
                  optimizer=tf.keras.optimizers.Adam(lr=lr), 
                  metrics=['accuracy'])
    
    return model


def data_generators(preprocess_fun=None, DA=True):
    if DA:
        print("Using DataAugmentation during the training")
        data_gen_train = ImageDataGenerator(rotation_range=40,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True,
                                        preprocessing_function=preprocess_fun)

    else:
        print("Dont use the DataAugmentation during the training")
        data_gen_train = ImageDataGenerator(preprocessing_function=preprocess_fun)

    data_gen_val = ImageDataGenerator(preprocessing_function=preprocess_fun)

    return data_gen_train, data_gen_val 


def train_model(model, train_flow, validation_flow, model_dir, label, epochs):

    checkpoint = ModelCheckpoint(model_dir + label + 'BEST.hdf5', 
                                 save_weights_only=True, mode='auto', 
                                 monitor='val_accuracy',
                                 save_best_only=True,
                                 verbose=1)
    
    start_time = time.time()
    
    history = model.fit(x=train_flow, 
                    steps_per_epoch=len(train_flow), 
                    epochs=epochs, 
                    validation_data=validation_flow, 
                    validation_steps=len(validation_flow),
                    callbacks=[checkpoint])
    
    end_time = time.time()
    
    print('\n\nTraining time: ')
    time_print(end_time - start_time)
    
    model.save(model_dir + label)
    # model.save_weights(model_dir + 'TEZINE_MODELA\\' + label +'.hdf5')


def train_combinedModel(odabrani_model, in_shape, 
                        train_flow, validation_flow, model_dir, label,
                        epochs_FE, epochs_FT, lr_FE, lr_FT,
                        br_klasa=6):
    
    print('Feature extractor training: \n' + '-'*20)
    print(f'LR = {lr_FE}')

    model_combined= make_model(odabrani_model,  in_shape=in_shape, lr=lr_FE, train_baseline=False,  br_klasa=br_klasa)
    model_combined.summary()
    train_model(model_combined, train_flow, validation_flow,
            model_dir=model_dir, label=label,
            epochs=epochs_FE)
    
    model_combined.layers[1].trainable = True
    # model_FT = make_model(odabrani_model,  in_shape=in_shape, lr=lr_FT, train_baseline=True,  br_klasa=br_klasa)

    # print('\n' + '#'*20 + '\nFINE_TUNING\n' + '#'*20)
    # for i in range(2,7):
    #    weights = model_FE.layers[i].get_weights()
    #    model_FT.layers[i].set_weights(weights)
    model_combined.summary()
    train_model(model_combined, train_flow, validation_flow,
            model_dir=model_dir, label=label,
            epochs=epochs_FT)


