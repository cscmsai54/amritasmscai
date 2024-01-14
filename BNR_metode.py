import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import time

def time_print(sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60
    print("{} h {} min {:.3f} sec".format(int(hours),int(mins),sec))   

def make_model(odabrani_model, br_klasa, in_shape, 
               train_baseline=False, lr=0.0001, pooling=None):
    
    baseline_model = odabrani_model(weights='imagenet', include_top=False, pooling=pooling, input_shape=in_shape)
   
    if pooling is not None:
        model = Sequential([
                    baseline_model,
                    Dense(256, activation='relu'),
                    BatchNormalization(),
                    Dense(128, activation='relu'),
                    BatchNormalization(),
                    Dense(br_klasa, activation='softmax')])
    else:
        model = Sequential([
                    baseline_model,
                    Flatten(),
                    Dense(256, activation='relu'),
                    BatchNormalization(),
                    Dense(128, activation='relu'),
                    BatchNormalization(),
                    Dense(br_klasa, activation='softmax')])
        
    baseline_model.trainable = train_baseline
    
    model.compile(loss='categorical_crossentropy', 
                  optimizer=Adam(lr=lr), 
                  metrics=['accuracy'])
    
    return model

def train_combinedModel(odabrani_model, BR_KLASA, IN_SHAPE, 
                        train_flow, validation_flow, model_dir, label,
                        epochs_FE, epochs_FT, lr_FE, lr_FT, pooling = None):
    print('Feature extractor training: \n' + '-'*20)
    print(f'LR = {lr_FE}')
    model_combined = make_model(odabrani_model, BR_KLASA, IN_SHAPE, train_baseline=False, lr=lr_FE, pooling=pooling)
    model_combined.summary()
    train_model(model_combined, train_flow, validation_flow,
            model_dir=model_dir, label=label,
            epochs=epochs_FE)
    model_combined.layers[0].trainable = True
    model_combined.compile(loss='categorical_crossentropy', 
                  optimizer=Adam(lr=lr_FT), 
                  metrics=['accuracy'])
    print('\n' + '#'*20 + '\nFINE_TUNING\n' + '#'*20)
    model_combined.summary()
    train_model(model_combined, train_flow, validation_flow,
            model_dir=model_dir, label=label,
            epochs=epochs_FT)

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


def train_model(model, train_flow, validation_flow,
                model_dir, label, epochs):
    
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


def evaluation_report(model, validation_flow, confusion_matrix=False):
    Y_pred = model.predict(validation_flow)
    y_pred = np.argmax(Y_pred, axis=1)
    if confusion_matrix:
        print('Confusion Matrix')
        print(confusion_matrix(validation_flow.classes, y_pred))
    print('\nClassification Report')
    target_names = ['Glass', 'Metal', 'No trash', 'Other', 'Plastic', 'Rubber']
    print(classification_report(validation_flow.classes, y_pred, target_names=target_names))


def evaluate_model(model, validation_flow):
    _, acc = model.evaluate(x=validation_flow)
    print('Točnost na skupu za testiranje: {:.2f}%'.format(acc*100))


def print_LR_separator(lr):
    print('#'*50)
    print('LR = ' + str(lr))
    print('#'*50 + '\n\n')


def CrossValidation_printCheck(odabrani_model, preprocess_fun, train_baseline,
                                BATCH_SIZE, IN_SHAPE, BR_KLASA, EPOCHS, pooling=None):
        if train_baseline:
            print('FINE-TUNING')
        else:
            print('FEATURE EXTRACTOR')
        print('BATCH_SIZE: ' + str(BATCH_SIZE))
        print('IN_SHAPE: ' + str(IN_SHAPE))
        print('EPOCHS: ' + str(EPOCHS))
        
        model = make_model(odabrani_model, BR_KLASA, IN_SHAPE, 
                                train_baseline=train_baseline, lr=0, pooling=pooling)
        model.summary()
        print('\n\n')
        
        #print('BR_KLASA: ' + str(BR_KLASA))



def CrossValidation(SPLITS_DIR, LRS, 
                    odabrani_model, preprocess_fun,  train_baseline,
                    BATCH_SIZE, IN_SHAPE, BR_KLASA, EPOCHS,
                    SPLIT_NOs=range(5), pooling=None):
    
    CrossValidation_printCheck(odabrani_model, preprocess_fun, train_baseline,
                                BATCH_SIZE, IN_SHAPE, BR_KLASA, EPOCHS, pooling=pooling)
    data_gen_train, data_gen_val = data_generators(preprocess_fun)
    sep = '\n\n' + '-'*50 + '\n'
    for lr in LRS:
        losses = []
        accs = []
        print_LR_separator(lr)
        for i in SPLIT_NOs:
            print('#'*10 + ' Split ' + str(i+1) + ' ' + '#'*10)
            print('#'*5 + 'lr = ' + str(lr) )
            TRAIN_DIR = SPLITS_DIR + '\\Split_' + str(i+1) +  '\\Train'
            VALID_DIR = SPLITS_DIR + '\\Split_' + str(i+1) +  '\\Valid'
           
            train_flow = data_gen_train.flow_from_directory(TRAIN_DIR, target_size=(IN_SHAPE[0], IN_SHAPE[1]),
                                                      batch_size = BATCH_SIZE, class_mode = 'categorical')
            validation_flow = data_gen_val.flow_from_directory(VALID_DIR, target_size=(IN_SHAPE[0], IN_SHAPE[1]),
                                                            batch_size=BATCH_SIZE, class_mode='categorical', 
                                                            shuffle=False)
            
            print('TRAIN_DIR' + TRAIN_DIR)
            print('VALID_DIR' + VALID_DIR)
            
            model = make_model(odabrani_model, BR_KLASA, IN_SHAPE, 
                                train_baseline=train_baseline, lr=lr, pooling=pooling)
            
            history = model.fit(x=train_flow, 
                            steps_per_epoch=len(train_flow), 
                            epochs=EPOCHS, 
                            validation_data=validation_flow, 
                            validation_steps=len(validation_flow))
            
            print(sep)
            loss, acc = model.evaluate(x=validation_flow)
            print('Gubitak na skupu za validaciju: {:.2f}, Točnost na skupu za Validaciju: {:.2f}%'.format(loss, acc*100))
            print(sep)
            losses.append(loss)
            accs.append(acc*100)
        
            print('\n\n' + '#'*50)
    
        print('\n\n' + '#'*50)
        avg_loss = np.mean(losses)
        std_loss = np.std(losses)
        print('LOSS --> mean: {:.2f}, std: {:.2f}'.format(avg_loss, std_loss))

        avg_acc = np.mean(accs)
        std_acc = np.std(accs)
        print('ACCURACY --> mean: {:.2f}%, std: {:.2f}%'.format(avg_acc, std_acc))
        print('\n\n' + '#'*50)
        print('\n\n' + '#'*50)

