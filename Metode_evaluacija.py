import numpy as np
from sklearn.metrics import classification_report
import time

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

