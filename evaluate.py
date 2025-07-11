import os
import matplotlib.pyplot as plt

from aiforge.datasets import normalize_data
from aiforge.metrics import plot_confusion_matrix
from aiforge.models import load_checkpoint
from aiforge.test import evaluate_model

def run(model, model_path, test_data, class_names=None):

    model.compile(optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    
    model = load_checkpoint(model, model_path)
    
    test_images, test_labels = test_data
    
    test_images = normalize_data(test_images)
    
    cm = evaluate_model(model, test_images, test_labels, confussion_matrix=True)
    
    plot_confusion_matrix(cm, class_names=class_names, save_path=os.path.join(os.path.dirname(model_path)), file_name='confusion_matrix.png')