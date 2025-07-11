import tensorflow as tf

from aiforge.datasets import normalize_data
from aiforge.train import train_model

def run(model, train_data, epochs, seed, model_path=None):
    train_images, train_labels = train_data
    
    callbacks = None
    
    if model_path is not None:
        
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                 save_weights_only=True,
                                                 verbose=1)
        callbacks = [cp_callback]
    
    # Normalize the data
    train_images = normalize_data(train_images)
    
    train_dict = {
        'optimizer': 'adam',
        'loss': 'sparse_categorical_crossentropy',
        'metrics': ['accuracy'],
        'train_labels': train_labels,
        'callbacks': callbacks
    }
       
    # Train the model
    
    train_model(model, train_images, epochs=epochs, seed=seed, **train_dict)