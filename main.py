import aiforge
aiforge.set_backend("tensorflow")
import argparse
import train
import evaluate
import os

from tensorflow import keras
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from aiforge.models import create_feed_forward_model

def load_config(overrides=None):
    if overrides is None:
        overrides = []
    with initialize_config_dir(config_dir=os.path.abspath("config"), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    return cfg

def train_model(overrides):
    """Train a model"""
    cfg = load_config(overrides)
    print("Training configuration:\n", OmegaConf.to_yaml(cfg))
    
    if cfg.dataset.name == "fashion_mnist":
        print("Using Fashion MNIST dataset")
        fashion_mnist = keras.datasets.fashion_mnist

        train_data, _ = fashion_mnist.load_data()
        
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    model = create_feed_forward_model(
        input_shape=train_data[0].shape[1:],
        num_classes=len(class_names),
        layers=cfg.model.layers,
        activations=cfg.model.activations,
        output_activation=cfg.model.output_activation
    )
    
    train.run(model, train_data, epochs=cfg.train.epochs, seed=cfg.seed, model_path=cfg.model.model_path) # Calls the actual training function

def evaluate_model(overrides):
    """Evaluate a trained model"""
    
    cfg = load_config(overrides)
    print("Evaluation configuration:\n", OmegaConf.to_yaml(cfg))
    
    if cfg.dataset.name == "fashion_mnist":
        print("Using Fashion MNIST dataset")
        fashion_mnist = keras.datasets.fashion_mnist

        train_data, test_data = fashion_mnist.load_data()
        
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    
    model = create_feed_forward_model(
        input_shape=train_data[0].shape[1:],
        num_classes=len(class_names),
        layers=cfg.model.layers,
        activations=cfg.model.activations,
        output_activation=cfg.model.output_activation
    )
    
    evaluate.run(model, cfg.model.model_path, test_data, class_names=class_names) # Calls evaluation function
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate the model.")
    parser.add_argument("mode", choices=["train", "evaluate"], help="Choose to train or evaluate the model")
    args, unknown = parser.parse_known_args()

    if args.mode == "train":
        train_model(unknown)
    elif args.mode == "evaluate":
        evaluate_model(unknown)