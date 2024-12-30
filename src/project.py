import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
from collections import Counter
import numpy as np
from keras.datasets import fashion_mnist, cifar100
from network.autoencoder_network import Autoencoder
from model.train_model import train
from visualization.utils import plot_metrics


N_CLASSES = 5
INPUT_SHAPE = (32, 32, 3)
EPOCHS = 10
BATCH_SIZE = 128

def main():
    (X_train, y_train), (X_test, y_test) = cifar100.load_data()
    X_train = X_train // 255
    #X_train = np.expand_dims(X_train, axis=-1)
    X_test = X_test// 255
    #X_test = np.expand_dims(X_test, axis=-1)
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"Training data class distributions: {Counter(y_train[:, 0])}")

    net = Autoencoder(N_CLASSES, INPUT_SHAPE)
    model = net.build()
    print(model.summary())
    H = train(model, X_train, X_train, X_test, X_test, EPOCHS, BATCH_SIZE)
    plot_metrics(H)

if __name__ == '__main__':
    main()