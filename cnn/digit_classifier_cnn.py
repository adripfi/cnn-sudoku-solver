import keras
from keras.layers import Activation
from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D
from sklearn.metrics import classification_report
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam

# Fix for internal CUDA error
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

class DigitModel:
    def __init__(self):
        self.model = self.get_model()

    @staticmethod
    def get_model():
        """
        Generate CNN
        """
        model = keras.models.Sequential()
        model.add(Conv2D(32, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        model.add(Dense(10))
        model.add(Activation("softmax"))

        return model

    def fit(self, **kwargs):
        """
        Wrapper for tf.keras.Model.fit() method
        """
        self.model.fit(**kwargs)

    def compile(self, **kwargs):
        """
        Wrapper for tf.keras.Model.compile() method
        """
        self.model.compile(**kwargs)

    def save_model(self, path):
        """
        Wrapper for tf.keras.Model.save() method
        """
        self.model.save(path)

    def load_model(self, path):
        """
        Wrapper for tf.keras.Model.load_model() method
        """
        del self.model
        self.model = keras.models.load_model(path)

    def predict(self, x):
        """
        Wrapper for tf.keras.Model.predict() method
        """
        return self.model.predict(x)


def main():
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    # scale images
    train_x = train_x.astype("float32") / 255.
    test_x = test_x.astype("float32") / 255.
    # reshape images
    train_x = train_x.reshape((-1, 28, 28, 1))
    test_x = test_x.reshape((-1, 28, 28, 1))

    # train model and save model
    m = DigitModel()
    m.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=["accuracy"])
    m.model.fit(train_x, train_y, batch_size=128, epochs=15, validation_split=10)
    m.save_model("digit_classifier.h5")

    # test model
    pred = m.predict(test_x)
    print(classification_report(test_y.argmax(axis=1), pred.argmax(axis=1)))


if __name__ == '__main__':
    main()