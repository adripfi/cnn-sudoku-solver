import keras
import numpy as np
from keras.layers import Activation
from keras.layers import Conv2D, BatchNormalization, Dense, Flatten, Reshape
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from data.process_sudoku_dataset import load_data

# Fix for internal CUDA error
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


class SudokuSolver:
    def __init__(self):
        self.model = self.get_model()

    @staticmethod
    def get_model():
        """
        Generate CNN architecture
        """
        model = keras.models.Sequential()
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(9, 9, 1)))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, kernel_size=(1, 1), activation='relu', padding='same'))
        model.add(Flatten())
        model.add(Dense(729))
        model.add(Reshape((-1, 9)))
        model.add(Activation('softmax'))

        return model

    def solve(self, puzzle):
        """
        Solve sudoku one cell at a time by only considering the prediction with the highest probability
        """
        while True:
            # forward pass
            output = self.model.predict(puzzle.reshape((1, 9, 9, 1))).squeeze()
            # get prediction for all cells as well as their probability
            prediction = np.argmax(output, axis=1).reshape((9, 9)) + 1
            probability = np.around(np.max(output, axis=1).reshape((9, 9)), 2)

            # denormalize input vector and reshape back to grid representation
            puzzle = (puzzle + 0.5) * 9
            puzzle = puzzle.reshape((9, 9))

            # check if sudoku is solved i.e. if there are any zeros left
            mask = (puzzle == 0)
            if not mask.sum():
                break

            # find cell with highest probability
            probability_new = probability * mask
            idx = np.argmax(probability_new)
            x, y = (idx // 9), (idx % 9)
            # update cell
            puzzle[x][y] = prediction[x][y]

            # normalize input again
            puzzle = puzzle / 9 - 0.5

        return prediction

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
        self.model = keras.models.load_model(path)


def main():
    # get dataset
    train_x, test_x, train_y, test_y = load_data()

    # train and save model
    m = SudokuSolver()
    m.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    m.model.fit(train_x, train_y, batch_size=64, epochs=2)
    m.save_model("cnn/trained_models/solver.h5")


if __name__ == '__main__':
    main()

