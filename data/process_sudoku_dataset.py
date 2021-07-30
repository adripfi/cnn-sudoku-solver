import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def pre_process_data(data_path="sudoku.csv"):
    """
    Process and save sudoku dataset for training by converting strings into (9, 9) arrays and normalizing them
    """
    sudoku_csv = pd.read_csv(data_path)
    data = sudoku_csv["quizzes"]
    labels = sudoku_csv["solutions"]

    # convert strings to int and reshape into 9x9 grids
    data_proc = []
    labels_proc = []
    for i in tqdm(range(len(data))):
        data_proc.append(np.array([int(d) for d in data[i]]).reshape((9, 9, 1)))
        labels_proc.append(np.array([int(d) for d in labels[i]]).reshape((81,1)))

    # zero mean
    data_proc = np.array(data_proc) / 9 - .5
    labels_proc = np.array(labels_proc) - 1

    train_x, test_x, train_y, test_y = train_test_split(data_proc, labels_proc,
                                                        test_size=0.1, train_size=0.9, shuffle=True)

    # save to disk
    with open("data/train_test_data_solver.npy", "wb") as f:
        np.save(f, train_x)
        np.save(f, test_x)
        np.save(f, train_y)
        np.save(f, test_y)

    return train_x, test_x, train_y, test_y


def load_data(data_path="data/train_test_data_solver.npy"):
    """
    Load processed sudoku dataset
    """
    with open(data_path, "rb") as f:
        train_x = np.load(f)
        test_x = np.load(f)
        train_y = np.load(f)
        test_y = np.load(f)

    return train_x, test_x, train_y, test_y


if __name__ == '__main__':
    pre_process_data()



