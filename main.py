from tabulate import tabulate
from cell_extraction import *
from cnn.digit_classifier_cnn import DigitModel
from cnn.sudoku_solver_cnn import SudokuSolver
import argparse


def main():
    parser = argparse.ArgumentParser(description="CNN OCR Sudoku Solver")
    parser.add_argument("-i", "--image", help="Path to sudoku image", metavar="IMAGE_PATH", required=True)
    args = parser.parse_args()

    # find sudoku cells
    print("Processing Image")
    extractor = CellExtractor(args.image)
    cells = np.array(extractor.extract())

    # load digit classifier CNN
    model = DigitModel()
    model.load_model("cnn/trained_models/digit_classifier.h5")

    # classify cells that hold digits
    mask = [True if d is not 0 else False for d in cells]
    digits = np.array(list(cells[mask])) / 255.
    pred = model.predict(np.expand_dims(digits, axis=3).astype("float"))
    cells[mask] = np.argmax(pred, axis=1)
    cells = cells.reshape((9, 9))

    print("Found Sudoku:")
    print(tabulate(cells, tablefmt="fancy_grid"))

    # load solver CNN
    solver = SudokuSolver()
    solver.load_model("cnn/trained_models/sudoku_solver.h5")

    # normalize input and solve sudoku
    cells = cells / 9 - .5
    solved = solver.solve(np.expand_dims(cells, axis=(0, 3)))

    print("\nSolved:")
    print(tabulate(solved, tablefmt="fancy_grid"))


if __name__ == '__main__':
    main()

