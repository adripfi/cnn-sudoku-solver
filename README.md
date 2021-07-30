# Overview 
The aim of this project was to develop an 9x9 soduko solver using traditional CV method as well as two CNNs, one for digit detection and one for solving the soduko. 

# Approach

## Grid detection
To detect the soduko grid OpenCV is used to find the biggest contour within the given image. Afterwards the warped grid is resized to a rectungular representation. Now individual cells can be extracted by simply deviding the rectungular grid into 9x9 cells. Each cell is then further proccesed to remove any artefacts and improve image quality for classification. 

## Digit recognition
After pre processing each cell is classified using a simple CNN that was trained on the MNIST dataset where it reached 99% accuracy. Additionaly if a cell has less than 3% nonzero pixels it is considered empty and will not be passed to the network. 


## Solving the Soduku
Of course there are very straight forward methods to solve sodukus e.g. backtracking algorithms or recursive approaches. But here an [approach](https://cs230.stanford.edu/files_winter_2018/projects/6939771.pdf) using an additional CNN shall be explored. The network is trained on a dataset of 0.96 million soduku puzzles. The loss is measured using crossentropy and training is stopped after two epochs with a batch size of 64. This resulted in the following metrics after training:
|   Loss           |  MSE    |        Accuracy | 
| -------------| ------------- | ------------- |
| 0.34 |          21.85         |      0.83      |

With only 83% accuracy, solving sudoku puzzles directly is not very reliable. To overcome this issue each sudoku is solved one cell at a time. In each iteration the predicted cell with the highest probability is accepted. This approach is also hinted at in the paper mentioned above. The method is tested on 40,000 randomly selected puzzles where it reached 99% accuracy.


# Example 
```
$ python main.py -i puzzle.jpg

Found Sudoku:
╒═══╤═══╤═══╤═══╤═══╤═══╤═══╤═══╤═══╕
│ 5 │ 3 │ 0 │ 0 │ 7 │ 0 │ 0 │ 0 │ 0 │
├───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ 6 │ 0 │ 0 │ 1 │ 9 │ 5 │ 0 │ 0 │ 0 │
├───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ 0 │ 9 │ 8 │ 0 │ 0 │ 0 │ 0 │ 6 │ 0 │
├───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ 8 │ 0 │ 0 │ 0 │ 6 │ 0 │ 0 │ 0 │ 3 │
├───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ 4 │ 0 │ 0 │ 8 │ 0 │ 3 │ 0 │ 0 │ 1 │
├───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ 7 │ 0 │ 0 │ 0 │ 2 │ 0 │ 0 │ 0 │ 6 │
├───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ 0 │ 6 │ 0 │ 0 │ 0 │ 0 │ 2 │ 8 │ 0 │
├───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ 0 │ 0 │ 0 │ 4 │ 1 │ 9 │ 0 │ 0 │ 5 │
├───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ 0 │ 0 │ 0 │ 0 │ 8 │ 0 │ 0 │ 7 │ 9 │
╘═══╧═══╧═══╧═══╧═══╧═══╧═══╧═══╧═══╛
Solved:
╒═══╤═══╤═══╤═══╤═══╤═══╤═══╤═══╤═══╕
│ 5 │ 3 │ 4 │ 6 │ 7 │ 8 │ 9 │ 1 │ 2 │
├───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ 6 │ 7 │ 2 │ 1 │ 9 │ 5 │ 3 │ 4 │ 8 │
├───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ 1 │ 9 │ 8 │ 3 │ 4 │ 2 │ 5 │ 6 │ 7 │
├───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ 8 │ 5 │ 9 │ 7 │ 6 │ 1 │ 4 │ 2 │ 3 │
├───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ 4 │ 2 │ 6 │ 8 │ 5 │ 3 │ 7 │ 9 │ 1 │
├───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ 7 │ 1 │ 3 │ 9 │ 2 │ 4 │ 8 │ 5 │ 6 │
├───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ 9 │ 6 │ 1 │ 5 │ 3 │ 7 │ 2 │ 8 │ 4 │
├───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ 2 │ 8 │ 7 │ 4 │ 1 │ 9 │ 6 │ 3 │ 5 │
├───┼───┼───┼───┼───┼───┼───┼───┼───┤
│ 3 │ 4 │ 5 │ 2 │ 8 │ 6 │ 1 │ 7 │ 9 │
╘═══╧═══╧═══╧═══╧═══╧═══╧═══╧═══╧═══╛
```



# References
[MNIST Dataset](https://keras.io/api/datasets/mnist/)<br/>
[Sudoku Dataset on Kaggle](https://www.kaggle.com/bryanpark/sudoku)<br/>
[Akin-David, Mantey. Solving Soduku with Nerual Networks. 2018](https://cs230.stanford.edu/files_winter_2018/projects/6939771.pdf)
