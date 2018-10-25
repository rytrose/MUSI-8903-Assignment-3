In this assignment, you will use PyTorch to build a recurrent neural network (RNN) and a convolutional recurrent neural network (CRNN) to rate musical performances. The overall task is to predict the scores given by human expert judges along 4 evaluation criteria: 

`0`: Musicality, `1`: Note Accuracy, `2`: Rhythmic Accuracy, `3`: Tone Quality

We will use pitch contours as our mid-level data representation to feed into a deep neural network which will be trained to perform a regression task to model the **Note Accuracy** criterion. 

Inside this folder, you will find the following:
* `./dat/`: This directory contains the raw dataset containing the pre-computed pitch contours and the ground truth ratings. We have a separate test set which we will use to test your models.
* `./dataLoaders/`: This directory contains the dataset and dataloader classes.
* `./models`: This directory contains your model class file titled `PCAssessNet.py`. You will implement your RNN and CRNN architectures here.
* `./saved`: This directory will contain your trained and saved models.
* `./runs/`: This directory will contain .txt files logging your training procedure.
* `script_train.py`: This script contains the overall training script will arguments for model type, hyperparameters and other initializations. 
* `train_utils.py`: This file contains some utility methods for training your models. You will implement some methods in this file.
* `eval_utils.py`: This file contains some utility methods to evaluate your models. You will implement some methods in this file.
* `readme.md`: This readme file which you are now reading.

Like assignment 2, we have provided the scaffolding for this assignment as well and you will need to write code within the spaces allocated. Do **NOT** edit any other part of the code in any way. The data pipeline is also taken care of. 

You will be training 2 models for this task. The first model will be a deep RNN and the second model will be a deep CRNN. For this assignment, you are free to choose your network architectures (number of layers, type of RNN cell, regularization etc.)

You will have to install the `dill` package for this assginment. Activate the conda environment you are using for this class (if any) and then run the ```pip install dill``` command.

### PART 1: Implement PitchRnn (30 points)
Implement all methods of the `PitchRnn class` in `PCAssessNet.py`. You will have to implement the class initialization method, the forward pass method and the hidden state initialization method. 

### PART 2: Implement Training & Evaluation Util Methods (30 points)
Here you will be implmenting various methods in the training and evaluation utility files:
* Complete the `train()` method in `train_utils.py`. (10 points)
* Complete the `adjust_learning_rate()`  methhod in `train_utils.py`. This takes the epoch number into consideration and adjusts the learning rate of your optimizer. You are free to experiment with and implement your own algorithm for this. (5 points)
* Complete the `eval_model()` method in `eval_utils.py`. (5 points)
* Complete the `eval_regression()` method in `eval_utils.py` (10 points).

### PART 3: Implement PitchCRnn (30 points)
Implement all methods of the `PitchCRnn` class in `PCAssessNet.py`. You will have to implement the class initialization method, the forward pass method and the hidden state initialization method. Note that the minimum sequence length that we consider here is `2000` and hence the convolutional layers of your CRNN should be designed so that they at least output `1` value for an input length of `2000`.  

### PART 4: Best Models (10 points)
Experiment with hyperparameter tuning and tweaking the model architectures to create your best `PitchRnn` and `PitchCRnn` model. These 10 points will be awarded based on your group's overall performance in the class. For e.g., the group with the best `PitchRnn` model will get 5 points, the next group will get 4.5 points and so on. The same procedure will be followed for `PitchCRnn` model. The evaluation will be based on the held-out test set that we have not provided you. 

### Submission Format
Submit all the following in a zip file named `Assign3_Group#` where # is your group number: 
* `./dataLoaders/`: Containing the dataset and dataloader python files
* `./models`: Directory containing your implemented `PCAssessNet.py` file.
* `./saved`: Directory containing your best `PitchRnn` and `PitchCRnn` models. Only include the 2 best saved models.
* `./runs`: Directory containing the training logs of your best `PitchRnn` and `PitchCRnn` models. Only include the 2 best train log files. 
* `script_train.py`
* `train_utils.py`
* `eval_utils.py`

Do **NOT** include the `./dat` folder 
