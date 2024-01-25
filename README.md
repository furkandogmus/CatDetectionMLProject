# Cat or Not Cat ML Project

This repository contains a simple machine learning project to classify images as either cat or non-cat. The project is implemented using a two-layer neural network.

## Dataset

The dataset consists of two files:

- `train_catvnoncat.h5`: Contains training set images and labels.
- `test_catvnoncat.h5`: Contains test set images and labels.

You can load the dataset using the provided `load_data` function. To load the dataset, run the following code:

```python
train_x_orig, train_y, test_x_orig, test_y, classes = load_data(r'C:\Users\Furkan\Desktop\Assignment-5\dataset\dataset')
# Example of a picture
index = 11
plt.imshow(train_x_orig[index])
print("y = " + str(train_y[0, index]) + ". It's a " + classes[train_y[0, index]].decode("utf-8") + " picture.")

# Explore your dataset
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print("Number of training examples: " + str(m_train))
print("Number of testing examples: " + str(m_test))
print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("train_x_orig shape: " + str(train_x_orig.shape))
print("train_y shape: " + str(train_y.shape))
print("test_x_orig shape: " + str(test_x_orig.shape))
print("test_y shape: " + str(test_y.shape))
```

## Data Preprocessing
The training and test examples are reshaped and standardized to have feature values between 0 and 1:
```python
# Reshape the training and test examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1
train_x = train_x_flatten / 255.
test_x = test_x_flatten / 255.

print("train_x's shape: " + str(train_x.shape))
print("test_x's shape: " + str(test_x.shape))
```

## Model Architecture

The neural network has two layers: 
one hidden layer with ReLU activation,
an output layer with sigmoid activation. 
The model's parameters are initialized using the **`initialize_parameters_deep`** function.
```python
n_x = 12288  # num_px * num_px * 3
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)
```

## Training the Model

The model is trained using the **`two_layer_model`** function, which implements gradient descent. 
The training process and cost evolution are visualized using the **`plot_loss`** function.
```python
parameters, costs = two_layer_model(train_x, train_y, layers_dims, learning_rate=0.0075, num_iterations=3000)
```

## Evaluation
The model's accuracy is evaluated on the test set using the predict function:
```python
accuracy = predict(test_x, test_y, parameters)
print(f"\nAccuracy is {accuracy}")
```

## Results

The training process will display the cost at regular intervals.<br>
After 3000 iterations, the model achieves a certain accuracy on the test set.<br>
Feel free to experiment with hyperparameters and dataset to further improve the model's performance.

# License
This project is licensed under the [MIT License](LICENSE). Feel free to use and modify the code as needed.
