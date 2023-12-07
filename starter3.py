import sys
import random
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from torch.optim.lr_scheduler import StepLR 
from sklearn.metrics import confusion_matrix, f1_score

def read_mnist(file_name):
    
    data_set = []
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i+1])
            data_set.append([label,attribs])
    return(data_set)

def show_mnist(file_name,mode):
    
    data_set = read_mnist(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ',end='')
                else:
                    print('*',end='')
            else:
                print('%4s ' % data_set[obs][1][idx],end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0],end='')
        print(' ')

def read_insurability(file_name):
    
    count = 0
    data = []
    with open(file_name,'rt') as f:
        for line in f:
            if count > 0:
                line = line.replace('\n','')
                tokens = line.split(',')
                if len(line) > 10:
                    x1 = float(tokens[0])
                    x2 = float(tokens[1])
                    x3 = float(tokens[2])
                    if tokens[3] == 'Good':
                        cls = 0
                    elif tokens[3] == 'Neutral':
                        cls = 1
                    else:
                        cls = 2
                    data.append([[cls],[x1,x2,x3]])
            count = count + 1
    return(data)
    
class FeedForward(nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(784, 1024)  
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.fc6 = nn.Linear(64, 10) 

    def forward(self, x):
        x = F.elu(self.bn1(self.fc1(x)))
        x = F.elu(self.bn2(self.fc2(x)))
        x = F.elu(self.bn3(self.fc3(x)))
        x = F.elu(self.bn4(self.fc4(x)))
        x = F.elu(self.bn5(self.fc5(x)))
        x = self.fc6(x)
        return x

class FeedForwardReg(nn.Module):
    def __init__(self):
        super(FeedForwardReg, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.dropout1 = nn.Dropout(0.2) 
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.2)  
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

def normalize_data(data):
    if len(data.shape) == 1:
        norm = np.linalg.norm(data, keepdims=True)
    else:
        norm = np.linalg.norm(data, axis=1, keepdims=True)
    return data / norm

def remove_unnamed_columns(df):
    unnamed_cols = df.columns.str.contains('Unnamed')
    df = df.loc[:, ~unnamed_cols]
    return df

def prepare_data_loaders():
    train = pd.read_csv('mnist_train.csv')
    valid = pd.read_csv('mnist_valid.csv')
    test = pd.read_csv('mnist_test.csv')

    train = remove_unnamed_columns(train)
    valid = remove_unnamed_columns(valid)
    test = remove_unnamed_columns(test)

    X_train, y_train = train.iloc[:, 1:].values / 255, train.iloc[:, 0].values
    X_valid, y_valid = valid.iloc[:, 1:].values / 255, valid.iloc[:, 0].values
    X_test, y_test = test.iloc[:, 1:].values / 255, test.iloc[:, 0].values

    X_train = normalize_data(X_train)
    X_valid = normalize_data(X_valid)
    X_test = normalize_data(X_test)

    X_train, y_train = torch.Tensor(X_train), torch.LongTensor(y_train)
    X_valid, y_valid = torch.Tensor(X_valid), torch.LongTensor(y_valid)
    X_test, y_test = torch.Tensor(X_test), torch.LongTensor(y_test)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    valid_loader = DataLoader(TensorDataset(X_valid, y_valid), batch_size=64, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

    return train_loader, valid_loader, test_loader

def train_network(model, train_loader, valid_loader, optimizer, criterion, scheduler, patience=5, epochs=50):
    best_loss = float('inf')
    best_model = None
    epochs_no_improve = 0

    # Lists to store metrics
    valid_losses = []
    valid_accuracies = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        # Training loop: iterate over the training data loader
        for inputs, labels in train_loader:
            # Reset the gradient to zero (PyTorch accumulates gradients)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward() # Perform backpropagation to compute gradients
            optimizer.step()
            train_loss += loss.item()

        # Validation loop: evaluate the model on the validation dataset
        valid_loss = 0.0
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in valid_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        valid_loss /= len(valid_loader)
        valid_accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}, Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.2f}%')
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)
     
        # Early Stopping: save the model if it performs better than previous epochs
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f'Early stopping triggered after epoch {epoch + 1}')
                model.load_state_dict(best_model)
                break
        
        # Update learning rate based on validation loss
        scheduler.step(valid_loss)

    epochs = range(1, len(valid_losses) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot validation loss on the left y-axis
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Validation Loss', color='tab:red')
    ax1.plot(epochs, valid_losses, label='Validation Loss', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Instantiate a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Validation Accuracy (%)', color='tab:blue')  
    ax2.plot(epochs, valid_accuracies, label='Validation Accuracy', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # Title and legend
    plt.title('Validation Loss and Accuracy over Epochs')
    fig.tight_layout()
    plt.show()

def evaluate_on_test(model, test_loader):
    correct = 0
    total = 0
    
    predictions = []
    true_labels = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            # Get the index of the highest probability class. This index corresponds to the predicted class.
            _, predicted = torch.max(outputs.data, 1)
            predictions.append(predicted)
            # Increment the total count by the number of labels in this batch
            total += labels.size(0)
            true_labels.append(labels)
            # Increment the correct count by the number of correctly predicted labels
            correct += (predicted == labels).sum().item()
    
    predictions = torch.cat(predictions).numpy()
    true_labels = torch.cat(true_labels).numpy()
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    test_accuracy = 100 * correct / total
    print(f'Accuracy on test set: {test_accuracy:.2f}%')

    
def classify_mnist():
    train_loader, valid_loader, test_loader = prepare_data_loaders()
    model = FeedForward()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3)
    print('---------without regularizer------------')
    train_network(model, train_loader, valid_loader, optimizer, criterion, scheduler)
    evaluate_on_test(model, test_loader)
    
def classify_mnist_reg():
    train_loader, valid_loader, test_loader = prepare_data_loaders()
    model = FeedForwardReg()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-8)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3)

    print('---------with regularizer------------')
    train_network(model, train_loader, valid_loader, optimizer, criterion, scheduler)
    evaluate_on_test(model, test_loader)



def check_class_balance(file_name):
    data = read_insurability(file_name)
    class_counts = [0, 0, 0]

    for item in data:
        cls = item[0][0]
        class_counts[cls] += 1

    return class_counts


# check the balance of the data set
file_name = 'three_train.csv'
class_counts = check_class_balance(file_name)


# print("Class counts:", class_counts)


class FFNNClassifier:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001, use_bias=True, lr_decay=None,
                 early_stopping=False, early_stopping_rounds=10):
        self.model = FFNNClassifier.FFNN(input_size, hidden_size, output_size, use_bias)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        self.lr_decay = lr_decay
        self.early_stopping = early_stopping
        self.learning_rate = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.best_valid_loss = float('inf')
        self.epochs_no_improve = 0
        self.train_losses = []
        self.valid_losses = []

    class FFNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, use_bias=True):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size, bias=use_bias)
            self.fc2 = nn.Linear(hidden_size, output_size, bias=use_bias)

        def forward(self, x):
            x = torch.sigmoid(self.fc1(x))
            x = self.softmax(self.fc2(x))
            return x

        @staticmethod
        def softmax(x):
            exp_x = torch.exp(x - torch.max(x))
            return exp_x / torch.sum(exp_x)

    @staticmethod
    def to_one_hot(labels, num_classes=3):
        one_hot_labels = torch.zeros((len(labels), num_classes))
        for i, label in enumerate(labels):
            one_hot_labels[i, label] = 1
        return one_hot_labels

    def adjust_learning_rate(self, epoch):
        if self.lr_decay:
            new_lr = self.learning_rate * (0.9 ** epoch)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

    def should_stop_early(self, valid_data):
        total_valid_loss = 0
        self.model.eval()
        with torch.no_grad():
            for labels, inputs in valid_data:
                inputs = torch.tensor(inputs, dtype=torch.float32)
                labels = self.to_one_hot(labels, 3)

                outputs = self.model(inputs)
                loss = -torch.log(outputs) * labels
                loss = torch.sum(loss, dim=1).mean()
                total_valid_loss += loss.item()

        avg_valid_loss = total_valid_loss / len(valid_data)

        if avg_valid_loss < self.best_valid_loss:
            self.best_valid_loss = avg_valid_loss
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1

        if self.epochs_no_improve >= self.early_stopping_rounds:
            return True
        else:
            return False

    def train(self, train_data, valid_data, epochs=100):
        self.train_losses, self.valid_losses = [], []
        # apply adjust_learning_rate at the end of each epoch
        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0
            for labels, inputs in train_data:
                inputs = torch.tensor(inputs, dtype=torch.float32)
                labels = self.to_one_hot(labels, 3)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = -torch.log(outputs) * labels
                loss = torch.sum(loss, dim=1).mean()

                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()

            self.train_losses.append(total_train_loss / len(train_data))

            # Validation phase
            self.model.eval()
            total_valid_loss = 0
            with torch.no_grad():
                for labels, inputs in valid_data:
                    inputs = torch.tensor(inputs, dtype=torch.float32)
                    labels = self.to_one_hot(labels, 3)

                    outputs = self.model(inputs)
                    loss = -torch.log(outputs) * labels
                    loss = torch.sum(loss, dim=1).mean()
                    total_valid_loss += loss.item()

            self.valid_losses.append(total_valid_loss / len(valid_data))

            self.adjust_learning_rate(epoch)

            if self.early_stopping and self.should_stop_early(valid_data):
                print("Early stopping triggered after", epoch, "epochs.")
                break

    def evaluate(self, data_loader):
        self.model.eval()
        true_labels, predicted_labels = [], []

        with torch.no_grad():
            for labels, inputs in data_loader:
                inputs = torch.tensor(inputs, dtype=torch.float32).view(1, -1)
                labels = torch.tensor(labels, dtype=torch.long)

                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                true_labels.extend(labels.tolist())
                predicted_labels.extend(predicted.tolist())

        return true_labels, predicted_labels


def classify_insurability():
    train = read_insurability('three_train.csv')
    valid = read_insurability('three_valid.csv')
    test = read_insurability('three_test.csv')

    learning_rates = [0.01, 0.001, 0.0001]
    use_bias_options = [True, False]
    early_stopping_rounds_options = [5, 10, 15]

    best_model = None
    best_f1_score = 0
    best_params = {}

    for lr in learning_rates:
        for use_bias in use_bias_options:
            for early_stopping_rounds in early_stopping_rounds_options:
                print(f"Training with lr={lr}, use_bias={use_bias}, early_stopping_rounds={early_stopping_rounds}")

                classifier = FFNNClassifier(input_size=3, hidden_size=2, output_size=3, learning_rate=lr,
                                            use_bias=use_bias, early_stopping=True,
                                            early_stopping_rounds=early_stopping_rounds)
                classifier.train(train, valid, epochs=100)

                true_labels, predicted_labels = classifier.evaluate(valid)
                f1 = f1_score(true_labels, predicted_labels, average='weighted')

                if f1 > best_f1_score:
                    best_f1_score = f1
                    best_model = classifier
                    best_params = {'learning_rate': lr, 'use_bias': use_bias,
                                   'early_stopping_rounds': early_stopping_rounds}

    print("Best F1 Score:", best_f1_score)
    print("Best Hyperparameters:", best_params)

    true_labels_test, predicted_labels_test = best_model.evaluate(test)
    cm_test = confusion_matrix(true_labels_test, predicted_labels_test)
    f1_test = f1_score(true_labels_test, predicted_labels_test, average='weighted')

    print("Test Confusion Matrix:\n", cm_test)
    print("Test F1 Score:", f1_test)
    return best_model


def plot_learning_curves(train_losses, valid_losses):
    plt.plot(train_losses, label='Training loss')
    plt.plot(valid_losses, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

class ManualNNClassifier:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, class_counts):
        seed_value = 42
        torch.manual_seed(seed_value)
        np.random.seed(seed_value)
        random.seed(seed_value)
        self.weights1 = torch.randn(input_size, hidden_size, requires_grad=True)
        self.weights2 = torch.randn(hidden_size, output_size, requires_grad=True)
        self.initialize_weights_xavier()
        # Learning rate and class counts
        self.learning_rate = learning_rate
        self.class_counts = class_counts


        self.m1 = torch.zeros_like(self.weights1)
        self.v1 = torch.zeros_like(self.weights1)
        self.m2 = torch.zeros_like(self.weights2)
        self.v2 = torch.zeros_like(self.weights2)

    def initialize_weights_xavier(self):
        for param in [self.weights1, self.weights2]:
            if param.requires_grad:
                torch.nn.init.xavier_uniform_(param.data)

    @staticmethod
    #def sigmoid(x):
    #    return 1 / (1 + torch.exp(-x))

    @staticmethod
    def softmax(x):
        exp_x = torch.exp(x - torch.max(x, dim=1, keepdim=True).values)
        return exp_x / torch.sum(exp_x, dim=1, keepdim=True)

    def forward(self, x):

        x = x.view(1,-1)  # Reshapes x to have shape [ 1, number of features]

        # Perform matrix multiplication
        x = torch.mm(x,self.weights1)
        x = torch.tanh(x)

        x = torch.mm(x,self.weights2)

        return x

    @staticmethod
    def to_one_hot(labels, num_classes=3):
        one_hot_labels = torch.zeros((len(labels), num_classes))
        for i, label in enumerate(labels):
            one_hot_labels[i, label] = 1
        return one_hot_labels

    @staticmethod
    def calculate_class_weights(class_counts):
        total_samples = sum(class_counts)
        class_weights = [total_samples / c for c in class_counts]
        return torch.tensor(class_weights, dtype=torch.float32)

    def calculate_gradients(self):
        d_weights1 = self.weights1.grad.clone() if self.weights1.grad is not None else None
        d_weights2 = self.weights2.grad.clone() if self.weights2.grad is not None else None
        return d_weights1, d_weights2

    def update_weights_with_adam(self, lr, beta1, beta2, epsilon, t):
        self.weights1, self.m1, self.v1 = self._update_weight_with_adam(self.weights1, self.m1, self.v1, lr, beta1,
                                                                        beta2, epsilon, t)
        self.weights2, self.m2, self.v2 = self._update_weight_with_adam(self.weights2, self.m2, self.v2, lr, beta1,
                                                                        beta2, epsilon, t)

    @staticmethod
    def _update_weight_with_adam(weight, m, v, lr, beta1, beta2, epsilon, t):
        gradient = weight.grad
        if gradient is None:
            return weight, m, v

        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * gradient ** 2
        m_corrected = m / (1 - beta1 ** t)
        v_corrected = v / (1 - beta2 ** t)
        weight.data = weight.data - lr * m_corrected / (v_corrected ** 0.5 + epsilon)
        return weight, m, v


    def train(self, train_features, train_labels, epochs):
        class_weights = self.calculate_class_weights(self.class_counts)
        lr = self.learning_rate
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        t = 0

        for epoch in range(epochs):
            for i in range(len(train_features)):
                inputs = torch.tensor([train_features[i]], dtype=torch.float32)
                labels = torch.tensor([train_labels[i]], dtype=torch.long)
                logits = self.forward(inputs)
                loss = F.cross_entropy(logits, labels)


                self.weights1.grad, self.weights2.grad = None, None
                loss.backward()
                d_weights1, d_weights2 = self.calculate_gradients()

                t += 1
                self.update_weights_with_adam(lr, beta1, beta2, epsilon, t)

    def evaluate(self, data_loader):
        true_labels, predicted_labels = [], []

        for labels, inputs in data_loader:
            inputs = torch.tensor(inputs, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)

            outputs = self.forward(inputs)
            _, predicted = torch.max(outputs, 1)
            true_labels.extend(labels.tolist())
            predicted_labels.extend(predicted.tolist())

        return true_labels, predicted_labels

class NNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NNClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        return self.fc2(x)

def train_pytorch_model(model, train_features, train_labels, learning_rate, epochs):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        for i in range(len(train_features)):
            inputs = torch.tensor([train_features[i]], dtype=torch.float32)
            labels = torch.tensor([train_labels[i]], dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()



def classify_insurability_manual():
    def process_data(data):
        labels = [item[0][0] for item in data]
        features = [item[1] for item in data]
        return features, labels

    # Read data from files
    train_data = read_insurability('three_train.csv')
    valid_data = read_insurability('three_valid.csv')
    test_data = read_insurability('three_test.csv')

    train_features, train_labels = process_data(train_data)
    valid_features, valid_labels = process_data(valid_data)
    test_features, test_labels = process_data(test_data)

    # Define learning rate and class counts
    # Initialize and train the PyTorch model
    pytorch_classifier = NNClassifier(3, 64, 3)
    train_pytorch_model(pytorch_classifier, train_features, train_labels, 0.001, 1)

    # CHECK: check your gradient calculations by comparing before/after update weight matrices against a parallel implementation using a PyTorch optimizer.
    manual_classifier = ManualNNClassifier(3, 64, 3, 0.001, [551, 911, 538])
    manual_classifier.train(train_features, train_labels, 1)

    # Compare weights
    pytorch_weights1_transposed = pytorch_classifier.fc1.weight.data.T

    pytorch_weights2_transposed = pytorch_classifier.fc2.weight.data.T
    manual_weights1 = manual_classifier.weights1.data
    manual_weights2 = manual_classifier.weights2.data

    #print("Difference in First Layer Weights:", torch.norm(pytorch_weights1_transposed - manual_weights1))
    #print("Difference in Second Layer Weights:", torch.norm(pytorch_weights2_transposed - manual_weights2))

    classifier = ManualNNClassifier(input_size=3, hidden_size=64, output_size=3, learning_rate=0.001,
                                    class_counts=[551, 911, 538])
    classifier.train(train_features, train_labels, epochs=100)


    # Evaluate the model on the validation dataset
    true_labels_valid, predicted_labels_valid = classifier.evaluate(valid_data)
    cm_valid = confusion_matrix(true_labels_valid, predicted_labels_valid)
    f1_valid = f1_score(true_labels_valid, predicted_labels_valid, average='weighted')

    # Evaluate the model on the test dataset
    true_labels_test, predicted_labels_test = classifier.evaluate(test_data)
    cm_test = confusion_matrix(true_labels_test, predicted_labels_test)
    f1_test = f1_score(true_labels_test, predicted_labels_test, average='weighted')

    # Print evaluation results
    print("Validation Confusion Matrix:\n", cm_valid)
    print("Validation F1 Score:", f1_valid)
    print("Test Confusion Matrix:\n", cm_test)
    print("Test F1 Score:", f1_test)

def main():
    classify_insurability()
    classify_mnist()
    classify_mnist_reg()
    classify_insurability_manual()

if __name__ == "__main__":
    main()