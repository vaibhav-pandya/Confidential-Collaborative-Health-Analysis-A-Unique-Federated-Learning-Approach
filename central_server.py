# First, we need some functions to download the dataset, to read the dataset text file, and to parse the lines and fields of the dataset:
import urllib.request

def download_url(url, save_as):
    response = urllib.request.urlopen(url)
    data = response.read()
    file = open(save_as, 'wb')
    file.write(data)
    file.close()
    response.close()
    
def read_binary_file(file):
    f = open(file,'rb')
    block = f.read()
    return block.decode('utf-16')

def split_text_in_lines(text):
    return text.split('\r\n')

def split_by_tabs(line):
    return line.split('\t')

# Then, we download the 2 files related to this dataset: diagnosis.names and diagnosis.data.

names_link = 'https://archive.ics.uci.edu/ml/machine-learning-databases/acute/diagnosis.names'
data_link = 'https://archive.ics.uci.edu/ml/machine-learning-databases/acute/diagnosis.data'
diagnosis_names = 'diagnosis.names'
diagnosis_data = 'diagnosis.data'
download_url(names_link, diagnosis_names)
download_url(data_link, diagnosis_data)

# We define some functions to parse the lines and fields of this dataset:
import numpy as np

def parse_double(field):
    field = field.replace(',', '.')
    return float(field)

def parse_boolean(field):
    return 1. if field == 'yes' else 0.

def read_np_array(file = diagnosis_data):
    text = read_binary_file(file)
    lines = split_text_in_lines(text)
    rows = []
    for line in lines:
        if line == '': continue
        line = line.replace('\r\n', '')
        fields = split_by_tabs(line)
        row = []
        j = 0
        for field in fields:
            value = parse_double(field) if j == 0 else parse_boolean(field)
            row.append(value)
            j += 1
        rows.append(row)
    matrix = np.array(rows, dtype = np.float32)
    # print(matrix)
    # print(matrix.shape)     #(120,8)
    return matrix


# We define some functions in order to randomly split this dataset in 2: Training dataset (96 samples) and testing dataset (24 samples).
def get_random_indexes(n):
    indexes = list(range(n))
    random_indexes = []
    for i in range(n):
        r = np.random.randint(len(indexes))
        random_indexes.append(indexes.pop(r))
    return random_indexes

def get_indexes_for_2_datasets(n, training = 80):
    indexes = get_random_indexes(n)
    train = int(training / 100. * n)   #96
    return indexes[:train], indexes[train:]

matrix = read_np_array()
n_samples, n_dimensions = matrix.shape
# print(n_samples)    #120
# print(n_dimensions)   #8
train_indexes, test_indexes = get_indexes_for_2_datasets(n_samples)
train_data = matrix[train_indexes]    #96
test_data = matrix[test_indexes]      #24

def print_dataset(name, data):
    print('Dataset {}. Shape: {}'.format(name, data.shape))
    print(data)

# We print the training dataset. It has 96 samples (96 rows), and 6 features and 2 diagnoses (8 columns).

# As we can see, the temperature is parsed into a real number. And the boolean values (yes or no) are parsed into real numbers as well: 1. and 0., respectively.
# print_dataset('Train', train_data)

# We print the testing dataset. It has 24 samples (24 rows), and 6 features and 2 diagnoses (8 columns).
# print_dataset('Test', test_data)

# We define our machine learning model, which is a logistic regression model. Why? Because this medical dataset is linearly separable, which simplifies things a lot.

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

input_size = 6
learning_rate = 0.01
num_iterations = 20000

class LogisticRegression(torch.nn.Module):

    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))
    
# We define some functions to train the machine learning model while keeping track of the training loss and the training accuracy.
def decide(y):
    return 1. if y >= 0.5 else 0.

decide_vectorized = np.vectorize(decide)

to_percent = lambda x: '{:.2f}%'.format(x)

def compute_accuracy(model, input, output):
    prediction = model(input).data.numpy()[:, 0]
    n_samples = prediction.shape[0] + 0.
    prediction = decide_vectorized(prediction)
    equal = prediction == output.data.numpy()
    return 100. * equal.sum() / n_samples

def get_input_and_output(data):
    input = Variable(torch.tensor(data[:, :6], dtype = torch.float32))
    output1 = Variable(torch.tensor(data[:, 6], dtype = torch.float32))
    output2 = Variable(torch.tensor(data[:, 7], dtype = torch.float32))
    return input, output1, output2

input, output1, output2 = get_input_and_output(train_data)
# print(input)
# print(len(input))
# print(output1)
# print(output2)
test_input, test_output1, test_output2 = get_input_and_output(test_data)
# print(test_input)
# print(len(test_input))
# print(test_output1)
# print(test_output2)

import matplotlib.pyplot as plt

diagnosis_title1 = 'Inflammation of Urinary Bladder'
diagnosis_title2 = 'Nephritis of Renal Pelvis Origin'

def plot_graphs(diagnosis_title, losses, accuracies):
    plt.plot(losses)
    plt.title(f"{diagnosis_title} - Training Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Training Loss")
    plt.show()
    plt.plot(accuracies)
    plt.title(f"{diagnosis_title} - Training Accuracy")
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy (Percent %)")
    plt.show()

def train_model(diagnosis_title, input, output, test_input, test_output):
    model = LogisticRegression()     #Model created
    # criterion = torch.nn.BCELoss(size_average=True)
    criterion = torch.nn.BCELoss(reduction='mean')  # Binary Cross Entropy Loss with reduction argument
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  
    losses = []
    accuracies = []
    n_samples, _ = input.shape
    output = output.view(-1, 1)  # Reshape target to match model output size
    for iteration in range(num_iterations):
            optimizer.zero_grad()
            prediction = model(input)
            loss = criterion(prediction, output)
            loss.backward()
            optimizer.step()
            if iteration % 500 == 0:
                train_acc = compute_accuracy(model, test_input, test_output)
                train_loss = loss.item()
                losses.append(train_loss)
                accuracies.append(train_acc)
                print('iteration={}, loss={:.4f}, train_acc={}'.format(iteration, train_loss, to_percent(train_acc)))
    plot_graphs(diagnosis_title, losses, accuracies)
    test_acc = compute_accuracy(model, test_input, test_output)
    print('\nTesting Accuracy = {}'.format(to_percent(test_acc)))
    return model


# We actually train the machine learning model to diagnose the Inflammation of Urinary Bladder. As you can see in the graphs, the training loss drops quickly to almost zero and the training accuracy reaches the 100%. The testing accuracy is also 100%. Notice that this machine learning system diagnoses this disease in a perfect way; whereas human doctors can commit mistakes.
model = train_model(diagnosis_title1, input, output1, test_input, test_output1)

# We actually train the machine learning model to diagnose the Nephritis of Renal Pelvis Origin. As you can see in the graphs, the training loss drops quickly to almost zero and the training accuracy reaches the 100%. The testing accuracy is also 100%. Notice that this machine learning system diagnoses this disease in a perfect way; whereas human doctors can commit mistakes.
model = train_model(diagnosis_title2, input, output2, test_input, test_output2)

# So far, we have used machine learning in an insecure way. Now, let's implement the techniques of Secure and Private AI, the main topic of SPAIC (Spike-based artificial intelligence computing framework).

##FEDERATED LEARNING WITH A TRUSTED AGGREGATOR##

# In this demo, there are 4 hospitals. (The dataset will be split in 4, randomly.) There could be more hospitals. The 4 hospitals cannot share the cases of their patients because they are competitors and it is necessary to protect the privacy of patients. Hence, the ML model will be learned in a federated way.

# How? Federated learning is iterated 1000 times. At each iteration, a copy of the shared model is sent to all the 4 hospitals. Each hospital trains its own local model with its own local dataset, in 5 local iterations. Each local model improves a little bit in its own direction. Then we compute the local losses and local accuracies to keep track of them and to make graphs of them. We send the local models to the trusted aggregator that will average all the model updates. This averaged model is the shared model that is sent to all the 4 hospitals at the begining of each iteration.

# In this way, only the ML model will be shared. Whereas the local cases of each hospital will be kept private and they will be used to train model updates in a local way. Federated learning will protect the privacy of datasets in each hospital and at the same time, we will generate a more robust machine learning model, which will benefit all hospitals. This shared ML model preserves the privacy of individual patients and at the same time, reveals important statistics of stereotypical cases.

# Federated Learning - Image taken from https://www.intel.ai/federated-learning-for-medical-imaging/

# import fed_server
# n_samples = train_data.shape[0]
# n_hospitals = len(hospitals) 
# samples_per_hospital = int((n_samples + 0.5) / n_hospitals)
# hospital_features = []
# hospital_targets1 = []
# hospital_targets2 = []
# train_data = fed_server.th.tensor(train_data, dtype = torch.float32, requires_grad=True)
# for i in range(n_hospitals):
#     train_data2 = train_data[i * samples_per_hospital:(i + 1) * samples_per_hospital].clone().detach().requires_grad_(True)
#     features = train_data2[:, :6].clone().detach().requires_grad_(True)
#     targets1 = train_data2[:, 6][:, None].clone().detach() 
#     targets2 = train_data2[:, 7][:, None].clone().detach() 
#     hospital_features.append(features.send(fed_server.hospitals[i]))
#     hospital_targets1.append(targets1.send(fed_server.hospitals[i]))
#     hospital_targets2.append(targets2.send(fed_server.hospitals[i]))

# print(model)