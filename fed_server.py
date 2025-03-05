# First, we start by creating the virtual workers that simulate the computers of each hospital. And then we establish communications among all of them.
import syft as sy
import torch as th
# hook = sy.TorchHook(th)
# from torch import nn, optim

def initialize_workers(n_hospitals):
    hospitals = []
    for i in range(n_hospitals):
        hospital_name = 'hospital{}'.format(i)
        hospital = sy.Worker(name=hospital_name)
        hospitals.append(hospital)
    secure_name = 'Secure{}'.format(n_hospitals)  
    secure_worker = sy.Worker(name=secure_name)
    return hospitals, secure_worker

def get_workers_names(workers):
    return [worker.name for worker in workers]

def add_and_print_workers(worker, workers):
    print('workers of {} = {}'.format(worker.name, get_workers_names(workers)))

if __name__ == "__main__":
    n_hospitals = 4
    hospitals, secure_worker = initialize_workers(n_hospitals)

    for i in range(n_hospitals):
        workers = [hospitals[j] for j in range(n_hospitals) if j != i] + [secure_worker]
        add_and_print_workers(hospitals[i], workers)
    add_and_print_workers(secure_worker, hospitals)

# Each virtual worker in each hospital is supposed to have some cases of patients, their own local datasets. The 4 hospitals cannot share the cases of their patients because they are competitors and it is necessary to protect the privacy of patients. So, the local cases of each hospital will be kept private and they will be used to train model updates in a local way.

# In order to simulate this process, we start by dividing the dataset in 4 parts, randomly. And then we send each part to each hospital. Each part is composed of the features, the diagnosis for disease 1, and the diagnosis for disease 2.

import central_server
n_hospitals = len(hospitals)  
n_samples = central_server.train_data.shape[0]   #returns the size of the first dimension of the tensor
# print(n_samples)  #96
# print(help(th))
samples_per_hospital = int((n_samples + 0.5) / n_hospitals)
hospital_features = []
hospital_targets1 = []
hospital_targets2 = []
train_data = th.tensor(central_server.train_data, dtype = th.float32, requires_grad=True)
# print(train_data)
for i in range(n_hospitals):
    train_data2 = train_data[i * samples_per_hospital:(i + 1) * samples_per_hospital].clone().detach().requires_grad_(True)
    features = train_data2[:, :6].clone().detach().requires_grad_(True)
    targets1 = train_data2[:, 6][:, None].clone().detach() 
    targets2 = train_data2[:, 7][:, None].clone().detach() 
# print(type(features))
    # hospital_features.append(features.scatter(hospitals[i]))
    # hospital_targets1.append(targets1.scatter(hospitals[i]))
    # hospital_targets2.append(targets2.scatter(hospitals[i]))
     # Assign extracted data to individual lists for each hospital
    hospital_features.append(features)
    hospital_targets1.append(targets1)
    hospital_targets2.append(targets2)

print(central_server.model)


