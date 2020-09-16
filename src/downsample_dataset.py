import os
import sys
import csv
import random
import argparse
import shutil

# Functions

# counts how many samples in one given class
def _count_samples(dict_train, c):
    count = 0
    for f in dict_train[c].keys():
        count = count + len(dict_train[c][f])

    return count

# removes one element from a given class
# works in loco
def _remove_one_from_largest_set(dict_train, c):
    max_key = None
    max_set_size = 0

    for f in dict_train[c].keys():
        if len(dict_train[c][f]) > max_set_size:
            max_key = f
            max_set_size = len(dict_train[c][f])

    if max_key is not None:
        del dict_train[c][max_key][-1]
    
    if len(dict_train[c][max_key]) == 0:
        del dict_train[c][max_key]

# loads the dictionary to be reduced
def load_csv(fname):
    dict_train = {}
    for i in range(12):
        dict_train[i] = {}

    with open(fname) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            label = int(row[0])
            img_local = row[2]
            img_name = row[1]
            if row[2] not in dict_train[label]:
                dict_train[label][img_local] = []

            (dict_train[label][img_local]).append(img_name)

    return dict_train

def initialize_folders(basic_folder,label_folder):
    #create
    try:
        # Create target Directory
        os.mkdir(basic_folder)
        print("Directory " , basic_folder ,  " Created ")
        for folder in label_folder:
            os.mkdir(basic_folder +'/'+ folder)
    except OSError:
        print("Directory " , basic_folder ,  " already exists")

        
def save_dataset(dict_set, sub_folder, dataset_path, reduced_dataset_name):

    string_samples_list = []
    string_samples_source_list = []

    for label in dict_set.keys():
        # print("%d: %d total, %d unique locals" % (label,_count_samples(dict_set, label),len(dict_set[label].keys())))

        for fname in dict_set[label].keys(): 
            # print("filename = %s" % fname)
            if len(dict_set[label][fname]):
                local_list = dict_set[label][fname]
                # print("Total %d" % len(local_list))

                for loc_fname in local_list:                    
                    new_loc_fname = "%s/%s/%s" % (reduced_dataset_name, sub_folder, (loc_fname.split('/'))[-1])                    
                    string_samples_list += [("%d,%s" % (label,new_loc_fname))] 
                    string_samples_source_list += [("%d,%s,%s" % (label,new_loc_fname,fname))]
                    print(loc_fname,new_loc_fname)
                    shutil.copy2(dataset_path+'/'+loc_fname, dataset_path+'/'+new_loc_fname)
                
    # save csv files
    # shuffle list
    random.shuffle(string_samples_list)
    random.shuffle(string_samples_source_list)
        
    # save labels.csv 
    _write_csv(("%s/%s/%s/labels.csv" % (dataset_path,reduced_dataset_name,sub_folder)), string_samples_list) 
        
    # save source_sample_labels.csv
    _write_csv(("%s/%s/%s/src_labels.csv" % (dataset_path,reduced_dataset_name,sub_folder)), string_samples_source_list) 
            

def _write_csv(file_name, data):
    with open(file_name,'w') as myf:
        for row in data:
            myf.write(row + '\n')
    myf.close()

# reduces the content of the given dictionary per class, in loco
def reduce_in_loco(dict_train, n):
    for c in dict_train.keys():
        while _count_samples(dict_train, c) > n:
            _remove_one_from_largest_set(dict_train, c)

            

# main

parser = argparse.ArgumentParser(description='DOWN SAMPLING DATASET')
parser.add_argument('--dataset_path', type=str, help="Dataset path")  
parser.add_argument('--dataset_name', type=str, help="Dataset name")  
parser.add_argument('--dataset_group_id', type=int, help="Dataset Group Id")  
parser.add_argument('--dataset_size_list', type=str, help="Dataset size list")
parser.add_argument('--train_percent', default=0.7, type=float, help="Percentual of train set samples")

args = parser.parse_args()
dataset_path = args.dataset_path
dataset_size_list = list(map(int, args.dataset_size_list.split(',')))
dataset_name = args.dataset_name
dataset_group_id = args.dataset_group_id
train_percent = args.train_percent
label_folder = ["training", "test", "categorize"]

print(dataset_path)
print(dataset_size_list)
print(dataset_name)
print(train_percent)

# load train and test list
train_fname = "%s/%s/%s/src_labels.csv" % (dataset_path, dataset_name, label_folder[0])
test_fname = "%s/%s/%s/src_labels.csv" % (dataset_path, dataset_name, label_folder[1])
cat_fname = "%s/%s/%s/src_labels.csv" % (dataset_path, dataset_name, label_folder[2])
dict_cat = load_csv(cat_fname)

for new_num_samples in dataset_size_list:
    
    reduced_dataset_name = "msh_tanzania_bal-%d-%d" % (dataset_group_id, new_num_samples)
    
    # save the reduced dataset
    new_dataset_path = dataset_path + '/' + reduced_dataset_name

    # number of samples of the new train and test sets
    nn_train = int(round(new_num_samples * train_percent))
    nn_test = new_num_samples - nn_train
    
    print(new_dataset_path,nn_train,nn_test)
    
    
    # load train and test list
    dict_train = load_csv(train_fname)
    dict_test = load_csv(test_fname)

    # reducing train and test set discts
    reduce_in_loco(dict_train, nn_train)
    reduce_in_loco(dict_test, nn_test)

    initialize_folders(new_dataset_path,label_folder)

    save_dataset(dict_train, label_folder[0], dataset_path, reduced_dataset_name)
    save_dataset(dict_test, label_folder[1], dataset_path, reduced_dataset_name)
    save_dataset(dict_cat, label_folder[2], dataset_path, reduced_dataset_name)

    dict_train.clear()
    dict_test.clear()
    

dict_cat.clear()
           
