import csv
import argparse

# export csv_a=../datasets/1/msh_tanzania_bal-1-25/training/src_labels.csv 
# export csv_b=../datasets/1/msh_tanzania_bal-1-25/test/src_labels.csv 
# python analise.py --csv_a $csv_a  --csv_b $csv_b
# python analise.py --csv_a src_labels_train_350.csv  --csv_b src_labels_test_400.csv


# loads the dictionary 
def load_csv(fname):
    dict_train = {}
    for i in range(12):
        dict_train[i] = {}

    with open(fname) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            label = int(row[0])
            img_local = (row[2].split('/'))[-1]
            img_name = (row[1].split('/'))[-1]
            if row[2] not in dict_train[label]:
                dict_train[label][img_local] = []

            (dict_train[label][img_local]).append(img_name)

    return dict_train


parser = argparse.ArgumentParser(description='Analise')
parser.add_argument('--csv_a', type=str)   
parser.add_argument('--csv_b', type=str)   

args = parser.parse_args()
name_a = args.csv_a
name_b = args.csv_b

dict_a = load_csv(name_a)
dict_b = load_csv(name_b)

print("Size A | Size B | intersection size | difference size")

for label in dict_a:
    print("\nLabel %d" % label)
    names1 = sorted (dict_a[label].keys())  
    names2 = sorted (dict_b[label].keys())  
    print(set(names2).issubset(set(names1)))

    count_intersec = 0
    count_diff = 0
    for name1 in names1:
        name1_ex = False
        for name2 in names2:  
            if name1==name2:
                name1_ex = True
        if name1_ex:
            count_intersec += 1 
        else:
            count_diff += 1
        #    print("%s" % name1)
    print(len(names1), len(names2), count_intersec, count_diff)
