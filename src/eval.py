import os
import sys
import numpy as np

def eval(**kwargs):
    """
    Eval the ensembled test for the PADS CNN model
    
    args: **kwargs (dict) keyword arguments
    """

    # Roll out the parameters
    dataset_name = kwargs["dataset_name"]
    dataset_size_list = kwargs["dataset_size_list"]
    dataset_group_id = kwargs["dataset_group_id"]
    drug_label_fname = kwargs["drug_label_fname"]
    num_seeds = kwargs["num_seeds"]
    output_path = kwargs["output_path"]
    prediction_path = kwargs["prediction_path"]
    dataset_sizes = list(map(int, dataset_size_list.split(',')))
    
    """   
    # inputs
    drug_label_fname = "../datasets/msh_tanzania_blank_drugs.csv"
    dataset_name = "msh_tanzania_bal"
    prediction_path = "/scratch365/pmoreira/pads_ensembled_test_v2/output/predictions" 
    num_seeds = 20
    dataset_sizes = [20,50]
    output_path = "/scratch365/pmoreira/pads_ensembled_test_v2/output/test_results"
    """
    
    v_acc1 = np.zeros((len(dataset_sizes),num_seeds))
    v_acc2 = np.zeros((len(dataset_sizes),num_seeds))

    # general acc
    acc_fname = ("%s/prediction_%s_acc_all.csv" % (output_path, dataset_name))

    # drugs and distractors acc 
    acc1_fname = ("%s/prediction_%s_acc1_mean.csv" % (output_path, dataset_name))

    # only drugs acc
    acc2_fname = ("%s/prediction_%s_acc2_mean.csv" % (output_path, dataset_name))

    # drugs and distractors [max]  
    acc_max_fname = ("%s/prediction_%s_acc_max.csv" % (output_path, dataset_name))

    # drugs and distractors [min, mean, max] acc1 
    acc1_minmax_fname = ("%s/prediction_%s_acc1_min_max.csv" % (output_path, dataset_name))

    # drugs [min, mean, max] acc2 
    acc2_minmax_fname = ("%s/prediction_%s_acc2_min_max.csv" % (output_path, dataset_name))



    # file head - acc1 and acc2 by task, by dataset 
    write_to_file = "dataset_size,task_id,acc_1,acc_2(only drugs)\n"
    csv_file = open(acc_fname, "w")
    csv_file.write(write_to_file)
    csv_file.close()


    # file head - acc max value by dataset
    write_to_file = "dataset_size,acc_1_max,acc_2_max(only drugs)\n"
    csv_file = open(acc_max_fname, "w")
    csv_file.write(write_to_file)
    csv_file.close()

    for dataset_id in range(len(dataset_sizes)):
        dataset_size = dataset_sizes[dataset_id]
        for task_id in range(num_seeds):
            # preditcion file name
            prediction_fname = ("%s/prediction_msh_tanzania_bal-%d-%d_%d.csv" % (prediction_path, dataset_group_id, dataset_size, task_id+1)) 

            # get acc
            acc1, acc2 = get_acc(drug_label_fname,prediction_fname)

            v_acc1[dataset_id,task_id] = acc1
            v_acc2[dataset_id,task_id] = acc2        

            # print results to file
            write_to_file = "%d,%d,%.2f,%.2f\n" % (dataset_size,task_id+1,acc1,acc2)

            csv_file = open(acc_fname, "a+")
            csv_file.write(write_to_file)
            csv_file.close()

        acc1 = v_acc1[dataset_id,:]
        acc2 = v_acc2[dataset_id,:]

        # ACC1 mean file to plot error bar
        write_to_file = "%d,%.2f,%.2f\n" % (dataset_size,np.mean(acc1),np.std(acc1))
        csv_file = open(acc1_fname, "a+")
        csv_file.write(write_to_file)
        csv_file.close()

        # ACC2 mean file to plot error bar
        write_to_file = "%d,%.2f,%.2f\n" % (dataset_size,np.mean(acc2),np.std(acc2))
        csv_file = open(acc2_fname, "a+")
        csv_file.write(write_to_file)
        csv_file.close()

        # ACC2 (min,mean,max) file to plot error bar
        write_to_file = "%d,%.2f,%.2f,%.2f\n" % (dataset_size,np.min(acc2),np.mean(acc2),np.max(acc2))
        csv_file = open(acc2_minmax_fname, "a+")
        csv_file.write(write_to_file)
        csv_file.close()

        # ACC1 (min,mean,max) file to plot error bar
        write_to_file = "%d,%.2f,%.2f,%.2f\n" % (dataset_size,np.min(acc1),np.mean(acc1),np.max(acc1))
        csv_file = open(acc1_minmax_fname, "a+")
        csv_file.write(write_to_file)
        csv_file.close()


        # ACC max file to plot error bar
        write_to_file = "%d,%.2f,%.2f\n" % (dataset_size,np.max(acc1),np.max(acc2))
        csv_file = open(acc_max_fname, "a+")
        csv_file.write(write_to_file)
        csv_file.close()

def get_acc(drug_label_fname,prediction_fname):
    
    # open drug labels
    try:
        d_file = open(drug_label_fname, "r") #open(sys.argv[1], "r")
        d_lines = d_file.readlines()
    except:
        print("Missing drug label file.")
        exit(-1)

    drugs = {}
    drug_or_distractor = {}

    #loop
    for dl in d_lines:
        data = dl.split(",")
        try:
            drugs[int(data[1])] = data[0]
            drug_or_distractor[int(data[1])] = int(data[3])
        except:
            continue

    no_drugs = len(drugs)

    # open the prediction file
    try:
        file = open(prediction_fname, "r") #open(sys.argv[2], "r")
        cat_lines = file.readlines()
    except:
        print("Missing user file: %s" % prediction_fname)
        exit(-1)


    
    #counters
    missing = 0
    count_pos_all_drugs = 0
    count_total_all_drugs = 0
    count_pos = 0
    count_total = 0
    count_pos_drugs = [0] * no_drugs
    count_total_drugs = [0] * no_drugs


    #loop
    for catl in cat_lines:
        data = catl.split(",")

        #get filename
        data_file = os.path.basename(data[1].strip())
        label = [int(s) for s in data_file.split('_') if s.isdigit()][0]

        #get prediction
        prediction = int(data[0])

        count_total += 1
        count_total_drugs[label] += 1

        #all drugs, only add if not distractor
        if drug_or_distractor[label] == 1:
            count_total_all_drugs += 1

        #correct?
        if label == prediction:
            count_pos += 1
            count_pos_drugs[label] += 1
            #all drugs, only add if not distractor
            if drug_or_distractor[label] == 1:
                count_pos_all_drugs += 1

    #print accuracy
    if count_total > 0:
        acc1 = round(float(count_pos)/float(count_total) * 100, 2)
    else:
        acc1 = 0
    #print accuracy for drugs not distractors
    if count_total_all_drugs > 0:
        acc2 = round(float(count_pos_all_drugs)/float(count_total_all_drugs) * 100, 2)
    else:
        acc2 = 0
        
    return acc1,acc2
