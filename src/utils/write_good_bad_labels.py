#!/usr/bin/python
import datetime, os, math, argparse
import sys
import subprocess
import MySQLdb
import getopt
from PIL import Image, ImageEnhance, ImageStat
import random
import numpy as np


#how to use
# python write_good_bad_labels.py --dataset_path msh_tanzania_bal-400 --samples_per_drug 400 > categorize__good_samples.csv

#set to my folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def write_csv(file_name, data):
    with open(file_name,'w') as myf:
        for row in data:
            myf.write(row + '\n')
    myf.close()


# read image
def get_img(filename):
    #try to open file
    try:
        img = Image.open(filename)
    except Exception:
        print "Cannot open", filename

    return img

# function to return average brightness of an image
# Source: http://stackoverflow.com/questions/3490727/what-are-some-methods-to-analyze-image-brightness-using-python
def brightness(im):
    stat = ImageStat.Stat(im)
    r,g,b = stat.mean
    #return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))   #this is a way of averaging the r g b values to derive "human-visible" brightness
    return math.sqrt(0.577*(r**2) + 0.577*(g**2) + 0.577*(b**2))

# preprocess image
def preprocess_img(img, img_brightness, img_size):
    #crop image to get active area
    img = img.crop((71, 359, 71+636, 359+490))

    #fix brightness
    if img_brightness > 0:
        #massage image
        bright = brightness(img)
        imgbright = ImageEnhance.Brightness(img)
        img = imgbright.enhance(img_brightness/bright)

    #resize in img_size set
    if img_size > 0:
        size = (img_size, img_size)
        img = img.resize((size), Image.ANTIALIAS)

    return img


# get random number that in not in the taken
def get_random_number(taken):
    r = random.randrange(1,32767)
    while r in taken:
        r = random.randrange(1,32767)
    return(r)


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



# randomly split a list in n elements 
def rand_select_no_repeat(a_list, n):
    selection = []
    remaining = []

    available = list(range(len(a_list)))
    while len(selection) < n and len(available) > 0:
        index = random.randint(0, len(available) - 1)
        selection.append(a_list[available[index]])
        available.remove(available[index])

    for i in available:
        remaining.append(a_list[i])

    return selection, remaining


# save image and data
def save_dataset(label_names, basic_folder, label_folder, bal_label_lists,img_brightness, img_size):

    num_taken = {}

    initialize_folders(basic_folder,label_folder)

    for sub_folder in label_folder:
        
        print("\nFolder [%s/%s/]" % (basic_folder,sub_folder))
        
        string_samples_list = [] 
        string_samples_source_list = [] 

        # for each class
        for actual_label in range(len(label_names)):
            print("generating for label %d: %d samples" % (actual_label, len(bal_label_lists[sub_folder][actual_label])))
            for filename in bal_label_lists[sub_folder][actual_label]:
                """# get  image 
                # get  image 
                img = get_img(filename) 
               
                # preprocess image 
                img = preprocess_img(img, img_brightness, img_size) """                

                # define new filename img
                rand_num = get_random_number(num_taken)
                num_taken[rand_num] = 1
                new_filename = ("%s/%s/%d_%d.png" % (basic_folder,sub_folder,actual_label,rand_num))
                """ 
                # save img
                img.save(new_filename)
                """
                # add new_filename (used in labels.csv)
                string_samples_list += [("%d,%s" % (actual_label,new_filename))] 
                string_samples_source_list += [("%d,%s,%s" % (actual_label,new_filename,filename))] 
        
        
        # shuffle list
        random.shuffle(string_samples_list)
        random.shuffle(string_samples_source_list)
        
        # save labels.csv 
        write_csv(("%s/%s/labels.csv" % (basic_folder,sub_folder)), string_samples_list) 
        
        # save source_sample_labels.csv
        write_csv(("%s/%s/src_labels.csv" % (basic_folder,sub_folder)), string_samples_source_list) 



# Access credentials to connect the database
def get_db(credential_fname,db_name):
    #get database credentials (db=pad)
    with open(credential_fname) as f:
        line = f.readline()
        split = line.split(",")
    f.close()

    #open database
    db = MySQLdb.connect(host="localhost", # your host, usually localhost
                         user=split[0], # your username
                          passwd=split[1], # your password
                          db=db_name) # name of the data base

    # you must create a Cursor object. It will let
    #  you execute all the queries you need
    return  db


#--------------------------------------------
# create all_sample_dict
#--------------------------------------------
def get_initial_sample_dict(pad_analysis_cur):

    # sql
    pad_analysis_cur.execute('SELECT DISTINCT `sample_actual` FROM `Analysis`')

    #grab a list of drugs and distractors
    all_sample_dict = {}

    #fill in dictionary
    for row in pad_analysis_cur.fetchall() :
        all_sample_dict[row[0].strip()] = 0

    return all_sample_dict


#-------------------------------------------------
# [Samples which belong to the categorize set]
#-------------------------------------------------
def get_categorize_data(pad_cur):
    
    # sql
    pad_cur.execute('SELECT `id`, `processed_file_location`, `sample_name` FROM `card` WHERE `training_set` & 96 = 96') 

    # grab data
    data = pad_cur.fetchall()

    return(data)


#------------------------------------------------
# [Samples do not belong to categorize set] 
# (used in the train and test(validation) sets)
#------------------------------------------------
def get_train_test_data(pad_cur):
    
    # sql
    pad_cur.execute('SELECT `id`, `processed_file_location`, `sample_name` FROM `card` WHERE `training_set` & 96 != 96 AND (`category`=\'MSH Tanzania\' OR `sample_name` LIKE "Blank PAD%" OR `sample_name`="DI water") AND `processed_file_location`!=""')

    # grab data
    data = pad_cur.fetchall()

    return(data)

# 
def process_sql_data(data, pad_analysis_cur):

    # create  dictionary for drugs and  distractors
    all_sample_dict = get_initial_sample_dict(pad_analysis_cur)
    all_writen_sample_dict = get_initial_sample_dict(pad_analysis_cur)

    # create 2nd dictionary for drugs no distractors
    drugs_sample_dict = {}
    missing = 0
    DIwater_sample = 0
    BlankPAD_sample = 0
    num_samples = 0

    data_samples = np.zeros((len(data),3), dtype='|S200')

    # row1 = id 
    # row2 = sample_name ( nota: na tabela Analysis tem o mesmo valor que actual_label ) 
    # row3 = proc_file_loc
 

    for row in data:
        sample_name = row[2].strip()

        #print "ROW2 " + str(row[0])
        if sample_name in drugs_sample_dict:
            drugs_sample_dict[sample_name] += 1  
        elif "Blank PAD" not in sample_name  and "DI water" not in sample_name:
            drugs_sample_dict[sample_name] = 1
       
        pad_analysis_cur.execute('SELECT `id`,`sample_actual`, `sample_name` FROM `Analysis` WHERE `id`=' + str(row[0]))
        
        found_it = False
        for row_analysis in pad_analysis_cur.fetchall():
            found_it = True
            # sample_actual
            actual_sample_name = row_analysis[1].strip()
            if actual_sample_name in all_sample_dict:
                all_sample_dict[actual_sample_name] += 1
                data_samples[num_samples,:] = [actual_sample_name,row[1].strip(),row_analysis[2].strip()]
                num_samples+=1

            """# sample_name (label is written in the card)
            written_sample_name = row_analysis[2].strip()
            if written_sample_name in all_sample_dict:
                all_writen_sample_dict[written_sample_name] += 1
                data_samples[num_samples,:] = [actual_sample_name, row[1].strip()]
                num_samples+=1 """

        if not found_it:       
            # [NOTE]
            # as amostras perdidas acabam sendo a de water e blanki
            if "DI water" in sample_name:
                DIwater_sample += 1
                data_samples[num_samples,:] = ["DI water", row[1].strip(), "DI water"]
                num_samples += 1
                #print(DIwater_sample, sample_name)
            elif "Blank PAD" in sample_name:
                BlankPAD_sample += 1
                data_samples[num_samples,:] = ["Blank PAD", row[1].strip(), "Blank PAD"]
                num_samples += 1
                #print(BlankPAD_sample, sample_name)
            else:
                #print(missing,sample_name)
                # print("perdido " + sample_name)
                missing += 1


    # print
    """          
    print("\n")
    print("All Sample: Drugs and DI", len(all_sample_dict))
    print(all_sample_dict)
    print("\n")
    print("Drugs only", len(drugs_sample_dict))
    print(drugs_sample_dict)
    print("\n")
    print("Missing sample", missing)
    """

    #generate numeric labels
    sample_dict_lookup = {}

    #lookup value
    sample_count = 0 

    #loop over and print sample_dict
    for sample in all_sample_dict:
        if all_sample_dict[sample] > 0:
            sample_dict_lookup[sample] = sample_count
            sample_count += 1

    # add DI water
    all_sample_dict['DI water'] = DIwater_sample
    sample_dict_lookup['DI water'] = sample_count 
    sample_count += 1

    # add Blank PAD
    all_sample_dict['Blank PAD'] = BlankPAD_sample
    sample_dict_lookup['Blank PAD'] = sample_count
    sample_count += 1

    # print 
    """i
    sum_all = 0
    for id in range(sample_count):
        sample = sample_dict_lookup.keys()[sample_dict_lookup.values().index(id)]
        print(sample + "," + str(sample_dict_lookup[sample]) + "," + str(all_sample_dict[sample]))
        
        print("%20s, %2d, %3d" % (sample, id, all_sample_dict[sample]))
        sum_all += all_sample_dict[sample]
    """

    # criar uma lista com os indices para a matrix  data_samples 
    label_lists = {}

    # criar uma lista com os indices para a matrix  data_samples 
    written_label_lists = {}


    # label name list
    label_names = {}

    for i in range(sample_count):
        label_lists[i] = []
        written_label_lists[i] = []
        label_name = sample_dict_lookup.keys()[sample_dict_lookup.values().index(i)]
        label_names[i] = label_name 
     
    print "#good/bad,id_actual,id_written,actual_sample_name,written_sample_name,sample_path"

    for i in range(num_samples):
        #print("%d, %s" % (sample_dict_lookup[data_samples[i,0]], data_samples[i,0]))
        
        # sample_actual
        actual_sample_name = data_samples[i,0]

        # sample_name (written in the card)
        written_sample_name = data_samples[i,2]

        # id_actual
        id_actual = sample_dict_lookup[actual_sample_name]

        # id_written
        id_written = sample_dict_lookup[written_sample_name]

        # sample_path
        sample_path = data_samples[i,1]
        label_lists[id_actual].append(sample_path)

        if id_actual != id_written:
            print "bad," + str(id_actual) + "," + str(id_written) + "," + actual_sample_name + "," +  written_sample_name + "," +  sample_path
        else:
            print "good," + str(id_actual) + "," + str(id_written) + "," + actual_sample_name + "," +  written_sample_name + "," +  sample_path
        # sample_name 
        written_label_lists[id_actual].append(written_sample_name)

    return (label_lists, label_names, written_label_lists)
    


#generate datasets from images in database
def generate_dataset(basic_folder, number_of_samples, img_size,  img_brightness, label_folder, sample_split, trainset_perc):

     
    """
    #[ parameters ]

    # image size
    img_size = 227

    # image Brightness
    image_brightness = 165.6

    # percentage of trainset size
    trainset_perc = 0.7

    #select how many images we require per drug
    number_of_samples = 50 # number_of_samples

    # main folder
    basic_folder = 'msh_tanzania_bal-50-xdebug'
    """

    # data base conections 
    pad_analysis_db = get_db('credentials_analysis.txt','pad_analysis')
    pad_analysis_cur = pad_analysis_db.cursor()
    pad_db = get_db('credentials.txt','pad')
    pad_cur = pad_db.cursor()

    # 
    bal_label_lists = {}
    bal_label_lists[label_folder[0]] = {}
    bal_label_lists[label_folder[1]] = {}
    bal_label_lists[label_folder[2]] = {}


    # Get data for categorize set from database
    data = get_categorize_data(pad_cur)
    [label_lists_c, label_names, wrt_label_lists_c] = process_sql_data(data, pad_analysis_cur)

    # Get data for train and test sets from database
#    data = get_train_test_data(pad_cur)
#    [label_lists, label_names, wrt_label_lists] = process_sql_data(data, pad_analysis_cur)


    # Close all cursors
    pad_cur.close()
    pad_analysis_cur.close()
    # Close all databases
    pad_db.close()
    pad_analysis_db.close()


#seed random for putting images into training and test arrays
#random.seed(1234)
#random.seed(5678)
random.seed(9012)

#~~~~Defines~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#define sub folders
# must be unique as label file allways called label.csv
# Order is training, test, catagory
arg_label_folder = ["training", "test", "categorize"]

#split trauining, test, catagorize
arg_sample_split = [0.6, 0.25, 0.15]

#~~~~End defines~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#set to my folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#get args
parser = argparse.ArgumentParser(description='Generate Datasets script')
parser.add_argument('--dataset_path', default="msh_tanzania_bal-20", type=str, help="Dataset path")
parser.add_argument('--samples_per_drug', default=20, type=int, help="Number of samples per drug")
parser.add_argument('--image_size', default=227, type=int, help="Size of output image")
parser.add_argument('--image_brightness', default=165.6, type=float, help="Image brightness level")
parser.add_argument('--train_percent', default=0.7, type=float, help="Percentual of train set samples")
args = parser.parse_args()

#get path to store
arg_folder = args.dataset_path

#get number of samples per drug
arg_number_of_samples = args.samples_per_drug
#print "samples_per_drug", arg_number_of_samples

#Get image size
arg_img_size = args.image_size

#Get image Brightness
arg_img_brightness = args.image_brightness
#print "Brightness", arg_img_brightness

# percentage of trainset size
arg_trainset_perc = args.train_percent 
#trainset_perc = 0.7

#run it
generate_dataset(arg_folder, arg_number_of_samples, arg_img_size, arg_img_brightness, arg_label_folder, arg_sample_split, arg_trainset_perc)






