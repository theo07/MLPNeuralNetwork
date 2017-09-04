import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import array
from functools import reduce
import datetime
import multiprocessing
from multiprocessing import Manager
import unittest
manager = Manager()

#the number of categories for each specific categorical column in the 
#simple author data
position_size = 15
industry_size = 30
country_size = 254
access_size = 26
max_sess_seq  = 100000
#embedding the categorical data
embed_position = np.random.uniform(low=0, high=1, size=(position_size, 2))
embed_industry = np.random.uniform(low=0, high=1, size=(industry_size, 2))
embed_country = np.random.uniform(low=0, high=1, size=(country_size, 3))
embed_access = np.random.uniform(low=0, high=1, size=(access_size, 2))

#thread safe collection
#this will be a list of tuples
data_tuple_concurrency = manager.list([])

def normalise(value, max_value):
    return value/max_value

def one_hot_encode(x, n_classes):
        """
        One hot encode a 
        : x: category id
        : n_classes: Number of classes
        """
        x = int(x)
        verts=array.array('i',(0,)*n_classes)
        myLabel = verts.tolist()
        myLabel[x - 1] = 1
        return myLabel

def thread_preprocess(chunk):
        chunk_data_x = []
        chunk_data_y = []  
        new_list = chunk.values
        #use current to print thread's name
        #current = multiprocessing.current_process()
        for row in new_list:
            embed_row = []
            embed_row.extend(embed_position[row[2]])
            embed_row.extend(embed_industry[row[4]])
            embed_row.extend(embed_country[row[6]]) 
            embed_row.extend(embed_access[row[3]]) 
            
            embed_row.append(normalise(row[5],max_sess_seq))
            
            x_y_tuple = (embed_row, one_hot_encode(row[0], 2))
            data_tuple_concurrency.append(x_y_tuple)

class Items(object):
    #creating init of the data object
    #the object will have data(meaning the feature vectors) corresponding labels
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.batch_id = 0
        self.data_created = False
        #counting the number of training epochs
        self.epochsNo = 0
        self.data_frame = []
        # create as many processes as there are CPUs on your machine
        self.num_processes = multiprocessing.cpu_count()
    def reset(self):
        self.data = []
        self.labels = []
        self.batch_id = 0
    def createData(self):
        #use the path to the DataInput.csv file found adjacent to this file
        #this loop will try to read the csv file given an input string
        #/home/teo/Desktop/authorData.csv
        while not self.data_created:
            inputPath = input("Please enter the file path of the authorData.csv provided: ")
            try:
                df = pd.read_csv(inputPath, error_bad_lines=False)
                self.data_created = True
            except Exception as e:
                print('Cannot open file | File I|O Error | Try Again | Error as ', e)
        df = df.fillna(0)
            
        del df['position_desc']
        del df['country_desc']
        del df['industry_desc']
        s = pd.value_counts(df.session_sequence)

        # calculate the chunk size as an integer
        chunk_size = int(df.shape[0]/self.num_processes)
        chunks = [df.iloc[df.index[i:i + chunk_size]] for i in range(0, df.shape[0], chunk_size)]
        
        pool = multiprocessing.Pool(processes=self.num_processes)
        pool.map(thread_preprocess, chunks)

        x_data = []
        y_data = []
        print('Total number of the population: ', len(data_tuple_concurrency))
        for pair in data_tuple_concurrency:
            x_data.append(pair[0])
            y_data.append(pair[1])
        self.data = x_data
        self.labels = y_data
    def splitTrainTest(self):
        #splitting the data/labels/sequence with 60% for train 40% for test
        
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=0.4)
        #creatin the train and test items which will be fed to the model
        train_items = Items(X_train, y_train)
        test_items = Items(X_test, y_test)
        print('Total training number of entries: ', len(train_items.data))
        return train_items, test_items
    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.epochsNo += 1
            print("Epoch no: ",  self.epochsNo)
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels
"""
#the tests I have used to ensure robustness of the simple experiment
def init_data():
    my_item = Items([],[])
    my_item.createData()
    df = pd.read_csv(r'/home/teo/Desktop/authorData.csv', error_bad_lines=False)
    df = df.fillna(0)

    del df['position_desc']
    del df['country_desc']
    del df['industry_desc']
    df = df.values
    return my_item, df
class TestPreprocessing(unittest.TestCase):
    def test_preprocessing_concurrency(self):
        my_item, df = init_data()
        #there are 77186 entries in the CSV file
        #hence concurrency should merge 77186 
        self.assertEqual(len(my_item.data), 77186)
        #also pandas should read 77186 rows
        self.assertEqual(len(df), 77186)
        self.assertEqual(len(my_item.data), len(df))
        
        #I do not expect concurrency rows to be in the given CSV order
        #nevertheless I will test by SQL Unique ID; which will be removed
        #after tests
        
        for i in range(0, len(my_item.data)):
            #picking 10 random IDs from the CSV
            #testing that concurrency is right
            #testing that labels are right
            item = my_item.data[i]
            if item[0] == 917164:
                CSV_ITEM = [917164, 10, 25, 26, 371, 13]
                print('THE LABEL', my_item.labels[i])
                self.assertEqual(my_item.labels[i], one_hot_encode(0, 2))
                self.assertEqual(np.array_equal(CSV_ITEM, item), True)
            elif item[0] == 738160:
                CSV_ITEM = [738160, 7, 25, 10, 42, 82]
                self.assertEqual(my_item.labels[i], one_hot_encode(0, 2))
                self.assertEqual(np.array_equal(CSV_ITEM, item), True)
            elif item[0] == 889426:
                CSV_ITEM = [889426, 12, 25, 26, 15, 13]
                self.assertEqual(my_item.labels[i], one_hot_encode(0, 2))
                self.assertEqual(np.array_equal(CSV_ITEM, item), True)
            elif item[0] == 708758:
                CSV_ITEM = [708758, 1, 25, 27, 110, 224]
                self.assertEqual(my_item.labels[i], one_hot_encode(0, 2))
                self.assertEqual(np.array_equal(CSV_ITEM, item), True)
            elif item[0] == 709332:
                CSV_ITEM = [709332, 4, 20, 25, 12, 224]
                self.assertEqual(my_item.labels[i], one_hot_encode(1, 2))
                self.assertEqual(np.array_equal(CSV_ITEM, item), True)
            elif item[0] == 1402192:
                CSV_ITEM = [1402192, 0, 25, 0, 14, 13]
                self.assertEqual(my_item.labels[i], one_hot_encode(0, 2))
                self.assertEqual(np.array_equal(CSV_ITEM, item), True)
            elif item[0] == 321279:
                CSV_ITEM = [321279, 4, 20, 0, 43, 224]
                self.assertEqual(my_item.labels[i], one_hot_encode(1, 2))
                self.assertEqual(np.array_equal(CSV_ITEM, item), True)
            elif item[0] == 1497434:
                CSV_ITEM = [1497434, 0, 25, 0, 16, 249]
                self.assertEqual(my_item.labels[i], one_hot_encode(0, 2))
                self.assertEqual(np.array_equal(CSV_ITEM, item), True)
            elif item[0] == 1031938:
                CSV_ITEM = [1031938, 0, 25, 0, 12, 224]
                self.assertEqual(my_item.labels[i], one_hot_encode(0, 2))
                self.assertEqual(np.array_equal(CSV_ITEM, item), True)
            elif item[0] == 1309124:
                CSV_ITEM = [1309124, 4, 25, 25, 17, 13]
                self.assertEqual(my_item.labels[i], one_hot_encode(0, 2))
                self.assertEqual(np.array_equal(CSV_ITEM, item), True)
            elif item[0] == 1162270:
                CSV_ITEM = [1162270, 8, 25, 16, 38, 224]
                self.assertEqual(my_item.labels[i], one_hot_encode(0, 2))
                self.assertEqual(np.array_equal(CSV_ITEM, item), True)
"""
