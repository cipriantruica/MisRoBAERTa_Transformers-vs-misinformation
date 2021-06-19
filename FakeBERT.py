# coding: utf-8

__author__ = "Ciprian-Octavian Truică"
__copyright__ = "Copyright 2021, University Politehnica of Bucharest"
__license__ = "GNU GPL"
__version__ = "0.1"
__email__ = "ciprian.truica@upb.ro"
__status__ = "Production"

import tensorflow as tf
from tensorflow.keras import layers, Model
from keras.utils import np_utils
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import sys
import time
import re

# pip install bert-for-tf2 tensorflow-hub

import bert
import tensorflow_hub as hub

classes = {
    "reliable": 0,
    "fake": 1,
    "bias": 2,
    "clickbait": 3,
    "conspiracy": 4,
    "hate": 5,
    "junksci": 6,
    "political": 7,
    "satire": 8,
    "unreliable": 9
}


class FakeBERT(tf.keras.Model):

    def __init__(self, vocabulary_size, embedding_dimensions=1000, cnn_filters=128, model_output_classes=2, dropout_rate=0.2, training=False, name="FakeBERT"):
        super(FakeBERT, self).__init__(name=name)

        self.embedding = layers.Embedding(vocabulary_size, embedding_dimensions)
        self.cnn_layer1 = layers.Conv1D(filters=cnn_filters, kernel_size=3, padding="valid", activation="relu", name='CNN-1')
        self.cnn_layer2 = layers.Conv1D(filters=cnn_filters, kernel_size=4, padding="valid", activation="relu", name='CNN-2')
        self.cnn_layer3 = layers.Conv1D(filters=cnn_filters, kernel_size=5, padding="valid", activation="relu", name='CNN-3')
        self.cnn_layer4 = layers.Conv1D(filters=cnn_filters, kernel_size=5, padding="valid", activation="relu", name='CNN-4')
        self.cnn_layer5 = layers.Conv1D(filters=cnn_filters, kernel_size=5, padding="valid", activation="relu", name='CNN-5')
        self.pool = layers.MaxPool1D(pool_size=5)
        self.flatten = layers.Flatten()
        self.dense_1 = layers.Dense(units=384, activation="relu")
        self.dense_2 = layers.Dense(units=128, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        if model_output_classes == 2:
            self.last_dense = layers.Dense(units=1, activation="relu")
        else:
            self.last_dense = layers.Dense(units=model_output_classes, activation="relu")

    def call(self, inputs, training):
        l = self.embedding(inputs)

        l_1 = self.cnn_layer1(l)
        l_1 = self.pool(l_1)

        l_2 = self.cnn_layer2(l)
        l_2 = self.pool(l_2)

        l_3 = self.cnn_layer3(l)
        l_3 = self.pool(l_3)

        concatenated = tf.concat([l_1, l_2, l_3], axis=-1) # (batch_size, 3 * cnn_filters)
        concatenated = self.cnn_layer4(concatenated)
        concatenated = self.pool(concatenated)

        concatenated = self.cnn_layer5(concatenated)
        concatenated = self.pool(concatenated)

        concatenated = self.flatten(concatenated)
        concatenated = self.dense_1(concatenated)
        concatenated = self.dropout(concatenated, training)
        concatenated = self.dense_2(concatenated)
        concatenated = self.dropout(concatenated, training)
        output = self.last_dense(concatenated)

        return output

# nohup python FakeBERT.py fake-news.csv 2 > output_FakeBERT 2>&1 &
if __name__ == "__main__":
    FILE_NAME = sys.argv[1]
    # MODEL_TYPE = sys.argv[2]
    # NO_GPU = int(sys.argv[3])
    NO_TESTS = int(sys.argv[2])

    accuracy = []
    precision_micro = []
    precision_macro = []
    precision_weighted = []
    recall_micro = []
    recall_macro = []
    recall_weighted = []
    execution_time = []
    for i in range(0, NO_TESTS):

        dataSet = pd.read_csv(FILE_NAME, encoding = "utf-8")

        # print(dataSet)
        dataSet['labels'] = 0
        for elem in classes:
            dataSet.loc[dataSet['label'] == elem, 'labels'] = classes[elem]

        dataSet.drop(['label'], axis=1, inplace=True)
        dataSet.rename(columns = {'content': 'text'}, inplace = True)

        # print(dataSet.head())

        BertTokenizer = bert.bert_tokenization.FullTokenizer
        bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3", trainable=False)
        vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
        to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
        tokenizer = BertTokenizer(vocabulary_file, to_lower_case)

        VOCAB_LENGTH = len(tokenizer.vocab)
        # print(VOCAB_LENGTH)
        EMB_DIM = 1000
        CNN_FILTERS = 128
        OUTPUT_CLASSES = dataSet['labels'].nunique()
        # print(OUTPUT_CLASSES)
        DROPOUT_RATE = 0.2
        NB_EPOCHS = 10

        def tokenize_reviews(text_reviews):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_reviews))

        start_time = time.time()

        dataSet['text'] = dataSet['text'].apply(tokenize_reviews)

        train_df, test_df = train_test_split(dataSet, test_size=0.2, stratify=dataSet['labels'])

        train_list = train_df.values.tolist()
        train_list_len = [[elem[0], np_utils.to_categorical(elem[1], OUTPUT_CLASSES), len(elem[0])] for elem in train_list]
        train_list_len.sort(key=lambda x: x[2])
        sorted_train_list = [(elem[0][:1000], elem[1]) for elem in train_list_len]

        test_list = test_df.values.tolist()
        test_list_len = [[elem[0], np_utils.to_categorical(elem[1], OUTPUT_CLASSES), len(elem[0])] for elem in test_list]
        test_list_len.sort(key=lambda x: x[2])
        sorted_test_list = [(elem[0][:1000], elem[1]) for elem in test_list_len]

        processed_train = tf.data.Dataset.from_generator(lambda: sorted_train_list, output_types=(tf.int32, tf.int32))
        processed_test = tf.data.Dataset.from_generator(lambda: sorted_test_list, output_types=(tf.int32, tf.int32))

        BATCH_SIZE = 128
        train_data = processed_train.padded_batch(BATCH_SIZE, padded_shapes=((1000, ), (OUTPUT_CLASSES,)))
        test_data = processed_test.padded_batch(BATCH_SIZE, padded_shapes=((1000, ), (OUTPUT_CLASSES,)))

        # print(next(iter(train_data)))
        # print(next(iter(test_data)))

        model = FakeBERT(vocabulary_size=VOCAB_LENGTH, embedding_dimensions=EMB_DIM, cnn_filters=CNN_FILTERS, model_output_classes=OUTPUT_CLASSES, dropout_rate=DROPOUT_RATE)

        if OUTPUT_CLASSES == 2:
            model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        else:
            model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        model.fit(train_data, epochs=NB_EPOCHS)

        y_pred = model.predict(test_data)
        predictions = np.argmax(y_pred, axis=1)

        end_time = time.time()

        acc = accuracy_score(test_df['labels'], predictions)
        accuracy.append(acc)

        micro_prec = precision_score(test_df['labels'], predictions, average='micro')
        precision_micro.append(micro_prec)

        macro_prec = precision_score(test_df['labels'], predictions, average='macro')
        precision_macro.append(macro_prec)

        weighted_prec = precision_score(test_df['labels'], predictions, average='weighted')
        precision_weighted.append(weighted_prec)

        micro_rec = recall_score(test_df['labels'], predictions, average='micro')
        recall_micro.append(micro_rec)

        macro_rec = recall_score(test_df['labels'], predictions, average='macro')
        recall_macro.append(macro_rec)

        weighted_rec = recall_score(test_df['labels'], predictions, average='weighted')
        recall_weighted.append(weighted_rec)

        exec_time = end_time - start_time
        execution_time.append(exec_time)

        print('Test', i, 'Accuracy:', acc)
        print('Test', i, 'Micro Precision:', micro_prec)
        print('Test', i, 'Macro Precision:', macro_prec)
        print('Test', i, 'Weighted Precision:', weighted_prec)
        print('Test', i, 'Micro Recall:', micro_rec)
        print('Test', i, 'Macro Recall:', macro_rec)
        print('Test', i, 'Weighted Recall:', weighted_rec)
        print('Test', i, 'Execution Time:', exec_time)
        print(classification_report(test_df['labels'], predictions))
        print(confusion_matrix(test_df['labels'], predictions))

        results = model.evaluate(test_data)
        print(results)

    print("==========================================================================================")
    print("Accuracy", np.mean(np.array(accuracy)), np.std(np.array(accuracy)))
    print("Micro Precision", np.mean(np.array(precision_micro)), np.std(np.array(precision_micro)))
    print("Macro Precision", np.mean(np.array(precision_macro)), np.std(np.array(precision_macro)))
    print("Weighted Precision", np.mean(np.array(precision_weighted)), np.std(np.array(precision_weighted)))
    print("Micro Recall", np.mean(np.array(recall_micro)), np.std(np.array(recall_micro)))
    print("Macro Recall", np.mean(np.array(recall_macro)), np.std(np.array(recall_macro)))
    print("Weighted Recall", np.mean(np.array(recall_weighted)), np.std(np.array(recall_weighted)))
    print("Execution Time", np.mean(np.array(execution_time)), np.std(np.array(execution_time)))
    print("==========================================================================================")