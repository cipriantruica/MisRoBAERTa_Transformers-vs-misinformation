# coding: utf-8

__author__ = "Ciprian-Octavian Truică"
__copyright__ = "Copyright 2021, University Politehnica of Bucharest"
__license__ = "GNU GPL"
__version__ = "0.1"
__email__ = "ciprian.truica@upb.ro"
__status__ = "Production"

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import sys
import time

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

trans_model = {
    "bert": ["bert-base-cased", "bert-large-cased"],
    "xlnet": ["xlnet-base-cased", "xlnet-large-cased"],
    "roberta": ["roberta-base", "roberta-large", "distilroberta-base"],
    "distilbert": ["distilbert-base-cased"],
    "electra": ["google/electra-base-discriminator", "google/electra-large-discriminator"],
    "albert": ["albert-base-v1", "albert-base-v2", "albert-xxlarge-v1", "albert-xxlarge-v2"],
    "xlm": ["xlm-mlm-100-1280"],
    "xlmroberta": ["xlm-roberta-base", "xlm-roberta-large" ],
    "bart": ["facebook/bart-base", "facebook/bart-large"],
    "deberta": ["microsoft/deberta-base", "microsoft/deberta-large"]
}

if __name__ == "__main__":
    FILE_NAME = sys.argv[1]
    MODEL_TYPE = sys.argv[2]
    NO_GPU = int(sys.argv[3])
    NO_TESTS = int(sys.argv[4])

    dataSet = pd.read_csv(FILE_NAME, encoding = "utf-8")

    print(dataSet)
    dataSet['labels'] = 0
    for elem in classes:
        dataSet.loc[dataSet['label'] == elem, 'labels'] = classes[elem]

    dataSet.drop(['label'], axis=1, inplace=True)
    dataSet.rename(columns = {'content': 'text'}, inplace = True)

    model_args = ClassificationArgs()

    model_args.early_stopping_delta = 0.01
    model_args.early_stopping_consider_epochs = True
    model_args.early_stopping_metric = "eval_loss"
    model_args.early_stopping_metric_minimize = True
    model_args.early_stopping_patience = 5
    model_args.use_early_stopping = True
    model_args.evaluate_during_training_steps = 10000
    model_args.num_train_epochs = 10
    model_args.overwrite_output_dir = True
    model_args.save_eval_checkpoints = False
    model_args.save_model_every_epoch = False

    for model_name in trans_model[MODEL_TYPE]:
        model_args.output_dir = "outputs_" + model_name + "/"
        accuracy = []
        precision_micro = []
        precision_macro = []
        precision_weighted = []
        recall_micro = []
        recall_macro = []
        recall_weighted = []
        execution_time = []
        for i in range(NO_TESTS):
            print(MODEL_TYPE, model_name)
            train_df, test_df = train_test_split(dataSet, test_size=0.2, stratify=dataSet['labels'])
            print(train_df.shape)
            print(test_df.shape)

            start_time = time.time()

            model = ClassificationModel(model_type=MODEL_TYPE, model_name=model_name, use_cuda=True, cuda_device=NO_GPU, num_labels=10, args=model_args)
            model.train_model(train_df)
            result, model_outputs, wrong_preds = model.eval_model(test_df)

            end_time = time.time()

            predictions = []
            for x in model_outputs:
                predictions.append(np.argmax(x))

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

            print(model_name, 'Test', i, 'Accuracy:', acc)
            print(model_name, 'Test', i, 'Micro Precision:', micro_prec)
            print(model_name, 'Test', i, 'Macro Precision:', macro_prec)
            print(model_name, 'Test', i, 'Weighted Precision:', weighted_prec)
            print(model_name, 'Test', i, 'Micro Recall:', micro_rec)
            print(model_name, 'Test', i, 'Macro Recall:', macro_rec)
            print(model_name, 'Test', i, 'Weighted Recall:', weighted_rec)
            print(model_name, 'Test', i, 'Execution Time:', exec_time)
        print("==========================================================================================")
        print(model_name, "Accuracy", np.mean(np.array(accuracy)), np.std(np.array(accuracy)))
        print(model_name, "Micro Precision", np.mean(np.array(precision_micro)), np.std(np.array(precision_micro)))
        print(model_name, "Macro Precision", np.mean(np.array(precision_macro)), np.std(np.array(precision_macro)))
        print(model_name, "Weighted Precision", np.mean(np.array(precision_weighted)), np.std(np.array(precision_weighted)))
        print(model_name, "Micro Recall", np.mean(np.array(recall_micro)), np.std(np.array(recall_micro)))
        print(model_name, "Macro Recall", np.mean(np.array(recall_macro)), np.std(np.array(recall_macro)))
        print(model_name, "Weighted Recall", np.mean(np.array(recall_weighted)), np.std(np.array(recall_weighted)))
        print(model_name, "Execution Time", np.mean(np.array(execution_time)), np.std(np.array(execution_time)))
        print("==========================================================================================")
