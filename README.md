
## MisRoBÆRTa: Transformers versus Misinformation


### Deployment: 

Python v3.7.x

Packages:
* numpy
* pandas
* matplotlib
* scikit-learn
* simpletransformers
* sentence-transormers
* keras
* tensorflow
* pytorch
* bert-for-tf2 
* tensorflow-hub
* transformers

### To run MisRoBÆRTa use "FakeBERT.py" script as follows:

`python MisRoBÆRTa.py FILE_NAME USE_CUDA NO_TESTS`

Where:
* FILE_NAME - is the a csv file with 2 columns: content and label
* USE_CUDA - 0 - False, 1 - True
* NO_TESTS - how many test to perform

### To run the transformers classificatin use "transfs_misinformation.py" script as follows:

python -u transfs_misinformation.py FILE_NAME MODEL_TYPE NO_GPU NO_TESTS  

Where:
* FILE_NAME - is the a csv file with 2 columns: content and label
* MODEL_TYPE - the transformer model name given with lowercase, e.g., bart
* NO_GPU - the GPU to run the code on
* NO_TESTS - how many test to perform


### To run the BART models either replace or copy the marked code in the "classification_model.py" and "bart_model.py" as follows:

For "classification_model.py" there are new lines added to this file

Place it here to overwrite:

` $PYTHON_HOME/lib/python3.7/site-packages/simpletransformers/classification/`

We higly recommand to add the lines marked with " # line for BART " in the existing "classification_model.py" file, and not overwrite the file.

For "bart_model.py" place this file here:

` $PYTHON_HOME/lib/python3.7/site-packages/simpletransformers/classification/transformer_models/`

### To run FakeBERT use "FakeBERT.py" script as follows:

`python FakeBERT.py FILE_NAME NO_TESTS`

Where:
* FILE_NAME - is the a csv file with 2 columns: content and label
* NO_TESTS - how many test to perform

