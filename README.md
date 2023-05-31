# Automatic-Essay-Grader-using-NLTK
 ## Table of Content
 + [Dataset Explaing](#data)
 + [Preprocessing](#prep)
 + [Building Model](#build)
 + [Test the Lambda Function](#test)
 
 ## Dataset Explaining <a name="data"></a>
 
 Training data is provided for each essay prompt.The number of training essays does vary.  For example, the lowest amount of training data is 1,190 essays, randomly selected from a total of 1,982.  The data will contain ASCII formatted text for each essay followed by one or more human scores, and (where necessary) a final resolved human score.
There are a total of 10 Datasets and each dataset has its own scoring guide. There are two established score point metrics used :

### Writing Applications Rubrics:

Score Point 1: An underdeveloped response that may take a position but offers no more than very minimal support

Score Point 2: An underdeveloped response that may or may not take position.

Score Point 3: A minimally developed response that may take position but with inadequate supports and details

Score Point 4: A somewhat developed response that takes a position and provides adequate support.

Score Point 5: A developed response that takes a clear position and offers reasonably persuasive support.

Score Point 6: A well developed response that takes a clear, thoughtful position and offers persuasive support

### Language Conventions Rubrics:

Score Point 1:Errors are serious and numerous. The reader may need to stop and reread parts of the sample and may struggle to discern the writer's meaning.

Score Point 2:Errors are typically frequent and may occasionally impede the flow of communication.

Score Point 3: Errors are occasional and often of the first draft variety.They have minor impact on the flow of communication.

Score Point 4: There are no errors that impacts the flow of communication

## Preprocessing <a name="prep"></a>
 ### Dependencies needed
```
pip install numpy
pip install pandas
pip install nltk
pip install scikit-learn
```
1. We first load the [training_set_rel3.tsv](Dataset/training_set_rel3.tsv) using pandas library. The columns containing null values are dropped and the dataset is cleaned to contain columns : essay_id, essay_set, essay, domain_score
2. The column "essay" contains the training essays. These essays contain a lot of stopwords and special words. Firstly, we remove the stopwords using the following code 
```
nltk.download('stopwords')
words=set(stopwords.words("english"))
def remove_stopwords(essay):
  wtoken= word_tokenize(essay)
  filter=[]
  for w in wtoken:
    if w not in words:
      filter.append(w)
      return ' '.join(filter)
 ```
 The remove_stopwords function takes each essay cell as input.
 word_tokenize function splits each sentence in an essay into words using the NLTK library and stores it in the variable wtoken.
 A for loop checks if the words are stopwords. The words which are not stopwords are stored in a list "filter"
 We then remove all punctuation using the following code:
 ```
def remove_puncs(essay):
    essay = re.sub("[^A-Za-z ]","",essay)
    return essay
 ```
 The re python package is used. re.sub is used to remove all the punctuations.
 
 We then remove all the special words starting with @ using th code 
  ```
  def clean_essay(essay):
    x=[]
    for i in essay.split():
        if i.startswith("@"):
            continue
        else:
            x.append(i)
    return ' '.join(x)
   ```  
   3. Once the dataframe is cleaned, the next step is to extract the features from the essays.
   The features that we have extracted are :
   * Number of words in a essay 
   ```  
   def noOfWords(essay):
    count=0
    for i in essay2word(essay):
        count=count+len(i)
    return count
 ```
 * Number of characters in a essay:
  ```
  def noOfChar(essay):
    count=0
    for i in essay2word(essay):
        for j in i:
            count=count+len(j)
    return count
  ```
  * The average word length in a essay:
  ```
  def avg_word_len(essay):
    return noOfChar(essay)/noOfWords(essay)
  ```
 * The number os sentences in anessay
  ```
  def noOfSent(essay):
    return len(essay2word(essay))
  ```
 * The number of verbs, nouns, adjectives and adverbs in an essay
  ```
  def count_pos(essay):
    sentences = essay2word(essay)
    noun_count=0
    adj_count=0
    verb_count=0
    adverb_count=0
    for i in sentences:
        pos_sentence = nltk.pos_tag(i)
        for j in pos_sentence:
            pos_tag = j[1]
            if(pos_tag[0]=='N'):
                noun_count+=1
            elif(pos_tag[0]=='V'):
                verb_count+=1
            elif(pos_tag[0]=='J'):
                adj_count+=1
            elif(pos_tag[0]=='R'):
                adverb_count+=1
    return noun_count,verb_count,adj_count,adverb_count
  ```
  * The number of spelling errors in an essay
  ```
  data = open(r'C:\Users\JEC\Desktop\6th Semester\big.txt').read()
words = re.findall('[a-z]+', data.lower())

def check_spell_error(essay):
    essay=essay.lower()
    new_essay = re.sub("[^A-Za-z0-9]"," ",essay)
    new_essay = re.sub("[0-9]","",new_essay)
    count=0
    all_words = new_essay.split()
    for i in all_words:
        if i not in words:
            count+=1
    return count
  ```
  * the average sentence length in an essay
   ```
   def avg_sent_len(essay):
    return noOfWords(essay)/noOfSent(essay)
  ```
  4. We create a csv file by applying the feature extracting functions to the dataframe
   ```
   dat=df.copy()
   dat["wordcount"]= dat["essay"].apply(noOfWords)
dat["charcount"]= dat["essay"].apply(noOfChar)
dat["avglen"]= dat["essay"].apply(avg_word_len)
dat["sentencecount"]= dat["essay"].apply(noOfSent)
dat["spellerror"]= dat["essay"].apply(check_spell_error)
dat["avg_sent"]=dat["essay"].apply(avg_sent_len)
dat['noun_count'], dat['adj_count'], dat['verb_count'], dat['adv_count'] = zip(*dat['essay'].map(count_pos))
dat.to_csv("processed_data.csv")
   
  ```
 The csv file is saved as Processed_data.csv
  
  ## Building Model <a name = "build"></a>
  
  ### Dependencies
   ```
   pip install numpy
   pip install pandas
   pip install gensim
   pip install keras
   pip install scikit-learn
   ```
   1. We load the dataset "training_set_rel3.tsv" and the csv file we created after preprocessing the dataset, "Processed_data.csv". Adding the domain_score column from the Processed_data to our dataset, we created X set and Y set  where Y constituted of all the cells in the "domain_score" column and X constituted of the dataframe excluding the "domain_score" column
   2. The training set and testing set are built from the X and Y set using code 
    ```
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42
  ```
  The train_test_split function splits the dataset into random train and test sets.
  3. The essays in the set X_train and X_test are converted to lists and each essay in the list is further converted to lists comprising of split words.
  4. This list of words is fed to a word2vec model which converts it to vectors using code 
  ```
  def makeVec(words, model, num_features):
    vec = np.zeros((num_features,),dtype="float32")
    noOfWords = 0.
    index2word_set = set(model.wv.index_to_key)   #new attribute index_to_key
    for i in words:
        if i in index2word_set:
            noOfWords += 1
            vec = np.add(vec,model.wv[i])   #word fetch in model
            np.seterr(invalid='ignore')

    vec = np.divide(vec,noOfWords)
    return vec
  ```
 5. The vectors are added to list "training_vectors" and "testing_vectors". Since the LSTM model takes input having 3 dimensions, we convert our input by reshaping it to 3 dimensions
 ```
 training_vectors = np.array(training_vectors)
testing_vectors = np.array(testing_vectors)

# Reshaping train and test vectors to 3 dimensions. (1 represnts one timestep)
training_vectors = np.reshape(training_vectors, (training_vectors.shape[0], 1, training_vectors.shape[1]))
testing_vectors = np.reshape(testing_vectors, (testing_vectors.shape[0], 1, testing_vectors.shape[1]))
lstm_model = get_model()
 ```
 6. Our LSTM model has 4 layers: LSTM, LSTM_1, Dropout and Dense. We feed this model our training set and train it using code
 ```
 lstm_model.fit(training_vectors, y_train, batch_size=40, epochs=50)
 ```
 Once our model is trained, we save it in a H5 file.

## Test Lambda Function<a name="test"></a>
 
 ### Dependencies
 You need the pip install all the files mentioned earlier for the model
```
 import json
 import site
 import numpy as np
 import pandas as pd
 import nltk
 import re
 from nltk.corpus import stopwords
 from nltk.tokenize import sent_tokenize, word_tokenize
 import gensim.models.word2vec
 from keras.layers import Embedding, LSTM, Dense, Dropout, Lambda, Flatten
 from keras.models import Sequential, load_model, model_from_config
 import keras.backend as K
 from sklearn.model_selection import train_test_split
 from sklearn.metrics import mean_squared_error
 from sklearn.metrics import cohen_kappa_score
 from gensim.models.keyedvectors import KeyedVectors
 from keras import backend as K
```

###Lambda Function
```
def lambda_handler(event, context):

    try:
        result = convertToVec(event['body'])
        return {
            'statusCode': 200,
            'body': json.dumps({'SCORE': result})
            # 'headers': {'Content-Type': 'application/json'}

        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

1. app.py file shows the lambda function 'lambda_handler' that'll execute the model file 'word2vec.bin' and 'final_model.h5'.
2. The test handler file handler.py will import the event.json for the input and the app.py for the lambda function. This is the main test function which you need to run in your platform. The essay you'll input can be editted in the event.json.
