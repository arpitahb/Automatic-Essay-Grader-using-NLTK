# Automatic-Essay-Grader-using-NLTK
 ## Table of Content
 + [Dataset Explaing](#data)
 
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

## Preprocessing
 ### Dependencies needed
```
pip install numpy
pip install pandas
pip install nltk
pip install gensim
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
