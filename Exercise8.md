# EXERCISE 8
## OBJECTIVE

Case Study on Text classification on “IMDB movie review” .Create a CNN model to predict if the movie review is positive or negative. Download the  “IMDB movie review” dataset. This dataset has 50k reviews of different movies. It is a benchmark dataset used in text-classification to train and test the Machine Learning and Deep Learning model.
Perform following operations to build the CNN model for text classification
Preprocessing the Data
Input and output label selection
Build the model
Explore Vanishing Gradients
Model compilation
## RESOURCE/REQUIREMENTS
windows operating system ,python-editor/colab, python-interpreter
## PROGRAM LOGIC
1. Load “IMDB movie review” dataset
2. Process the data
	    3. Build the CNN model
	    4. Compile the model
	    5. Explore Vanishing Gradients
	    6. model compilation
## DESCRIPTION / PROCEDURE


Load the dataset click on : https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
Source Code:
```python
import tensorflow as tf
from tensorflow.keras import layers
class CNN:
    __version__ = '0.2.0'

    def __init__(self, embedding_layer=None, num_words=None, embedding_dim=None,
                 max_seq_length=100, kernel_sizes=[3, 4, 5], feature_maps=[100, 100, 100],
                 use_char=False, char_embedding_dim=50, char_max_length=200, alphabet_size=None, char_kernel_sizes=[3, 10, 20],
                 char_feature_maps=[100, 100, 100], hidden_units=100, dropout_rate=None, nb_classes=None):
        """
        Arguments:
            embedding_layer    : If not defined with pre-trained embeddings it will be created from scratch (default: None)
            num_words          : Maximal amount of words in the vocabulary (default: None)
            embedding_dim      : Dimension of word representation (default: None)
            max_seq_length     : Max length of word sequence (default: 100)
            filter_sizes       : An array of filter sizes per channel (default: [3,4,5])
            feature_maps       : Defines the feature maps per channel (default: [100,100,100])
            use_char           : If True, char-based model will be added to word-based model
            char_embedding_dim : Dimension of char representation (default: 50)
            char_max_length    : Max length of char sequence (default: 200)
            alphabet_size      : Amount of differnent chars used for creating embeddings (default: None)
            hidden_units       : Hidden units per convolution channel (default: 100)
            dropout_rate       : If defined, dropout will be added after embedding layer & concatenation (default: None)
            nb_classes         : Number of classes which can be predicted
        """

        # WORD-level
        self.embedding_layer = embedding_layer
        self.num_words       = num_words
        self.max_seq_length  = max_seq_length
        self.embedding_dim   = embedding_dim
        self.kernel_sizes    = kernel_sizes
        self.feature_maps    = feature_maps
        
        # CHAR-level
        self.use_char           = use_char
        self.char_embedding_dim = char_embedding_dim
        self.char_max_length    = char_max_length
        self.alphabet_size      = alphabet_size
        self.char_kernel_sizes  = char_kernel_sizes
        self.char_feature_maps  = char_feature_maps
        
        # General
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.nb_classes   = nb_classes

    def build_model(self):
        """
        Build the model
        Returns:
            Model : Keras model instance
        """

        # Checks
        if len(self.kernel_sizes) != len(self.feature_maps):
            raise Exception('Please define `kernel_sizes` and `feature_maps` with the same amount.')
        if not self.embedding_layer and (not self.num_words or not self.embedding_dim):
            raise Exception('Please define `num_words` and `embedding_dim` if you not using a pre-trained embedding.')
        if self.use_char and (not self.char_max_length or not self.alphabet_size):
            raise Exception('Please define `char_max_length` and `alphabet_size` if you are using char.')

        # Building word-embeddings from scratch
        if self.embedding_layer is None:
            self.embedding_layer = layers.Embedding(
                input_dim    = self.num_words,
                output_dim   = self.embedding_dim,
                input_length = self.max_seq_length,
                weights      = None,
                trainable    = True,
                name         = "word_embedding"
            )

        # WORD-level
        word_input = layers.Input(shape=(self.max_seq_length,), dtype='int32', name='word_input')
        x = self.embedding_layer(word_input)
        
        if self.dropout_rate:
            x = layers.Dropout(self.dropout_rate)(x)
        
        x = self.building_block(x, self.kernel_sizes, self.feature_maps)
        x = layers.Activation('relu')(x)
        prediction = layers.Dense(self.nb_classes, activation='softmax')(x)

        
        # CHAR-level
        if self.use_char:
            char_input = layers.Input(shape=(self.char_max_length,), dtype='int32', name='char_input')
            x_char = layers.Embedding(
                input_dim    = self.alphabet_size + 1,
                output_dim   = self.char_embedding_dim,
                input_length = self.char_max_length,
                name         = 'char_embedding'
            )(char_input)
            
            x_char = self.building_block(x_char, self.char_kernel_sizes, self.char_feature_maps)
            x_char = layers.Activation('relu')(x_char)
            x_char = layers.Dense(self.nb_classes, activation='softmax')(x_char)

            prediction = layers.Average()([prediction, x_char])
            return tf.keras.Model(inputs=[word_input, char_input], outputs=prediction, name='CNN_Word_Char')

        return tf.keras.Model(inputs=word_input, outputs=prediction, name='CNN_Word')

    def building_block(self, input_layer, kernel_sizes, feature_maps):
        """
        Creates several CNN channels in parallel and concatenate them
        Arguments:
            input_layer : Layer which will be the input for all convolutional blocks
            kernel_sizes: Array of kernel sizes (working as n-gram filter)
            feature_maps: Array of feature maps
        Returns:
            x           : Building block with one or several channels
        """
        channels = []
        for ix in range(len(kernel_sizes)):
            x = self.create_channel(input_layer, kernel_sizes[ix], feature_maps[ix])
            channels.append(x)

        # Check how many channels, one channel doesn't need a concatenation
        if (len(channels) > 1):
            x = layers.concatenate(channels)
        
        return x

    def create_channel(self, x, kernel_size, feature_map):
        """
        Creates a layer, working channel wise
        Arguments:
            x           : Input for convolutional channel
            kernel_size : Kernel size for creating Conv1D
            feature_map : Feature map
        Returns:
            x           : Channel including (Conv1D + {GlobalMaxPooling & GlobalAveragePooling} + Dense [+ Dropout])
        """
        x = layers.SeparableConv1D(
            feature_map,
            kernel_size      = kernel_size,
            activation       = 'relu',
            strides          = 1,
            padding          = 'valid',
            depth_multiplier = 4
        )(x)

        x1 = layers.GlobalMaxPooling1D()(x)
        x2 = layers.GlobalAveragePooling1D()(x)
        x  = layers.concatenate([x1, x2])

        x  = layers.Dense(self.hidden_units)(x)
        if self.dropout_rate:
            x = layers.Dropout(self.dropout_rate)(x)
        return x
#install Required modules
!pip3 install utils
import urllib3
import os
import re
import string
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from io import BytesIO
from zipfile import ZipFile
from nltk.corpus import stopwords

def clean_doc(doc):
    """
    Cleaning a document by several methods:
        - Lowercase
        - Removing whitespaces
        - Removing numbers
        - Removing stopwords
        - Removing punctuations
        - Removing short words
    Arguments:
        doc : Text
    Returns:
        str : Cleaned text
    """
    
    #stop_words = set(stopwords.words('english'))
    
    # Lowercase
    doc = doc.lower()
    # Remove numbers
    #doc = re.sub(r"[0-9]+", "", doc)
    # Split in tokens
    tokens = doc.split()
    # Remove Stopwords
    #tokens = [w for w in tokens if not w in stop_words]
    # Remove punctuation
    #tokens = [w.translate(str.maketrans('', '', string.punctuation)) for w in tokens]
    # Tokens with less then two characters will be ignored
    #tokens = [word for word in tokens if len(word) > 1]
    return ' '.join(tokens)

def read_files(path):
    """
    Read in files of a given path.
    This can be a directory including many files or just one file.
    
    Arguments:
        path : Filepath to file(s)
    Returns:
        documents : Return a list of cleaned documents
    """
    
    documents = list()
    # Read in all files in directory
    if os.path.isdir(path):
        for filename in os.listdir(path):
            with open(f"{path}/{filename}") as f:
                doc = f.read()
                doc = clean_doc(doc)
                documents.append(doc)
    
    # Read in all lines in one file
    if os.path.isfile(path):        
        with open(path, encoding='iso-8859-1') as f:
            doc = f.readlines()
            for line in doc:
                documents.append(clean_doc(line))
                
    return documents

def char_vectorizer(X, char_max_length, char2idx_dict):
    """
    Vectorize an array of word sequences to char vector.
    Example (length 15): [test entry] --> [[1,2,3,1,4,2,5,1,6,7,0,0,0,0,0]]
    Arguments:
        X               : Array of word sequences
        char_max_length : Maximum length of vector
        char2idx_dict   : Dictionary of indices for converting char to integer
    Returns:
        str2idx : Array of vectorized char sequences
    """
    
    str2idx = np.zeros((len(X), char_max_length), dtype='int64')
    for idx, doc in enumerate(X):
        max_length = min(len(doc), char_max_length)
        for i in range(0, max_length):
            c = doc[i]
            if c in char2idx_dict:
                str2idx[idx, i] = char2idx_dict[c]
    return str2idx

def create_glove_embeddings(embedding_dim, max_num_words, max_seq_length, tokenizer):
    """
    Load and create GloVe embeddings.
    Arguments:
        embedding_dim : Dimension of embeddings (e.g. 50,100,200,300)
        max_num_words : Maximum count of words in vocabulary
        max_seq_length: Maximum length of vector
        tokenizer     : Tokenizer for converting words to integer
    Returns:
        tf.keras.layers.Embedding : Glove embeddings initialized in Keras Embedding-Layer
    """
    
    print("Pretrained GloVe embedding is loading...")
    
    if not os.path.exists("data"):
        os.makedirs("data")
    
    if not os.path.exists("data/glove"):
        print("No previous embeddings found. Will be download required files...")
        os.makedirs("data/glove")
        http = urllib3.PoolManager()
        response = http.request(
            url     = "http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip",
            method  = "GET",
            retries = False
        )

        with ZipFile(BytesIO(response.data)) as myzip:
            for f in myzip.infolist():
                with open(f"data/glove/{f.filename}", "wb") as outfile:
                    outfile.write(myzip.open(f.filename).read())
                    
        print("Download of GloVe embeddings finished.")

    embeddings_index = {}
    with open(f"data/glove/glove.6B.{embedding_dim}d.txt") as glove_embedding:
        for line in glove_embedding.readlines():
            values = line.split()
            word   = values[0]
            coefs  = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
    
    print(f"Found {len(embeddings_index)} word vectors in GloVe embedding\n")

    embedding_matrix = np.zeros((max_num_words, embedding_dim))

    for word, i in tokenizer.word_index.items():
        if i >= max_num_words:
            continue
        
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return tf.keras.layers.Embedding(
        input_dim    = max_num_words,
        output_dim   = embedding_dim,
        input_length = max_seq_length,
        weights      = [embedding_matrix],
        trainable    = True,
        name         = "word_embedding"
    )

def plot_acc_loss(title, histories, key_acc, key_loss):
    """
    Generate a plot for visualizing accuracy and loss
    Arguments:
        title     : Title of visualization
        histories : Array of Keras metrics per run and epoch
        key_acc   : Key of accuracy (accuracy, val_accuracy)
        key_loss  : Key of loss (loss, val_loss)
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    # Accuracy
    ax1.set_title(f"Model accuracy ({title})")
    names = []
    for i, model in enumerate(histories):
        ax1.plot(model[key_acc])
        ax1.set_xlabel("epoch")
        names.append(f"Model {i+1}")
        ax1.set_ylabel("accuracy")
    ax1.legend(names, loc="lower right")
    
    # Loss
    ax2.set_title(f"Model loss ({title})")
    for model in histories:
        ax2.plot(model[key_loss])
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('loss')
    ax2.legend(names, loc='upper right')
    fig.set_size_inches(20, 5)
    plt.show()

    
def visualize_features(ml_classifier, nb_neg_features=15, nb_pos_features=15):
    """
    Visualize trained coefficient of log regression in respect to vectorizer.
    Arguments:
        ml_classifier   : ML-Pipeline including vectorizer as well as trained model
        nb_neg_features : Number of features to visualize
        nb_pos_features : Number of features to visualize
    """

    feature_names = ml_classifier.get_params()['vectorizer'].get_feature_names()
    coef = ml_classifier.get_params()['classifier'].coef_.ravel()

    print('Extracted features: {}'.format(len(feature_names)))

    pos_coef = np.argsort(coef)[-nb_pos_features:]
    neg_coef = np.argsort(coef)[:nb_neg_features]
    interesting_coefs = np.hstack([neg_coef, pos_coef])

    # Plot
    plt.figure(figsize=(20, 5))
    colors = ['red' if c < 0 else 'green' for c in coef[interesting_coefs]]
    plt.bar(np.arange(nb_neg_features + nb_pos_features), coef[interesting_coefs], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(
        np.arange(nb_neg_features+nb_pos_features),
        feature_names[interesting_coefs],
        size     = 15,
        rotation = 75,
        ha       = 'center'
    );
    plt.show()
#Dataset
DATASET = "imdb_reviews" # datasets: "imdb_reviews"

# WORD-level
MAX_NUM_WORDS  = 15000
EMBEDDING_DIM  = 300
MAX_SEQ_LENGTH = 200
USE_GLOVE      = True
KERNEL_SIZES   = [3,4,5]
FEATURE_MAPS   = [200,200,200]

# CHAR-level
USE_CHAR           = False
ALPHABET           = " abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
ALPHABET_SIZE      = len(ALPHABET)
CHAR_EMBEDDING_DIM = 300
CHAR_MAX_LENGTH    = 2500
CHAR_KERNEL_SIZES  = [5,10,20]
CHAR_FEATURE_MAPS  = [200,200,200]

# GENERAL
DROPOUT_RATE = 0.5
HIDDEN_UNITS = 250
NB_CLASSES   = 2

# LEARNING
BATCH_SIZE = 100
NB_EPOCHS  = 10
RUNS       = 5
VAL_SIZE   = 0.2
import tensorflow_datasets as tfds
from utils import clean_doc

# Load train dataset
train = tfds.as_numpy(tfds.load(DATASET, data_dir=f"data/{DATASET}", split="train", batch_size=-1))
X_train, y_train = [clean_doc(x.decode()) for x in train["text"]], train["label"]

# Load test dataset
test = tfds.as_numpy(tfds.load(DATASET, data_dir=f"data/{DATASET}", split="test", batch_size=-1))
X_test, y_test = [clean_doc(x.decode()) for x in test["text"]], test["label"]

print(f"Train samples: {len(X_train)}")
print(f"Test samples:  {len(X_test)}")
# Preprocessing for word-based model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
plt.style.use('seaborn')

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(X_train)

sequences  = tokenizer.texts_to_sequences(X_train)
word_index = tokenizer.word_index
result     = [len(x.split()) for x in X_train]

# Plot histogram
plt.figure(figsize=(20,5))
plt.title("Document length")
plt.hist(result, 200, density=False, range=(0,np.max(result)))
plt.show()

print("Text informations:")
print(f" - max length:   {np.max(result)}")
print(f" - min length:   {np.min(result)}")
print(f" - mean length:  {np.mean(result)}")
print(f" - limit length: {MAX_SEQ_LENGTH}")

# Padding all sequences to same length of `MAX_SEQ_LENGTH`
word_data = tf.keras.preprocessing.sequence.pad_sequences(
    sequences,
    maxlen  = MAX_SEQ_LENGTH,
    padding = 'post'
)
#Preprocessing for char-based model
from utils import char_vectorizer
if USE_CHAR:
    char2idx_dict = {}
    idx2char_dict = {}

    for idx, char in enumerate(ALPHABET):
        char2idx_dict[char] = idx + 1

    idx2char_dict = dict([(i+1, char) for i, char in enumerate(char2idx_dict)])
    
    # Get informations about char sequence length
    result = [len(x) for x in X_train]
    plt.figure(figsize=(20,5))
    plt.title("Char length")
    plt.hist(result, 200, density=False, range=(0,np.max(result)))
    plt.show()
    print("Text informations:")
    print(f" - max:   {np.max(result)}")
    print(f" - min:   {np.min(result)}")
    print(f" - mean:  {np.mean(result)}")
    print(f" - limit: {CHAR_MAX_LENGTH}") 
    char_data = char_vectorizer(X_train, CHAR_MAX_LENGTH, char2idx_dict)
#Train Model
from sklearn.model_selection import train_test_split
from utils import create_glove_embeddings
histories = []
for i in range(RUNS):
    print(f"Running iteration {i+1}/{RUNS}")
    random_state = np.random.randint(1000)
    
    _X_train, _X_val, _y_train, _y_val = train_test_split(
        word_data, 
        tf.keras.utils.to_categorical(y_train),
        test_size    = VAL_SIZE, 
        random_state = random_state
    )
    if USE_CHAR:
        _X_train_c, _X_val_c, _, _ = train_test_split(
            char_data,
            tf.keras.utils.to_categorical(y_train),
            test_size    = VAL_SIZE,
            random_state = random_state
        )
        
        _X_train = [_X_train, _X_train_c]
        _X_val   = [_X_val,   _X_val_c]
   emb_layer = None
    if USE_GLOVE:
        emb_layer = create_glove_embeddings(
            embedding_dim  = EMBEDDING_DIM,
            max_num_words  = MAX_NUM_WORDS,
            max_seq_length = MAX_SEQ_LENGTH,
            tokenizer      = tokenizer       )
    
    model = CNN(
        embedding_layer    = emb_layer,
        num_words          = MAX_NUM_WORDS,
        embedding_dim      = EMBEDDING_DIM,
        kernel_sizes       = KERNEL_SIZES,
        feature_maps       = FEATURE_MAPS,
        max_seq_length     = MAX_SEQ_LENGTH,
        use_char           = USE_CHAR,
        char_embedding_dim = CHAR_EMBEDDING_DIM,
        char_max_length    = CHAR_MAX_LENGTH,
        alphabet_size      = ALPHABET_SIZE,
        char_kernel_sizes  = CHAR_KERNEL_SIZES,
        char_feature_maps  = CHAR_FEATURE_MAPS,
        dropout_rate       = DROPOUT_RATE,
        hidden_units       = HIDDEN_UNITS,
        nb_classes         = NB_CLASSES
    ).build_model()
    
    model.compile(
        loss      = "categorical_crossentropy",
        optimizer = tf.optimizers.Adam(),
        metrics   = ["accuracy"]
    )
    
    history = model.fit(
        _X_train, _y_train,
        epochs          = NB_EPOCHS,
        batch_size      = BATCH_SIZE,
        validation_data = (_X_val, _y_val),
        callbacks       = [tf.keras.callbacks.ModelCheckpoint(
            filepath       = f"model-{i+1}.h5",
            monitor        = "val_loss",
            verbose        = 1,
            save_best_only = True,
            mode           = "min"
        )]
    )

    histories.append(history.history)
def get_avg(histories, his_key):
    tmp = []
    for history in histories:
        tmp.append(history[his_key][np.argmin(history['val_loss'])])
    return np.mean(tmp)
print(f"Training: \t{get_avg(histories,'loss')} loss / {get_avg(histories,'accuracy')} acc")
print(f"Validation: \t{get_avg(histories,'val_loss')} loss / {get_avg(histories,'val_accuracy')} acc")
from utils import plot_acc_loss
plot_acc_loss('training', histories, 'accuracy', 'loss')
plot_acc_loss('validation', histories, 'val_accuracy', 'val_loss')
sequences_test = tokenizer.texts_to_sequences(X_test)
X_test_word    = tf.keras.preprocessing.sequence.pad_sequences(
    sequences_test,
    maxlen  = MAX_SEQ_LENGTH,
    padding = 'post'
)
if USE_CHAR:
    X_test_word = [X_test_word, char_vectorizer(X_test, CHAR_MAX_LENGTH, char2idx_dict)]
else:
    X_test_word = X_test_word
import cnn_model
test_loss = []
test_accs = []
for i in range(0, RUNS):
    cnn_ = tf.keras.models.load_model(f"model-{i+1}.h5")
    score = cnn_.evaluate(X_test_word, tf.keras.utils.to_categorical(y_test), verbose=0)
    test_loss.append(score[0])
    test_accs.append(score[1])
    print(f"Running test with model {i+1}: {score[0]} loss / {score[1]} acc")
    
print(f"\nAverage loss / accuracy on testset: {np.mean(test_loss)} loss / {np.mean(test_accs)} acc")
print(f"Standard deviation: (+-{np.std(test_loss)}) loss / (+-{np.std(test_accs)}) acc")
from sklearn.feature_extraction.text import TfidfVectorizer

tf_idf = TfidfVectorizer(
    ngram_range  = (1,2),
    min_df       = 0.0005,
    max_df       = 0.1,
    lowercase    = False,
    preprocessor = None,
    sublinear_tf = True,
    use_idf      = True,
)
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier(
    max_iter            = 1000,
    tol                 = 1e-3,
    validation_fraction = 0.2
)
from sklearn.pipeline import Pipeline

ml_classifier = Pipeline([
    ('vectorizer', tf_idf),
    ('classifier', sgd)
])

X_train, _X_val, _y_train, _y_val = train_test_split(X_train, y_train, test_size=VAL_SIZE)
ml_classifier.fit(_X_train, _y_train)
val_pred = ml_classifier.predict(_X_val)
print(f"Accuracy: {np.mean(_y_val==val_pred)}")

from utils import visualize_features
visualize_features(ml_classifier)
preds = ml_classifier.predict(X_test)
print(f"Accuracy: {np.mean(y_test==preds)}")
```


Output:
```
Running test with model 1: 0.11171921902698906 loss / 0.9570000171661377 acc
Running test with model 2: 0.11396337144527781 loss / 0.9580526351928711 acc
Running test with model 3: 0.11552080318390538 loss / 0.9570000171661377 acc
Running test with model 4: 0.11119069827779343 loss / 0.957447350025177 acc
Running test with model 5: 0.11308792875589509 loss / 0.9575263261795044 acc

Average loss / accuracy on testset: 0.11309640413797215 loss / 0.9574052691459656 acc
Standard deviation: (+-0.0015593439988272734) loss / (+-0.0003908878716174513) acc
________________________________________Accuracy: 0.9398947368421052
```

