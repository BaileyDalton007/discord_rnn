print('Loading Model')

# Keeps Tensorflow from logging into the console
from os import environ
from sys import exit

environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.layers import Embedding, LSTM, Dense, Activation
from tensorflow.keras.models import Sequential

from gensim.models.word2vec import Word2Vec as w2v

import numpy as np
# Gets rid of "log divide by zero" error in console
np.seterr(divide = 'ignore') 

word_model = w2v.load('word2vec_100.model')

# Whether or not to print model input to the console
VERBOSE = False

TEMP = 0.7

# Defining some variables and functions to interact with the w2v model
pretrained_weights = word_model.wv.vectors
vocab_size, embedding_size = pretrained_weights.shape

def word2idx(word):
  return word_model.wv.key_to_index[word]
def idx2word(idx):
  return word_model.wv.index_to_key[idx]

# Load RNN model
model = Sequential()

# Embedding layer for w2v model
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[pretrained_weights]))

model.add(LSTM(units=embedding_size))
model.add(Dense(units=vocab_size))
model.add(Activation('softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Change to path of model that should be loaded
model.load_weights('./simple_model_2/model.ckpt')

def sample(preds, temperature=1.0):
  if temperature <= 0:
    return np.argmax(preds)

  preds = np.asarray(preds).astype('float64')
  preds = np.log(preds) / 0.7
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)
  return np.argmax(probas)

# Generates next num_generated words from the text
def generate_next(text, num_generated=10):
  word_idxs = [word2idx(word) for word in text.lower().split()]

  for i in range(num_generated):
    x = np.zeros([1, 20])
    for i, idx in enumerate(word_idxs):
      x[0, i] = idx

    prediction = model.predict(x, verbose=0)

    idx = sample(prediction[-1], temperature=TEMP)
    word_idxs.append(idx)

  return ' '.join(idx2word(idx) for idx in word_idxs)

def generation_handler(input):
    words = input.split()

    model_input_str = ""

    for word in words:
        if word in word_model.wv.key_to_index:
            model_input_str += f'{word} '
    
    if VERBOSE:
        print(f'What the model sees: {model_input_str}')

    print(generate_next(model_input_str))

def command_handler(input):
    if '.help' in input:
        print("=============================================================================================")
        print("List of Commands:")
        print(" .help: this, duh")
        print(" .verbose: how much output is given for a prompt")
        print(" .temp <num>: change sampling temperature to <num> parameter, amount of variation in output")
        print(" .stop: stop the program")
        print(" ")
        print("See README for more comprehensive list of commands")
        print("=============================================================================================")

    elif '.verbose' in input:
        global VERBOSE

        if VERBOSE == True:
            print('Will no longer show model input')
            VERBOSE = False
        else:
            print('Will now show filtered input to the model')
            VERBOSE = True

    elif '.temp' in input:
        global TEMP
        try:
            TEMP = float(input.split()[1])
            print('Sampling temperature is now ' + str(TEMP))

        except:
            print('Be sure to use correct syntax, ex: .temp 0.7')

    elif '.stop' in input:
        exit()
    
    else:
        generation_handler(input)
        
####################################################################################

print('DISCLAIMER: this model was trained on messages from many different servers, some of which are fairly unmoderated.\n I am in no way endorsing or condoning messages from this model. Thanks!')
print('Model Loaded')

print('Type a command or prompt to get started (type .help for help)')

while True:
    prompt = input('> ')

    command_handler(prompt)