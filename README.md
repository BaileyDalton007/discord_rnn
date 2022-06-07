
# Discord Neural Network

This project uses downloaded discord chats to train a neural network to respond to messages.



## Background

Discord is a messaging application that utilizes two main forms of communications, direct and server based messaging. 

![discord_gif](https://user-images.githubusercontent.com/59097689/171918248-2f74dcd9-b56b-4868-a22f-2d25bb656333.gif)


I have used Discord for well over four years, so I thought that my messages over that time would make an interesting dataset to train a neural network on.




## Project Goals
- Download and parse Discord chat logs.
    - Formed dataset into csv's in a few different ways bringing about differing results.
- Use a word embedding model to encode the messages to feed to the neural network for training.
- Train a neural network to give a semi-comprehensible output response message.

## Personal Goals
- To learn about and implement a word embedding model.
- To learn and experiment with how different deep learning models handle textual input and generation.
- To document my method and findings throughly through this write-up.

## Methodology
The first problem to tackle was categorical encoding, more specifically, generating word embeddings.

#### Word Embeds
I decided to use word vectorization to perserve the meaning and 
context of words. See [References](#References) for a comprehensive
guide for understanding word vectorization. 

After downloading direct message logs and server channels from 
Discord using an external program (see [References](#References)),
they came in the following csv format.

```csv
             AuthorID         Author                Date                                            Content Attachments Reactions
0  806534603145871401   baileyD#0000  Feb-03-21 09:44 AM  wanted to get a diff account for myo friends a...         NaN       NaN
1  293462127799173121  KibblesK#0000  Feb-03-21 09:44 AM                                            Lmao ok         NaN       NaN
2  806534603145871401   baileyD#0000  Feb-03-21 09:44 AM         so i dont need to keep switchging accounts         NaN       NaN
3  293462127799173121  KibblesK#0000  Feb-03-21 09:44 AM                                             Gotcha         NaN       NaN
4  806534603145871401   baileyD#0000  Feb-10-21 09:57 AM            i completely locked myself out of my...         NaN       NaN
```

Of course most of this data will not be needed for the word
embedding model, so I wrote `discord_export_w2v_training.py` to extract just the message
content. See [Usage](#Usage) for instructions on using the scripts in this repositiory.

The output of running the script:
```
Done with 369906 of lines checked and 116283 of lines output!
```
This means we will have `116,283` messages to train our w2v model on.


Now we can feed that training data to a Word2Vec model.
I decided to create a seperate model for word embedding instead of 
just adding an embedding layer into the neural network so I could
explore them both independently of one another.
See `discord_rnn_word_embeddings.ipynb`
for implementation. After tokenizing the training data, I passed
it to the model with the insructions to only use words that are in
the dataset atleast twice, to generate a vector of 100 dimensions
for each word in the vocabulary, and to have a context window of 5.
My hope is that raising `min_count` could help filter out typos.
```python
# Train Word2Vec model, may take a bit
model = w2v(msg_array, size=100, min_count=2, window=5, iter=100)
    > Word2Vec(vocab=18935, size=100, alpha=0.025)
```

We can see that the model generated vectors for a vocabulary of ~19000.
The whole point of using a Word2Vec model is to preserve word context
in N dimensional space, which we can see using Cosing Similarity.
```python
model.wv.most_similar('knee')
    > [
        ('stomach', 0.6018036603927612),
        ('neighborhood', 0.566454291343689),
        ('wardrobe', 0.5427192449569702),
        ('ear', 0.5401051044464111),
        ('bankai', 0.5262097716331482),
        ('uncle', 0.513681173324585),
        ('pockets', 0.5117610692977905),
        ('lap', 0.5110880136489868),
        ('leg', 0.5026416778564453),
        ('blanket', 0.5007407665252686)
      ]
```
Of course, this model is flawed, but we can see that a few other
similar body parts to `knee` have been listed, such as `stomach`,
`ear`, `lap` and `leg`. There is still work to be done, but we can
begin to see glimpses of intelligence. Lets save our word embedding
model and move onto the generation neural network.



And here is a PCA reduced representation of the word vectors, colored
by their number of uses, see `w2v_plot.html` for interactive version:


Here is a top-level diagram of the process I have described thus far:
![image](https://user-images.githubusercontent.com/59097689/171921352-c09c7cd8-01aa-42c8-8d84-0eaa1513a4dd.png)



#### Dense Neural Network Approach
Disclaimer, this method went about as well as you could expect,
horribly horrible in just about every way. I was curious what
would happen if I tried using a DNN on a task so obviously suited
for a RNN. The output I got was incoherent and in the next section
I will be implementing it correctly. I learned so much from this
though and still find the results quite interesting, but if you
have no interest in what not to do, go ahead and skip to the
Recurrent Neural Network section.

##### Preparing Training Data

With this approach, I was trying to make this a classification task,
despite it very clearly not being one. I decided to take each message
I sent from the downloaded data, and extract the last three messages
before for context to pass into the model. The model would (theoretically)
use the input messages to classify a type of response. Once again, I
know this will not work well, but I was curious nonetheless.

Here is the output from the data preperation script `discord_export_cleanup.py`
Notice: script will no longer output this as it has been changed for
the format of RNN input, so if for some reason you want to replicate
this, look back in commit history.
```
Tmsg0,  Tmsg1,            Tmsg2,             Umsg
Ahh,    or  notif,        no its only ping,  allo
Okay,   But ... anymore,  I have ... mins,   Are ... tournements
```

The last column, `Umsg` is the user sent message, in this case, what I sent.
The other 3 columns, `Tmsg0, Tmsg1, Tmsg2` are input messages 1-3.


![image](https://user-images.githubusercontent.com/59097689/172444544-2d4e6971-e264-4945-a849-cb1b6d878868.png)

The output of `discord_export_cleanup.py` tells us that we have `6353` rows of training data
to train our model on, which is quite low. That is another problem that will be fixed in the next
section.


Before messages can be passed to the model, they need to be
padded to a uniform length, tokenized, and made into word vectors,
which is all handled in the Data Preprocessing section of `discord_dnn_model.inpynb`.

##### Training the Model

After preprocessing we know the input and output shape of our model.
```
IN: (VECTOR_DIM, WORD_COUNT, DEPTH)
OUT: (VECTOR_DIM, WORD_COUNT)
```
In this case, we have 100 dimensional word vectors, padded our messages
to 20 words, and passed in 3 messages for training. So our shapes are:

```
IN: (100, 20, 3)
OUT: (100, 20)
```

Now to the architecture of the model. After seeing the output of
the DNN, I knew there was no saving it, it was a fundementally
flawed approach, so I did not spend much time experimenting with
the model architecture. Here is the version defined in `discord_dnn_model.ipynb`:
```python
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(512, activation='relu', input_shape=(3, WORD_COUNT, VECTOR_DIM)))
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(2000, activation='relu'))

model.add(tf.keras.layers.Reshape((WORD_COUNT, VECTOR_DIM)))
```
Not much interesting going on here, just 2 dense layers followed by
a flatten and another dense layer, before being reshaped to the
format of an output message.

I decided to train the model for 150 epochs, even though accuracy
flattens out around 15. I did this for no good reason, was just curious
what effects training that long would have on a model using a relatively small
dataset.

Here are performance graphs trained on 15 epochs:

![image](https://user-images.githubusercontent.com/59097689/172443611-6b2011dc-e3ed-40e9-89af-f02beec990fe.png)

And here are the same graphs trained on 150 epochs:

![image](https://user-images.githubusercontent.com/59097689/172443411-c2a6cc65-4867-4c2c-92c0-c172614aa54a.png)

For fun here is some of the output I got:
```
input_msgs = [['okay so first iteration of neural network is just mid stroke'],
              ['woah thats really sick'],
              ['huh it came out']]

> ['but', 'sourmatt', '🤔', 'exam', 'has', 'a', 'like', 'lime', 'vanguard', 'to', 'excited',
    'sync', 'grounded', 'consist', 'dias', 'channels', 'busy', 'and', 'because', 'more']
```
Here I got my first somewhat relevant response, as the first 3 words
of the output are a geniune response to the input messages.
```
input_msgs = [['Also this is news to me but snoop dogg has the record now for most solo
                kills in warzone'],
              ['1 game'],
              ['how many']]
> ['obv', 'enemies', 'inactive', 'done', 'argument', 'vocabulary', '“the', 'prescribed',
    'because', 'nearby', 'and', 'rough', 'nebula', 'current', 'advantage', 'ingrained',
     'that', 'hassle', 'because', 'unrelated']
```

Intelligence fades just as quickly as it appears, seems as though that
last one was a fluke.
```
input_msgs = [['valorant is my fav game known to man'],
              ['bro aint no way you like valorant more than minecraft you monkey'],
              ['i have the game taste of an infant child']]
> ['i', 'yea', 'recoil', 'tryouts', 'issues', 'reacting', 'that', 'recording', 'past',
    'yea', 'since', 'that', 'experience', 'some', 'rough', 'shown', 'obv', 'a',
    'because', 'since']
              
```
The outputs are just as I expected, comically bad. As I stated before,
I tried to treat this as a classification problem when it clearly wasn't
and the effects of that can be seen in the horrendous accuracy and loss
graphs shown above.

#### Recurrent Neural Network Approach
placeholder

## Usage

```python
placeholder
```


## References

 - [Understaning Word Vectors](https://gist.github.com/aparrish/2f562e3737544cf29aaf1af30362f469)
 - [Discord Chat Exporter Repository](https://github.com/Tyrrrz/DiscordChatExporter)
