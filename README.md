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

Here is a top-level diagram of the process I have described thus far:
![image](https://user-images.githubusercontent.com/59097689/171921352-c09c7cd8-01aa-42c8-8d84-0eaa1513a4dd.png)

## Usage

```python
placeholder
```


## References

 - [Understaning Word Vectors](https://gist.github.com/aparrish/2f562e3737544cf29aaf1af30362f469)
 - [Discord Chat Exporter Repository](https://github.com/Tyrrrz/DiscordChatExporter)
