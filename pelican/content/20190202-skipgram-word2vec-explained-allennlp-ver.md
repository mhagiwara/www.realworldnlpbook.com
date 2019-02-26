Title: SkipGram (word2vec) Model Explained (AllenNLP ver.)
Date: 2019-02-02 00:00
Category: Word Embeddings
Tags: Word Embeddings, word2vec, AllenNLP
Cover: example.png
slug: skipgram-word2vec-explained-allennlp-ver

## Word Embeddings

As I explained in the previous post, a word embedding is a continuous vector representation of a word. If you are not familiar with the mathematical concept, imagine assigning an array of floating point numbers to each word: 

- `vec("dog") = [0.8, 0.3, 0.1]`
- `vec("cat") = [0.7, 0.5, 0.1]`
- `vec("pizza") = [0.1, 0.2, 0.8]`

In this example, I just made up those three-dimensional vectors, but you can see the first element of each word represents some sort of "animal-ness." If you want to calculate some semantic similarity between words, you can do it by looking at the "angle" between two vectors (more technically, this is called a cosine similarity). If you want to train another NLP model on top of those representations, you can use them as the input to your machine learning model. 

Now, there's one important piece of information missing from the discussion so far. How do you come up with those float numbers? It would be virtually impossible to assign them by hand. There are hundreds of thousands of unique words in a typical large corpus, and the arrays should be at least around 100-dimensional long to be effective, which means there are more than tens of millions of numbers that you need to tweak. But more importantly, what should those numbers look like? How do you determine whether you should assign a 0.8 to the first element of the "dog" vector, or 0.7, or any other numbers? That's exactly what the SkipGram model is designed to do, which I'll explain below.

## SkipGram

One possible idea to do this without teaching the computer what "dog" means is to use its context. For example, what words tend to appear together with the word "dog" if you look its appearances in a large text corpus? "Pet," "tail," "smell," "bark," "puppy," ... there can be countless options. How about "cat"? Maybe "pet," "tail," "fur," "meow," "kitten," and so on. Because "dog" and "cat" have a lot in common conceptually (they are both popular pet animals with a tail, etc. etc.), these two sets of context words also have large overlap. In other words, you can guess how close two words are to each other by looking at what other words appear in the same context. This is called _the distributional hypothesis_ and has a long history in NLP.

We are now one step closer. If two words have a lot of context words in common, we can give similar vectors to those two words. You can think of a word vector as a "compressed" representation of its context words. Then the question becomes: how can you "de-compress" a word vector to obtain their context words? How can you even represent a set of context words mathematically? Conceptually, we'd like to come up with a model that does something like this: 

```text
"dog" -> (0.8, 0.3, 0.1) -> (de-compressor) -> {"pet", "tail", "smell", "bark", ...}  
``` 

One way to represent a set of words mathematically is to assign a score to each word in the vocabulary. Instead of representing context words as a set, we can think of it as an associative array from words to their "scores":

```text
{"bark": 1.4, "chocolate": 0.1, ..., "pet": 1.2, ..., "smell": 0.6, ...} 
```

The only remaining piece of the model is how to come up with those "scores."  If you look at these scores, they can be conveniently represented by a N-dimensional vector, where N is the size of the entire vocabulary (the number of unique context words we consider). All the "de-compressor" needs to do is expand the word embedding vector (which has three dimensions) to another vector of N dimensions.

This may sound very familiar to some of you—yes, it's exactly what linear layers (aka fully-connected layers) do. They convert a vector of one size to another of different size in a linear fashion. Putting everything together, the architecture of the SkipGram model looks like the following figure:

<figure style="text-align: center">
	<img src="images/skipgram.png"/>
	<figcaption>Figure: SkipGram model</figcaption>
</figure>

## Softmax

Hopefully I successfully convinced you that SkipGram is actually a lot simpler than most people think. Now, let's talk about how to "train" it and learn the word embeddings we want. The key here is to turn this into a classification task, where the network predicts what words appear in the context. This is actually a "fake" task because we are not interested in the prediction of the model per se, but rather in the by-product (word embeddings) produced by training the model.

It is relatively easy to make a neural network solve a classification task. You need two things:

* Modify the network so that it produces a probability distribution
* Use cross entropy as the loss function

You use something called _softmax_ to do the first. Softmax is a function that turns a vector of K float numbers to a probability distribution, by first "squashing" the numbers so that they fit a range between 0.0-1.0, and then normalizing them so that the sum equals 1. Softmax does all this while preserving the relative ordering of the input float numbers, so large input numbers still have large probability mass in the output distribution. The following figure illustrates this conceptually:

<figure style="text-align: center">
	<img src="images/softmax.png"/>
	<figcaption>Figure: Converting a K-dimensional real vector to a probability distribution using Softmax</figcaption>
</figure>

Cross entropy is a loss function used to measure the distance between two probability distributions. It returns zero if two distributions match exactly, and higher values if the two diverge. For classification tasks, we use cross entropy to compare:

1. the predicted probability distribution produced by the neural network (output of softmax) and,
2. the "target" probability distribution where the probability of the correct class is 1.0 and everything else is 0.0

The predictions made by the SkipGram model get closer and closer to the actual context words, and word embeddings are learned at the same time. 

## Negative Sampling

Theoretically, you can now build your own SkipGram model and train word embeddings. In practice, however, there is one issue in doing so—speed.  

## Subsampling

```python
def _subsample_tokens(self, tokens):
    """Given a list of tokens, runs sub-sampling.

    Returns a new list of tokens where rejected tokens are replaced by Nones.
    """
    new_tokens = []
    for token in tokens:
        reject_prob = self.reject_probs.get(token, 0.)
        if random.random() <= reject_prob:
            new_tokens.append(None)
        else:
            new_tokens.append(token)

    return new_tokens
```

## Evaluating Word Embeddings
