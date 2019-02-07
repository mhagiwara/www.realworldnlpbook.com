Title: SkipGram (word2vec) Model Explained (AllenNLP ver.)
Date: 2019-02-02 00:00
Category: Word Embeddings
Tags: Word Embeddings, word2vec, AllenNLP
Cover: example.png
slug: 20190222-skipgram-word2vec-explained-allennlp-ver

## Word Embeddings

As I explained in the previous post, a word embedding is a continuous vector representation of a word. If you are not familiar with the mathematical concept, imagine assigning an array of floating point numbers to each word: 

- v("dog") = (0.8, 0.3, 0.1)
- v("cat") = (0.7, 0.5, 0.1)
- v("pizza") = (0.1, 0.2, 0.8)

In this example, I just made up those three-dimensional vectors, but you can see the first element of each word represents some sort of "animal-ness." If you want to calculate some semantic similarity between words, you can do it by looking at the "angle" between two vectors (more technically, this is called a cosine similarity). If you want to train another NLP model on top of those representations, you can use them as the input to your machine learning model. 

Now, there's one important piece of information missing from the discussion so far. How do you come up with those float numbers? It would be virtually impossible to assign them by hand. There are hundreds of thousands of unique words in a typical large corpus, and the arrays should be at least 100-dimensional long to be effective, which means there are more than tens of millions of numbers that you need to tweak. But more importantly, what should those numbers look like? How do you determine whether you  should assign a 0.8 to the first element of the "dog" vector, or 0.7, or any other numbers? That's exactly what the SkipGram model is designed to do, which I'll explain below.

## SkipGram

One possible idea to do this without teaching the computer what "dog" means is to use its context. For example, what words tend to appear together with the word "dog" if you look its appearances in a large text corpus? "Pet," "tail," "smell," "bark," "puppy," ... there can be countless options. How about "cat"? Maybe "pet," "tail," "fur," "meow," "kitten," and so on. Because "dog" and "cat" have a lot in common conceptually (they are both popular pet animals with a tail, etc. etc.), these two sets of context words also have a large overlap. In other words, you can know how close two words are to each other by looking at what other words appear in the same context. This is called the distributional hypothesis and has a long history in NLP.

Now, we are one step closer. If two words have a lot of context words in common, we can give similar vectors to those two words. You can think of a word vector as a "compressed" representation of its context words. Then the question becomes: how can you "de-compress" a word vector to obtain their context words? How can you even represent a set of context words mathematically? Conceptually, we'd like to come up with a model that does something like this: 

```text
"dog" -> (0.8, 0.3, 0.1) -> (de-compressor) -> {"pet", "tail", "smell", "bark", ...}  
``` 

One way to represent a set of words mathematically is to assign a score to each word in the vocabulary. Instead of representing context words as a set, we can think of it as an associative array from words to their "scores":

```text
{"bark": 1.4, "chocolate": 0.1, ..., "pet": 1.2, ..., "smell": 0.6, ...} 
```

Now the only remaining piece of the model is how to come up with those "scores."  If you think of them as some sort of semantic closeness between the word in question ("dog") and the associated word ("bark"), you can derive them as a similarity between two vectors. This can be done by assigning another set of embedding vectors to context vectors, and calculating the inner product between the two:

```text
"dog" -> (0.8, 0.3, 0.1) -> (score = inner product = 3.2) <- (1.5, 0.6, 0.2) <- "bark"
```

## Softmax

## Negative Sampling

## Subsampling

## Evaluating Word Embeddings
