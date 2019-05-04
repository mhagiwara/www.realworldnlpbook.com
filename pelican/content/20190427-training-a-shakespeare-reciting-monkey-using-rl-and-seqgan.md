Title: Training a Shakespeare Reciting Monkey using RL and SeqGAN
Date: 2019-02-02 00:00
Category: Text Generation
Tags: Text Generation, RL, SeqGAN, AllenNLP
Cover: images/skipgram.png
slug: training-a-shakespeare-reciting-monkey-using-rl-and-seqgan

## Tackle the Monkey First

There is this article titled [Tackle the monkey first](https://blog.x.company/tackle-the-monkey-first-90fd6223e04d) written by Astro Teller, who leads Google's moonshot lab "X." The article begins "Let’s say you’re trying to teach a monkey how to recite Shakespeare while on a pedestal. How should you allocate your time and money between training the monkey and building the pedestal?"

(Figure - Shakespeare reciting monkey)

The moral of the story is that you should spend zero time building the pedestal and focus on the most difficult part, which is training the Shakespeare reciting monkey, although many people tend to rush off and start building the pedestal instead just to have a sense of progress and to have at least something to show to their boss. A Shakespeare reciting monkey without a pedestal is still an incredible achievement, while a pedestal is just a pedestal without the monkey.

I just love this article—I can't remember how many times I was tempted to "build the pedestal" first when I had choices between projects or some components and I realized I was about to do something easier instead of tackling the most difficult part. Re-reading this article almost always helps me go back on track. "When in doubt, do something more difficult." is one of my life mottoes. In fact, I liked this article so much that now I write "tackle the monkey first" at the top of my to-do list to constantly remind me of the most important thing every day.

So, I decided to tackle the monkey first. 

## Infinite Monkey Theorem

(Figure - Monkeys at typewriters)

I don't know where Astro Teller got the inspiration of Shakespeare reciting monkeys, but the concept is not entirely new. In mathematics and probability theory, there's a famous thought experiment called "Infinite Monkey Theorem," which states that a monkey sitting at a typewriter hitting keys at random will eventually produce the complete works of Shakespeare, such as the entire text of Hamlet, purely by chance given an infinite amount of time. 

This is somewhat counter-intuitive, but there's a nice explanation in the Wikipedia page on why this is the case, so we can borrow it here—suppose there are 50 keys on the typewriter and the monkey is pressing the keys uniformly and independently. The chance of this monkey accidentally producing a word "banana" is 1/50 to the sixths power, which is less than one in 15 billion. Let this probability be p. 

Now, let's think about giving this monkey a second chance. The chance of this monkey producing the word "banana" within two trials is 1 - (1 - p)^2, which is about one in 8 billion. Still extremely small likelihood, but not zero, and it increases as the number of trials increases. After one million trials, this probability is about 0.9999, and after ten billion, it is about 0.5273. The probability 1 - (1 - p)^n asymptotically reaches 1.0 as n gets bigger. This informally proves that a monkey hitting random keys will eventually produce a word "banana" purely by chance.

But we are not just talking about producing a single word—it's about producing the entire work of Shakespeare. For example, there are about 130,000 characters in Hamlet. The probability of a monkey producing the entire text of Hamlet by chance after the first trial is one in 3.4 x 10^183,946. To get a feel of how small this probability is, let's quote a paragraph from the Wikipedia page—

> Even if every proton in the observable universe were a monkey with a typewriter, typing from the Big Bang until the end of the universe (when protons might no longer exist), they would still need a still far greater amount of time – more than three hundred and sixty thousand orders of magnitude longer – to have even a 1 in 10^500 chance of success.

This is a bit discouraging—in theory, a typewriter-hitting monkey will produce Hamlet at some point, but in practice, none of us can afford a universe full of monkeys nor billions of years. If it takes a universe full of monkeys more than billions of years to have an even slightest chance of reciting Shakespeare, it is understandable that you would rather be building the pedestal instead. 

## Training Monkeys

But if you think of it, a monkey hitting typewriter keys uniformly and independently may be an interesting mathematical device, but not a realistic one. Researchers found that certain types of monkeys, especially chimpanzees, are highly intelligent. You can probably train the monkey so that they have a higher chance of reproducing Shakespeare before you wait till the end of the universe.

For example, in English text, the letter E appears a lot more frequently than, say, the letter Z does. If you want to reproduce Hamlet, for example, hitting the E key more often than other letters is a good strategy. If you can teach this simple bias over the distribution of English letters to the monkey (let's call him George), he will have a far greater chance of reciting Shakespeare.

And here's the tool that many animal trainers agree is useful—rewards. You can reinforce George's positive behavior by giving him some bananas. Specifically, when the text George produced resembles Shakespeare's work, you'll give him some bananas. The more it looks like Shakespeare's work, the more bananas he'll be rewarded with. On the other hand, when he produces some garbage that doesn't look anything from Shakespeare, he'll get no bananas. George has very little idea what he's done right or wrong, but if you keep doing this long enough, you expect that he'll figure out how he should be typing in order to get more rewards, and what he produces will start to look like Shakespeare's work.

## Reinforcemenet Learnin (RL)

In machine learning, this framework is called Reinforcement Learning (RL). There are many good courses, books, and Internet articles that you can refer to to learn 

There is an agent (= George the monkey) in an environment. The agent takes some actions (= typing keys) based on its state (= what George's thinking), and is given some rewards (= bananas) depending on the outcome (= produced text). The goal here is to optimize the actions of the agent so that it can maximize the future expected rewards.

Specifically, we are going to use one family of reinforcement learning algorithms called policy gradient. In policy gradient, the policy (what actions the agent should take in a particular state) is explicitly modeled and optimized. We'll be using one particular type of policy gradient algorithm called REINFORCE, which uses Monte-Carlo methods to estimate the expected rewards, although we are not going in to the details of the algorithm in this post.

## Implementing a Shakespeare reciting monkey

In order to implement the Shakespeare reciting monkey, I'm going to use a recurrent neural network (RNN), namely, an LSTM RNN. At each timestep, the RNN receives a vector, updates its internal states, and produces another vector. The output vector is then fed into a linear layer to expand (or shrink) it to another vector, from which predictions are made.

Since we are simulating a monkey typing a typewriter, we generate and feed individual characters to the RNN. But as with typical language generation models, there is no actual "input," so we feed the generated character at the previous timestep as the input to the next timestep. The length of the output vector is the same as the number of all the alphabets we consider. The following figure illustrates the architecture.

I implemented this RNN language generation model as follows. It's a mix of AllenNLP and PyTorch, but hopefully it's not terribly difficult to read. 

```python
class RNNLanguageModel(Model):
    def __init__(self, vocab: Vocabulary) -> None:
        super().__init__(vocab)

        token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                    embedding_dim=EMBEDDING_SIZE)
        self.embedder = BasicTextFieldEmbedder({"tokens": token_embedding})

        self.rnn = PytorchSeq2SeqWrapper(
            torch.nn.LSTM(EMBEDDING_SIZE, HIDDEN_SIZE, batch_first=True))

        self.hidden2out = torch.nn.Linear(in_features=self.rnn.get_output_dim(),
                                          out_features=vocab.get_vocab_size('tokens'))


    def generate(self, max_len: int) -> Tuple[List[str], Tensor]:

        start_symbol_idx = self.vocab.get_token_index(START_SYMBOL, 'tokens')
        end_symbol_idx = self.vocab.get_token_index(END_SYMBOL, 'tokens')

        log_likelihood = 0.
        words = []
        state = (torch.zeros(1, 1, HIDDEN_SIZE), torch.zeros(1, 1, HIDDEN_SIZE))

        word_idx = start_symbol_idx

        for i in range(max_len):
            tokens = torch.tensor([[word_idx]])

            embeddings = self.embedder({'tokens': tokens})
            output, state = self.rnn._module(embeddings, state)
            output = self.hidden2out(output)

            log_prob = torch.log_softmax(output[0, 0], dim=0)
            dist = torch.exp(log_prob)

            word_idx = torch.multinomial(
                dist, num_samples=1, replacement=False).item()

            log_likelihood += log_prob[word_idx]

            if word_idx == end_symbol_idx:
                break

            words.append(self.vocab.get_token_from_index(word_idx, 'tokens'))

        return words, log_likelihood
```

After computing the logits (`output`) by applying the linear layer, they are converted to a probability distribution (by applying `log_softmax` then `exp`), from which the next word is sampled by `multinomial`. 

Make sure that the generate function returns the log likelihood of the generated text. We will need it later. The log likelihood of the sequence is simply the sum of log of individual probabilities (which you already calculated by `log_softmax` above).

Now, how do we calculate the reward? There could be a million different ways to do this, but I chose to use one of my favorite metrics for textual similarity, [chrF](https://aclweb.org/anthology/W15-3049). chrF is simply the F-measure of character n-grams between the prediction and the reference. The more similar these two are, the higher the value will be. My code below random samples 100 lines from the training set (Shakespeare's Hamlet), calculates [the chrF metric using NLTK](https://www.nltk.org/_modules/nltk/translate/chrf_score.html) between the generated text and each sampled line, and returns the average: 

```python 

def calculate_reward(generated: str, train_set: List[str], num_lines=100) -> float:
    line_ids = np.random.choice(len(train_set), size=num_lines)

    chrf_total = 0.
    for line_id in line_ids:
        line = train_set[line_id]
        chrf = sentence_chrf(line, generated, min_len=2, max_len=6, beta=1.,
                             ignore_whitespace=False)

        chrf_total += chrf

    return chrf_total / num_lines
```

Finally, here's the key to reinforcement learning: in order to encourage positive behavior, you optimize the LSTM RNN to maximize the log likelihood, but you scale this value by the reward you just calculated. If the reward is large, the network is adjusted so that it will repeat the similar behavior next time around. If it's small, little adjustment happens. This is equivalent to using the negative log likelihood scaled by the reward as the loss function:

```
loss = -1. * reward * log_likelihood
``` 
 
```python
model.zero_grad()

log_likelihoods = []
rewards = []

for _ in range(BATCH_SIZE):
    words, log_likelihood = model.generate(max_len=60)
    reward = calculate_reward(''.join(words), train_set)

    log_likelihoods.append(log_likelihood)
    rewards.append(reward)

baseline = sum(rewards) / BATCH_SIZE
avr_loss = sum(-1. * (reward - baseline) * log_likelihood
               for reward, log_likelihood in zip(rewards, log_likelihoods))
avr_loss /= num_instances

avr_loss.backward()
optimizer.step()
```

## SeqGAN

## Implementing SeqGAN

