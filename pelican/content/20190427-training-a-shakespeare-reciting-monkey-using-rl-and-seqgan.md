Title: Training a Shakespeare Reciting Monkey using RL and SeqGAN
Date: 2019-02-02 00:00
Category: Text Generation
Tags: Text Generation, RL, SeqGAN, AllenNLP
Cover: images/skipgram.png
slug: training-a-shakespeare-reciting-monkey-using-rl-and-seqgan

There is this article titled [Tackle the monkey first](https://blog.x.company/tackle-the-monkey-first-90fd6223e04d) written by Astro Teller, who leads Google's moonshot lab "X." The article begins "Let’s say you’re trying to teach a monkey how to recite Shakespeare while on a pedestal. How should you allocate your time and money between training the monkey and building the pedestal?"

(Figure - Shakespeare reciting monkey)

The moral of the story is that you should spend zero time building the pedestal and focus on the most difficult part, which is training the Shakespeare reciting monkey, although many people tend to rush off and start building the pedestal instead just to have a sense of progress and to have at least something to show to their boss. A Shakespeare reciting monkey without a pedestal is still an incredible achievement, while a pedestal is just a pedestal without the monkey.

I just love this article—I can't remember how many times I was tempted to "build the pedestal" first when I had choices between projects or some components and I realized I was about to do something easier instead of tackling the most difficult part. Re-reading this article almost always helps me go back on track. "When in doubt, do something more difficult." is one of my life mottoes. In fact, I liked this article so much that now I write "tackle the monkey first" at the top of my to-do list to constantly remind me of the most important thing every day.

So, I decided to tackle the monkey first. 

## Infinite Monkey Theorem

(Figure - Monkeys at typewriters)

I don't know where Astro Teller got the inspiration of Shakespeare reciting monkeys, but the concept is not entirely new. In mathematics and probability theory, there's a famous thought experiment called "Infinite Monkey Theorem," which states that a monkey sitting at a typewriter hitting keys at random will eventually produce the complete works of Shakespeare, such as the entire text of Hamlet, purely by chance given an infinite amount of time. 

This is somewhat counter-intuitive, but there's a nice explanation in the Wikipedia page on why this is the case, so we can borrow it here—suppose there are 50 keys on the typewriter and the monkey is pressing the keys uniformly and independently. The chance of this monkey accidentally producing a word "banana" is 1/50 to the sixths power, which is less than one in 15 billion. Let this probability be p. 

Now, let's think about giving this monkey a second chance. The chance of this monkey producing the word "banana" after two trials is 1 - (1 - p)^2, which is about one in 8 billion. Still extremely small likelihood, but not zero, and it increases as the number of trials increases. After one million trials, this probability is about 0.9999, and after ten billion, it is about 0.5273. The probability 1 - (1 - p)^n asymptotically reaches 1.0 as n gets bigger. This informally proves that a monkey hitting random keys will eventually produce a word "banana" purely by chance.

But we are not just talking about producing a single word—it's about producing the entire work of Shakespeare. For example, there are about 130,000 characters in Hamlet. The probability of a monkey producing the entire text of Hamlet by chance after the first trial is one in 3.4 x 10^183,946. To get a feel of how small this probability is, let's quote a paragraph from the Wikipedia page—

> Even if every proton in the observable universe were a monkey with a typewriter, typing from the Big Bang until the end of the universe (when protons might no longer exist), they would still need a still far greater amount of time – more than three hundred and sixty thousand orders of magnitude longer – to have even a 1 in 10^500 chance of success.

This is a bit discouraging—in theory, a typewriter-hitting monkey will produce Hamlet at some point, but in practice, none of us can afford a universe full of monkeys nor billions of years. If it takes a universe full of monkeys more than billions of years to have an even slightest chance of reciting Shakespeare, it is understandable that you would rather be building the pedestal instead. 

## Using Reinforcement Learning (RL)

But if you think of it, a monkey hitting typewriter keys uniformly and independently may be an interesting mathematical device, but not a realistic one. Researchers found that certain types of monkeys, especially chimpanzees, are highly intelligent. You can probably train the monkey so that they have a higher chance of reproducing Shakespeare before you wait till the end of the universe.

For example, in English text, the letter E appears a lot more frequently than, say, the letter Z does. If you want to reproduce Hamlet, for example, hitting the E key more often than other letters is a good strategy. If you can teach this simple bias over the distribution of English letters to the monkey, he/she will have a far greater chance of reciting Shakespeare.

And here's the tool that many animal trainers agree is useful—rewards. 

## 