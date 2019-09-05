# Categorical features
There are many ways to transform a categorical variable with high cardinality. I will describe the following methods here:

one-hot — the simplest of all techniques, very useful in a number of settings with low cardinality

rare-word tagging — this allows the cardinality to vary

frequency binning — often of great use in anomaly detection, fraud prevention and intrusion detection

random embedding — the grandparent of modern semantic embedding

the hash trick — random embedding for people who like binary features

Luduan features — how to encode operational structure efficiently by observing cooccurrence.

## Background
But before we get into all this too much, let’s settle some terminology. Let’s take “low cardinality” to be less than 10 or so, “medium cardinality” to be from 10 to 100, “high cardinality” to be 100 to 1000 and “ultra high cardinality” to be above 1000. These boundaries aren’t hard and fast and we should be willing to wiggle a bit on them. Some categorical variables are ordered (birth year) and some are not (car make). We should also keep in mind categorical variables where we do not know the full cardinality, be it low, medium, high or ultra high.

Examples of features in these different cardinality ranges include:

low cardinality — gender, rotation direction (CW or CCW), cardinal points (N, S, E, W, NE, etc), phone type (land line, cell, VOIP)

medium cardinality — car make, telephone brand, key on keyboard, US state

high cardinality — country of birth, birth year

ultra high cardinality — word from text, URL, domain name, IP address, post code

Examples of categorical variables where we can’t easily know the full cardinality with absolute certainty or where we might wish to allow for change might include brand names, countries and gender. Examples of categorical variables where the cardinality is not just not currently known, but is growing continually include domain names, IP addresses and words from text.

There are lots of techniques and tricks for handling variables of this general class. Which techniques work best depend a lot on the rough cardinality and whether the cardinality is fixed. When cardinality is even moderately high, you start to encounter problems due to the fact that some values will be much more rare than others. As you get to ultra high cardinality, this problem becomes very severe as frequencies can vary between different values by many orders of magnitude.

When the cardinality is not fixed or is simply not yet known, a different problem arises in that essentially all machine learning techniques want to deal with a fixed number of input variables. That means that we have to figure out some way to convert an unbounded kind of input into a strictly bounded number of inputs without losing information. With numerical features, we also have a large cardinality, but the mathematical structure of numbers such as distance and ordering usually allows us to treat such inputs much more simply. With true categorical values, we have to discover or impose this structure.

# One-hot Encoding

In the simplest of all cases, we have low and fixed cardinality. In such a case, we can have a model feature for each possible value that the variable can take on and set all of these features to zero except for the one corresponding to the value of our categorical feature. This works great and is known as one-hot encoding. This might lead to encoding days of the week as Monday = (1,0,0,0,0,0,0), Tuesday = (0,1,0,0,0,0,0) and so on.

## Rare-word Collapse

As the cardinality increases, however, this works less and less well, largely because some values will be much more rare than other values and increasing the number of features to a model beyond a few thousand generally has very bad effects on the ability to build a model. Even worse, high cardinality generally goes hand in hand with indefinite cardinality. Even so, it is common in natural language models to simply group all but the 𝑘 most common values of a categorical variable as a single “RARE-WORD” value. This reduction allows us to have a 𝑘+1-hot encoding. If 𝑘 is big enough, this will work pretty well because the “RARE-WORD” value will itself be pretty rare.

## Frequency Binning

We can take this idea of collapsing to a radical and surprisingly effective extreme. This is done by reducing a high cardinality categorical feature to a single number that represents the frequency of the value of the feature. Alternately, you might use the quantile of the rank of the frequency, or bin the frequency of the value. In any case, this works in applications where a specific value isn’t as important as the fact that you have seen a surprisingly rare value. Consider, network intrusion detection where suddenly seeing lots of data going to a previously almost unknown external network address could be very informative. It doesn’t really matter which previously unknown address is being used, just that it is previously unknown or nearly so. Note that you can combine this kind of frequency feature with other features as well so that you not only get these desirable novelty effects, but you can keep the precise resolution about exactly which categorical value was seen.

## Random Embedding

Another way to keep a fixed sized encoding with values of large or unknown cardinality without collapsing rare values together is to use a random embedding or projection. One simple way to do this is convert each possible value to a 50–300 dimensional vector. Commonly, these vectors will be constrained to have unit length You can actually do this in a consistent way without knowing the categorical values ahead of time by using the actual value as a seed for a random number generator and then using that generator to sample a “random” unit vector. If the dimension of the vector is high enough (say 100 to 500 dimensions or more) then the vectors corresponding to any two categorical values will be nearly orthogonal with high probability. This quasi-orthogonality of random vectors is very handy since it makes each different value be sufficiently different from all other values so that machine learning algorithms can pick out important structure.

These random vectors can also be tuned somewhat using simple techniques to build a semantic space, or using more advanced techniques to get some very fancy results. Such random projections can be used to do linear algebraic decompositions as well.

## The Hash Trick

We can use different random projections to get something much more like the one-hot encoding as well without having to collapse rare features or, indeed, without having to even know which features are rare. For each distinct value, we can encode that value using a 𝑛 binary values of which exactly 𝑘 randomly chosen values are set to 1 with the rest set to 0 using the same seeding trick as before. Commonly 𝑛 is taken to be a few thousand while 𝑘 can be relatively small, typically less than 20. When 𝑘=1, we get one-hot encoding again. This technique works because of the same mathematical techniques as random projection, but is generally described more in terms of analogies to Bloom filters.

## Luduan Features

Finally, you can derive a numerical features by grouping values that have anomalous correlation with some objective observation and then weighting by the underlying frequency of the feature value (or the inverse log of that frequency). This reduction is known as a Luduan feature and is based on the use of log-likelihood ratio tests for finding interesting cooccurrence. I gave a talk on using these techniques for transaction mining some time ago that described how to do this.
