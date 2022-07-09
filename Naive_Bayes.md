## Discriminative classifiers: 

Discriminative classifiers divide the feature space into regions that separate
the data belonging to different classes, such that every separate region contains samples belonging
to a single class. The decision boundary is determined by constructing and solving equations of the form

$$ \mathbf{y} = \mathbf{w}^T\mathbf{\phi(X)}+b $$

In practice equations of this kind may not always be solvable, so discriminative classifiers
minimize some loss to solve an approximation of this equation. Algorithms of this type are not
necessarily concerned with the probability distributions of features or labels.

Additionally, determining
probabilities using the conditional distribution of y given x, can give us an idea of how strong the
possibility of a data instance belonging to a specific class is.

## Generative classifiers:
Generative classifiers are concerned with finding the joint probability of features and labels, i.e.
they try to estimate the probabilities of all pairs of features and labels found in the training
data. Generative models assume that there is an underlying probability distribution from which the
data has been sampled, and try to estimate the parameters of this distribution. Thus, they see the
data as being "generated" through an estimated distribution (discriminative models, in contrast,
need not make this assumption).

# What is the Naïve Bayes method?

The Naïve Bayes algorithm is a set of generative classifiers. The
fundamental assumption in a Naïve Bayes algorithm is that
**_conditional on the class, features are independent_**. This means that one feature value appearing in a given
class is independent of the value of any other feature in the same class, and consequently within a class, any change in the value of one feature does not impact the value of another.

E.g. Given that a sample is drawn from the "setosa" class of the iris dataset, knowing it's sepal
length should tell us nothing about its petal width.


This is impractical in real life, since features are likely to be interrelated (e.g. in the Titanic
dataset, given that a passenger survived, if they paid a low fare, we might be able to conclude that they
are a woman. In text data, we know that occurences of words are not independent of each other.
Hence it is a ‘naïve’ (oversimplifying) assumption.

Thus, the algorithm:

* assumes that the features in each class in the dataset is sampled from a distribution
* estimates the parameters for each such distribution

Ultimately, the model ends up with one unique joint distribution for each class.
Note that these distributions are all from from the same family (i.e. Normal, Binomial, Gamma, etc).

Then, from the set of features in the test dataset, the algorithm estimates the probabilities that the new sample belongs to each of the class-wise distributions, and the class that has the highest probability is the predicted class.


# Mathematical model

The goal of the prediction is to find the probability that a feature vector belongs to a given
class. In other words, for a set of $k$ 
classes ${c_1, c_2, \dots, c_k}$, the prediction involves
computing the following probabilities:

$$
P(c_i\|\mathbf{x}) \forall i \in [1, k]
$$

where $\mathbf{x}$ denotes the feature vector and $c_i$ is the $i^th$ class. This expression denotes
the posterior probability that a feature vector belongs to a given class. 

However, in the training dataset, we only have the prior probabilities, i.e. we can only compute::

$$
P(\mathbf{x}|c_i)
$$

which denotes the probability of finding a feature vector in a given class.

So, the algorithm uses the Bayes’ theorem to convert the prior probabilities into the posterior
probabilities.

## Estimating class-wise distributions

As in typical machine learning problems, the parameters of the class-wise distributions are unknown.
These are typically computed with maximum likelihood estimation. 

Using Bayes' theorem we expand the expression as follows:

$\hspace{50pt}\tiny\color{blue}\fbox{The likelihood of observing the sample (or features) given the label}$
$\hspace{90pt}\tiny\color{orange}\fbox{The likelihood of observing the sample (or features) given the label}$
$\hspace{2cm}P(y=c_i|\mathbf{x}) = \frac{\color{blue}P(\mathbf{x}|y=c_i; \mathbf{\theta})}{\color{green}P(\mathbf{x};\mathbf{\theta})} \times \color{orange}P(y=c_{i};\mathbf{\theta})$

$\hspace{70pt}\tiny\color{green}\fbox{The likelihood of observing the sample (or features) given the label}$

$\mathbf{\theta}$ represents the unknown parameters, which we estimate by maximising the likelihood, i.e we differentiate the likelihood and prior expressed in terms of such unknown parameters and equate them to 0.

For derivation of parameter estimate formula using maximum log likelihood – Refer lecture slides 30,31 & 36-39.
## Prediction
Once we have estimated the parameters $\mathbf{\theta}$ as above using the training data, we would want to use the same to predict the label for a test data sample. 
For a sample $\mathbf{x}$ 
with features, say, $[x_1, x_2, x_3, x_4]$ 
and label $y$,

we want to find the probability of the  sample belonging to class $c_{i}$ i.e
required evaluation is

$$
P(y=c_i|\mathbf{x}; \mathbf{\theta})
$$

- the  probability of the class being $c_i$ 
given the sample $\mathbf{x}$, 
parameterized by $\mathbf{\theta}$.

(Note: $\mathbf{\theta}$ is a vector containing the parameters of the corresponding probability
distributions. It could denote a vector of $[\mu, \sigma]$ for a Gaussian distribution, or a vector
of $k, n, p$ for a Binomial distribution, etc.)

Using the Bayes' theorem expansion we have the following (Parameters of the feature distribution are now known):

$$
P(y=c_i|\mathbf{x}) = \frac{P(\mathbf{x}|y=c_i)}{P(\mathbf{x})} \times P(y=c_{i})
$$

We know the following quantities from the training data:

1. the prior probability $P(\mathbf{x}|y=c_i)$ from the training data, and
2. the class priors $P(y=c_i) \forall i \in [1, k]$

The denominator in the RHS can be ignored, since it is only a normalizing factor, and will not
affect the relative probabilities of a sample belonging to different classes.

The expression for the prior probability, $P(\mathbf{x}|y=c_i)$ can be expanded as

$$
P(x_1, x_2, x_3, x_4 | y=c_i)
$$

(note that $\mathbf{x}$ is a vector containing 4 elements). Using the naive assumption of
class-conditional independence, this expression can be expanded as follows:

$$
P(x_1|y=c_i)P(x_2|y=c_i)P(x_3|y=c_i)P(x_4|y=c_i)
$$

Now, this is a representation which can be computed from the estimated parameters of the training data.

With all these quantities in place for each class, we can compute the LHS for all $k$ classes.
Whichever class has the highest probability then becomes our prediction for the sample $\mathbf{x}$.


# Illustrating parameter estimation using an example

Let us try to understand the model with an example. The following is a small subset of the dataset capturing details of passengers aboard the ‘Titanic’. The dataset can be found in the UCI Machine learning repository. 



| PClass    | Sex | Age | Parents/Children Aboard| Survived
| ----------- | ----------- |-------------|----------|---------|
|$\color{orange}3$ |$\color{green}female$ |$\color{violet}31$| $\color{brown}0$| $\color{purple}0$|
|$\color{orange}3$ |$\color{green}female$ |$\color{violet}31$| $\color{brown}0$| $\color{blueviolet}1$|
|$\color{Gold}1$ |$\color{green}female$ |$\color{violet}31$| $\color{brown}0$| $\color{blueviolet}1$|
|$\color{Gold}1$ |$\color{green}female$ |$\color{violet}31$| $2$| $\color{blueviolet}1$|
|$\color{orange}3$ |$\color{green}female$ |$\color{violet}31$| $\color{olive}1$| $\color{blueviolet}1$|
|$\color{grey}2$ |$\color{blue}male$ |$\color{violet}31$| $\color{brown}0$| $\color{purple}0$|
|$\color{grey}2$ |$\color{green}female$ |$\color{violet}31$| $\color{brown}0$| $\color{blueviolet}1$|
|$\color{grey}2$ |$\color{blue}male$ |$\color{violet}31$| $\color{olive}1$| $\color{purple}0$|
|$\color{gold}1$ |$\color{blue}male$ |$\color{violet}31$| $\color{brown}0$| $\color{purple}0$|
|$\color{grey}2$ |$\color{blue}male$ |$\color{violet}31$| $\color{brown}0$| $\color{blueviolet}1$|
|$\color{gold}1$ |$\color{blue}male$ |$\color{violet}31$| $\color{brown}0$| $\color{blueviolet}1$|
|$\color{orange}3$ |$\color{blue}male$ |$\color{violet}31$| $\color{brown}0$| $\color{blueviolet}1$|
|$\color{orange}3$ |$\color{blue}male$ |$\color{violet}31$| $\color{brown}0$| $\color{purple}0$|
|$\color{grey}2$ |$\color{green}female$ |$\color{violet}31$| $\color{olive}1$| $\color{blueviolet}1$|
|$\color{gold}1$ |$\color{blue}male$ |$\color{red}52$| $\color{olive}1$| $\color{purple}0$|
|$\color{gold}1$ |$\color{blue}male$ |$\color{red}52$| $\color{brown}0$| $\color{blueviolet}1$|
|$\color{gold}1$ |$\color{green}female$ |$\color{red}52$| $\color{brown}0$| $\color{blueviolet}1$|
|$\color{grey}2$ |$\color{blue}male$ |$\color{red}52$| $\color{brown}0$| $\color{purple}0$|
|$\color{gold}1$ |$\color{green}female$ |$\color{red}52$| $\color{olive}1$| $\color{blueviolet}1$|

Suppose we want to find out the probability of survival of a male passenger aged 31 years with 0 parent/children aboard and travelling by the first class (Let's call this the target probability). Here is where Naïve Bayes could be used.
Probability of Survived = 1 given that Pclass = 1 and Sex = male and Age = 31 and Parents/Children Aboard = 0 

$\frac{P(Pclass \space =\space 1 \space and \space  Sex \space =\space  male \space  and \space  Age \space  = \space 31 \space and \space Parents/Children \space  Aboard \space  = \space 0\space |Survived \space =\space 1)} {P(Pclass \space=\space 1 \space and \space Sex \space= \space male \space and \space Age \space = \space 31 \space and \space Parents/Children \space Aboard \space = \space 0)} \times{P(Survived=1)}$

P(Survived = 1) = 12/19 = 0.632

$ P(Pclass \space=\space 1 \space and \space Sex \space= \space male \space and \space Age \space = \space 31 \space and \space Parents/Children \space Aboard \space = \space 0) = 2/19 = 0.105 $

To evaluate the conditional probabilities given Survived = 1, we need to first consider a subset of the data.

| PClass    | Sex | Age | Parents/Children Aboard| Survived
| ----------- | ----------- |-------------|----------|---------|
|$\color{orange}3$ |$\color{green}female$ |$\color{violet}31$| $\color{brown}0$| $\color{blueviolet}1$|
|$\color{Gold}1$ |$\color{green}female$ |$\color{violet}31$| $\color{brown}0$| $\color{blueviolet}1$|
|$\color{Gold}1$ |$\color{green}female$ |$\color{violet}31$| $2$| $\color{blueviolet}1$|
|$\color{orange}3$ |$\color{green}female$ |$\color{violet}31$| $\color{olive}1$| $\color{blueviolet}1$|
|$\color{grey}2$ |$\color{green}female$ |$\color{violet}31$| $\color{brown}0$| $\color{blueviolet}1$|
|$\color{grey}2$ |$\color{blue}male$ |$\color{violet}31$| $\color{brown}0$| $\color{blueviolet}1$|
|$\color{gold}1$ |$\color{blue}male$ |$\color{violet}31$| $\color{brown}0$| $\color{blueviolet}1$|
|$\color{orange}3$ |$\color{blue}male$ |$\color{violet}31$| $\color{brown}0$| $\color{blueviolet}1$|
|$\color{grey}2$ |$\color{green}female$ |$\color{violet}31$| $\color{olive}1$| $\color{blueviolet}1$|
|$\color{gold}1$ |$\color{blue}male$ |$\color{red}52$| $\color{brown}0$| $\color{blueviolet}1$|
|$\color{gold}1$ |$\color{green}female$ |$\color{red}52$| $\color{brown}0$| $\color{blueviolet}1$|
|$\color{gold}1$ |$\color{green}female$ |$\color{red}52$| $\color{olive}1$| $\color{blueviolet}1$|

Individual probabilities are as below:
P(Sex = male) = 4/12 = 0.333
P(Age = 31) = 9/12 = 0.75
P(Pclass =1) = 6/12 = 0.5
P(Parents/Children Aboard=0) =  8/12 = 0.67

$Target Probability =$ 
$\frac{(0.333 * 0.75 * 0.5 * 0.67)}{0.105}\times 0.632$
$=  0.503$

The probability turns out to be marginally higher than 0.5, and we predict that the sample belongs to class 1 i.e the passenger with these attributes survived. 

What happens when we want to calculate the probability of survival of a female passenger aged 25 travelling by the 3rd class and with 0 parents/siblings?
We observe that the training data has no rows with females aged 25. If we were to proceed computing this in a similar manner
$Probability =$ 
$(0.67 * 0 * 0.25 * 0.67)\times0.632$ 
$=  0$

How would the algorithm in this case, predict the class of the new sample (As probabilities of the sample belonging to all classes would be 0)? This becomes particularly important for instance in the case of text classification, where we are trying to compare similarity of one document to another, and are likely to have many unseen words in the test sample. 
This is why we use a technique called Laplace smoothing. In estimating the parameters, we augment the numerator by a constant and the denominator by the constant*total number of categories, such that the probability does not become 0. 
For instance in the example above, if we used c=1, then the probability would be computed as 

$Probability =$ 
$\frac{(0.67 * 0.25 * 0.5 * 0.67)}{0.10}\times 0.632$
$=  0.355$

# Summarising types of Naive Bayes Classifiers


| Distribution of feature     | Number of parameters for each feature | Probability formula | Formula for parameter estimation| Some practical applications
| ----------- | ----------- |-------------|----------|---------|
| Bernoulli    | 2       |	$μ_{jc}^{(x_j)}(1 - μ_{jc})^{(1-x_j)}$|	$w_{jyr} = \genfrac (){}{0} {\displaystyleΣ_{i=1}^n 1(y^{(i)} =y_r)x_j^{(i)}+c}{\displaystyleΣ_{i=1}^n1(y^{(i)}=y_r)+2c}$|Binary feature values
| Categorical   | >2        |$μ_{j1c1}^{(xj=v1)}μ_{j2c1}^{(xj=v2)}μ_{j3c1}^{(xj=v3)}……..μ_{jec1}^{(xj=ve)}$|	$w_{jyr} = \genfrac (){}{0} {\displaystyleΣ_{i=1}^n 1(y^{(i)} =y_r)1(x_j^{(i)}=v)+c}{\displaystyleΣ_{i=1}^n1(y^{(i)}=y_r)+ce}$	| Image Classification
| Multinomial  | >2        |$\genfrac (){}{0}{l!}{x_1!x_2!…x_m!}\displaystyle∏_{j=1}^m μ_{jc}^{x_j}$	| $w_{jyr} = \genfrac (){}{0} {\displaystyleΣ_{i=1}^n 1(y^{(i)} =y_r)x_j^{(i)}+c}{\displaystyleΣ_{i=1}^n1(y^{(i)}=y_r)\displaystyleΣ_{j=1}^m x_j^{(i)}+2m}$	| Text classification, Spam filtering, Sentiment analysis
| Gaussian   | 2        |$\genfrac (){}{0} 1{\sqrt {2π}σ_{jc}} \ exp(\genfrac (){}{0}{-(x-μ_{jc})^2)}{2σ_{jc}^2}$| $μ_{jr} = \genfrac (){}{0} 1{n_r} 1(y^{(i)} =y_r)(x_j^{(i)}) \newline σ_{jr} = \genfrac (){}{0} 1{n_r} \displaystyleΣ_{i=1}^n1(y^{(i)}=y_r)(x_j^{(i)}-μ_{jr})^2$|Continuous features


## Advantages 
*	Can be used for small datasets (n<=m)
* Naïve Bayes classifiers generally have high bias and low variance. They generalise the
training dataset properties. For example in the Titanic dataset considered above, variables like
Name, Age and Fare take a large range of values and hence are highly prone to overfitting if we use
classifiers like Linear regression, logistic regression or Decision trees.  
* Naïve Bayes is a good
baseline model owing to it’s simple assumption and generalisation property.  Naïve Bayes models
train quickly. 
* Can be beneficial for data having categorical features (Bernoulli/Categorical NB)
* No requirement of transformation of non-linear datasets 

## Disadvantages 
* The features
need not necessarily be conditionally independent and the naive assumption fails in most cases.
* Could produce lower accuracy.  


# Further Reading
Sources: Generative vs discriminative models https://cs229.stanford.edu/summer2019/cs229-notes2.pdf
https://stats.stackexchange.com/questions/12421/generative-vs-discriminative Naïve Bayes vs Logistic
regression: https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf Naïve bayes as a
baseline classifier: https://www.cl.cam.ac.uk/teaching/1617/MLRD/handbook/nb.pdf Also refer:
https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote05.html
