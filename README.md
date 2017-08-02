# IBM Watson Developer Certification Study Guide

A one page study guide for the below Watson Certification exam:

[Exam Test C7020-230 - IBM Watson V3 Application Development](http://ibm.co/2liIJG1)

The certification exam covers all of the aspects of building an application that uses Watson services. This includes a basic understanding of cognitive technologies, as well as a practical knowledge of the core APIs. Get up to speed with these resources:
- [Cognitive Computing Primer](http://ibm.co/2k5PAxf)
- [Watson Developer Cloud API Documentation](http://ibm.co/2mU4Bnp)

The aim of this doc is to provide a more consolidated view of the required reading and study that is outlined in the [IBM Watson Professional Certification Program Study Guide Series](http://ibm.co/2iYtyP9). 

The Watson services are constantly evolving so always reference back to the [Watson Documentation](http://ibm.co/2mU4Bnp). Please also feel free to contribute or provide feedback if you see anything that is incorrect. 

[Watson is accessed through IBM Bluemix](http://ibm.co/2jdqk8s)
### [Check out and play with Watson services on Bluemix](http://bit.ly/2jtpOUB)
[![IBM Bluemix Watson](http://watson.codes/IBM_Watson_Logo_300.gif)](http://bit.ly/2jtpOUB)

## High-level Exam Objectives

- [Section 1 - Fundamentals of Cognitive Computing](#section-1---fundamentals-of-cognitive-computing)
 - [1.1 Define the main characteristics of a cognitive system.](#11-define-the-main-characteristics-of-a-cognitive-system)
 - [1.2 Explain neural nets.](#12-explain-neural-nets)
 - [1.3 Explain machine learning technologies (supervised, unsupervised, reinforcement learning approaches).](#13-explain-machine-learning-technologies)
 - [1.4 Define a common set of use cases for cognitive systems.](#14-define-a-common-set-of-use-cases-for-cognitive-systems)
 - [1.5 Define Precision, Recall, and Accuracy.](#15-define-precision-recall-and-accuracy)
 - [1.6 Explain the importance of separating training, validation and test data.](#16-explain-the-importance-of-separating-training-validation-and-test-data)
 - [1.7 Measure accuracy of service.](#17-measure-accuracy-of-service)
 - [1.8 Perform Domain Adaption using Watson Knowledge Studio (WKS).](#18-perform-domain-adaption-using-watson-knowledge-studio-wks)
 - [1.9 Define Intents and Classes.](#19-define-intents-and-classes)
 - [1.10 Explain difference between ground truth and corpus.](#110-explain-difference-between-ground-truth-and-corpus)
 - [1.11 Define the difference between the user question and the user intent.](#111-define-the-difference-between-the-user-question-and-the-user-intent)

- [Section 2 - Use Cases of Cognitive Services](#section-2---use-cases-of-cognitive-services)
 - [2.1 Select appropriate combination of cognitive technologies based on use-case and data format.](#21-select-appropriate-combination-of-cognitive-technologies-based-on-use-case-and-data-format)
 - [2.2 Explain the uses of the Watson services in the Application Starter Kits.](#22-explain-the-uses-of-the-watson-services-in-the-application-starter-kits)
 - [2.3 Describe the Watson Conversational Agent.](#22-explain-the-uses-of-the-watson-services-in-the-application-starter-kits)
 - [2.4 Explain use cases for integrating external systems (such as Twitter, Weather API).](#24-explain-use-cases-for-integrating-external-systems-such-as-twitter-weather-api)
 - [2.5 Describe the IBM Watson Discovery Service](#25-describe-the-ibm-watson-discovery-service)

- [Section 3 – Fundamentals of IBM Watson Developer Cloud](#section-3--fundamentals-of-ibm-watson-developer-cloud)
 - [3.1 Distinguish cognitive services on WDC for which training is required or not.](#31-distinguish-cognitive-services-on-wdc-for-which-training-is-required-or-not)
 - [3.2 Provide examples of text classification using the NLC.](#32-provide-examples-of-text-classification-using-the-nlc)
 - [3.3 Explain the Watson SDKs available as part of the services on Watson Developer Cloud.](#33-explain-the-watson-sdks-available-as-part-of-the-services-on-watson-developer-cloud)
 - [3.4 Explain the Watson REST APIs available as part of the services on Watson Developer Cloud.](#34-explain-the-watson-rest-apis-available-as-part-of-the-services-on-watson-developer-cloud)
 - [3.5 Explain and configure Natural Language Classification.](#35-explain-and-configure-natural-language-classification)
 - [3.6 Explain and configure Visual recognition.](#36-explain-and-configure-visual-recognition)
 - [3.7 Explain how Personality Insights service works.](#37-explain-how-personality-insights-service-works)
 - [3.8 Explain how Tone Analyzer service works.](#39-explain-and-execute-alchemy-language-services)
 - [3.9 Explain and execute IBM Watson Natural Language Understanding.](#39-explain-and-execute-ibm-watson-natural-language-understanding-services)
 - [3.10 Explain and configure Watson Discovery Service](#310explain-setup-configure-and-query-the-ibm-watson-discovery-service)
 - [3.11 Explain and configure Watson Conversation Service](#311explain-and-configure-the-ibm-watson-conversation-service)

- [Section 4 - Developing Cognitive applications using Watson Developer Cloud Services](#section-4---developing-cognitive-applications-using-watson-developer-cloud-services)
 
 - [4.1 Call a Watson API to analyze content.](#41-call-a-watson-api-to-analyze-content)
 - [4.2 Describe the tasks required to implement the Conversational Agent / Digital Bot.](#42-describe-the-tasks-required-to-implement-the-conversational-agent--digital-bot)
 - [4.3 Transform service outputs for consumption by other services.](#43-transform-service-outputs-for-consumption-by-other-services)
 - [4.4 Define common design patterns for composing multiple Watson services together (across APIs).](#44-define-common-design-patterns-for-composing-multiple-watson-services-together-across-apis)
 - [4.5 Design and execute a use case driven service choreography (within an API).](#45-design-and-execute-a-use-case-driven-service-choreography-within-an-api)
 - [4.6 Deploy a web application to IBM Bluemix.](#46-deploy-a-web-application-to-ibm-bluemix)
 - [4.7 Explain the advantages of using IBM Bluemix as the cloud platform for cognitive application development and deployment.](#47-explain-the-advantages-of-using-ibm-bluemix-as-the-cloud-platform-for-cognitive-application-development-and-deployment)

- [Section 5 - Administration & DevOps for applications using IBM Watson Developer Cloud Services](#section-5---administration--devops-for-applications-using-ibm-watson-developer-cloud-services)

 - [5.1 Describe the process of obtaining credentials for Watson services.](#51-describe-the-process-of-obtaining-credentials-for-watson-services)
 - [5.2 Examine application logs provided on IBM Bluemix.](#52-examine-application-logs-provided-on-ibm-bluemixs)

## Section 1 - Fundamentals of Cognitive Computing
### 1.1. Define the main characteristics of a cognitive system.

#### 1.1.1. Cognitive systems understand, reason and learn
##### 1.1.1.1. Must understand structured and unstructured data
##### 1.1.1.2. Must reason by prioritizing recommendations and ability to form hypothesis
##### 1.1.1.3. Learns iteratively by repeated training as it build smarter patterns
#### 1.1.2. Cognitive systems are here to augment human knowledge not replace it
#### 1.1.3. Cognitive systems employ machine learning technologies
#### 1.1.4. Cognitive systems use natural language processing

### 1.2 Explain neural nets.

https://github.com/cazala/synaptic/wiki/Neural-Networks-101

- Neural Nets mimic how neurons in the brain communicate. 
- Neural networks are models of biological neural structures. 

Neurons are the basic unit of a neural network. In nature, neurons have a number of dendrites (inputs), a cell nucleus (processor) and an axon (output). When the neuron activates, it accumulates all its incoming inputs, and if it goes over a certain threshold it fires a signal thru the axon.. sort of. The important thing about neurons is that they can learn.

Artificial neurons look more like this:

![Artificial neurons](https://camo.githubusercontent.com/8b87e593fb9382c16a81cc059d994adec259a1c4/687474703a2f2f692e696d6775722e636f6d2f643654374b39332e706e67)

Video:
[![Neural Networks Demystified - Part 1: Data and Architecture](https://i.ytimg.com/vi/bxe2T-V8XRs/maxresdefault.jpg)](https://www.youtube.com/watch?v=bxe2T-V8XRs?v=VID)
[Neural Networks Demystified - Part 1: Data and Architecture](https://www.youtube.com/watch?v=bxe2T-V8XRs?v=VID)

So how does a Neural Network learn?
A neural network learns by training. The algorithm used to do this is called backpropagation. After giving the network an input, it will produce an output, the next step is to teach the network what should have been the correct output for that input (the ideal output). The network will take this ideal output and start adjusting the weights to produce a more accurate output next time, starting from the output layer and going backwards until reaching the input layer. So next time we show that same input to the network, it's going to give an output closer to that ideal one that we trained it to output. This process is repeated for many iterations until we consider the error between the ideal output and the one output by the network to be small enough.

#### 1.2.1.1. Explain the role of synapse and neuron

- A nueron operates by recieving signals from other nuerons through connections called synapses. 

#### 1.2.1.2. Understand weights and bias

- For each nueron input there is a weight (the weight of that specific connection).
- When a artifical neuron activates if computes its state by adding all the incoming inputs multiplied by it's corresponding connection weight. 
- But Neurons always have one extra input, the bias which is always 1 and has it's own connection weight. THis makes sure that even when all other inputs are none there's going to be activation in the nueron. 

![Weights and Bias](https://qph.ec.quoracdn.net/main-qimg-31d260a826ec73fce99ae098be5a7351)

#### 1.2.1.3. List various approaches to neural nets

[Types of artificial neural networks](https://en.wikipedia.org/wiki/Types_of_artificial_neural_networks)

#### 1.2.1.4. Explain forward and backward propagation

##### Feed Forward Propagation 

A feedforward neural network is an artificial neural network wherein connections between the units do not form a cycle. As such, it is different from recurrent neural networks.
In this network, the information moves in only one direction, forward, from the input nodes, through the hidden nodes (if any) and to the output nodes. There are no cycles or loops in the network.

Video:
[![Neural Networks Demystified Part 2: Forward Propagation](https://i.ytimg.com/vi/UJwK6jAStmg/maxresdefault.jpg)](https://www.youtube.com/watch?v=UJwK6jAStmg?v=VID)
[Neural Networks Demystified Part 2: Forward Propagation](https://www.youtube.com/watch?v=UJwK6jAStmg?v=VID)

##### Back Propagation 

Backpropagation, an abbreviation for "backward propagation of errors", is a common method of training artificial neural networks used in conjunction with an optimization method such as gradient descent. It calculates the gradient of a loss function with respect to all the weights in the network, so that the gradient is fed to the optimization method which in turn uses it to update the weights, in an attempt to minimize the loss function.

Backpropagation requires a known, desired output for each input value in order to calculate the loss function gradient – it is therefore usually considered to be a supervised learning method; nonetheless, it is also used in some unsupervised networks such as autoencoders. It is a generalization of the delta rule to multi-layered feedforward networks, made possible by using the chain rule to iteratively compute gradients for each layer. Backpropagation requires that the activation function used by the artificial neurons (or "nodes") be differentiable.

But how does the backpropagation work?
This algorithm adjusts the weights using Gradient Descent calculation. Let's say we make a graphic of the relationship between a certain weight and the error in the network's output:

![Artificial neurons](https://camo.githubusercontent.com/e6a0e02bd080acc585a622d2c03ca6e44a9e9adc/687474703a2f2f692e696d6775722e636f6d2f36565a6542706e2e706e67)

This algorithm calculates the gradient, also called the instant slope (the arrow in the image), of the actual value of the weight, and it moves it in the direction that will lead to a lower error (red dot in the image). This process is repeated for every weight in the network.

The goal of any supervised learning algorithm is to find a function that best maps a set of inputs to its correct output. An example would be a classification task, where the input is an image of an animal, and the correct output would be the name of the animal.
The goal and motivation for developing the backpropagation algorithm was to find a way to train a multi-layered neural network such that it can learn the appropriate internal representations to allow it to learn any arbitrary mapping of input to output.

Video:
[![Neural Networks Demystified - Part 4: Backpropagation](https://i.ytimg.com/vi/GlcnxUlrtek/maxresdefault.jpg)](https://www.youtube.com/watch?v=GlcnxUlrtek?v=VID)
[Neural Networks Demystified - Part 4: Backpropagation](https://www.youtube.com/watch?v=GlcnxUlrtek?v=VID)

##### 1.2.1.5 Gradient Descent 

Video:
[![Neural Networks Demystified - Part 3: Gradient Descent](https://i.ytimg.com/vi/5u0jaA3qAGk/maxresdefault.jpg)](https://www.youtube.com/watch?v=5u0jaA3qAGk?v=VID)
[Neural Networks Demystified - Part 3: Gradient Descent](https://www.youtube.com/watch?v=5u0jaA3qAGk?v=VID)

(ivp: see also stochastic vs mini-batches, epoch)

### 1.3 Explain machine learning technologies (supervised, unsupervised, reinforcement learning approaches).

##### 1.3.1. Explain the connection between Machine learning and Cognitive systems
Reference: [Computing, cognition and the future of knowing](http://www.research.ibm.com/software/IBMResearch/multimedia/Computing_Cognition_WhitePaper.pdf)

Machine learning is a branch of the larger discipline of Artificial Intelligence, which involves the design and construction of computer applications or systems that are able to learn based on their data inputs and/or outputs. The discipline of machine learning also incorporates other data analysis disciplines, ranging from predictive analytics and data mining to pattern recognition. And a variety of specific algorithms are used for this purpose, frequently organized in taxonomies, these algorithms can be used depending on the type of input required. 

Many products and services that we use every day from search-engine advertising applications to facial recognition on social media sites to “smart” cars, phones and electric grids are beginnin to demonstrate aspects of Artificial Intelligence. Most consist of purpose-built, narrowly focused applications, specific to a particular service. They use a few of the core capabilities of cognitive
computing. Some use text mining. Others use image recognition with machine learning. Most are limited to the application for which they were conceived. 

Cognitive systems, in contrast, combine five core capabilities:
- 1. They create deeper human engagement.
- 2. They scale and elevate expertise.
- 3. They infuse products and services with cognition.
- 4. They enable cognitive processes and operations.
- 5. They enhance exploration and discovery.

Large-scale machine learning is the process by which cognitive systems improve with training and use.

Cognitive computing is not a single discipline of computer science. It is the combination of multiple academic fields, from hardware architecture to algorithmic strategy to process design to industry expertise.

Many of these capabilities require specialized infrastructure that leverages high-performance computing, specialized hardware architectures and even new computing paradigms. But these technologies must be developed in concert, with hardware, software, cloud platforms and applications that are built expressly to work together in support of cognitive solutions.

##### 1.3.2. Describe some of the main machine learning concepts:

- [A Tour of Machine Learning Algorithms](http://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/)
- [List of machine learning concepts](https://en.wikipedia.org/wiki/List_of_machine_learning_concepts)
- [Supervised learning, unsupervised learning and reinforcement learning: Workflow basics](http://stats.stackexchange.com/questions/144154/supervised-learning-unsupervised-learning-and-reinforcement-learning-workflow)

##### 1.3.2.1. Supervised learning:

- We are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and output. 

##### 1.3.2.1.1.Classification

- In a classification problem we are trying to predict results in a discrete output. In other words we are trying to map input variables into categories. 

##### 1.3.2.1.2.Regression/Prediction

- In a regression problem we are trying to predict results with a continous output meaning that we are trying to map input variables to some continous function. 

##### 1.3.2.1.3.Semi-supervised learning

- Semi-supervised learning are tasks and techniques that also make use of unlabeled data for training – typically a small amount of labeled data with a large amount of unlabeled data.

##### 1.3.2.2. Unsupervised learning:

- Unsupervised learning is a type of machine learning algorithm used to draw inferences from datasets consisting of input data without labeled responses. Input data is not labeled and does not have a known result.

##### 1.3.2.2.1. Artificial neural network

- An Artificial Neural Network (ANN) is an information processing paradigm that is inspired by the way biological nervous systems, such as the brain, process information. Different kinds of ANN can be used for supervised or unsupervised learning. An example of ANN for unsupervised learning - autoencoder.

##### 1.3.2.2.2.Association rule learning

- Association rule learning is a rule-based machine learning method for discovering interesting relations between variables in large databases. It is intended to identify strong rules discovered in databases using some measures of interestingness. 
(see also lectures form AI@edX
https://www.youtube.com/watch?v=MZtHAy3mpnk
https://www.youtube.com/watch?v=wdaGeXRJpeQ)

##### 1.3.2.2.3.Hierarchical clustering

- Hierarchical clustering is a method of cluster analysis which seeks to build a hierarchy of clusters. Strategies for hierarchical clustering generally fall into two types:

 - Agglomerative: This is a "bottom up" approach: each observation starts in its own cluster, and pairs of clusters are merged as one moves up the hierarchy.
 - Divisive: This is a "top down" approach: all observations start in one cluster, and splits are performed recursively as one moves down the hierarchy.
 
 ![Clustered Iris dataset](https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Iris_dendrogram.png/800px-Iris_dendrogram.png)

##### 1.3.2.2.4.Cluster analysis

- Cluster analysis or clustering is the task of grouping a set of objects in such a way that objects in the same group (called a cluster) are more similar (in some sense or another) to each other than to those in other groups (clusters). 
(for example see K-Means algorithm https://www.youtube.com/watch?v=_aWzGGNrcic)

##### 1.3.2.2.5.Outlier Detection
In data mining, anomaly detection (also outlier detection) is the identification of items, events or observations which do not conform to an expected pattern or other items in a dataset. Typically the anomalous items will translate to some kind of problem such as bank fraud, a structural defect, medical problems or errors in a text. Anomalies are also referred to as outliers, novelties, noise, deviations and exceptions.

Three broad categories of anomaly detection techniques exist. Unsupervised anomaly detection techniques detect anomalies in an unlabeled test data set under the assumption that the majority of the instances in the data set are normal by looking for instances that seem to fit least to the remainder of the data set. Supervised anomaly detection techniques require a data set that has been labeled as "normal" and "abnormal" and involves training a classifier (the key difference to many other statistical classification problems is the inherent unbalanced nature of outlier detection). Semi-supervised anomaly detection techniques construct a model representing normal behavior from a given normal training data set, and then testing the likelihood of a test instance to be generated by the learnt model.

https://en.wikipedia.org/wiki/Anomaly_detection
https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm#k-NN_outlier

-------
- The local outlier factor is based on a concept of a local density, where locality is given by {\displaystyle k} k nearest neighbors, whose distance is used to estimate the density. By comparing the local density of an object to the local densities of its neighbors, one can identify regions of similar density, and points that have a substantially lower density than their neighbors. These are considered to be outliers.

##### 1.3.2.3. Reinforcement learning
- These algorithms choose an action, based on each data point and later learn how good the decision was. Over time, the algorithm changes its strategy to learn better and achieve the best reward. Thus, reinforcement learning is particularly well-suited to problems which include a long-term versus short-term reward trade-off. 

### 1.4. Define a common set of use cases for cognitive systems.

Customer Call Centers
#### 1.4.1. Agent Assist: email based Q&A
1.4.1.1. **Problem Solved**: Provides a natural language help system for call agents to rapidly retrieve answers to customer questions

1.4.1.2. **Capabilities needed for this use case**: Conversation, natural language answer retrieval, keyword extraction, and entity extraction

1.4.1.3. **Services used**: Natural Language Classifier, Watson Discovery

1.4.1.4. **Benefits**:

1.4.1.4.1. Detect the topic of a ticket and route to the appropriate department to handle it (Example: room service, maintenance, housekeeping)

1.4.1.4.2. Escalate support tickets based on customer sentiment

1.4.1.4.3. Route support requests to agents that already solved similar problems by detecting natural language similarities between new customer tickets and resolved ones.

1.4.1.4.4. Automation: Customer/Technical Support Tickets
Routing

1.4.1.5. **Watson Services used**: natural language (text) classification, keyword extraction, entity extraction, and sentiment/tone analysis

#### 1.4.2. Physicians
##### 1.4.2.1. Expert Advisor:
1.4.2.1.1. **Example**: Watson Discovery Advisor

1.4.2.1.2. **Problem Solve**d: Provides relevant medical suggestions and insights in natural language so
physicians can more accurately diagnose patients.
1.4.2.1.3. **Services used**: Conversation, Natural Language Classifier, Watson Discovery
 
#### 1.4.3. Social Media Data Insights:

1.4.3.1. **Partner**: Ground Signal

1.4.3.2. **Problem Solved**: Extract useful insights from social media such as Instagram and Twitter by determining the content of photos and topics/sentiment of user posts.

 1.4.3.3. **Services used**: Natural Language Classifier, Watson Discovery, Visual Recognition
  
### 1.5. Define Precision, Recall, and Accuracy.

#### 1.5.1. [Precision:](#https://en.wikipedia.org/wiki/Precision_and_recall)
- Definition: Precision is the percentage of documents labelled as positive that are actually positive.
- Formula: True Positives/(True Positives + False Positives)

#### 1.5.2. [Recall:](#https://en.wikipedia.org/wiki/Precision_and_recall)
- Recall is the percent of documents labelled as positive were successfully retrieved.
- Formula: True Positives/(True Positives + False Negatives)

#### 1.5.3. Accuracy:
- Accuracy is the fraction of documents relevant to a query that were successfully retrieved.
- Formula: (True Positives + True Negatives)/Total Document Count

see also: https://nlp.stanford.edu/IR-book/pdf/08eval.pdf

#### 1.5.4. Diagrams like this are often useful in capturing the True/False
Positive/Negatives described above:
[https://www.quora.com/What-is-the-best-way-to-understand-the-terms-precision-and-recall](#https://www.quora.com/What-is-the-best-way-to-understand-the-terms-precision-and-recall)

### 1.6. Explain the importance of separating training, validation and test data.

#### 1.6.1. One school of thoughts: partition the data into above three

Normally to perform supervised learning you need two types of data sets:
 1. In one dataset (your "gold standard") you have the input data together with correct/expected output, This dataset is usually duly prepared either by humans or by collecting some data in semi-automated way. But it is important that you have the expected output for every data row here, because you need for supervised learning.
 2. The data you are going to apply your model to. In many cases this is the data where you are interested for the output of your model and thus you don't have any "expected" output here yet.

While performing machine learning you do the following:
 1. Training phase: you present your data from your "gold standard" and train your model, by pairing the input with expected output.
 2. Validation/Test phase: in order to estimate how well your model has been trained (that is dependent upon the size of your data, the value you would like to predict, input etc) and to estimate model properties (mean error for numeric predictors, classification errors for classifiers, recall and precision for IR-models etc.)
 3. Application phase: now you apply your freshly-developed model to the real-world data and get the results. Since you normally don't have any reference value in this type of data (otherwise, why would you need your model?), you can only speculate about the quality of your model output using the results of your validation phase.

The validation phase is often split into two parts:

 1. In the first part you just look at your models and select the best performing approach using the validation data (=validation)
 2. Then you estimate the accuracy of the selected approach (=test).

Hence the separation to 50/25/25.

In case if you don't need to choose an appropriate model from several rivaling approaches, you can just re-partition your set that you basically have only training set and test set, without performing the validation of your trained model. I personally partition them 70/30 then.

#### 1.6.2 Another: Usinh Bootstrap method vs Cross Validation
##### 1.6.2.1. the bootstrap method allows us to simulate the process of obtaining new data sets, so that we can estimate the error/ variability of our estimate without generating additional samples

(ivp:
resampling methods: cross-validation and the bootstrap. These methods refit a model of interest to samples formed
from the training set, in order to obtain additional
information about the fitted model.

Сross validation is a technique for validating the model performance, and it’s done by split the training data into k parts. We take k-1 parts as our training set and use the “held out” part as our test set. We repeat that k times differently (we hold out different part every time). Finally we take the average of the k scores as our performance estimation.

Cross validation can suffer bias or variance. if we increase the number of splits (k), the variance will increase and bias will decrease. On contrast, if we decrease (k), the bias will increase and variance will decrease. Generally 10-fold CV is used but of course it depends on the size of the training data.
----
Bootstrapping is a technique that helps in many situations like validation of a predictive model performance, ensemble methods, estimation of bias and variance of the model.

It works by sampling with replacement from the original data, and take the “not chosen” data points as test cases (or use the whole original dataset as test set (ivp)). We can make this several times and calculate the average score as estimation of our model performance.

-----
1. ‘simple’ bootstrap. This involves creating resamples with replacement from the original data, of the same size; applying the modelling strategy to the resample; using the model to predict the values of the full set of original data and calculating a goodness of fit statistic (eg either R-squared or root mean squared error) comparing the predicted value to the actual value. Note - Following Efron, Harrell calls this the “simple bootstrap”, but other authors and the useful caret package use “simple bootstrap” to mean the resample model is used to predict the out-of-bag values at each resample point, rather than the full original sample.
2. ‘enhanced’ bootstrap. This is a little more involved and is basically a method of estimating the ‘optimism’ of the goodness of fit statistic. There’s a nice step by step explanation by thestatsgeek which I won’t try to improve on.
3. repeated 10-fold cross-validation. 10-fold cross-validation involves dividing your data into ten parts, then taking turns to fit the model on 90% of the data and using that model to predict the remaining 10%. The average of the 10 goodness of fit statistics becomes your estimate of the actual goodness of fit. One of the problems with k-fold cross-validation is that it has a high variance ie doing it different times you get different results based on the luck of you k-way split; so repeated k-fold cross-validation addresses this by performing the whole process a number of times and taking the average.
)

#### 1.6.3. Training Process: 

##### 1.6.3.1. Data = Training Data + Cross-Validation Data + Test Data 

(ivp: 

Well, most ML models are described by two sets of parameters. The 1st set consists in “regular” parameters that are “learned” through training. The other parameters, called hyperparameters or meta-parameters are parameters which values are set before the learning starts (think, for example, the learning rate, the regularisation parameter, the number of layers or neurons in a layer for ANN etc.)

Obviously, different values for those parameters may lead to different (sometimes by a lot) generalisation performance for our Machine Learning model therefore we need to identify a set of optimal values for them and this is done by training multiple models with different values for the hyperparameters (how to chose those values falls under the name of hyperparameter optimisation - Hyperparameter optimization - Wikipedia)

Now, imagine you have you data and you need to run a supervised ML algorithm on it. You split the data into:

training - this is the data for which your algorithm knows the “labels” and which you will feed it to the training process to build your model.
test - this is a portion of the data that you keep hidden from your algorithm and only use it after the training takes places to compute some metrics that can give you a hint on how your algorithm behaves. For each item in you test dataset you predict its “value” using the built model and compare against the real “value”
Now, back to the context of hyperparameter optimisation. If you run the same algorithm (train on training, evaluate on test) for multiple sets of hyperparameters and chose the model with the best “performance” on the test set you risk overfitting this test set. To avoid this problem of overfitting the test data, the training set is split once more into:

actual training - a subset of the training set that is used to optimise the model
validation - another subset of the training set that is used to evaluate the model performance for each run / set of hyperparameter values.
Multiple training sessions are run on the actual training set, for various hyperparameter values and the models are evaluated agains the validation dataset. The model with the best performance is then chosen - remember that so far the algorithm has not yet seen the test data therefore there is no suspicion of overfitting it.

After choosing the best model (and implicitly the values for the hyperparameters) this model is evaluated agains the test dataset and the performance is reported.

Long story short: split your data into 3 subsets: training, validation, test. Train multiple variations of your model on the training dataset, chose the one with the best performance on the validation set and report how it generalise to the test set (important - the test set is kept hidden throughout the training process).)

###### 1.6.3.1.1. Data = Inputs + Outputs 
###### 1.6.3.1.2. Input + Output Sets  a set of functions that map input to output 
##### 1.6.3.2. We train these functions using the training data 
##### 1.6.3.3. We select which function gives less errors or better classification or prediction by feeding them each the validation data / blind (cross-validate)  
##### 1.6.3.4. We select the best outcome 
##### 1.6.3.5. We test the best outcome (function / neural net (weights,etc)) with the test data 
 
### 1.7. Measure accuracy of service.

The goal of the ML model is to learn patterns that generalize well for unseen data instead of just memorizing the data that it was shown during training. Once you have a model, it is important to check if your model is performing well on unseen examples that you have not used for training the model. To do this, you use the model to predict the answer on the evaluation dataset (held out data) and then compare the predicted target to the actual answer (ground truth).

A number of metrics are used in ML to measure the predictive accuracy of a model. The choice of accuracy metric depends on the ML task. It is important to review these metrics to decide if your model is performing well.

###### 1.7.2.1  Sample size of training set data, dangers of over fitting, curated content 
Small Data problems
Problems of small-data are numerous, but mainly revolve around high variance:
Over-fitting becomes much harder to avoid
You don’t only over-fit to your training data, but sometimes you over-fit to your validation set as well.
Outliers become much more dangerous.
Noise in general becomes a real issue, be it in your target variable or in some of the features.

#### 1.7.3. Explain factors that affect accuracy of unsupervised learning 
##### 1.7.3.1. Sample size, curse of dimensionality, over/under fitting 
Curse of dimentionality: With a fixed number of training samples, the predictive power reduces as the dimensionality increases, and this is known as Hughes phenomenon
##### 1.7.4. Running a Blind set test 

**Training Set**: Data used to train the ML algorithm. The developer may
also look at this data to help design the system. This is usually the
largest subset.
**Tuning Set**: Data set aside to assess how well the program performs on
unseen data and/or to set parameters. Helps to minimize overfitting.
Blind Test Set: Data set aside to perform a final evaluation of how well
the program performs on new data. The developer should never look
at these texts!

https://www.ibm.com/watson/developercloud/doc/wks/improve-ml.html#wks_mamanagedata 

Why do I need a blind set? 
Because you use test data to assess accuracy in detail, you get to know the documents and their features after a while. For example, you start to know which entity types, relation types, and text types in the documents are best understood by the machine learning model, and which are not. This information is important because it helps you focus on making the right improvements - refining the type system, supplementing the training data to fill gaps, or adding dictionaries, for example. As the test documents get used iteratively to improve the model, they can start to influence the model training indirectly. That's why the "blind" set of documents is so important.

#### 1.7.5. Importance of iterative training using feedback that has diminished costs derivative  
(ivp: TODO)

### 1.8. Perform Domain Adaption using Watson Knowledge Studio (WKS).

There is a great YouTube video series for Watson Knowledge Studio here:

Video:
[![Teach Watson with Watson Knowledge Studio](https://i.ytimg.com/vi/XBwpU97D5aE/maxresdefault.jpg)](https://www.youtube.com/watch?v=XBwpU97D5aE?v=VID)
[Teach Watson with Watson Knowledge Studio](https://www.youtube.com/watch?v=XBwpU97D5aE?v=VID)

IBM Watson Knowledge Studio is a cloud-based application that enables developers and domain experts to collaborate on the creation of custom annotator components that can be used to identify mentions and relations in unstructured text.
Watson Knowledge Studio is:
- Intuitive: Use a guided experience to teach Watson nuances of natural language without writing a single line of code
- Collaborative: SMEs work together to infuse domain knowledge in cognitive applications

Use Watson™ Knowledge Studio to create a machine-learning model that understands the linguistic nuances, meaning, and relationships specific to your industry.

To become a subject matter expert in a given industry or domain, Watson must be trained. You can facilitate the task of training Watson with Watson Knowledge Studio. It provides easy-to-use tools for annotating unstructured domain literature, and uses those annotations to create a custom machine-learning model that understands the language of the domain. The accuracy of the model improves through iterative testing, ultimately resulting in an algorithm that can learn from the patterns that it sees and recognize those patterns in large collections of new documents.

The following diagram illustrates how it works.
[![Watson™ Knowledge Studio](https://www.ibm.com/watson/developercloud/doc/wks/images/wks-ovw-anno.png)]

1. Based on a set of domain-specific source documents, the team creates a type system that defines entity types and relation types for the information of interest to the application that will use the model.
2. A group of two or more human annotators annotate a small set of source documents to label words that represent entity types, words that represent relation types between entity mentions, and to identify coreferences of entity types. Any inconsistencies in annotation are resolved, and one set of optimally annotated documents is built, which forms the ground truth.
3. The ground truth is used to train a model.
4. The trained model is used to find entities, relations, and coreferences in new, never-seen-before documents.

Deliver meaningful insights to users by deploying a trained model in other Watson cloud-based offerings and cognitive solutions.

Watson services integration

Share domain artifacts and models between IBM Watson Knowledge Studio and other Watson services.

Use Watson Knowledge Studio to perform the following tasks:
- Bootstrap annotation by using the AlchemyLanguage entity extraction service to automatically find and annotate entities in your documents. When human annotators begin to annotate the documents, they can see the annotations that were already made by the service and can review and add to them. See Pre-annotating documents with IBM AlchemyLanguage for details.
- Import industry-specific dictionaries that you downloaded from IBM® Bluemix® Analytics Exchange.
- Import analyzed documents that are in UIMA CAS XMI format. For example, you can import UIMA CAS XMI files that were exported from IBM Watson Explorer content analytics collections or IBM Watson Explorer Content Analytics Studio.
- Deploy a trained model to use with the AlchemyLanguage service.
- Export a trained model to use in IBM Watson Explorer.

### 1.9. Define Intents and Classes.

- The Natural Language Classifier service available via WDC, enables clustering or classification based on some measure of inherent similarity or distance given the input data. Such clustering is known as intents or classes.

- Where classes may include images, intent is a similar clustering for written utterances in unstructured natural language format.

### 1.10. Explain difference between ground truth and corpus.

- Ground truth is used in both supervised and unsupervised machine learning approaches, yet portray different values and formats. For example, in a typical supervised learning system, ground truth consisted of inputs (questions) and approved outputs (answers). With the aide of logistical regression and iterative training the system improves in accuracy.

- In unsupervised approach, such as NLC, the ground truth consists of a comma-separated csv or a JSON file that lists hundreds of sample utterances and a dozen or so intents (or classes) classifying those utterances.

### 1.11. Define the difference between the user question and the user intent.

To answer correctly, we need to understand the intent behind the question, in order to first classify it then take action on it (e.g., with a Dialog API)
- The user question is the verbatim question
- The user intent maps the user question to a known classification
- This is a form of classifying question based on search goals
- Intents are the superset of all actions your users may want your cognitive system to undertake. Put another way, questions are a subset of user intents. Questions usually end in "?", but sometimes we need to extract the user intent from the underlying context.
 - Common examples of user intents:
  - Automation: “Schedule a meeting with Sue at 5pm next Tuesday.”
  - Declarative: “I need to change my password.”
  - Imperative: “Show me the directions to my the nearest gas station.”
  
## Section 2 - Use Cases of Cognitive Services
### 2.1. Select appropriate combination of cognitive technologies based on use-case and data format.
#### 2.1.1. Agent-assist for email-based customer call center
##### 2.1.1.1. Data: customer emails
##### 2.1.1.2. Features needed: Q&A, Text classification, entity extraction and, keyword extraction
##### 2.1.1.3. Watson-specific Services addressing the requirements: Natural Language Classifier, Watson Discovery, Natural Language Understanding
#### 2.1.2. Agent-assist for phone-based customer call center
##### 2.1.2.1. Data: customer voice recordings
##### 2.1.2.2. Features needed: Q&A, Speech recognition, text-to-speech, text classification, entity extraction, keyword extraction
##### 2.1.2.3. Watson-specific Services addressing the requirements: Watson Conversation, Watson Discovery, Natural Language Understanding, Watson Text to Speech, Watson Speech to Text
#### 2.1.3. Expert advisor use case for physicians
##### 2.1.3.1. Data: natural language intents
##### 2.1.3.2. Features needed: Q&A, Text classification, entity extraction and keyword extraction
##### 2.1.3.3. Watson-specific Services addressing the requirements: Natural Language Classifier, Watson Conversation, Watson Discovery, Natural Language Understanding
#### 2.1.4. Data insights for Instagram images
##### 2.1.4.1. Data: images
##### 2.1.4.2. Features needed: Image classification and natural OCR
##### 2.1.4.3. Watson-specific: Visual Recognition
#### 2.1.5. Data insights for Twitter
##### 2.1.5.1. Data: tweets
##### 2.1.5.2. Features needed: Text classification, entity extraction, keyword extraction, personality profile
##### 2.1.5.3. Watson-specific Services addressing the requirements: Natural Language Classifier, Natural Language Understanding, Personality Insights and Watson Discovery

#### 2.2. Explain the uses of the Watson services in the Application Starter Kits.

##### Chatbot with Long Tail Search
Use a conversational interface to answer both simple, common questions and complex, less common questions by adding search capability to a chatbot application.

Services Used: Discovery, Conversation

##### News Intelligence
Build applications that uncover insights from pre-enriched news content. Use a dashboard to visualize the latest connections and trends for companies mentioned in the news.

Services Used: Discovery

##### Social Customer Care
Social Customer Care monitors social media, understands brand customer needs or requests and responds proactively.

Services Used: Personality Insights, Natural Language Classifier, Tone Analyzer

##### Text Message Chatbot
This starter kit uses Watson Conversation, Watson Natural Language Understanding, and the Weather API to demonstrate how to create an intuitive natural language conversation chatbot that connects to other services.

Services Used: Conversation, Natural Language Understanding

##### Voice of the Customer
Analyze consumer reviews and extract valuable insights.

Services Used: Discovery

##### Knowledge Base Search
Use cognitive search to uncover the best answers to natural language questions by taking advantage of the Discovery Service's embedded natural language processing and powerful query language.

Services Used: Discovery

##### Answer Retrieval
Find and surface the most relevant responses to natural language queries from a large set of unstructured data.

Services Used: Retrieve and Rank

[You can view the list of Watson Starter Kits here](https://www.ibm.com/watson/developercloud/starter-kits.html)

### 2.3. Describe the Watson Conversational Agent.

For section 2.2 and 2.3, we deep dive into the Watson services currently available and stated in the study guide. By understanding the services individually, it will help with knowing what services would work for different scenarios. 

## (Unsorted)


### [Natural Language Classifier](https://www.ibm.com/watson/developercloud/doc/natural-language-classifier/index.html)

The IBM Watson™ Natural Language Classifier service uses machine learning algorithms to return the top matching predefined classes for short text inputs. The service interprets the intent behind text and returns a corresponding classification with associated confidence levels. The return value can then be used to trigger a corresponding action, such as redirecting the request or answering a question.

##### Intended Use

The Natural Language Classifier is tuned and tailored to short text (1000 characters or less) and can be trained to function in any domain or application.

- Tackle common questions from your users that are typically handled by a live agent.
- Classify SMS texts as personal, work, or promotional
- Classify tweets into a set of classes, such as events, news, or opinions.
- Based on the response from the service, an application can control the outcome to the user. For example, you can start another application, respond with an answer, begin a dialog, or any number of other possible outcomes.

Here are some other examples of how you might apply the Natural Language Classifier:
- Twitter, SMS, and other text messages
 - Classify tweets into a set of classes, such as events, news, or opinions.
 -  Analyze text messages into categories, such as Personal, Work, or Promotions.
- Sentiment analysis
 - Analyze text from social media or other sources and identify whether it relates positively or negatively to an offering or service.

##### You input
Text to a pre-trained model

##### Service output
Classes ordered by confidence

##### How to use the service
The process of creating and using the classifier:
![Natural Language Classifier](https://www.ibm.com/watson/developercloud/doc/natural-language-classifier/images/classifier_process.png)

##### CSV training data file format
Make sure that your CSV training data adheres to the following format requirements:
- The data must be UTF-8 encoded.
- Separate text values and each class value by a comma delimiter. Each record (row) is terminated by an end-of-line character, which is a special character or sequence of characters that indicate the end of a line.
- Each record must have one text value and at least one class value.
- Class values cannot include tabs or end-of-line characters.
- Text values cannot contain tabs or new lines without special handling. To preserve tabs or new lines, escape a tab with \t, and escape new lines with \r, \n or \r\n.
- For example, Example text\twith a tab is valid, but Example text    with a tab is not valid.
- Always enclose text or class values with double quotation marks in the training data when it includes the following characters:
- Commas ("Example text, with comma").
- Double quotation marks. In addition, quotation marks must be escaped with double quotation marks ("Example text with ""quotation""").

##### Size limitations
There are size limitations to the training data:
- The training data must have at least five records (rows) and no more than 15,000 records.
- The maximum total length of a text value is 1024 characters.

##### Supported languages

The classifier supports English (en), Arabic (ar), French (fr), German (de), Italian (it), Japanese (ja), Korean (ko), Portuguese (Brazilian) (pt), and Spanish (es). The language of the training data must match the language of the text that you intend to classify. Specify the language when you create the classifier.

##### Guidelines for good training
The following guidelines are not enforced by the API. However, the classifier tends to perform better when the training data adheres to them:
- Limit the length of input text to fewer than 60 words.
- Limit the number of classes to several hundred classes. Support for larger numbers of classes might be included in later versions of the service.
- When each text record has only one class, make sure that each class is matched with at least 5 - 10 records to provide enough training on that class.
- It can be difficult to decide whether to include multiple classes for a text. Two common reasons drive multiple classes:
- When the text is vague, identifying a single class is not always clear.
- When experts interpret the text in different ways, multiple classes support those interpretations.
- However, if many texts in your training data include multiple classes, or if some texts have more than three classes, you might need to refine the classes. For example, review whether the classes are hierarchical. If they are hierarchical, include the leaf node as the class.

[More detailed documentation for Natural Language Classifier](https://www.ibm.com/watson/developercloud/doc/natural-language-classifier/index.html)

### [Tone Analyzer](https://www.ibm.com/watson/developercloud/doc/tone-analyzer/index.html)

The IBM Watson™ Tone Analyzer Service uses linguistic analysis to detect three types of tones from text: emotion, social tendencies, and language style. Emotions identified include things like anger, fear, joy, sadness, and disgust. Identified social tendencies include things from the Big Five personality traits used by some psychologists. These include openness, conscientiousness, extroversion, agreeableness, and emotional range. Identified language styles include confident, analytical, and tentative.

##### Intended Use

##### Common uses for the Tone Analyzer service include:

- Personal and business communications - Anyone could use the Tone Analyzer service to get feedback about their communications, which could improve the effectiveness of the messages and how they are received.
- Message resonance - optimize the tones in your communication to increase the impact on your audience
- Digital Virtual Agent for customer care - If a human client is interacting with an automated digital agent, and the client is agitated or angry, it is likely reflected in the choice of words they use to explain their problem. An automated agent could use the Tone Analyzer Service to detect those tones, and be programmed to respond appropriately to them.
- Self-branding - Bloggers and journalists could use the Tone Analyzer Service to get feedback on their tone and fine-tune their writing to reflect a specific personality or style.

##### You input
Any Text

You submit JSON, plain text, or HTML input that contains your written content to the service. The service accepts up to 128 KB of text, which is about 1000 sentences. The service returns JSON results that report the tone of your input. You can use these results to improve the perception and effectiveness of your communications, ensuring that your writing conveys the tone and style that you want for your intended audience. The following diagram shows the basic flow of calls to the service.

##### Service output
JSON that provides a hierarchical representation of the analysis of the terms in the input message

example https://tone-analyzer-demo.mybluemix.net/

Mored detailed documentation for [Tone Analyzer](https://www.ibm.com/watson/developercloud/doc/tone-analyzer/index.html)


### [Watson Conversation](https://www.ibm.com/watson/developercloud/doc/conversation/index.html)

Watson Conversation combines a number of cognitive techniques to help you build and train a bot - defining intents and entities and crafting dialog to simulate conversation. The system can then be further refined with supplementary technologies to make the system more human-like or to give it a higher chance of returning the right answer. Watson Conversation allows you to deploy a range of bots via many channels, from simple, narrowly focused Bots to much more sophisticated, full-blown virtual agents across mobile devices, messaging platforms like Slack, or even through a physical robot.

##### Suggested uses

- Add a chatbot to your website that automatically responds to customers’ most frequently asked questions.
- Build Twitter, Slack, Facebook Messenger, and other messaging platform chatbots that interact instantly with channel users.
- Allow customers to control your mobile app using natural language virtual agents.

##### You input

Your domain expertise in the form of intents, entities and crafted conversation

##### Service output

A trained model that enables natural conversations with end users

With the IBM Watson™ Conversation service, you can create an application that understands natural-language input and uses machine learning to respond to customers in a way that simulates a conversation between humans.

##### How to use the service

This diagram shows the overall architecture of a complete solution:

![Watson Conversation](https://raw.githubusercontent.com/Bluemix-Watson-Labs/conversation-api-overview/master/conversation_arch_overview.png)

- Users interact with your application through the user interface that you implement. For example, A simple chat window or a mobile app, or even a robot with a voice interface.
- The application sends the user input to the Conversation service.
 - The application connects to a workspace, which is a container for your dialog flow and training data.
 - The service interprets the user input, directs the flow of the conversation and gathers information that it needs.
 - You can connect additional Watson services to analyze user input, such as Tone Analyzer or Speech to Text.
- The application can interact with your back-end systems based on the user’s intent and additional information. For example, answer question, open tickets, update account information, or place orders. There is no limit to what you can do.

More detailed documentation for [Watson Conversation](https://www.ibm.com/watson/developercloud/doc/conversation/index.html)

### [Language Translator](https://www.ibm.com/watson/developercloud/doc/language-translator/)

The Watson Language Translator service provides domain-specific translation utilizing Statistical Machine Translation techniques that have been perfected in our research labs over the past few decades. The service offers multiple domain-specific translation models, plus three levels of self-service customization for text with very specific language. (Note: The Watson Language Translation service has been rebranded as the Language Translator service. The Language Translator service provides the same capabilities as the Language Translation service, but with simpler pricing.)

##### Intended use

What can be done with Watson Language Translator? As an example, an English-speaking help desk representative can assist a Spanish-speaking customer through chat (using the conversational translation model). As another example, a West African news website can curate English news from across the globe and present it in French to its constituents (using the news translation model). Similarly, a patent attorney in the US can effectively discover prior art (to invalidate a patent claims litigation from a competitor) based on invention disclosures made in Korean with the Korean Patent Office. Another example would be that a bank can translate all of their product descriptions from English to Arabic using a custom model tailored to that bank's product names and terminology. All of these examples (and more) can benefit from the real-time, domain-specific translation abilities of the Language Translator service.

##### You input

Plain text in one of the supported input languages and domains.

##### Service output

Plain text in the target language selected.

More detailed documentation for [Language Translator](https://www.ibm.com/watson/developercloud/doc/language-translator/)

### [Personality Insights](https://www.ibm.com/watson/developercloud/doc/personality-insights/)

Personality Insights extracts and analyzes a spectrum of personality attributes to help discover actionable insights about people and entities, and in turn guides end users to highly personalized interactions. The service outputs personality characteristics that are divided into three dimensions: the Big 5, Values, and Needs. We recommend using Personality Insights with at least 1200 words of input text.

##### Intended Use

The Personality Insights service lends itself to an almost limitless number of potential applications. Businesses can use the detailed personality portraits of individual customers for finer-grained customer segmentation and better-quality lead generation. This data enables them to design marketing, provide product recommendations, and deliver customer care that is more personal and relevant. Personality Insights can also be used to help recruiters or university admissions match candidates to companies or universities. For more detailed information, see the "Use Cases" section of the Personality Insights documentation.

##### You input

JSON, or Text or HTML (such as social media, emails, blogs, or other communication) written by one individual

##### Service output

A tree of cognitive and social characteristics in JSON or CSV format

##### Personality Insights basics
The Personality Insights service offers a set of core analytics for discovering actionable insights about people and entities. The following sections provide basic information about using the service.

##### The personality models
The Personality Insights service is based on the psychology of language in combination with data analytics algorithms. The service analyzes the content that you send and returns a personality profile for the author of the input. The service infers personality characteristics based on three models:

- **Big Five** personality characteristics represent the most widely used model for generally describing how a person engages with the world. The model includes five primary dimensions:
 - Agreeableness is a person's tendency to be compassionate and cooperative toward others.
 - Conscientiousness is a person's tendency to act in an organized or thoughtful way.
 - Extraversion is a person's tendency to seek stimulation in the company of others.
 - Emotional Range, also referred to as Neuroticism or Natural Reactions, is the extent to which a person's emotions are sensitive to the person's environment.
 - Openness is the extent to which a person is open to experiencing a variety of activities.
- Each of these top-level dimensions has six facets that further characterize an individual according to the dimension.
- **Needs** describe which aspects of a product will resonate with a person. The model includes twelve characteristic needs: Excitement, Harmony, Curiosity, Ideal, Closeness, Self-expression, Liberty, Love, Practicality, Stability, Challenge, and Structure.
- **Values** describe motivating factors that influence a person's decision making. The model includes five values:Self-transcendence / Helping others, Conservation / Tradition, Hedonism / Taking pleasure in life, Self-enhancement / Achieving success, and Open to change / Excitement.

More detailed documentation for [Personality Insights](https://www.ibm.com/watson/developercloud/doc/personality-insights/)

### [Retrieve and Rank](https://www.ibm.com/watson/developercloud/doc/retrieve-rank/)

This service helps users find the most relevant information for their query by using a combination of search and machine learning algorithms to detect "signals" in the data. Built on top of Apache Solr, developers load their data into the service, train a machine learning model based on known relevant results, then leverage this model to provide improved results to their end users based on their question or query.

The Retrieve and Rank Service can be applied to a number of information retrieval scenarios. For example, an experienced technician who is going onsite and requires help troubleshooting a problem, or a contact center agent who needs assistance in dealing with an incoming customer issue, or a project manager finding domain experts from a professional services organization to build out a project team.

##### You input

Your documents
Queries (questions) associated with your documents
Service Runtime: User questions and queries

##### Service output

Indexed documents for retrieval
Machine learning model (Rank)
Service Runtime: A list of relevant documents and metadata

##### Overview of the Retrieve and Rank service
The IBM Watson™ Retrieve and Rank service combines two information retrieval components in a single service: the power of Apache Solr and a sophisticated machine learning capability. This combination provides users with more relevant results by automatically reranking them by using these machine learning algorithms.

##### How to use the service
The following image shows the process of creating and using the Retrieve and Rank service:

![RR](https://www.ibm.com/watson/developercloud/doc/retrieve-rank/images/retrieve_rank_process.png)

For a step-by-step overview of using the Retrieve and Rank service, [see the Tutorial page.](https://www.ibm.com/watson/developercloud/doc/retrieve-rank/tutorial.shtml)

##### Technologies
The purpose of the Retrieve and Rank service is to help you find documents that are more relevant than those that you might get with standard information retrieval techniques.

- **Retrieve**: Retrieve is based on Apache Solr. It supports nearly all of the default Solr APIs and improves error handling and resiliency. You can start your solution by first using only the Retrieve features, and then add the ranking component.
- **Rank**: The rank component (ranker) creates a machine-learning model trained on your data. You call the ranker in your runtime queries to use this model to boost the relevancy of your results with queries that the model has not previously seen.

The service combines several proprietary machine learning techniques, which are known as learning-to-rank algorithms. During its training, the ranker chooses the best combination of algorithms from your training data.

##### Primary uses
The core users of the Retrieve and Rank service are customer-facing professionals, such as support staff, contact center agents, field technicians, and other professionals. These users must find relevant results quickly from large numbers of documents:
- Customer support: Find quick answers for customers from your growing set of answer documents
- Field technicians: Resolve technical issues onsite
- Professional services: Find the right people with the right skills for key engagements

##### Benefits
The Retrieve and Rank service can improve information retrieval as compared to standard results.
- The ranker models take advantage of rich data in your documents to provide more relevant answers to queries.
- You benefit from new features developed both by the open source community and from advanced information retrieval techniques that are built by the Watson algorithm teams.
- Each Solr cluster and ranker is highly available in the Bluemix environment. The scalable IBM infrastructure removes the need for you to staff your own highly available data center.

##### About Apache Solr
As previously mentioned, the Retrieve part of the Retrieve and Rank service is based on Apache Solr. When you use Retrieve and Rank, you need to be knowledgeable about Solr as well as about the specifics of the Retrieve and Rank service. For example, when Solr passes an error code to the service, the service passes it to your application without modification so that standard Solr clients can correctly parse and act upon it. You therefore need to know about Solr error codes when writing error-handling routines in your Retrieve and Rank application.

More detailed documentation for [Retrieve and Rank](https://www.ibm.com/watson/developercloud/doc/retrieve-rank/)

### [Speech to Text](https://www.ibm.com/watson/developercloud/doc/speech-to-text/)

Watson Speech to Text can be used anywhere there is a need to bridge the gap between the spoken word and its written form. This easy-to-use service uses machine intelligence to combine information about grammar and language structure with knowledge of the composition of an audio signal to generate an accurate transcription. It uses IBM's speech recognition capabilities to convert speech in multiple languages into text. The transcription of incoming audio is continuously sent back to the client with minimal delay, and it is corrected as more speech is heard. Additionally, the service now includes the ability to detect one or more keywords in the audio stream. The service is accessed via a WebSocket connection or REST API.

##### Intended Use

The Speech to Text service can be used anywhere voice-interactivity is needed. The service is great for mobile experiences, transcribing media files, call center transcriptions, voice control of embedded systems, or converting sound to text to then make data searchable. Supported languages include US English, UK English, Japanese, Spanish, Brazilian Portuguese, Modern Standard Arabic, and Mandarin. The Speech to Text service now provides the ability to detect the presence of specific keywords or key phrases in the input stream.

##### You input

Streamed audio with Intelligible Speech
Recorded audio with Intelligible Speech

##### Service output

Text transcriptions of the audio with recognized words

##### Continuous transmission
 
By default, the service stops transcription at the first end-of-speech (EOS) incident, which is denoted by a half-second of non-speech (typically silence) or when the stream terminates. Set the continuous parameter to true to instruct the service to transcribe the entire audio stream until the stream terminates. In this case, the results can include multiple transcript elements to indicate phrases separated by pauses. You can concatenate the transcript elements to assemble the complete transcription of the audio stream.

More detailed documentation for [Speech to Text](https://www.ibm.com/watson/developercloud/doc/speech-to-text/)

### 2.4. Explain use cases for integrating external systems (such as Twitter, Weather API).

When systems communicate with each other, this is considered Internet of Things

(ivp: IoT use cases:
#### Predictive maintenance
By leveraging streaming data from sensors and devices to quickly assess current conditions, recognize warning signs, deliver alerts and automatically trigger appropriate maintenance processes, IoT turns maintenance into a dynamic, rapid and automated task
#### Smart metering
A smart meter is an internet-capable device that measures energy, water or natural gas consumption of a building or home, according to Silicon Labs.
Traditional meters only measure total consumption, whereas smart meters record when and how much of a resource is consumed. Power companies are deploying smart meters to monitor consumer usage and adjust prices according to the time of day and season.
#### IoT in the supply chain
Keeping track of and predicting product demand can be tricky and mistakes have a cost implication – either through over-supply (and therefore wastage), or under-supply (and the attendant missed sales opportunities.) IoT technology could track the location of individual components, and broadcast supply and demand on a shared blockchain. Data as to demand request, supply volumes, or part expiration dates could be instantly accessed and analysed to identify where production needs to be stepped up

#### Energy consumption
For those who want to offset their carbon footprint on the carbon credit exchange, IoT provides a way of accurately tracking and trading carbon credits and energy consumption.
)

- [Explain the components of systems communicating with one another](https://console.ng.bluemix.net/docs/services/IoT/index.html)
- [Use case of Twitter and sentiment analysis](https://www.ibm.com/blogs/bluemix/2016/06/cognitive-apis-with-watson-sentiment-analysis/)
- [Use case of the Weather APIs and mission critical decision that are impacted by weather](https://www.ibm.com/blogs/bluemix/2015/10/ibm-insights-weather-available-in-bluemix/?HavasWatsonStudyGuide)

### 2.5. Describe the IBM Watson Discovery Service
SUBTASK(S):
#### 2.5.1. List the functionalities that the Watson Discovery service provides
IBM Watson™ Discovery brings together a functionally rich set of integrated, automated Watson APIs to:

- Crawl, convert, enrich and normalize data.
- Securely explore your proprietary content as well as free and licensed public content.
- Apply additional enrichments such as concepts, relations, and sentiment through natural language processing.
- Simplify development while still providing direct access to APIs.
------
- ingest
- normalize
- enrich 
- query
- analyze

#### 2.5.2. Explain the components of the Discovery service
- environment - think of an environment as the warehouse where you are storing all your boxes of documents
- configuration - defines how to process & enrich the documents
- collection - think of a collection as a box where you will store your documents in your environment

#### 2.5.3. Use the Discovery service via APIs or the Discovery Tooling
##### 2.5.3.1. Explain how to setup the environment
The first time that you click this button, you need to choose an environment size from the list of options. Your Discovery service environment is where all your data that is stored will live. Think of an environment as the warehouse where you are storing all your boxes of documents. Make your selection and your environment will be created.
Note: You have several environment sizes to choose from, see the Discovery catalog for details. If you find later that you need a larger environment, you can upgrade to a larger one then. (Your original source files will not count against your file size limit.)

Example of environment metadata:
```{
    "environment_id": "cc62bccd-4dd9-4e75-84dd-0a89f12c4f3f",
    "name": "byod",
    "description": "",
    "created": "2017-07-18T14:33:04.326Z",
    "updated": "2017-07-18T14:33:04.326Z",
    "status": "active",
    "read_only": false,
    "size": 0,
    "index_capacity": {
        "disk_usage": {
            "used_bytes": 54943860,
            "total_bytes": 4294967296,
            "used": "52.4 MB",
            "total": "4 GB",
            "percent_used": 1.28
        },
        "memory_usage": {
            "used_bytes": 394587080,
            "total_bytes": 1056309248,
            "used": "376.31 MB",
            "total": "1007.38 MB",
            "percent_used": 37.36
        }
    }
}
```

##### 2.5.3.2. Explain how to configure the Discovery service
- create a service (choose plan) 
- (optional) create configuration with description of documents transformations and enrichments (Note:  Before you start changing settings, you should upload the sample document that you identified at the start of this task)
- create a collection

##### 2.5.3.3. Explain how to add content
- via API: POST /v1/environments/{environment_id}/collections/{collection_id}/documents or with SDK
- via Tooling: 
  1. Click on the file  icon in the top left of your screen and select your collection.
  2. Choose appropriate configuration
  3. Go to Add data to this collection at the right of the screen and start uploading your documents.
- via Data Crawler (command line tool that will help you take your documents from the repositories where they reside (for example: file shares, databases, Microsoft SharePoint® ) and push them to the cloud, to be used by the Discovery Service)

##### 2.5.3.4. Explain how to build queries
- Click on the magnifying glass icon to open the query page
- Select your collection and click Get started
- Click 'Build your own query'
- Click 'Use the Discovery Query Language'

###### Query structure
![Query structure](https://www.ibm.com/watson/developercloud/doc/discovery/images/query_structure2.png)
 
Operators table - https://www.ibm.com/watson/developercloud/doc/discovery/query-reference.html#operators

Query Examples:
- `enriched_text.sentiment.document.label:positive` - Will return all documents that have a positive sentiment.
- `enriched_text.entities.type:company` - Will return all documents that contain an entity of the type company
- `enriched_text.categories.label::"health and fitness"` - Will return all documents in the health and fitness category. (The operator :: specifies an exact match.)
- `enriched_text.entities.text::Watson` - Will return all documents that contain the entity Watson. (The operator :: specifies an exact match. By using an exact match we don't get false positives on similar entities, for example Watson Health and Dr. Watson would be ignored.)
- `enriched_text.entities.text:IBM,enriched_text.entities.text:!Watson` - Will return all documents that contain the entity IBM, but not the entity Watson (The operator :! specifies "does not contain".)

###### Combined Queries & Filters
You can combine query parameters together to build more targeted queries. For example use both the filter and query parameters.
When used in tandem with queries, filters execute first to narrow down the document set and speed up the query. Adding these filters will return unranked documents, and after that the accompanying query will run and return the results of that query ranked by relevance.

Filter examples: 
- `enriched_text.sentiment.document.label:positive` - Will return only the documents in the collection with the sentiment of positive.
- `enriched_text.entities::(relevance>0.8,text::IBM)` - Will return only the documents that contain the entity IBM with a relevance score above 80% for that entity.

###### Aggregations
Aggregation queries return a set of data values; for example, top keywords, overall sentiment of entities, and more.

 ![Aggregation structure](https://www.ibm.com/watson/developercloud/doc/discovery/images/aggregation_structure.png)

Aggregation types: term, filter, nested, histogram, timeslice, top_hits, max, min, average, sum

https://www.ibm.com/watson/developercloud/doc/discovery/query-reference.html#aggregations

`nested(...)` expression sets the context to all following expressions of the aggregation  
Good explantation of nested aggregation - https://youtu.be/pcNwV9prfmY?t=3m48s (especially note the explanation on 6:03)

Reference:
https://www.ibm.com/watson/developercloud/doc/discovery/index.html

https://www.youtube.com/watch?v=FikHwoJ6_FE

https://www.youtube.com/watch?v=fmIPeopG-ys&t=1s

## Section 3 – Fundamentals of IBM Watson Developer Cloud

### 3.1. Distinguish cognitive services on WDC for which training is required or not.

Some IBM Watson services work out-of-the-box as they were pre-trained in a specific domain (domain-adapted). Other Watson services require training. For pre-trained services, it’s critical to know the adapted domains as they indicate the areas in which the service will perform best.

#### 3.1.1. Some IBM Watson services work out-of-the-box as they were pre-trained
in a specific domain (domain-adapted). Other Watson services require
training. For pre-trained services, it’s critical to know the adapted
domains as they indicate the areas in which the service will perform best.

Pre-trained Watson services:

##### 3.1.2.1. Watson Text-to-Speech
##### 3.1.2.2. Watson Speech-to-text
##### 3.1.2.3. Language Translator (conversational, news, and patent domains; however some one can also train custom models)
##### 3.1.2.4. Natural Language Understanding (however similar models can be trained in Watson Knowledge Studio)
##### 3.1.2.5. Tone Analyzer
##### 3.1.2.6. Personality Insights (social media domain)
##### 3.1.2.7. Watson Discovery News
##### 3.1.2.8. Watson Discovery Service

#### 3.1.3. Services requiring training:
##### 3.1.3.1. Natural Language Classifier
##### 3.1.3.2. Visual recognition (custom models, however it is supplied with some pre-built models)
##### 3.1.3.3. Watson Conversation

### 3.2. Provide examples of text classification using the NLC.
#### 3.2.1. Sentiment analysis
#### 3.2.2. Spam email detection
#### 3.2.3. Customer message routing
#### 3.2.4. Academic paper classification into technical fields of interest
#### 3.2.5. Forum post classification to determine correct posting category
#### 3.2.6. Patient reports for escalation and routing based on symptoms
#### 3.2.7. News article analysis
#### 3.2.8. Investment opportunity ranking
#### 3.2.9. Web page topic analysis

### 3.3. Explain the Watson SDKs available as part of the services on Watson Developer Cloud.

#### 3.3.1 Identify the programming languages with SDKs available

 - [Node SDK](https://www.npmjs.com/package/watson-developer-cloud)
 - [Java SDK](http://mvnrepository.com/artifact/com.ibm.watson.developer_cloud/java-sdk)
 - [Swift SDK](https://github.com/watson-developer-cloud/swift-sdk)
 - [Unity SDK](https://github.com/watson-developer-cloud/unity-sdk#installing-the-sdk-source-into-your-unity-project)
 - [Python SDK](https://pypi.python.org/pypi/watson-developer-cloud)
 - [.NET Standard SDK](https://github.com/watson-developer-cloud/dotnet-standard-sdk)

#### 3.3.2 Describe the advantage and disadvantages of using an SDK

 advantages: easier to start; less coding; conventient interface
 disadvantage: additional libraries in your project; limited to API methods which currently implemented in SDK; a bit less control over communication; SDK must be up-to-date
 
#### 3.3.3 Find the Watson SDKs and other resources on the WDC GitHub
  - [Watson Developer Cloud Github](https://github.com/watson-developer-cloud/)
  
###  3.4. Explain the Watson REST APIs available as part of the services on Watson Developer Cloud.

#### 3.4.1 Identify the Language services on WDC 

 - (Conversation)
 - Document Conversion 
 - (Discovery)
 - Language Translator 
 - Natural Language Classifier 
 - (Natural Language Understanding)
 - (Personality Insights) 
 - Retrieve and Rank 
 - (Tone Analyzer)
 
#### 3.4.2 Identify the Vision services on WDC

 - Visual Recognition
 
#### 3.4.3 Identify the Speech services on WDC

 - Speech to Text 
 - Text to Speech
 
#### 3.4.4 Identify the Data Insights services on WDC

 - Discovery 
 - Personality Insight
 - Tone Analyzer
 
###  3.5. Explain and configure Natural Language Classification.

#### 3.5.1. The service enables developers without a background in machine learning
or statistical algorithms to interpret the intent behind text.

#### 3.5.2. Configure:

3.5.2.1. Gather sample text from real end users (fake initially if you have to
but not much)

3.5.2.2. Determine the users intents that capture the actions/needs
expressed in the text

3.5.2.3. Classify your user text into these user intents

3.5.2.4. Separate your user text into train/test datasets

3.5.2.5. Train an NLC classifier on your training dataset

3.5.2.6. Pass the user input to an NLC classifier

3.5.2.7. Determine the accuracy, precision, and recall of the NLC classifier
using your test dataset

3.5.2.8. Improve the confidence level iteratively through back propagation (ivp: WTF?!)
or other means.
(ivp: https://console.bluemix.net/docs/services/natural-language-classifier/using-your-data.html#using-your-own-data)

Reference:
https://www.ibm.com/watson/developercloud/natural-language-classifier/api/v1/

### 3.6. Explain and configure Visual recognition.

#### 3.6.1. Describe the process for training a classifier
- via API: 

  1. `POST /v3/classifiers` 

  curl -X POST -F "beagle_positive_examples=@beagle.zip" -F "husky_positive_examples=@husky.zip" -F "goldenretriever_positive_examples=@golden-retriever.zip" -F "negative_examples=@cats.zip" -F "name=dogs" "https://gateway-a.watsonplatform.net/visual-recognition/api/v3/classifiers?api_key={api-key}&version=2016-05-20" 


  Positive example filenames require the suffix _positive_examples. In this example, the filenames are beagle_positive_examples, goldenretriever_positive_examples, and husky_positive_examples. The prefix (beagle, goldenretriever, and husky) is returned as the name of class.

  2. Check the training status periodically until you see a status of ready

#### 3.6.2. Explain how to identify images with a specified classifier

- when training: use `POST /v3/classifiers/{classifier_id}` to update existing classifier
- when classifying: use parameter `classifier_ids` to define which classifier should process given image

#### 3.6.3. Describe the capabilities of facial, gender, and age recognition

'Face Detection' classifier can detect faces on the image and predict geneder and age for each face

#### 3.6.4. Describe the capabilities of Natural Scene OCR

- Optical Character Recognition (OCR) is the electronic conversion of images of written or printed text into machine-encoded text
- Natural scene character recognition is challenging due to the cluttered background, which is hard to separate from text

as of 07/2017:
The Visual Recognition service used to have a text recognition feature. It was available in the earlier beta version of the service, but has since been moved to closed beta according to the release notes:

"Text recognition is now closed beta - We have gone back into a closed beta with the POST and GET /v3/recognize_text methods. We look forward to continuing to support BETA clients using the service, with no current plans for another open beta."

#### 3.6.5. Explain how collections are built and the use of similarity search
1. Create a collection with `POST /v3/collections`
2. Wait until status of collection changes to available
3. Call the beta `POST /v3/collections/{collection_id}/images` method to upload the image to the collection (maximum image size is 2 MB):
4. Call the beta `POST /v3/collections/{collection_id}/find_similar` method to upload the image file and search your collection to see if there are similar images.
(example curl -X POST -F "image_file=@silver-dress2.jpg" "https://gateway-a.watsonplatform.net/visual-recognition/api/v3/collections/{collection_id}/find_similar?api_key={api-key}&version=2016-05-20" )

Reference:
https://www.ibm.com/watson/developercloud/visual-recognition/api/v3/


###  3.7. Explain how Personality Insights service works.
#### 3.7.1. Describe the intended use of the Personality Insights service

The IBM Watson™ Personality Insights service provides an Application Programming Interface (API) that enables applications to derive insights from social media, enterprise data, or other digital communications. The service uses linguistic analytics to infer individuals' intrinsic personality characteristics, including Big Five, Needs, and Values, from digital communications such as email, text messages, tweets, and forum posts.

The service can automatically infer, from potentially noisy social media, portraits of individuals that reflect their personality characteristics. It can also determine individuals' consumption preferences, which indicate their likelihood to prefer various products, services, and activities.

#### 3.7.2. Describe the inputs and outputs of the Personality Insights service

see [Personality Insights](#personality-insights)


#### 3.7.3. Describe the personality models of the Personality Insights service 

see [Personality Insights](#personality-insights)

###  3.8. Explain how Tone Analyzer service works.
#### 3.8.1. Describe the common use cases of the Tone Analyzer service
see [Tone Analyzer](#tone-analyzer)

#### 3.8.2. Describe the basic flow of the Tone Analyzer service
- send text to general endpoint or customer engagement endpoint for analyze
- enjoy the results ;)

#### 3.8.3. Explain the three categories of tone scores and their sub-tones: emotional tone, social tone, and language tone.
- Emotion tones

  Emotion tones (ID emotion_tone) measure different types of emotions and feelings that people express.
  
    - Anger: Anger is evoked due to injustice, conflict, humiliation, negligence, or betrayal. If anger is active, the individual attacks the target, verbally or physically. If anger is passive, the person silently sulks and feels tension and hostility.
    - Disgust: Disgust is an emotional response of revulsion to something considered offensive or unpleasant. The sensation refers to something revolting.
    - Fear: Fear is a response to impending danger. It is a survival mechanism that is triggered as a reaction to some negative stimulus. Fear may be a mild caution or an extreme phobia.
    - Joy: Joy (or happiness) has shades of enjoyment, satisfaction, and pleasure. Joy brings a sense of well-being, inner peace, love, safety, and contentment.
    - Sadness: Sadness indicates a feeling of loss and disadvantage. When a person is quiet, less energetic, and withdrawn, it may be inferred that they feel sadness.

- Social tones

Social tones (ID social_tone) measure the social tendencies in people's writing. The tones are adopted from the Big Five personality model; for more information, see the Personality models described for the IBM Watson™ Personality Insights service.

    - Agreeableness: The tendency to be compassionate and cooperative towards others [Selfish, uncaring, uncooperative, self-interested, confrontational, skeptical, or arrogant ||	Caring, sympathetic, cooperative, compromising, trustworthy, or humble]
    - Conscientiousness:	The tendency to act in an organized or thoughtful way	[Spontaneous, laid-back, reckless, unmethodical, remiss, or disorganized ||	Disciplined, dutiful, achievement-striving, confident, driven, or organized]
    - Emotional range:	The extent to which a person's emotions are sensitive to their environment	[Calm, bland, content, relaxed, unconcerned, or careful ||	Concerned, frustrated, angry, passionate, upset, stressed, insecure, or impulsive]
    - Extraversion:	The tendency to seek stimulation in the company of others	[[Independent, timid, introverted, restrained, boring, or dreary	|| Engaging, seeking attention, needy, assertive, outgoing, sociable, cheerful, excitement-seeking, or busy]]
    - Openness:	The extent to which a person is open to experiencing a variety of activities	[No-nonsense, straightforward, blunt, or preferring tradition and the obvious over the complex, ambiguous, and subtle ||	Intellectual, curious, emotionally aware, imaginative, willing to try new things, appreciating beauty, or open to change]

- Language tones

Language tones (ID language_tone) describe a person's perceived writing style. A score of less than 0.5 indicates that the content offered little or no evidence of the tone.

    - Analytical: analytical	A person's reasoning and analytical attitude about things	[Intellectual, rational, systematic, emotionless, or impersonal]
    - Confident: confident	A person's degree of certainty	[Assured, collected, hopeful, or egotistical]
    - Tentative: tentative	A person's degree of inhibition	[Questionable, doubtful, or debatable]
    
#### 3.8.4. Explain how Tone Analyzer service is different from the Natural Language Understanding - Sentiment Analysis and Emotion Insights service
- NLU - Sentiment Analysis doesn't give you deep insight into authors personality/state. 
- NLU - Emotion Insights service is only part of functionality provided by Tone Analyzer.

References:
https://www.ibm.com/watson/developercloud/tone-analyzer/api/v3/
https://www.ibm.com/watson/developercloud/doc/tone-analyzer/index.html
https://www.ibm.com/blogs/watson/2016/02/293/

### 3.9. Explain and execute IBM Watson Natural Language Understanding services.

#### 3.9.1. Identify the capabilities of Natural Language Understanding

**Features:**
- Categories (e.g /news, /art and entertainment, /movies and tv/television, /news, /international news)
- Concepts (e.g. Linguistics, Natural language processing, Natural language understanding)
- Emotion (You can also enable emotion analysis for entities and keywords that are automatically detected by the service, e.g {"apples": joy, "oranges": anger})
- Entities (e.g. IBM: Company, Armonk: Location, New York: Location, United States: Location)
- Keywords (Australian Open, Tennis Australia, IBM SlamTracker analytics)
- Metadata (For HTML and URL input, get the author of the webpage, the page title, and the publication date. For example:)
- Relations ("awardedTo" relation between "Noble Prize in Physics" and "Albert Einstein"; "timeOf" relation between "1921" and "awarded")
- Semantic Roles (e.g. Input "In 2011, Watson competed on Jeopardy!" => { Subject: Watson, Action: competed, Object: on Jeopardy})
- Sentiment (e.g. Positive sentiment (score: 0.91))
- (additionally) Language Detection

**Supported Languages:**
Arabic, English, French, German, Italian, Japanese, Portuguese, Russian, Spanish, Swedish
(for support of concrete features per language see https://www.ibm.com/watson/developercloud/doc/natural-language-understanding/#supported-languages)

#### 3.9.2. Describe the text extraction features of Natural Language Understanding

see https://www.ibm.com/watson/developercloud/doc/natural-language-understanding/index.html#service-features

#### 3.9.3. Distinguish between keywords, entities, and concepts

The biggest difference is that concepts can pick up on things that aren't explicitly mentioned. For example, this article about Mount Rushmore returns (among others) the concept South Dakota, which is never explicitly mentioned. Keywords, on the other hand, are nouns and noun phrases that are pulled directly from the text based on their frequency or potential importance.

While we're on the subject, entities are similar to keywords but are more specific. Entity extraction wouldn't pick up park rangers (a keyword), but it does pick up National Park Service.

(https://www.quora.com/What-is-the-difference-between-keywords-and-concepts-in-the-AlchemyAPI-output)

#### 3.9.4. Distinguish between document-level and targeted sentiment
Document-level sentiment is a sentiment of the whole document, target sentiment is array of target analysis results. Each object contains the text of the target, sentiment score, and a label.

(see https://www.ibm.com/watson/developercloud/natural-language-understanding/api/v1/#sentiment)
(see demo app https://natural-language-understanding-demo.mybluemix.net/)

#### 3.9.5. Explain the difference between the taxonomy call and the knowledge graph
- Taxonomy is hierarchical way to cathegorize things (i.e. it's just a bunch of categories, which can be used to "describe" some objects; ~= metadata) 
- In contrast to it knowledge graph is information about concrete objects and their relationships (i.e. way to organize information about objects + this information; ~= matadata + data)

https://en.wikipedia.org/wiki/Knowledge_Graph
https://www.youtube.com/watch?v=mmQl6VGvX-c

#### 3.9.6. Explain disambiguation as it relates to entities
Example of ambiguaty: "Ford" can be a car or a person.

When NLU find entity and cannot unambiguously detect it's type it provides information which can help to resolve ambiguity: entity subType information, a common entity name, and a dbpedia_resource link if applicable.

Example 
```
"disambiguation": {
        "name": "IBM",
        "dbpedia_resource": "http://dbpedia.org/resource/IBM",
        "subtype": [
          "SoftwareLicense",
          "OperatingSystemDeveloper",
          "ProcessorManufacturer",
          "SoftwareDeveloper",
          "CompanyFounder",
          "ProgrammingLanguageDesigner",
          "ProgrammingLanguageDeveloper"
        ]
      }
```

#### 3.9.7. Explain how Emotion Analysis service works

##### 3.9.7.1. What emotions does Emotion Analysis detect?
  - sadness
  - joy
  - fear
  - disgust
  - anger

##### 3.9.7.2. Describe the main use cases for applying the Emotion Insights service
  - **Product Feedback and Campaign Effectiveness:** Monitor the emotional reaction of your target audience for your products, campaigns, and other marketing communications. 
- **Customer Satisfaction:** Analyze customer surveys, emails, chats, and reviews to determine the emotional pulse of your customers. 
- **Contact-center Management, Automated agents, and Robots:** Detect emotions in chats or other conversations and adapt to provide an appropriate response. For instance, direct a customer to a human agent if intense anger is detected.
   
https://developer.ibm.com/watson/blog/2016/02/29/another-step-closer-to-building-empathetic-systems/
 
##### 3.9.7.3. Describe the main types of positive/negative sentiment extracted from digital text

[see 3.9.4](#394-distinguish-between-document-level-and-targeted-sentiment)

##### 3.9.7.4. Describe the API types provided by the Sentiment Analysis service

The question is probably legacy taken over from Alechemy Language times (it used to have HTML, Text and WEB API [Sentiment Analysis API](http://web.archive.org/web/20160319211133/http://www.alchemyapi.com/api/sentiment/urls.html)). 

Currently NLU has all features combined in one main API endpoint /analyze (exists in 2 flavours - GET and POST) 

##### 3.9.7.5. Describe the differences between sentiment and emotion analyses

Emotion recognition is a special case of sentiment analysis. The output of sentiment analysis is produced in terms of either polarity (e.g., positive or negative) or in the form of rating (e.g., from 1 to 5). Emotions are a more detailed level of analysis in which the result are depicted in more expressive and fine-grained level of analysis.
Sentiment analysis deals with only text, while emotions can be expressed by text, images, audio, video, facial signs etc.

Reference:
https://www.ibm.com/watson/developercloud/natural-languageunderstanding/api/v1/


### 3.10.Explain, setup, configure and query the IBM Watson Discovery service.

#### 3.10.1. How to create the Data Collection repository

Inputs: environment_id, configuration_id

Example: 
```
curl -X POST -u "{username}":"{password}" -H "Content-Type: application/json" -d '{
  "name": "test_collection",
  "description": "My test collection",
  "configuration_id": "{configuration_id}",
  "language_code": "en"
}' "https://gateway.watsonplatform.net/discovery/api/v1/environments/{environment_id}/collections?version=2017-07-19"
```

##### 3.10.1.1. Explain the significance of working with sample documents

To make the configuration process more efficient, you can upload up to ten Microsoft Word, HTML, JSON, or PDF files that are representative of your document set. These are called sample documents. Sample documents are not added to your collection — they are only used to identify fields that are common to your documents and customize those fields to your requirements.

##### 3.10.1.2. Explain the difference between Default and switching to a new custom collection
*I guess the question supposed to ask about default and custom configurations (since there is no such as thing as default collection). (Ivan)*

------

The Discovery service includes a standard configuration file that will convert, enrich and normalize your data without requiring you to manually configure these options.

This default configuration file is named Default Configuration. It contains enrichments, plus standard document conversions based on font styles and sizes.

First the default enrichments. Discovery will enrich (add cognitive metadata to) the text field of your documents with semantic information collected by four Watson Enrichments — Entity Extraction, Sentiment Analysis, Category Classification, and Concept Tagging (learn more about them here).

[more details](https://www.ibm.com/watson/developercloud/doc/discovery/building.html#the-default-configuration)

------
**When you need a custom configuration**

**I understand that my documents may not be structured in the way the default configuration expects. How do I know if the default settings are right for me?**

The easiest way to see if the default works for you is to test it by Uploading sample documents. If the sample JSON results meet your expectations, then no additional configuration is required.

**I understand that default enrichments are added to the text field of my documents. Can I add additional enrichments to other fields?**

Absolutely, you can add additional enrichments to as many fields as you wish. See Adding enrichments for details.

##### 3.10.1.3. Explain when might you need more than one collection

A collection is a set of your documents. Why would I want more than one collection? There are a few reasons, including:

You may want multiple collections in order to separate results for different audiences
The data may be so different that it doesn't make sense for it all to be queried at the same time

#### 3.10.2. What are some of the steps required when you customize your configuration
##### 3.10.2.1. Identify sample documents

##### 3.10.2.2. Convert sample documents

##### 3.10.2.3. Add enrichments

##### 3.10.2.4. Normalize data

#### 3.10.3. What are the four standard document formats and explain the conversion flow (MS Word, PDF, HTML and JSON)

#### 3.10.4. Adding Enrichments, explain the following enrichments:

##### 3.10.4.1. Entity Extraction
Extracts people, companies, organizations, cities, geographic features, and more from this field.
[more details](https://www.ibm.com/watson/developercloud/doc/discovery/building.html#entity-extraction)

##### 3.10.4.2. Keyword Extraction
Determines important keywords in this field, ranks them, and optionally detects the sentiment.
[more details](https://www.ibm.com/watson/developercloud/doc/discovery/building.html#keyword-extraction)

##### 3.10.4.3. Taxonomy Classification (aka Category Classification)
Classifies this field into a hierarchy of categories that's five levels deep.
[more details](https://www.ibm.com/watson/developercloud/doc/discovery/building.html#category-classification)

##### 3.10.4.4. Concept Tagging
Identifies general concepts that aren’t necessarily directly referenced in this field
[more details](https://www.ibm.com/watson/developercloud/doc/discovery/building.html#concept-tagging)

##### 3.10.4.5. Relation Extraction (aka Semantic Role Extraction)
Parses sentences into subject, action, and object form and returns additional semantic information.
[more details](https://www.ibm.com/watson/developercloud/doc/discovery/building.html#semantic-role-extraction)

##### 3.10.4.6. Sentiment Analysis
Identifies the overall positive or negative sentiment within this field.
[more details](https://www.ibm.com/watson/developercloud/doc/discovery/building.html#sentiment-analysis)

##### 3.10.4.7. Emotion Analysis
Analyzes the emotions (anger, disgust, fear, joy, and sadness) in this field.
[more details](https://www.ibm.com/watson/developercloud/doc/discovery/building.html#emotion-analysis)

#### 3.10.5. Explain document size limitations

The maximum file size for individual documents in your collection is 50MB

#### 3.10.6. Explain the essence of the Normalization step, the last step in customizing your configuration
In the Normalize section of the Discovery tooling you can move, merge, copy or remove fields.
Empty fields (fields that contain no information) will be deleted by default. You can change that using the 'Remove empty fields' toggle.

#### 3.10.7. Explain some of the methods or ways of adding content after you are satisfied with the configuration work.
##### 3.10.7.1. Adding content through the API
```
curl -X POST -u "{username}":"{password}" -F file=@sample1.html "https://gateway.watsonplatform.net/discovery/api/v1/environments/{environment_id}/collections/{collection_id}/documents?version=2017-07-19"
```

##### 3.10.7.2. Adding content through the UI
Go to Add data to this collection at the right of the screen and start uploading your documents via drag and drop or browse.
##### 3.10.7.3. Adding content through the data crawler
if you want to have a managed upload of a significant number of files, or you want to extract content from a supported repository (such as a DB2 database).

1. Configure the Discovery service
2. Download and install the Data Crawler on a supported Linux system that has access to the content that you want to crawl.
3. Connect the Data Crawler to your content.
4. Configure the Data Crawler to connect to the Discovery Service.
5. Crawl your content.

#### 3.10.8. Querying your data
##### 3.10.8.1. Explain the three search query parameters (filter, query, aggregation)
- Query: A query search returns all documents in your data set with full enrichments and full text in order of relevance. A query also excludes any documents that don't mention the query content. (These queries are written using the Discovery Query Language.)
- Filter: When used in tandem with queries, filters execute first to narrow down the document set and speed up the query. Adding these filters will return unranked documents, and after that the accompanying query will run and return the results of that query ranked by relevance.
- Aggregation: Aggregation queries return a set of data values; for example, top keywords, overall sentiment of entities, and more

##### 3.10.8.2. Explain the three structure query parameters (count, offset, return)
- count: Sets the number of documents that you want returned in the response
- offset: The number of query results to omit from the start of the output. For example, if the count parameter is set to 10 and the offset parameter is set to 8, the query returns only the last two results. Do not use this parameter for deep pagination, as it impedes performance.
- return: A comma-separated list of the portion of the document hierarchy to return. Any of the document hierarchy are valid values.

##### 3.10.8.3. Explain Aggregations
[Building aggregations](https://www.ibm.com/watson/developercloud/doc/discovery/using.html#building-aggregations)

References:
https://www.ibm.com/watson/developercloud/discovery/api/v1/
https://www.ibm.com/watson/developercloud/doc/discovery/index.htmlhttps://ww
w.ibm.com/blogs/watson/2016/12/watson-discovery-service-understand-datascale-less-effort/
https://www.youtube.com/watch?v=fmIPeopG-ys&t=1s
https://www.ibm.com/blogs/bluemix/2016/11/watson-discovery-service/

### 3.11.Explain and configure the IBM Watson Conversation service

#### 3.11.1. Creating a workspace
#### 3.11.2. Define intents (user input)
#### 3.11.3. Define entities (relevant term or object)
#### 3.11.4. Build a dialog (branching conversation flow)
#### 3.11.5. Test the conversation agent / bind to application
https://console.bluemix.net/docs/services/conversation/deploy.html#deployment-overview
https://console.bluemix.net/docs/services/conversation/develop-app.html#building-a-client-application

#### 3.11.6. Know how other Watson services can add value to Conversation service
Discovery, RnR, Tone Analyzer

#### 3.11.7. Describe the difference between short tail and long tail conversational exchanges.
https://developer.ibm.com/dwblog/2017/chatbot-long-tail-questions-watson-conversation-discovery/

#### 3.11.8. Explain how Discovery can support long tail conversations in Watson Conversation.
https://developer.ibm.com/dwblog/2017/chatbot-long-tail-questions-watson-conversation-discovery/

Reference:
https://www.ibm.com/watson/developercloud/conversation/api/v1/
 
## Section 4 - Developing Cognitive applications using Watson Developer Cloud Services

### 4.1. Call a Watson API to analyze content.

#### 4.1.1. Natural Language Understanding - Create an instance of the Natural Language Understanding service in Bluemix

##### 4.1.1.1. Select the correct API to call for text extraction, sentiment analysis, or any of the Natural Language Understanding services.
Outdated question. In contrast to Alchemy Language, NLU has only one endpoint where you can select features you need

##### 4.1.1.2. Pass your content to your Alchemy services’ endpoint through a RESTful API call
```
curl -G -u "{username}":"{password}" -d "version=2017-02-27" -d "url=www.ibm.com" -d "features=keywords,entities" -d "entities.emotion=true" -d "entities.sentiment=true" -d "keywords.emotion=true" -d "keywords.sentiment=true" "https://gateway.watsonplatform.net/natural-language-understanding/api/v1/analyze"
```

##### 4.1.1.3. Natural Language Classifier
##### 4.1.1.4. Gather sample text from real end users (fake initially if you have to…but not much)
##### 4.1.1.5. Determine the users intents that capture the actions/needs expressed in the text
##### 4.1.1.6. Classify your user text into these user intents
##### 4.1.1.7. Separate your user text into train/test datasets
##### 4.1.1.8. Create an instance of the Natural Language Classifier service in Bluemix
##### 4.1.1.9. Train an NLC classifier on your training dataset
##### 4.1.1.10. Pass your content to your NLC services’ endpoint through a RESTful API call
##### 4.1.1.11. Determine the accuracy, precision, and recall of the NLC classifier using your test dataset

### 4.1.2. Personality Insights - Create an instance of the Personality Insights service in Bluemix
#### 4.1.2.1. Gather text from users in their own voice
#### 4.1.2.2. Ensure you meet the minimum limits for word count to limit sampling error.

You can send the service up to 20 MB of input content, but accuracy levels off at around 3000 words of input; additional content does not contribute further to the accuracy of the profile. Therefore, the service extracts and uses only the first 250 KB of content, not counting any HTML or JSON markup, from large requests

https://www.ibm.com/watson/developercloud/doc/personality-insights/user-overview.html#overviewGuidelines 

##### 4.1.2.3. Pass your content to your Personality Insight services’ endpoint through a RESTful API call
```
curl -X POST -u "{username}:{password}"
--header "Content-Type: application/json"
--data-binary @profile.json
"https://gateway.watsonplatform.net/personality-insights/api/v3/profile?version=2016-10-20&consumption_preferences=true&raw_scores=true"
```

###  4.2. Describe the tasks required to implement the Conversational Agent / Digital Bot.

#### 4.2.1. Scope if there is any historical data to draw upon in the creation of the digital agent.
#### 4.2.2. Set expectations of what your conversational agent will and will not do.
#### 4.2.3. Determine the user intents (What the end-user says to the bot) and entities (what they are talking about) in your conversation. Factor in historical data if applicable.
#### 4.2.4. Design and define ideal conversation flow by writing questions and responses
#### 4.2.5. Create a service instance of Watson Conversation in Bluemix
#### 4.2.6. Launch a workspace through the provisioned service instance
#### 4.2.7. Configure service to recognize user intents
#### 4.2.8. Configure service to recognize user entities
#### 4.2.9. Identify the usage of system entities
#### 4.2.10. Map user intents and entities and define responses within dialog nodes
#### 4.2.11. Build out the dialog conversation flow for the agent.
#### 4.2.12. Determine ways users could diverge from your conversation process flow and ways to redirect them back.
#### 4.2.13. Think on measures of conversation control and shaping. Does your agent conversation scope out and address what the end-user needs from the agent?
#### 4.2.14. Will the end-user be able to comprehend the vocabulary, terminology, and style of the digital agent?
#### 4.2.15. Define a conversation node to have multiple responses.
#### 4.2.16. Bind Conversation service instance to an application.
#### 4.2.17. Present your beta conversation agent to end users to capture real enduser interaction.
#### 4.2.18. Identify areas where your conversation agent misunderstood the user.
#### 4.2.19. Identify areas where users strayed outside the domain of your conversation agent through the Improve section of your Conversation workspace.
#### 4.2.20. Update your conversation agent with new intents or entities to strengthen interactions.
#### 4.2.21. Understand improvements of accuracy by utilizing training sets against themselves (K-Fold Cross Validation) and blind tests for when changes are made with your intents, entities, and dialog.
"Remove a random 10%-20% (depend on number of question). Do not look at these questions. You use these as your blind test. Testing these will give you a reasonable expectation of how it will likely perform in the real world (assuming questions are not manufactured).

In earlier versions of WEA, we had what was called an experiment (k-fold validation). The system would remove a single question from training, and then ask it back. It would do this for all questions. The purpose was to test each cluster, and see what clusters are impacting others.

NLC/Conversation doesn't do this. To do it would take forever. You can use instead a monte-carlo cross fold. To do this, you take a random 10% from your train set, train on the 90% and then test on the 10% removed. You should do this a few times and average the results (at least 3 times)."

https://stackoverflow.com/questions/39800476/best-practices-for-creating-intents-entities-with-ibm-conversation-service

###  4.3. Transform service outputs for consumption by other services.

#### 4.3.1. Natural Language Classifier
##### 4.3.1.1. Using classifiers from NLC to drive dialog selections in Dialog
#### 43.2. Personality Insights
##### 4.3.2.1. Use the service output from two different textual inputs and compare the personalities based on the results
#### 4.3.3. Speech to text
##### 4.3.3.1. Use the transcribed output from speed to text as input to language translation
#### 4.3.4. Language translation
##### 4.3.4.1. Use the translated text from language translation as input to text to speech
#### 4.3.5. Watson Discovery News
##### 4.3.5.1. Use the top article returned by the search from Watson Discovery News as input to Watson Natural Language Understanding Sentiment Analysis and Watson Tone Analyzer 
##### 4.3.5.2. Use the top article returned by the search from Watson Discovery News as input to relationship extraction to tell who is trending in the article

###  4.4. Define common design patterns for composing multiple Watson services together (across APIs).

Cognitive systems tend to gain more value as additional services are composed. With so many services, it’s sometimes hard to tell which services work best together.

#### 4.4.1. Conversation
##### 4.4.1.1. Goal: Engage user in back-and-forth dialog while detecting and acting on user intent. The specifics of the actions taken are guided by the entities discovered.
##### 4.4.1.2. Watson Conversation Service
#### 4.4.2. Q&A
##### 4.4.3.1. Goal: Answer a wide range of customer questions while offering precise answers for frequently asked facts and highly relevant passages for less frequent questions that may not have a single best answer
##### 4.4.3.2. Services: Watson Conversation Service + Watson Discovery Service
#### 4.4.4. Agent Assist
##### 4.4.4.1. Goal: Provide natural language help systems so call agents can rapidly retrieve answers to customer questions
##### 4.4.4.2. Services: Watson Conversation Service + Watson Speech to Text Service + Watson Text to Speech Service + Watson Discovery Service
#### 4.4.5. Automated Customer Support Routing
##### 4.4.5.1. Goal: Detect the topic of a ticket and route to the appropriate department to handle it. E.g. room service, maintenance, housekeeping in the case of hotel guest request routing.
##### 4.4.5.2. Watson Natural Language Understanding (Services: Keyword extraction and sentiment analysis)
I would probably also use NLC to train system about categorizing intents

#### 4.4.6. Social Monitoring
##### 4.4.6.1. Goal: Monitor all posts with specific keywords (e.g. for a company’s followers, sponsors, or critiques) to detect what’s being discussed and the sentiment/tone associated to it.
##### 4.4.6.2. Services used: Keyword extraction, entity extraction, and sentiment/tone analysis (Natural Language Understanding)
#### 4.4.7. Discovery Insights
##### 4.4.7.1. Goal: to prepare your unstructured data, create a query that will pinpoint the information you need, and then integrate those insights into your new application or existing solution.
##### 4.4.7.2. Services Used: (Watson Discovery Service)
 
### 4.5. Design and execute a use case driven service choreography (within an API).
#### 4.5.1. Natural Language Classifier
##### 4.5.1.1. Create a classifier
``curl -u "{username}":"{password}" -F training_data=@train.csv -F training_metadata="{\"language\":\"en\",\"name\":\"My Classifier\"}" "https://gateway.watsonplatform.net/natural-language-classifier/api/v1/classifiers"``
##### 4.5.1.2. Return label information
``curl -X POST -u "{username}":"{password}" -H "Content-Type:application/json" -d "{\"text\":\"How hot will it be today?\"}" "https://gateway.watsonplatform.net/natural-language-classifier/api/v1/classifiers/10D41B-nlc-1/classify"``
##### 4.5.1.3. List classifiers
``curl -u "{username}":"{password}" "https://gateway.watsonplatform.net/natural-language-classifier/api/v1/classifiers"``

#### 4.5.2. Discovery Service
##### 4.5.2.1. Crawl, convert, enrich and normalize data.
##### 4.5.2.2. Securely explore your proprietary content as well as free and licensed public content.
##### 4.5.2.3. Apply additional enrichments such as concepts, relations, and sentiment through natural language processing.
##### 4.5.2.4. Query and analyze your results

#### 4.5.3. Language Translator – Use a default language domain or customize a domain
##### 4.5.3.1. Choose the language to translate or allow Watson to auto-detect the language of your input
##### 4.5.3.2. Select the language for the translator to output
##### 4.5.3.3. Enter or paste text into the input to be translated

#### 4.5.4. Visual Recognition
##### 4.5.4.1. Gather and prepare the training data for classifiers or collections
##### 4.5.4.2. Train and create new classifier or collection by uploading the training data to the API
##### 4.5.4.3. Classify images or search your collection by uploading image files to search against your collection
##### 4.5.4.4. View the search results of the identified objects, scenes, faces, and text that meet minimum threshold. For collections, view the search returns of similar images to the ones used to search.
 
###  4.6. Deploy a web application to IBM Bluemix.

#### 4.6.1. Specific steps:
##### 4.6.1.1. Setup an account on IBM Bluemix
##### 4.6.1.2. Create an instance of an IBM Watson service in the Bluemix account. An instance can be created using either the Bluemix GUI or the Bluemix command line interface.
- via GUI
- via CLI: ``cf create-service service_name service_plan service_instance``

##### 4.6.1.3. Obtain credentials from the IBM Watson service from the service instance.
- via GUI
- via CloudFoundry CLI: ``cf service-key my-service my-credentials-1``

##### 4.6.1.4. These credentials can be obtained programmatically if the application is also hosted on IBM Reference: 
probably they meant ``cf bind-service``
##### 4.6.1.5. The credentials can also be viewed and recorded using the IBM Bluemix CLI or GUI interface.
##### 4.6.1.6. The application can then invoke the IBM Watson Service using the published REST API for that service. Many developers use the Watson API Explorer to create and view API invocations of the desired service using REST protocol.
https://watson-api-explorer.mybluemix.net/

#### 4.6.2. Overall Getting Started reference:
##### 4.6.2.1. “Getting started with Watson and Bluemix”
Reference
https://watson-api-explorer.mybluemix.net/
https://www.ibm.com/watson/developercloud/doc/common/getting-startedvariables.html
https://www.ibm.com/watson/developercloud/doc/common/index.html

### 4.7. Explain the advantages of using IBM Bluemix as the cloud platform for cognitive application development and deployment.
#### 4.7.1. Bluemix provides services for securely integrating on-premises and offpremises applications
Bluemix Local - on-premises
Bluemix Dedicated / Bluemix Public - off-premises

#### 4.7.2. The services in the Bluemix catalog can be used by applications to meet needs in many functional and non-functional areas
#### 4.7.3. Environmental, operational, and functional Security are built into the platform
https://console.bluemix.net/docs/security/index.html#security

#### 4.7.4. Scaling application capacity is simple for applications following modern application design (such as 12-factor methodology)
https://developer.ibm.com/courses/all/cloud-application-developer-certification-preparation-v2/?course=begin#9527

#### 4.7.5. Bluemix facilities the integration and configuration of applications and services
#### 4.7.6. Bluemix is built on open source technologies with community governance
e.g. Cloud Foundry, Docker and OpenStack.

Reference: “What is IBM Bluemix” at
https://console.ng.bluemix.net/docs/overview/whatisbluemix.html#bluemixoverview

## Section 5 - Administration & DevOps for applications using IBM Watson Developer Cloud Services

### 5.1. Describe the process of obtaining credentials for Watson services.

#### 5.1.1. [Use the Bluemix web interface](http://ibm.co/2jdqk8s)

#### 5.1.2. [Get service credentials in Bluemix](https://www.ibm.com/smarterplanet/us/en/ibmwatson/developercloud/doc/getting_started/gs-credentials.shtml)

#### 5.1.3. Get service credentials programmatically
https://docs.cloudfoundry.org/devguide/services/application-binding.html#use
https://www.ibm.com/smarterplanet/us/en/ibmwatson/developercloud/doc/getting_started/gs-credentials.shtml

#### 5.1.4  [Manage organizations, spaces, and assigned users in IBM Bluemix](https://console.ng.bluemix.net/docs/admin/adminpublic.html#administer)

#### 5.1.5 [Using tokens with Watson services](https://www.ibm.com/smarterplanet/us/en/ibmwatson/developercloud/doc/getting_started/gs-tokens.shtml)

#### 5.1.6 Obtain a token
The DataPower Edge Router is responsible for generating and validating authentication tokens. To obtain a token for a service, you send an HTTP ´´GET´´ request to the ´´token´´ method of the authorization API. The host depends on the service your are using. You can identify the host from the endpoint for the service's API.

##### 5.1.6.1 Use a token
You pass a token to the HTTP interface of a service by using the X-Watson-Authorization-Token request header, the watson-token query parameter, or a cookie. You must pass the token with each HTTP request.

##### 5.1.6.2 Get a token programmatically
just to call the ´´token´´ method of the authorizazion API from your app

### 5.2. Examine application logs provided on IBM Bluemix
#### 5.2.1. Log for apps running on Cloud Foundry
https://console.bluemix.net/docs/services/CloudLogAnalysis/log_analysis_ov.html#log_analysis_ov
#### 5.2.2. View logs from the Bluemix dashboard
#### 5.2.3. View logs from the command line interface
#### 5.2.4. Filter logs
https://console.bluemix.net/docs/services/CloudLogAnalysis/kibana/analize_logs_interactively.html#analize_logs_interactively

#### 5.2.5. Configure external logs hosts
https://console.bluemix.net/docs/services/CloudLogAnalysis/external/logging_external_hosts.html#thirdparty_logging

#### 5.2.6. View logs from external logs hosts
References:
https://console.ng.bluemix.net/docs/monitor_log/monitoringandlogging.html#moni
toring_logging_bluemix_apps
http://docs.cloudfoundry.org/devguide/deploy-apps/streaming-logs.html


----------------------

### 5.2. Monitor resource utilization of applications using IBM Watson services.

- [Monitor applications running on Cloud Foundry](https://console.ng.bluemix.net/docs/monitor_log/monitoringandlogging.html#monitoring_logging_bluemix_apps)
- [Monitor applications by using IBM Monitoring and Analytics for Bluemix](https://console.ng.bluemix.net/docs/services/monana/index.html#gettingstartedtemplate)

### 5.3. Monitoring application performance on IBM Bluemix.

- Configure performance monitoring
- [Monitor performance of applications](https://console.ng.bluemix.net/docs/monitor_log/monitoring/monitoring_bmx_ov.html#monitoring_bmx_ov)




