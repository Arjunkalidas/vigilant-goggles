# Conversational AI Chatbot using Deep Learning

## Group Name/Number: Vigilant Goggles (Group 14)
### Arjun Kalidas, Amith Yarra, Girish Kulkarni, Sachin Kulkarni

## Problem Description:

Human computer interaction and conversation is a very challenging task in AI. Especially, when this along with achieving success with Turing Test, is even harder. The success of a Chatbot is measured when it is able to provide the best answer for a query that it receives. They should be able to deliver the best responses, by providing additional information and make the conversation feel realistic. That is a challenge which is why that is the focal point in many researches these days. This involves including the emotions and reactions arising from a conversation between a human and a bot [5]. The interpretation from a conversation is used to calibrate the initial dataset and expanding the dictionary with new terms seemed to improve the originality significantly. When the emotions are analyzed and responses are sent as an error function and as a self-learning balancer the chatbots tend to self-learn better. Although these chatbots solve the general usage and behave according to a particular dataset backed by it, they are unaware of the humane aspects of a conversation. And we hope to achieve a chatbot that is purpose oriented in solving problems for software engineers, curious minds, and is capable of delivering entertainment that can engage a user in a casual conversation without coming out as an emotionless entity. Our research is targeted at large amount of English corpus data, Reddit, StackExchange and StackOverflow data, as a basis for the chatbot, monitoring the behavior, modeling and made to act as a true problem solver while also answering trivia. The ambitious goal of this project is to use a unique model for our bot than the ones  in commonly found bots, measure the performance and evaluation of the use case solving ability. We separate ourselves from other bots by being a unique bot for solving technological queries. From our extensive research, we did not come across one single bot that does what we are trying to achieve. Although we have designed this chatbot for the software engineering solution, there is no domain attributed or any other domain specific components involved, so this can be extended for other use cases as well.

## Related Work:

Our initial interest in Chatbots stems from the very basic idea that when multiple machine learning algorithms and libraries are involved in the development of a complex chatbot model, certain factors are overlooked. There are many chatbots that are already in the market to perform certain specific tasks. And most chatbot development tends to take a path of solving a specific problem that a company website addresses or double down as a customer service executive and so on. Many a times, that leads to less focus on the actual algorithm that goes into the model and more on the aesthetics and dialog flow. During our research we found that organizations and developers apply multiple tools, techniques and approaches and may confuse the development process to the level that chatbots behave sometimes as a random chance of luck. The consensus that deep learning as a single unified method can be applied to solve this problem [2] is a stand we find hard to agree with. Deep neural network is more useful as it processes data within many layers instead of single layer in traditional deep learning, which identifies the important features of the subject under study thereby, allowing us to create more generic models that can be scaled to high data volume. A chatbot can act as an assistant in solving specific problems as well as engage users in a more informative interaction while still maintaining the fun aspect of acting like a normal human to alleviate the barrier, this is a very important aspect that we would like to pursue [1]. And this would involve collecting and processing prior human knowledge. However, this model (Oriol Vinayls et. al) fails to deliver a realistic conversation and fails the Turing test due to lack of a coherent personality.

Hybrid model for chatbots, using retrieval-based system, sequence learning and Long-Term Short Memory, to build a life-like system ended up with occasional grammatical errors, and incomprehensible language, thereby hindering it to be a potential production bot [2]. In a Topic Aware Seq2Seq model [3] A dataset D where X, Y are source and target messages and K is the topic associated with X, the goal is to learn a response generation model from D. From this paper we wish to adapt the Seq2Seq in an encoder-decoder structure. Given a target message Y and a source message X, the model generates probability of Y conditioned on X. The encoder reads the input text X word by word and a context vector ‘c’ is represented through a recurrent neural network. But they fail to mention what is lacking and quote the misgivings of the framework developed. When the useful property of LSTM (Long Short-Term Memory) where it can map an input sequence of variable length into a fixed-dimensional vector representation comes into this equation it makes the model more powerful and impervious to noise [4]. This was a totally LSTM based approach, while they suggest RNN included in this could have made the model better, but they didn’t use the same. A good phrase based SMT (Statistical Machine Translation) model was achieved while not optimized for enough performance.

## Method:

Any query is just a Google search away nowadays. In our model although we leverage such a search engine, we wanted to stay away from a mundane text search as much as possible. Hence, we employ an interactive user friendly environment to encourage wider participation from the developer community. Our vision is to be a B2B environment, like how Slack is to team collaboration, our chatbot is to engineers in a technological environment. We have used several methodologies, tools and datasets such as below:
* English Corpora with NLTK (Natural Language Toolkit)
The main issue with text data is that it is all in text format (strings). However, Machine learning algorithms need some sort of numerical feature vector in order to perform the task.

* Tokenization: We are using RegEx expressions to tokenize the strings and eliminate/ignore the special characters or symbols. Replacing spaces and bad symbols with text and compacting the dataset file. (Tokenization is just the term used to describe the process of converting the normal text strings into a list of tokens i.e. words that are relevant for our context). Sentence tokenizer can be used to find the list of sentences and Word tokenizer can be used to find the list of words in strings.

* Removing Noise i.e. everything that isn’t in a standard number or letter (For example, removal of special characters or emojis that may be entered in the chat as it does not hold good for the context nor the question that we ask for the chatbot).

* Removing Stop words. Sometimes, some extremely common words which would appear to be of little value in helping select documents matching a user need are excluded from the vocabulary entirely. These words are called stop words.

* Stemming: Stemming is the process of reducing inflected (or sometimes derived) words to their stem, base or root form — generally a written word form. Example if we were to stem the following words: “Stems”, “Stemming”, “Stemmed”, “and Stemtization”, the result would be a single word “stem”.

* Lemmatization: A slight variant of stemming is lemmatization. The major difference between the both is that, stemming can often create non-existent words, whereas lemmas are actual words. So, your root stem, meaning the word you end up with, is not something you can just look up in a dictionary, but you can look up a lemma. Examples of Lemmatization are that “run” is a base form for words like “running” or “ran” or that the word “better” and “good” are in the same lemma, so they are considered the same.

* User Interface: Chatbot UI is built using Telegram API. An API is exposed by Telegram, and with an Auth Token, we access the chat interface.

* Developed Intent and Tag Classifiers
We classified the class of data points under consideration. We can classify the classes as targets/ labels or categories. Classification predictive modeling helped us with the task of approximating a mapping function (f) from input variables (X) to discrete output variables (y).


## Difficulties or Problems Experienced:
Challenges:
A major challenge we faced was developing the “intent” classifier for technical questions based on Stack Overflow data. Here, we had to perform a TF-IDF transformation apart from basic pre-processing and tokenization.
Data collection from various sources and assimilating them to be usable for our context.
Preprocessing the textual TSV data into Machine Learning Model recognizable forms.
Learning and understanding various Machine Learning Models and evaluating them to compare the performances.
Understanding the Gensim library and Google Word2Vec concept.


## Detailed Approaches:

User Interface:
We are using Telegram for the interface. The reason to choose telegram is,, it is a messaging app with a focus on speed and security. The API exposed by telegram is simple and free as well for anyone who is interested in using. Also, the sync feature is quite seamless on phone, Mac/PC and tablets.

Telegram provides us with a token when we register as a user. We are using this token to connect with Telegram backend and execute our chatbot logic. And since the UI is also on the same platform, the binding is strong and performant.

Flow of Logic:
With the driver file running in the background the chatbot is ready to talk (Start). Telegram provides the  user with an interface to interact with the bot. After asking the bot a question, the intent classifier determines whether the question falls under a regular conversation category or a programming language question category. If it is a regular non programming question the chatterbot library generates a response and sends it to the Telegram interface.
If the question falls under the programming language category the response is generated by looking for an answer whose question has the highest cosine similarity with the question being asked. The tag classifier uses TF-IDF to determine the response which is sent to the Telegram interface. The tags (programming language) and the questions-responses are stored in the pickle files generated using the pickle library. Storing the questions with database embeddings is performed by creating a model using Google’s pre-trained word2vec model (Gensim - an unsupervised learning model). This vectorizes the user input.



## Detailed Architecture:

Chatterbot Library:
We have implemented the ‘Chitchat’ functionality of our chatbot using the Chatterbot library.
ChatterBot is a Python library that makes it easy to generate automated responses to a user’s input. ChatterBot internally uses a wide array of machine learning algorithms to produce different types of reactions. This feature helped us in developing a skeletal structure for the chatbot quickly and efficiently, although there are a lot of improvements that are required and are underway. We currently use a standard English corpus data and train the model on the same.

## Implementation:
There are two classifiers used and they are in the form of ‘.pkl’ files.
Intent Classifier: This classifier will predict if a question is a Stack Overflow question or not. If it is not a Stack Overflow question, we let Chatterbot handle it.
Programming-Language (Tag) Classifier: This classifier will predict which language a question belongs to and we determine the intent, so that we can search for the respective language questions in our database.

More about classifiers.

- Intent Classifier: The development of natural language (NLP) helps chatbots to understand user requests. But the key to making the chatbot more interactive and providing users with customized communication experiences is the conversation engine system in NLP. The main aspect of this engine for chatbot communication is the identification of purpose. It's a very complex process in fact. Text input is specified by a software function called a "classifier" that will equate the information provided with a specific "intent" to provide a detailed explanation of the computer's terms.

A classifier is a way of categorizing pieces of data into several different categories-in this case, a sentence. Just as human beings classify items into sets, such as a guitar is an instrument, a shirt is a form of clothing, and happy is an emotion, chatbots classify each part of a phrase into categories to understand the meaning behind the feedback it got.

There are various ways to measure vectors from sentences provided by the user. Each approach has its advantages and disadvantages:
Word2Vec 
Weighted Average of Word2Vec from TF-IDF
Doc2Vec 
LSTM 

Choosing one depends on the function that is to be applied on vectors. In this approach, we have used TF-IDF vectorizer, which works as follows:

Text data requires special preparation before it can be used in any machine learning model. The text must be parsed to remove words, called tokenization. Then the words need to be encoded as integers or floating point values for use as input to a machine learning algorithm, called feature extraction (or vectorization).

This is an acronym than stands for “Term Frequency – Inverse Document” Frequency which are the components of the resulting scores assigned to each word.
Term Frequency: This summarizes how often a given word appears within a document.
Inverse Document Frequency: This downscales words that appear a lot across documents. 
TF-IDF are word frequency scores that try to highlight words that are more interesting, e.g. frequent in a document but not across documents.

The TF-IDF vectorizer will tokenize documents, learn the vocabulary and inverse document frequency weightings, and allows the programmer to encode new documents.

- Tag Classifier: The tag classifier classifies the user query to a particular class using the stackoverflow data.
 
The stackoverflow data consists of various user questions and their responses regarding a particular programming language, and it has about 10 such classes, where each class is related to a programming language.  To create the programming language classifier, the stackoverflow data is trained against a logistic regression model using TF-IDF features. This tag classifier is then transformed to a pickle model and saved. 

When a user queries the chatbot about a programming language,  every question is converted to an embedding and stored, so that calculating the embeddings for the whole dataset every time is not needed. Whenever the user asks a Stack Overflow question, a distance similarity measure is used to get the most similar question.

The word embeddings are stored in a folder. This folder has multiple pickle files with each file corresponding to a different programming language. (i.e, Stack Overflow dataset).

The closest answer related to the user query is fetched by the approach of TF-IDF and cosine similarity, below is the math behind the concept to help us understand better:

## TFIDF(i, d) := TF(i, d) / max_k tf(k, d) * log10(N/df(i))
where,
    i is a term
    d is a document (song)
    TF(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (songs)
    df(i) is the number of unique documents containing term i

The cosine similarity, defined as: 
## dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.

## Model:

We have a main python file, which we would like to call a driver file. The code snippets for the implementation can be found in the “Design” module below. In this driver file, we have defined a class that has its methods to perform various operations such as “initialization”, “gather similar questions” and “generate answers”. The “init” method initializes the “chatbot” functionality which is imported from “chatterbot” library. For convenience, we are using Google’s pre-trained Word2Vec model. Here we are initializing “Word2Vec” model and Word2vec is essentially about proportions of word occurrences in relations holding in general over large corpora of text. It is a breakthrough algorithm even though it is not about deep learning, as it demonstrated very effectively how to surface surprising seemingly semantic relationships by unsupervised processing of large amounts of text.

Subsequently we load intent classifier and tag classifier in the form of pickle files. And a vectorizer object that works on the principle of TF-IDF is loaded as another pickle file. We are using embeddings that allow words with similar meaning to have a similar representation. They are a distributed representation for text that is perhaps one of the key findings for the impressive performance of deep learning methods on challenging natural language processing problems. We leveraged the same and it provided us with a decent performance. Our word embedding algorithm is Word2Vec and we are using the “Gensim” library for topic modeling. The reason for choosing Word2Vec over Embedding Layer is because Word2Vec is a statistical method for efficiently learning a standalone word embedding from a text corpus. And Embedding layer using either multilayer perceptron or recurrent neural networks require a lot of training data and can be slow but will learn an embedding both targeted to the specific text data and the NLP task. [6]

In the main method, we use token generated by Telegram and call the “SimpleDialogueManager” method here and pass the same as a parameter to BotHandler. Here, we go on to do some basic checking to make sure the user’s question is valid or opening statement is on a lighter tone than directly jumping into the technical or non-technical query. This is to ensure that user feels like he or she is communicating with a human and not a bot. At least a warm initiation can lead to successful query resolution and user will be at ease.

Our training data contains approximately 6 GB of data, that includes files such as Google News vectors, embeddings, pickle files for intents, tags and TFIDFvectorizer. We use tab separated values (TSV) in our training data. All the intent classifiers and embeddings are trained and created respectively in a conda environment by means of Jupyter notebook. We generate all our intents and embeddings and store them to be later used by the driver file.

We cleaned the data fed to intent classifier and tag classifier. The data was tokenized and eliminated bad symbols, stopwords and removed special characters and spaces. We split the dataset in an 80/20 pareto rule fashion for training and testing respectively. And another challenge we faced was developing the “intent” classifier for technical questions based on StackOverflow data. Here, we had to perform a TF-IDF transformation apart from basic pre-processing and tokenization.


## Results:

The model when trained against a English corpus data, Stack Overflow and Google’s pre-processed Google News vectors, we received an accuracy of 0.78085 for the “Intent Classifier” and 0.75291 for “Programming Language Classifier”. Our model uses the supervised learning technique - Logistic Regression.

## Reflections:

# Review Comment 1:
Please work on your methods to clearly stress your difference. Still I cannot find one from your writing.
In a restricted environment or for ease of usability to reduce the number of hops between websites, it would be convenient to have a chatbot integrated with the system/website.
And just having a chatbot that caters a specific use case for software engineers or scholars that can be integrated directly in a Piazza or Canvas like environment is extremely useful. We don’t have a survey to back our claim, but we ourselves as engineers has faced this a lot and we are solving that problem.
Our chatbot is easily extensible to support various datasets and can be trained to be more context-aware and accurate although we have achieved a significantly high accuracy, precision and recall.
# Review Comment 2:
Good figure. Add caption to explain better
Added captions as a brief explanation below the diagram.

# Review Comment 3:
Content-aware chatbot is already there. This must be your objective or intention. Novelty/difference claim should focus on how you model or realize content awareness differently from other existing work. 
We are using TF-IDF vectorization, and Cosine Similarity to find the closest similar question that are stored in the embeddings. We have each word singled out and the number of times a word is repeated is noted in the form of dictionary. Instead, of trying to find an exact match and that resulting in no responses, unlike other chatbots that try too hard to achieve in the arena, we try to match with  the closest query and return the same. This can be considered analogous to a graceful exit approach taken in software engineering.
Content-aware chatbot is already available in the market. We are aware of the same, but our difference/novelty claim is that just like “StackOverflow” is integrated with Google Collaborate like in the image below, we would like to integrate our chatbot with IDEs such as Eclipse, IntelliJ and Jupyter Notebooks.


# Review Comment 4:
How is this chatbot different from other AI Assistants or bots?
Our chatbot is more specific to the queries that are thrown at it. And it is more tolerant to typo and wrong query phrases. We achieved a 98% accuracy for programming language queries even for wrongly phrased queries, but chatbot seemed to understand our query and returned a close answer. It was not because the chatbot was intelligent like an AI, but the training and testing made it resistant to erroneous queries. 
Most of the AI assistants,  currently available do not recognize certain slangs, languages and vocabularies. We have noticed ourselves that they can be taxing on the user and due to the high linguistic demands, users tend to give up. Google Assistant is by far the best performer of the lot.
AI assistants can help you with immediate queries, general knowledge, news, weather etc and you can use them on the go, but we intend to integrate our bot with a development environment.
# Review Comment 5:
Thanks for sharing your preliminary results. How is your model different from keyword search? 
Programmers or technical personnel tend to prefer a written or typed search rather than a voice search. In most cases, the queries will contain the errors from running a program, log or console. Why not a Google search then? Because you have to exit your current environment and search, but if the chatbot is present right next to you, then the convenience multiplies several folds.
The chatbot not only supports keyword, but phrases, sentences and a casual conversation.
# Review Comment 6:
Use regular reference styles. URL is not right format.
The reference styles are fixed as per the comment.

## Difference/Novelty:
When we see a disruption in the chatbot arena with bots that act as a companion for patients with Dementia, Insomnia and other illnesses [10]. There are bots in most popular websites to guide you through to perform certain activities on them. Bots by Disney and Marvel for entertainment, UNICEF for marginalized communities and medical companies launching their own bots to expedite diagnosing process for patients. We can see bots being developed by real estate giants, e-commerce companies and even Gaming engines come packed with them. Some chatbots can even help you while you are traveling [9], or even can act as a lawyer by clearing your legal doubts. In such a scenario where there are bots available to solve certain specific problems in most fields, we see a lack of support when it comes to Software Engineering, Data Science and other STEM fields. Although many models for chatbots exist, we employ a TF-IDF vectorization and Cosine Similarity, to retrieve the most closest answer available for the query. Additionally, our bot that can engage you in a fun casual conversation, while still being your teacher in answering your most technology related queries sourced from experts and the developer communities. Moreover, contextual awareness is lacking in most chatbots, and our implementation would aim to bring about the feature. This is not similar to the AI assistants that we see today like Google Assistant, Apple Siri or Cortana. We have AI assistants to do a particular task and help with various mobile related and smart home applications.


## Experiments:

The main python file which needs to be running for the chatbot to be active to respond.



Tags separating the data category of the questions asked
Intent classifier: This part of the notebook classifies the data based on the programming language type.




The chatbot is up and running after executing the main.py file. 



The chatbot is capable of classifying the data based on a casual question and a programming related question.
The chatbot is capable of distinguishing an entire question from just keywords. Since, there is a TF-IDF vectorization and cosine similarity employed here, it will take care of both keywords and full phrases or sentences. For example if you see below, we can note that in the same context we are asking for full questions as well as keywords.





## Metrics and Accuracy:
We have achieved a high level of accuracy while predicting and classifying the intent of the user, whether the question asked is related to programming or a normal conversation is taken care by the intent classifier.

The user queries are tested against the google pretrained vectors and a set of features is generated to determine whether the question is dialogue or a technical question. It is achieved via Tokenization, Stemming, Lemmatization and natural language processing of the user queries and performing TFIDF. Finally the query is classified using Logistic Regression to determine the user intent. Some of the metrics have been posted below to testify the accuracy of our model:

## Intent-classifier : Accuracy: ~ 98%

We will plot a confusion matrix below for the classifier which is useful for quickly calculating precision and recall given the predicted labels from a model. A confusion matrix for binary classification shows the four different outcomes: true positive, false positive, true negative, and false negative. The actual values from the columns, and the predicted values (labels) form the rows. The intersection of the rows and columns show one of the four outcomes. For example, if we predict a data point is positive, but it actually is negative, this is a false positive. Below is the confusion matrix for the intent classifier.



Going from the confusion matrix to the recall and precision requires finding the respective values in the matrix and applying the equations:

By applying the above formula, we calculated the precision and recall for the Intent classifier as below:

## Precision: 0.9918598302476795
## Recall: 0.9880346106304079

Once we have the intent classifier up and running, we have a programming language classifier which is a multiclass classifier being developed and put in place, The classifier accepts the user queries which is recognized as a technical question by the intent classifier. The questions are recognized and classified as queries pertaining to different programming languages based on the tags or keywords present in the question. The user query is broken down based on ngram model and TFIDF to look out for relevant words within the vectors.

To understand the accuracy of the multiclass classifier we have computed the precision, recall, f1 score and support by plotting the correlation matrix and using the concepts explained below:


The metrics are computed in a per datapoint manner. For each predicted label, only the score is computed, and then these scores are aggregated over all the data points.
Precision: It calculates the proportion of positive predictions, those are actually correct. The following formula is used to compute the precision of the model.

 
Recall: It calculates the proportion of actual positives that were identified correctly. The following formula is used to compute the recall of the model.

 
To calculate precision and recall for multiclass-multilabel classification. You can add the precision and recall separately for each class, then divide the sum by the number of classes. You will get the approximate calculation of precision and recall for them.
The below figure depicts precision , recall, f1 - score and support for the programming language classifier.
 

 
The accuracy score of the Logistic Regression for our multiclass classifier is 0.8043816124241622. 
 
As for the conversation with our chatbot, the library we used uses logic adapters, which are essentially just modules that take input and return a response. Each response that is returned has a confidence value associated with it. The confidence value is a numeric indicator of how accurate the logic adapter thinks the response is.
A confidence score is kind of a metric for accuracy but it probably better reflects the breadth of a particular bot's knowledge than the accuracy of it's responses. Arguably a chat bot that responds more confidently may also be more accurate, but that is based on the assumption that the set of logic adapters the bot uses are all completely accurate in their ability to judge the precision of the response they generate.
Because ChatterBot's logic adapters each encompass a modular process for selecting a response, there isn't necessarily a common way to gauge the accuracy of all of  them.

## Effort:
We performed an extensive literature survey to gather knowledge of Chatbots their inner workings, math and algorithms underlying the same. After analysing the models, we found that this arena is still maturing and there is a lack of cohesiveness when it comes to purpose oriented bots. Hence, we focussed on improving our model using basic math, with TF-IDF and Cosine Similarity. The task of developing the model was broken down into multiple steps, and they were - 
Collection, analysis, visualization and preprocessing data, along with Google’s pre-trained vectors, NLTK english corpus and popular forum data
 Applied all the relevant math, tokenized data, trained and tested the model
Generated embeddings and Pickle Files to be used as a comprehensive database
Engineered classifiers - Intent classifier, tag classifier and programming classifiers
Integrated the classifiers to build the final model, that retrieves the closest available reply for a query from embeddings and Pickle files.

## Conclusion:
Chatbots are very prevalent nowadays and can be seen playing a pivotal role in our everyday lives. Whether we are visiting e-commerce websites like Amazon, travel websites such as Expedia, banking apps such as Bank of America, Wells Fargo and even Niner Chat in UNC Charlotte Library Website, chatbots are present. In our project, we describe and attempt at developing one such Chatbot that solves a use case for software engineers and technology enthusiasts. We took into account, the various mathematical concepts, machine learning algorithms and human psychological attributes to better reply to the queries. In the development of this project, we sought to overcome the widely known problems and shortcomings of such a system. We encountered numerous bottlenecks along the way beginning with collection of conversational data, and many of them in various formats, understanding the attributes that are relevant for a Chatbot and to process them. We faced a major setback when it came to training the bot for a dialogue classifier, since we had to take into account various language related intricacies, nuances and colloquial variants.  In addition, we achieved very high precision, accuracy and recall values for the training and testing dataset, we hope to pursue our goal and make it more extensible to be used by an enterprise software company or an educational institution.

## Future Work:
We would be improving accuracy of the dialogue classifier to engage users in a deeply involved conversation. We also would like to handle edge cases such as training for other popular spoken languages, improve the speed and performance. This Chatbot can be extended to do daily tasks such as keyword-based intents. Examples: weather, soccer scores, newly-released movies etc. Integrate Telegram based Chatbot with the Canvas website and other in-house educational websites. Also, we would like to expose our chatbot API for the development community to leverage and build their own chatbots.


## References & Citation:
[1] Oriol Vinayls, Quoc V. Le, A Neural Conversational Model 
[2]  Vyas Vijay Bhagawat, Deep Learning for Chatbots 
[3]  Chen Xing, Wei Wu, Yu Wu, Jie Liu, Topic Aware Neural Response Generation 
[4]  Ilya Sutskever, Oriol Vinyals, Quoc V. Le, Sequence to Sequence Learning with Neural Networks 
[5] Serena Leggeri, Andrea Esposito , and Luca Iocchi , A Task-oriented Conversational Agent Self-learning Based on Sentiment Analysis 
[6] https://machinelearningmastery.com/what-are-word-embeddings/ 
[7]https://medium.com/@BhashkarKunal/conversational-ai-chatbot-using-deep-learning-how-bi-directional-lstm-machine-reading-38dc5cf5a5a3 
[8]https://towardsdatascience.com/nlp-sequence-to-sequence-networks-part-2-seq2seq-model-encoderdecoder-model-6c22e29fd7e1 
[9] https://chatbotslife.com/most-innovative-chatbot-on-the-web-bb27d13b1475 
[10] https://www.wordstream.com/blog/ws/2017/10/04/chatbots
[11] https://mlwhiz.com/blog/2019/04/15/chatbot/

 
 ## Dataset:
 https://drive.google.com/open?id=1rXcYSlFPglmBY3BqRzrq7ob7ViJFiihg
