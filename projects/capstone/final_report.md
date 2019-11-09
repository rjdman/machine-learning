# Machine Learning Engineer Nanodegree

## Capstone Project

Roger Man  
November 8th, 2019

## I. Definition
<!--_(approx. 1-2 pages)_-->

### Project Overview

ASR (automatic speech recognition) is described by [Techopedia](https://www.techopedia.com/definition/6044/automatic-speech-recognition-asr) as the use of software-based techniques to identify and process human voice. In most cases, ASR typically is used to convert spoken words into text. The transcription output can then be used in many practical domains, the most popularised being virtual assistants such as [Siri](https://www.apple.com/uk/siri/) and the [Google Assistant](https://assistant.google.com). As businesses and academic institutions begin to recognise the value of unstructured data, transcription will serve as an important feature to a wide variety of machine learning applications.

However, most transcription services available today require a prior language configuration before being able to provide a transcription. In an increasingly globalised world, automatic detection of language from audio will become a necessity.

Using Convolutional Neural Network architectures in audio classification is an effective and increasingly common approach to [audio classification](https://ai.google/research/pubs/pub45611). In this project, I hope to use open-source datasets and software packages in order to automate the detection of language from audio using Deep Neural Networks.
<!--In this section, look to provide a high-level overview of the project in layman’s terms. Questions to ask yourself when writing this section:
* _Has an overview of the project been provided, such as the problem domain, project origin, and related datasets or input data?_
* _Has enough background information been given so that an uninformed reader would understand the problem domain and following problem statement?_-->

### Problem Statement

Since widely available speech-to-text transcription APIs currently require a language to be pre-selected before use, there is an opportunity to automate this process.

The solution is to create a model which identifies language from speech audio. This will be a multiclass classification using audio-based features and the target can be one of many languages.

I tackled this problem in a number of steps. Firstly I performed an exploratory data analysis to determine the underlying characteristics of the data in order to find a robust dataset sample for this exercise. The next step was to understand the different transformations available for audio classification. Having settled on using mel spectograms as a feature, preprocessing the audio and extracting features into train/validation/test sets. I selected transfer learning as an approach to classifying audio in order to reduce the training time whilst still capturing more complex features from an image. Convolutional Neural Networks typically perform very well in this image classification space. Using the higher level bottleneck features obtained from the ResNet50 model, I trained a classification model using a simple Deep Neural Network.

Finally, I evaluated the performance of the final model with the test dataset.

<!--In this section, you will want to clearly define the problem that you are trying to solve, including the strategy (outline of tasks) you will use to achieve the desired solution. You should also thoroughly discuss what the intended solution will be for this problem. Questions to ask yourself when writing this section:
* _Is the problem statement clearly defined? Will the reader understand what you are expecting to solve?_
* _Have you thoroughly discussed how you will attempt to solve the problem?_
* _Is an anticipated solution clearly defined? Will the reader understand what results you are looking for?_-->

### Metrics

For this project, I will simply be using accuracy as an evaluation metric. This is the number of correct predictions divided by the total number of  predictions.

Accuracy can sometimes be misleading. In the case of imbalanced classes, we may get a high accuracy without solving the problem. I have taken care to ensure the labels of my dataset will be uniformly distributed, therefore accuracy will indeed be a good indicator of performance. In practice, we are likely to look at precision or recall to understand where we are misclassifying between classes. However, this would be just an aid to help us improve accuracy.

<!--In this section, you will need to clearly define the metrics or calculations you will use to measure performance of a model or result in your project. These calculations and metrics should be justified based on the characteristics of the problem and problem domain. Questions to ask yourself when writing this section:
* _Are the metrics you’ve chosen to measure the performance of your models clearly discussed and defined?_
* _Have you provided reasonable justification for the metrics chosen based on the problem and solution?_-->


## II. Analysis

<!--_(approx. 2-4 pages)_-->

### Data Exploration

The [Common Voice dataset](https://voice.mozilla.org/en/datasets) contains 1945 hours of validated recordings in 29 different languages. The English dataset alone is more than 30gb in storage size. Volunteers record voice clips by reading from a bank of donated sentences. These clips are validated through a voting process, only allowing audio clips with 2 or more votes through to the Common Voice dataset. This dataset is available under a [CCO licence](https://creativecommons.org/share-your-work/public-domain/cc0/). Enhancement and reuse of the works is permitted for any purposes without restriction under copyright or database law.

There is a master tsv (tab separated values) file for each language containing the associated metadata for the audio clips, including the following fields in string format:

* **client_id**: unique identifer of the contributor
* **path**: path to the audio file from the clips folder
* **sentence**: sentence spoken by the contributor
* **up_votes**: number of up votes from other contributors
* **down_votes**: number of down votes from other contributors
* **age**: age grouped by decade
* **gender**: male, female or other
* **accent**: regional accent if reported

The audio clips are all in a consistent mp3 format with a sample rate of 48 kHz. Since a variety of volunteers with different hardware have produced this dataset, there may be cause for concern with the consistency of recordings. The up vote metadata will be used as a proxy in finding the most consistent recordings. The mixed recording conditions may actually contribute positively to the training, producing a final model is practical and generalisable to the wider world.

The sheer size of the dataset cannot be computed using my limited time and resources. I chose to sample 1000 audio files from the three languages: English, Spanish and Chinese. These languages seems very different in terms of linguistics, tone, pitch which I hope will lead to more distinguishable features between the classes.

The Chinese dataset is heavily gender imbalanced with just 32 usable recordings made by females compared to the 4909 clips created by males. There is no opportunity to correct the gender imbalance with oversampling since there are so few records from the female Chinese population. To eliminate the possibility of this gender imbalance becoming a key driver behind the final model created, the decision was made to only use the male dataset for each language.

<!--In this section, you will be expected to analyze the data you are using for the problem. This data can either be in the form of a dataset (or datasets), input data (or input files), or even an environment. The type of data should be thoroughly described and, if possible, have basic statistics and information presented (such as discussion of input features or defining characteristics about the input or environment). Any abnormalities or interesting qualities about the data that may need to be addressed have been identified (such as features that need to be transformed or the possibility of outliers). Questions to ask yourself when writing this section:
* _If a dataset is present for this problem, have you thoroughly discussed certain features about the dataset? Has a data sample been provided to the reader?_
* _If a dataset is present for this problem, are statistics about the dataset calculated and reported? Have any relevant results from this calculation been discussed?_
* _If a dataset is **not** present for this problem, has discussion been made about the input space or input data for your problem?_
* _Are there any abnormalities or characteristics about the input space or dataset that need to be addressed? (categorical variables, missing values, outliers, etc.)_-->

### Exploratory Visualization

<!--In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
* _Have you visualized a relevant characteristic or feature about the dataset or input data?_
* _Is the visualization thoroughly analyzed and discussed?_
* _If a plot is provided, are the axes, title, and datum clearly defined?_-->

![Example mel spectogram][image]

[image]: images/mel spectogram.png "Example mel spectogram"

The image above is a visualisation plot from the mel spectogram extraction of an audio file. The plot was generated using the [Librosa](https://librosa.github.io) The following characteristics can be found:

* The y-axis shows the frequency bins in the mel scale going from 200Hz to 8000Hz. Any frequencies above or below are not in the human voice range in log scale.
* The x-axis illustrates the length of the audio in seconds.
* The colour bar on the right hand side denotes the amplitude in log scale.
* There is no silence on either side of the graph as the audio has been trimmed.

The rationale behind using the log scale of frequencies, is that it similar to the way in which humans process or perceive sound.

This visual representation of the audio, shows several dimensions of data relating to sound or voice: time, frequency and amplitude. In my opinion, we have encoded the audio information related to a voice or sound into an image. This would be a great input for an image classification model using Convolutional Neural Networks.

### Algorithms and Techniques

**ResNet50** is a state-of-the-art Deep Neural Network architecture with 50 layers trained on the ImageNet dataset. The model architecture was groundbreaking in solving the [vanishing gradient problem](https://medium.com/@anishsingh20/the-vanishing-gradient-problem-48ae7f501257) where deeper neural networks are outperformed by shallower neural networks.

![Resnet50 model architecture][architecture]

[architecture]: https://miro.medium.com/max/2792/1*hEU7S-EiVqcmtAlj6kgfRA.png "Resnet50 model architecture"
[source: medium, Priya Dwivedi](https://i.stack.imgur.com/msvse.png) Resnet50 model architecture

Similar to other Convolutional Neural Networks (CNNs) such as VGG16, ResNet50 is formed of convolutional layers. These convolutions give spatial context that fully-connected neural networks lack, so in cases where spatial context is important, for instance image classification, CNNs typically outperform fully-connected networks. Due to the max pooling layers, complex CNNs require fewer parameters to train and are less prone to overfitting.

However, traditional CNNs such as VGG16 still face vanishing gradients. ResNet50 eliminates this problem through the use of identity mapping. Identity mapping or [skip connections](https://towardsdatascience.com/understanding-and-coding-a-resnet-in-keras-446d7ff84d33), take the input from one layer and propagates to a layer further ahead than it normally would - the concept is outlined below in a residual block. This generates alternate paths for the gradient to move through the network - thus allowing us to build much deeper networks than traditionally possible.

![Residual block][block]

[block]: https://i.stack.imgur.com/msvse.png "Residual block"
[source: imgur](https://i.stack.imgur.com/msvse.png) Residual block

Since the ResNet50 model is a great classifier for images, **Transfer Learning** can be applied using the ResNet50 model. Rather than having to build a whole new model architecture and train weights from scratch, we can remove the fully-connected layers or the classifier from the ResNet50 model. We can then build our own model architecture on top then apply one of the following strategies (as shown in the image below):

* Train the entire model (initialising pre-trained weights or random)
* Freeze some of the initial layers and train the other layers
* Freeze the bottleneck features and train using the output of the convolution layers as input to your own model architecture

![Residual block][transferlearning]

[transferlearning]: https://miro.medium.com/max/5994/1*9t7Po_ZFsT5_lZj445c-Lw.png "Residual block"
[source: towardsdatascience, Pedro Marcelino
](https://towardsdatascience.com/transfer-learning-from-pre-trained-models-f2393f124751) Transfer learning strategies

Transfer learning offers the ability to leverage the learnings made by another model and apply those learnings to a similar use case. The high performance from the ResNet50 model in the image classification space is the main rationale behind using transfer learning with ResNet50.

In all strategies outlined above, we always have our own classifier with at least one fully-connected layer. The following parameters will be used to build, train and tune a classifier:

* **Model architecture**
  * Number of layers (depth of the network)
  * Layer types and their respective parameters (including activation functions)
    * dense (number of nodes, regularisers)
    * convolution (filters, stride, padding)
    * dropout (probability)
    * pooling (max pooling, average pooling)

* **Model Training parameters**
  * Number of epochs (number of complete passes)
  * Batch size (number of samples for every update)
  * Loss function (the error function e.g. MSE, categorical_crossentropy)
  * Optimser (function used to minimise the loss function e.g. RMSprop, ADAM, SGD)
  * Learning rate (default usually 0.001)
  * Momentum (for SGD)
  * Metrics (accuracy)

In terms of data inputs and the transformations into in the model, the flow of data would be as follows:

* Audio files are converted into mel spectograms and saved to disk. The number of bins has been set to 224 so that a image of dimensions width x 224 will be created - the height is well-suited for ResNet50.
* The keras function `preprocess_input` from `keras.applications.resnet50` was able to take the images with varying heights and resize into arrays of (224,224,3).
  * The width of the images were rescaled to fit the 224 width dimension.
  * Since these image were grayscale, the `preprocess_input` function manages to automatically create three channels to replicate RGB by copying the original image two times over.
* The ResNet50 model needs to initalised with the parameter `include_top=False`. This removes the final fully-connected in the original model, allowing us to create our own model architecture to classify on top.
* **[OPTIONAL]** If we want to freeze all the convolutional layers, we want a convenient way to store the botteneck features from the convolutional layers of ResNet50. Since the weights are preloaded from Keras, next step is use the `model.predict()` function with the (1800, 224,224,3).shape array as input - the 1800 denotes the number of training files. This would be the input to my own fully-connected layers.

<!--In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
* _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
* _Are the techniques to be used thoroughly discussed and justified?_
* _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_-->

### Benchmark

The benchmark model will be a classifier which defaults to one language. This would replicate the current scenario in using ASR where we have no prior knowledge of the audio language. Our data has been split into three equally balanced classes. Using the evaluation metric defined above, this benchmark would achieve a 33.3% accuracy score. Success would be achieved if the final model beats this score.
<!--In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
* _Has some result or value been provided that acts as a benchmark for measuring performance?_
* _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_-->

## III. Methodology
<!--_(approx. 3-5 pages)_-->

### Data Preprocessing

I chose to split the data into train/validation/test sets. The train set will be used to train the classifier, with the validation set providing context on the generalisation of the model. The test set will be kept to evaluate the performance of the final model.

There were four stages to preprocessing:

* Sampling
* Creating folders
* Generating and saving images

The process to generate my sample datasets from the metadata can be summarised in the following steps:

1. The audio metadata is ordered by up votes for each language 
2. The audio metadata is filtered by male
3. The top 1000 audio files are selected for each language
4. The sampled metadata is randomised for each language
5. The metadata is divided to create train/validation/test sets for each language
6. The train/validation/test sets for the three languages are combined together to create equally balanced classes

This generated 1800 training datapoints, 600 validation datapoints and 600 test datapoints.

With the train/validation/test split in mind, I used the following folder structure to manage my mel spectogram images:

* data/train-mel spectogram/
* data/validation-mel spectogram/
* data/test-mel spectogram/

From the audio file path from each of these metadatapoints above, I ran the following workflow:

1. Load audio file using path
2. Trim the silence from the beginning and end of the audio file
3. Extract the log mel-frequency spectogram array
4. Scale array elements between 0 and 255
5. Pivot the array (width represents time)
6. Initialise mel-frequency spectogram image from array
7. Save file to relevant folder outined above

I used the following packages to generate and save images:

* [Librosa](https://librosa.github.io) - in-built functionality to explore audio features including MFCCs, spectograms and mel spectograms.
* [Pillow](https://pillow.readthedocs.io/en/stable/) - functions to convert, save and crop images from numpy arrays
* [Numpy](https://numpy.org) - helpful in manipulating arrays efficiently

This generated 1800 training mel spectograms, 600 validation mel spectograms and 600 test mel spectograms. These images were all 224 pixels in height, though they did vary in width. I did not feel that this would be an issue since the CNN architectures would learn from many examples that this is not a distinguishing feature.

<!--In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
* _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
* _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
* _If no preprocessing is needed, has it been made clear why?_-->

### Implementation

**Data extraction and preprocessing**

I did spend a majority of my time working on feature selection and the subsequent data processing. Investigating the different kinds of audio features such as MFCCs, spectograms, mel-frequency and spectograms was fairly challenging with no background in signal processing. The different sources of data (Chinese, English, Spanish) and the number of files meant that a bottleneck in the process was the processing time for singular processes (creating the mel spectogram). To alleviate that bottleneck, I decided to create a multiprocessing function which made the reprocessing files up to 16 times faster.

**Modelling**

I used the [Keras](https://keras.io/why-use-keras/) python library in the creation of the final solution:

* **ResNet50:** A pre-trained ResNet50 model is available on Keras. The Keras API is intuituve with in-built presets for hyperparameters but with the flexibility for lower-level tweaking. This means that we can quickly build a baseline model which then can be refined.
* **Transfer learning:** Keras enables the application of transfer learning by allowing chosen layers to be frozen or unfrozen depending on the chosen strategy. The object `model.layers` contains all of the layers. Each of these layers has an attribute `trainable` which freezes the layer if set to `False` (vice versa).
* **Speed and versatility:** Although training the a neural network can be slow, this can be alleviated using cloud computing or powerful hardware. Using a GPU can greatly accelerate the training time with very little change in the code by running the code below.
  > `sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))` 
  
  [source](https://stackoverflow.com/questions/45662253/can-i-run-keras-model-on-gpu)

I used a ResNet50 model to apply transfer learning to the dataset. In the interests of training time, I froze all convolutional layers and kept the ImageNet weights to extract the bottleneck features which I then used as features into my own model architecture. The implementation steps are documented extensively in the *Algorithm and Techniques* section, so I will only provide brief commentary on what I previously covered and focus on the implementation details of the initial solution.

1. Initalise pre-trained ResNet50 model from Keras with `include_top=False` as an input parameter.
2. Using `image` class from `keras.preprocessing` module load mel spectogram images using the `image.load(<path>)` method. Scaling the RGB array elements between 0 and 1 in the same step.
3. Apply `preprocess_input` method to ensure the inputs will work as expected.
4. Extract bottleneck features from the initialised model using `predict` method.

At this point, the output from the convolutional layers (bottleneck features) were stored in an array with shape (1800, 7, 7, 2048). Steps 1-5 were also applied to the test and validation set. This numpy array represents the 1800 training files and the complex features generated from ResNet50.

*First model architecture*
![Model architecture from first model version][firstarchitecture]

[firstarchitecture]: images/firstarchitecture.png "Model architecture from first model version"

The overall model architecture was fairly lightweight.

* The input layer was a global average pooling layer. This had an effect of dimension reduction and greatly reduces the number of parameters for the next layers.
* A dropout layer was put in place next to reduce overfitting. This enables the random dropping of nodes - in this case the probability of dropping a ode was 50%.
* The next layer was a dense layer with 512 nodes. This is a fully-connected  layer in which every node in this layer is connected to every node in the next layer. The ReLU (Rectified Linear Unit) activation function was used. The ReLU activation function typically increases training speed and alleviates to some extent the vanishing gradient problem. This is the default activation function for most DNNs.
* A dropout layer with 0.2 probability.
* The output layer was a dense layer with three nodes (representing the three classes) and a softmax activation function. The output from the activation function adds up to 1.

At this point, I was ready to train the model. I used the following parameters for the first model:
* Loss function: `categorical_crossentropy`
* Optimiser: `rmsprop` (default)
* Metrics: `accuracy`
* Epochs: `100`
* Batch size: `25`

<!--In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
* _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
* _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
* _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_-->

### Refinement

<!--In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
* _Has an initial solution been found and clearly reported?_
* _Is the process of improvement clearly documented, such as what techniques were used?_
* _Are intermediate and final solutions clearly reported as the process is improved?_-->

![First train validation loss][firstloss]

[firstloss]: images/loss_61833_first.png "First train validation loss"

![First train validation accuracy][firstaccuracy]

[firstaccuracy]: images/accuracy_61833_first.png "First train validation accuracy"

The initial performance of the model was as follows: 61.8% accuracy on the test dataset. This is a good performance for a first pass. However, I was worried by the near convergence of the train validation loss. It seemed like there wasn't much I could do from just changing the hyperparameters.

In search of better performance, I experimented with different elements of the process defined above:

* Different model architectures (VGG16, MobileNet_V2). Using the same transfer learning method, these models produced slightly worse results. Accuracy on the test set came in around 55%.
* Unfreezing layers (last 4 layers - this took too long). My machine was not up to the task. This was physically impossible with my time constraints. This may have been possible with cloud computing.
* Further subsampling to find mel spectograms of consistent size and transformations in the form of cropping mel spectograms images. Accuracy using the same model was poor ~40% accuracy on the test set.
* Increased the size of training/test/validation dataset (up to 4000). However, the increased training time did not coincide with greater performance. Accuracy on the test set was 55% at best.

The only method which increased the model accuracy on the test set was tuning the hyperparameters and rebuilding the model architecture of the fully-connected layers. I believe the most important factors were as follows: 

* Changing the architecture to include more layers, less nodes and more dropouts proved to increase accuracy slightly and made the new model less prone to overfitting. Though the model did take longer to converge.
* Changing the optimser - ADAM (Adaptive Moment Optimization) seemed to perform better than RMSProp.
* Increasing the number of epochs. This allowed more time for the model to seek out gradients and allowing for the training and validation loss to converge.
* Increasing batch size did not have much of an effect on overall performance.

## IV. Results
<!--_(approx. 2-3 pages)_-->

### Model Evaluation and Validation

*Final model architecture*
![Model architecture from final model version][modelarchitecture]

[modelarchitecture]: images/architecture.png "Model architecture from final model version"

The overall model architecture was fairly lightweight. In total there were just 279,235 parameters to train which is fairly fast to train.

* The input layer was a global average pooling layer.
* A dropout layer with 0.5 probability of dropping nodes.
* A dense layer with 128 nodes and ReLU activation.
* A dropout layer with 0.5 probability of dropping nodes.
* A dense layer with 64 nodes and ReLU activation.
* A dropout layer with 0.4 probability of dropping nodes.
* A dense layer with 128 nodes and ReLU activation.
* A dropout layer with 0.2 probability of dropping nodes.
* The output layer consisted of a dense layer with three nodes and a softmax activation function.

I used the following parameters for the final model:
* Loss function: `categorical_crossentropy`
* Optimiser: `ADAM`
* Learning rate: `0.0001`
* Metrics: `accuracy`
* Epochs: `1000`
* Batch size: `50`

![Final train validation loss][finalloss]

[finalloss]: images/loss_67333_ADAM_1000.png "Final train validation loss"

![Final train validation accuracy][finalaccuracy]

[finalaccuracy]: images/accuracy_67333_ADAM_1000.png "Final train validation accuracy"

The final accuracy on the test dataset achieved was 67.3%.

I was hoping for a larger increase in accuracy from the initial model. This represents a 5.5% increase in performance from hyperparameter tuning and changes to the model architecture. The final parameters are appropriate as you can see the model starting to overfit in the 600th epoch, with the training loss exceeding validation loss. This seems to be the maximum we can extract from the data.

In order to validate the model. I created another test set 'unseen' to verify the results we obtained in this exercise. This consisted of files which had less up votes in comparison to the previous dataset the final model was trained upon so there may be a difference in quality.

Putting through 3000 sample files from English, Chinese and Spanish leads to a unseen accuracy of 56.4%. This is not the headline 67.3% recorded by the test set but it is in line when we consider the data may be less clean in comparison to the data we trained upon.

It would be interesting to see the performance of the model when taking recordings from the real world - not in a test environment. I think the model will struggle in such a situation since the underlying data the model is built upon is not generalised to a great amount of noise.

<!--In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
* _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
* _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
* _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
* _Can results found from the model be trusted?_-->

### Justification

At 67.3% accuracy, the final model is more than twice as accurate as the benchmark model (33.3%). This is more than twice as good as the benchmark. We have met our goal that we set out to achieve.

Although the solution is not robust enough to be part of a workign product, the results prove that language identification can be done and I am sure that with more research, time and better data we can create a credible solution that can be used commerically. With an accuracy of just 67.3%, it is unlikely anyone will use this solution in their day to day. Those who do use ASR today are unlikely to need automatic language identification. By configuring their language settings (whether it be a voice assistant or something else) to the language they use day to day. In that case, our 67.3% pales in comparison to an almost 100% accuracy in real life.

Nevertheless, this is a great proof of concept in identifying/applying a viable method to solve an issue which I feel will be important to solve in the era of automation.
<!--In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
* _Are the final results found stronger than the benchmark result reported earlier?_
* _Have you thoroughly analyzed and discussed the final solution?_
* _Is the final solution significant enough to have solved the problem?_-->


## V. Conclusion
<!--_(approx. 1-2 pages)_-->

### Free-Form Visualization

<!--In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
* _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
* _Is the visualization thoroughly analyzed and discussed?_
* _If a plot is provided, are the axes, title, and datum clearly defined?_-->

The visualisation below shows the raw input features fed into the Keras `preprocess` function for ResNet50 models. A short analysis of the spectograms identifies the following:

* Consistent heights, but vastly different widths of the mel spectogram images. The huge differences between the image widths implored me to experiment with cropping files as outlined previously (to no avail).
* Large sections of white pixels - an indication of silence. Although the audio files have been trimmed, there seems to exist pauses in speech. This could be a time-related feature that can be picked up by convolutional layers. However, in future we could also approach this in a different way by using voice activity detection methods to remove all silence the mel spectograms.

*Preprocessed grayscale mel spectogram features for each language*
![Example grayscale mel spectogram features][features]

[features]: images/features.png "Example grayscale mel spectogram features"

### Reflection

<!--In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
* _Have you thoroughly summarized the entire process you used for this project?_
* _Were there any interesting aspects of the project?_
* _Were there any difficult aspects of the project?_
* _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_-->

The summary of the steps taken within the project are listed out as follows:

1. Identified problem statement and dataset
2. Exploratory data analysis
3. Sample robust dataset
4. Audio feature transformation analysis
5. Preprocessing mel spectograms for train/validation/test sets
6. ResNet50 bottleneck feature extraction from mel spectograms
7. Build model architecture
8. Iterate
    * Train
    * Validate
    * Tweak hyperparameters
9. Evaluate model with test dataset

* The most interesting part of the project was creating a solutions to micro problems on the journey to finding a larger solution. Although frustrating, some elements of the code were quite useful (such as my multiprocessing script `parallel.py`) and I will likely be able to use these learnings in my future projects.
* The most difficult part of the project was implementing a model in a deeply technical subject matter that I was unfamiliar with - audio. I had to deal with two areas of research deep learning and audio signaling. This was difficult to balance with my day to day. In addition, comparative to image classification, there is very little in terms of material for purely audio classification.
* The final solution does meet expectations. At 67.3%, it is more than twice as accurate as the benchmark model (33.3%). However, I do not believe this is a solution that will be beneficial for day to day applications. The amount of error is just too great for the model to be relied upon - just imagine a voice assistant which gets your language right two thirds of the time! There are also practical considerations in using this tool day to day in terms of the difficult data transformation pipeline.

### Improvement

The current model does provide better results than the benchmark model, nevertheless there are still a lot of potential improvements to be made.

* Data cleansing activities could have made a direct impact on the performance of the final model. Retrospectively, I do feel the Chinese dataset did not have the same depth as the Spanish and English sets. There were comparatively fewer votes for the Chinese dataset. If I could choose the languages again, I may have gone for Russian, Italian, German, English and Spanish since they had the most complete data. These languages would have also opened the doors to further increases in the train/validation/test set size and might have led to better accuracy or generalisation at least.

* I would like to use cloud computing next time I apply deep learning to a problem. It seems that deep learning requires many examples to train on in order to become robust. In light of that, cloud computing gives me an oportunity to train with more data and thus is likely to improve results.

* If I used my final solution as the new benchmark, there most definitely is a better solution. There are three areas in which my solution falls down, usability, speed and performance. Google enables multiple languages to be used on it's own voice assistant and language detection though limited to a certain extent. Even if Google's performance is worse, the implementation in terms of outreach is far better than my locally built model. That can be changed with efforts in creating a more efficient pipeline and serving my model through a website (or via API at least).

<!--In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
* _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
* _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
* _If you used your final solution as the new benchmark, do you think an even better solution exists?_-->

-----------

<!--**Before submitting, ask yourself. . .**

* Does the project report you’ve written follow a well-organized structure similar to that of the project template?
* Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
* Would the intended audience of your project be able to understand your analysis, methods, and results?
* Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
* Are all the resources used for this project correctly cited and referenced?
* Is the code that implements your solution easily readable and properly commented?
* Does the code execute without error and produce results similar to those reported?-->

References:

[*Deep Residual Learning for Image Recognition*](https://arxiv.org/pdf/1512.03385.pdf) Kaiming He et al.

[*Using CNNs and RNNs for Music Genre Recognition*](https://towardsdatascience.com/using-cnns-and-rnns-for-music-genre-recognition-2435fb2ed6af) Priya Dwivedi

[*Accuracy, Precision, Recall & F1 Score: Interpretation of Performance Measures*](https://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/) Renuka Joshi

[*A COMPARISON OF AUDIO SIGNAL PREPROCESSING METHODS FOR DEEP NEURAL NETWORKS ON MUSIC TAGGING*](https://arxiv.org/pdf/1709.01922.pdf) Keunwoo Choi et al.

[*Deep Learning with Audio Thread*](https://forums.fast.ai/t/deep-learning-with-audio-thread/38123) Robert Bracco

[*Combining High-Level Features of Raw Audio Waves and Mel-Spectrograms for Audio Tagging*](https://www.groundai.com/project/combining-high-level-features-of-raw-audio-waves-and-mel-spectrograms-for-audio-tagging/1) Marcel Lederle, Benjamin Wilhelm

[*A Comprehensive Hands-on Guide to Transfer Learning with Real-World Applications in Deep Learning*](https://towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a) Dipanjan (DJ) Sarkar

[*Getting to Know the Mel Spectrogram*](https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0) Dalya Gartzman

[*Audio Classification using FastAI and On-the-Fly Frequency Transforms*](https://towardsdatascience.com/audio-classification-using-fastai-and-on-the-fly-frequency-transforms-4dbe1b540f89) John Hartquist

[*An overview of gradient descent optimization algorithms*](https://ruder.io/optimizing-gradient-descent/) Sebastian Ruder

[*A bunch of tips and tricks for training deep neural networks*](https://towardsdatascience.com/a-bunch-of-tips-and-tricks-for-training-deep-neural-networks-3ca24c31ddc8) Long Ang
