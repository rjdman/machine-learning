# Machine Learning Engineer Nanodegree
## Capstone Proposal
Roger Man  
November 4th, 2019

## Proposal
<!-- _(approx. 2-3 pages)_ -->

### Domain Background
<!-- _(approx. 1-2 paragraphs)_ -->

ASR (automatic speech recognition) is described by [Techopedia](https://www.techopedia.com/definition/6044/automatic-speech-recognition-asr) as the use of software-based techniques to identify and process human voice. In most cases, ASR typically is used to convert spoken words into text. The transcription output can then be used in many practical domains, the most popularised being virtual assistants such as [Siri](https://www.apple.com/uk/siri/) and the [Google Assistant](https://assistant.google.com). As businesses and academic institutions begin to recognise the value of unstructured data, transcription will serve as an important feature to a wide variety of machine learning applications.

However, most transcription services available today require a prior language configuration before being able to provide a transcription. In an increasingly globalised environment, automatic detection of language from audio will become a necessity.

<!-- In this section, provide brief details on the background information of the domain from which the project is proposed. Historical information relevant to the project should be included. It should be clear how or why a problem in the domain can or should be solved. Related academic research should be appropriately cited in this section, including why that research is relevant. Additionally, a discussion of your personal motivation for investigating a particular problem in the domain is encouraged but not required.-->

### Problem Statement
<!-- _(approx. 1 paragraph)_-->

Since widely available speech-to-text transcription APIs currently require a language to be pre-selected before use, there is an opportunity to automate this process. The solution is to create a model which identifies language from speech audio. This will be a multiclass classification with audio-based features and the target can be one of many languages.

<!-- In this section, clearly describe the problem that is to be solved. The problem described should be well defined and should have at least one relevant potential solution. Additionally, describe the problem thoroughly such that it is clear that the problem is quantifiable (the problem can be expressed in mathematical or logical terms) , measurable (the problem can be measured by some metric and clearly observed), and replicable (the problem can be reproduced and occurs more than once).-->

### Datasets and Inputs
<!-- _(approx. 2-3 paragraphs)_-->

The [Common Voice dataset](https://voice.mozilla.org/en/datasets) contains 1945 hours of validated recordings in 29 different languages. Volunteers record voice clips by reading from a bank of donated sentences. These clips are validated through a voting process, only allowing audio clips with 2 or more votes through to the Common Voice dataset.

The audio clips are all in a consistent mp3 format with a sample rate of 48 kHz. Since a variety of volunteers with different hardware have produced this dataset, there may be cause for concern with the consistency of recordings.

Each audio clip has an associated record of metadata, including the following fields:
- client_id
- path
- sentence
- up_votes
- down_votes
- age
- gender
- accent

The upvote metadata will be valuable in finding the most consistent recordings. In addition, the mixed recording conditions may contribute positively to the training, producing a final model is practical and generalisable to the wider world.

The distribution of audio clips in the dataset is not evenly split across languages, gender, age or accent. For instancem more than a third of the dataset is in English. The sheer volume of data alongside the rich metadata will enable us to subsample or cherry pick suitable datapoints for the project without too much trouble.

This dataset is available under a [CCO licence](https://creativecommons.org/share-your-work/public-domain/cc0/). Enhancement and reuse of the works is permitted for any purposes without restriction under copyright or database law.

<!--In this section, the dataset(s) and/or input(s) being considered for the project should be thoroughly described, such as how they relate to the problem and why they should be used. Information such as how the dataset or input is (was) obtained, and the characteristics of the dataset or input, should be included with relevant references and citations as necessary It should be clear how the dataset(s) or input(s) will be used in the project and whether their use is appropriate given the context of the problem.-->

### Solution Statement
<!-- _(approx. 1 paragraph)_-->

The solution is a multiclass classification model with the means to identify the language being spoken within an audio clip. I will attempt to use audio-based python packages such as Librosa to explore different audio preprocessing techniques (spectogram, mel-spectogram, MFCC) and then extract the most suitable features for training. Using image representations of audio captures the complex, unstructured nature of audio into a machine readable format - ripe for deep learning. At this point, I am favouring the use of a pre-trained model such as VGG16 or Resnet50 in order to reduce the training time whilst still capturing more complex features from an image.

<!--In this section, clearly describe a solution to the problem. The solution should be applicable to the project domain and appropriate for the dataset(s) or input(s) given. Additionally, describe the solution thoroughly such that it is clear that the solution is quantifiable (the solution can be expressed in mathematical or logical terms) , measurable (the solution can be measured by some metric and clearly observed), and replicable (the solution can be reproduced and occurs more than once).-->

### Benchmark Model
<!-- _(approximately 1-2 paragraphs)_-->

The benchmark model will be a classifier which defaults to one language. This would be the current scenario in using ASR where we have no prior knowledge of the audio language.

<!--In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail.-->

### Evaluation Metrics
<!-- _(approx. 1-2 paragraphs)_-->

I propose the models will be evaluated based on weighted accuracy since this is a multiclass classification problem. This will give equal weighting to every class.


<!--In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms).-->

### Project Design
<!-- _(approx. 1 page)_-->

* **Programming language:** Python 3
* **Libraries:** Pandas, Numpy, Keras, Sklearn, Librosa
* **Workflow:**
  * Explore the dataset and analyse the datapoints, providing distribution based on metadata.
  * Subsample from the dataset to create well balanced classes.
  * Extract several audio based features from the dataset and determine which features are be more suitable for image classification.
  * Create model architecture using a pre-trained model.
  * Fine tune hyperparameters and architecture to optimise performance.

<!--In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.-->

-----------

<!--**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?-->