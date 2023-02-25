![alt text](https://th.bing.com/th/id/R.37b89b3df6c6cb7f7c714b538d73ed7b?rik=PPzbQhbNzyVFlQ&riu=http%3a%2f%2flomonteandcollings.ca%2fwp-content%2fuploads%2f2016%2f11%2fIntact.png&ehk=mCNrLu%2fN6fY0e4n8ersNFFZH7tvr%2f84zCPBhHBnlhB8%3d&risl=&pid=ImgRaw&r=0)

# CxC 2023 - Intact Challenge Submission

## Inspiration
As Canada's largest P&C insurer, Intact leverages the power of data to address various problems. One such problem is the classification of medical documents into medical specialties. Intact receives a large number of medical claims on a daily basis that must be managed and sorted in an effective manner to ensure that consumers are receiving their rightful claims. Machine Learning can be utilized to develop a model that classifies the text from these documents into a medical specialty.

## What it does
This notebook outlines the process of developing a medical document classifier.

## How we built it
This model was developed using HuggingFace's 'XLNet-base-cased' NLP model, trained on the data provided by Intact. The models were run in a Jupyter Notebook on Google Collab, using a standard GPU hardware accelerator (this model can be run without the accelerator, however, it will be **very** slow).

## Challenges we ran into
-   The data was extremely imbalanced. Due to the nature of different medical documents, the extremely imbalanced training dataset omitted many terms that could exist in these documents. While preprocessing the transcripts and weighing the classes helped emphasize the underrepresented classes, the model will greatly benefit from more samples in the underrepresented classes.
-   Optuna was inconsistent. The hyperparameters used in the optimizer yielded a different f1_macro score than the final model when being judged against the validation data. Simpletransformers is better optimized for Wandb (another hyperparameter tuning library), however, the library ran into many issues with Google Collab, mainly when using a GPU/TPU accelerator. When switching to a CPU backend without using CUDA cores, it ran at an extremely slow pace which led to the optimizer being timed out.
* Google Collab was limited in its capabilities. The model was trained on Google Collab with additional computing cores. Due to the limitations in RAM and storage, the amount of training that could be performed was severely limited. The hyperparameters optimized using Optuna had to be significantly restricted to ensure that they could run on the limited hardware available. Additionally, Google Collab timed out often, meaning that most of the training steps in this notebook had to be repeated. We did try using Azure Notebooks, however, there was a compatibility issue that prevented the easytransformer library from being imported. 

## Accomplishments that we're proud of
* Cleaning the data so that it can easily be used to train the model.
* Attempting to fine-tune the parameters using optuna
* Utilizing XLNet over bag-of-words to improve the model (testing different models)

## What we learned
* How to prepare and clean data for NLP tasks
* How to use HuggingFace's library of NLP Models
* Communicating my thoughts in an effective manner to a technical audience

## What's next for j4noronh_Intact_CxC2023
In the future I would want to learn how to better optimize my training process to make it more consistent and efficient.
