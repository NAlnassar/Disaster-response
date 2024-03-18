# Disaster Message Classifier

This repository contains a model for classifying disaster messages, along with the necessary data used for training and a web application to interact with the trained model.
Hoping this project can help the community by speeding up the process of classifying each and every message as fast as possible to allow for the least number of injuries and casualties.
even a single life saved is worth the effort!


## Quickstart

> [!NOTE]
> Instructions to:
>
> 
>  process data:
>
> `cd /path/to/repo`
>
> process data and save to database:
>
> `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/dismessages.db`
>
>  Train model from database and save the model:
> 
> `python models/train_classifier.py data/dismessages.db models/classifier.pkl`
>
> Run The Web Application:
>  `python run.py`

### Installation

Install the required libraries by running the following command:
`pip install -r requirements.txt`

## Libraries Used in the Repository

- [Pandas](https://pandas.pydata.org/): A data manipulation library for Python, providing data structures and functions for efficiently handling and analyzing **structured** data.
- [Numpy](https://numpy.org/): A fundamental package for scientific computing with Python, supporting arrays, matrices, and tools needed for machine learning tasks and scientific projects in general.
- [Scikit-learn](https://scikit-learn.org/): A versatile machine learning library for Python, featuring various algorithms for classification, regression, clustering, along with utilities for model evaluation and data preprocessing. Even though it is pretty high-level compared to PyTorch or TensorFlow, it gets the job done.
- [NLTK](https://www.nltk.org/): NLTK is a Python library for natural language processing tasks. It offers tools for text processing, such as tokenization and tagging.

## Files in the repository
-Root Directory
  -app 1
  
    -templates 1.1
    
      -go.html 1.1.1
      -master.html 1.1.2
      
    -run.py 1.2
    
  -data 2

    -Disaster_categories.csv 2.1
    -Disaster_messages.csv 2.2
    -dismessages.db 2.3
    -process_data.py 2.4
    
  -models 3
  
    -classifier.pkl 3.1
    -train_classifier.py 3.2

## Contribution Guidelines

This repository is not open to contributions. It has been created for personal use and specific purposes, and external contributions are not being accepted. Feel free to fork this repository for your personal use or reference.

## Reporting Issues

While contributions are not accepted, you can report any issues or bugs you encounter by creating an issue. However, please understand that these issues may not be actively addressed or resolved.

## License

This repository is licensed under the [MIT License](https://opensource.org/licenses/MIT).

