import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load data from csv files

    Args:
    messages_filepath: path to messages csv file
    categories_filepath: path to categories csv file

    Returns:
    messages: messages dataframe
    '''
    return pd.read_csv(messages_filepath), pd.read_csv(categories_filepath)


def clean_data(df):
    '''
    Clean data
    
    Args:
    df: dataframe
    
    Returns:
    df: cleaned dataframe
    '''
    #Merges the messages and categories datasets using the common id
    df = pd.merge(df[0], df[1], on='id')
    #Create dataframe with individual category columns 
    categories = df['categories'].str.split(';', expand=True)
    #Select the first row of the categories dataframe
    row = categories.iloc[0]
    #Extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x[:-2]).tolist()
    #Rename the columns of `categories`
    categories.columns = category_colnames
    # drop rows where values that are not 0 or 1 and Convert category values to binary
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        if categories[column].nunique() > 2:
            categories[column] = categories[categories[column]<2][column]
    #Drop the original categories column from `df`
    df = df.drop('categories', axis=1)
    #Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    #Remove duplicates
    df = df.drop_duplicates()
    return df

    


def save_data(df, database_filename):
    '''
    Save data to database
    
    Args:
    df: dataframe
    database_filename: path to database
    
    Returns:
    None
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace') 
    


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()