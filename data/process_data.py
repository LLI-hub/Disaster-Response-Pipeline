
'''
Import all the packages needed 
'''
import sys
import pandas as pd
import re
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load disaster_messages.csv into a dataframe called messages
    Load disaster_categories.csv into a dataframe called categories
    
    create df by merging messages and categories
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath , encoding='latin-1')
    # load categories dataset
    categories = pd.read_csv(categories_filepath, encoding='latin-1')
    
    # merge datasets
    df = pd.merge(messages,categories,
                  how='outer', on='id')
    return df


def clean_data(df):
    '''
    Split categories into separate category columns.
    - Split the values in the categories column on the ; character so that each value becomes a separate column.
    - Use the first row of categories dataframe to create column names for the categories data.
    - Rename columns of categories with new column names
    '''
    # create a dataframe of the individual category columns
    categories = df.categories.str.split(";", expand=True)
    
    # select the first row of the categories dataframe
    row = categories.loc[df.index == 0].values.flatten().tolist()
    
    # extract a list of new column names for categories.
    for i, val in enumerate(row):
        row[i] = row[i].split("-")[0]
    category_colnames = row
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    '''
    Convert category values to just numbers 0 or 1.
    Iterate through the category columns in df to keep only the last character of each string (the 1 or 0).
    '''
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.strip().str[-1]
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
        categories[column]=categories[column].map(lambda x: 1 if x >= 2 else x)
    
    '''
    Replace categories column in df with new category columns.
    - Drop the categories column from the df dataframe since it is no longer needed.
    - Concatenate df and categories data frames.
    '''
    # drop the original categories column from `df`
    df = df.drop('categories', 1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)
    
    # drop duplicates
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    '''
    Save the clean dataset into an sqlite database
    - With pandas to_sql method combined with the SQLAlchemy library. 
    - We have imported SQLAlchemy's in the first part of this notebook to use it here.
    - create_engine
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
        
    df.to_sql('df_clean', engine, index=False, if_exists='replace')


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