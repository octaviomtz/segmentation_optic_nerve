import numpy as np 
import pandas as pd

def create_df(df):
    '''modify df'''
    df.columns = ['img', 'lbl'] 
    df['img'] = df['img'].apply(lambda x:x.replace('JPEGImages', 'images'))
    df['lbl'] = df['lbl'].apply(lambda x:x.replace('Annotations', 'labels'))
    return df