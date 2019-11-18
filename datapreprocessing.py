#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 20:52:25 2019

@author: yilinwang
"""

import pandas as pd
import email
import re
import numpy as np

warnings.filterwarnings("ignore")


#load the dataset
df = pd.read_csv('/Users/.../emails.csv')


df_inbox = df[df['file'].str.contains('inbox')]
df_inbox.to_csv('inbox.csv')


# get email content
def get_text(msg):
    res = []
    for text in msg.walk():
        if text.get_content_type() == 'text/plain':
            res.append(text.get_payload())
    return ''.join(res)

def split_email_addr(line):
    if line:
        addrs = line.split(',')
        addrs = frozenset(map(lambda x:x.strip(), addrs))
    else:
        addrs = None
    
    return addrs

# Extract info from email content
msg = list(map(email.message_from_string, df_inbox['message']))
keys = msg[0].keys()

for key in keys:
    df_inbox[key] = [cont[key] for cont in msg]
# Parse content from emails
df_inbox['content'] = list(map(get_text, msg))
df_inbox['From'] = df_inbox['From'].map(split_email_addr)
df_inbox['To'] = df_inbox['To'].map(split_email_addr)

# Extract user and file infomation
df_inbox['user'] = df_inbox['file'].map(lambda x:x.split('/')[0])


# Drop unnecessary variables
df_inbox.drop(['message','file','Message-ID',
              'Mime-Version',
              'Content-Type',
              'Content-Transfer-Encoding'],
              axis = 1,inplace = True)

df_inbox['Date'] = pd.to_datetime(df_inbox['Date'], infer_datetime_format=True)

# save dataframe into csv file
df_inbox.to_csv('inbox.csv')






