#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Import packages
import pandas as pd
import email

# Load the Data
df = pd.read_csv('emails.csv')
# try with first 20 rows of data
# df = df[:20]
df_copy = df.copy()

# Shape and preview of the dataframe
print(df.info())
print(df.head(3))

# Collect the email content
def get_email_content(message):
    contents = []
    for content in message.walk():
        if content.get_content_type() == "text/plain":
            contents.append(content.get_payload().lower())
    
    return ''.join(contents)


# Extract info from email content
msg = list(map(email.message_from_string, df_copy['message']))
keys = msg[0].keys()

for key in keys:
    df_copy[key] = [cont[key] for cont in msg]
# Parse content from emails
df_copy['content'] = list(map(get_email_content, msg))

# Extract user and file infomation
df_copy['user'] = df_copy['file'].map(lambda x:x.split('/')[0])

# Drop unnecessary variables
df_copy.drop(['message',
              'Mime-Version',
              'Content-Type',
              'Content-Transfer-Encoding'],
              axis = 1,inplace = True)

# save dataframe into csv file
df_copy.to_csv('dataprep.csv')

