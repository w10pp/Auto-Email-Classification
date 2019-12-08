# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd

#load the data
email = pd.read_csv("emails.csv")

print(email.head())
print(email.message[1])

# subset, inbox emails
inbox = email[email["file"].str.contains("inbox")]

#create dataframe
columns = ["Message-ID", "Date", "From", "To", "Subject", "Text"]

df_message = pd.DataFrame([me.replace("\n\t","").split("\n")[0].lstrip("Message-ID:"), 
                           me.replace("\n\t","").split("\n")[1].lstrip("Date:"), 
                           me.replace("\n\t","").split("\n")[2].lstrip("From:"), me.split("\n")[3].lstrip("To:"), 
                           me.replace("\n\t","").split("\n")[4].lstrip("Subject:"),
                           "".join(me.replace("\n\t","").split("\n")[15:])] for me in inbox.message)


df_message.columns = columns
print(df_message.shape)

df_message.to_csv("inbox.csv", header=True)
