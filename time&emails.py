#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Load the data
df_clean = pd.read_csv('inboxdata.csv')
# Date reformat
df_clean['Date'] = pd.to_datetime(df_clean['Date'], utc=True)

# Year
year = df_clean.groupby(df_clean['Date'].dt.year)['content'].count()
year_plot = plt.plot(year)
x_label = plt.xlabel('Year', fontsize = 15)
y_label = plt.ylabel('Number of Emails', fontsize = 15)
x_range = plt.xlim((1979, 2010))


# Month
month = df_clean[df_clean['Date'].dt.year==2001].groupby(df_clean['Date'].dt.month)['content'].count()
month_plot = plt.plot(month)
x_label_m = plt.xlabel('Month', fontsize = 15)
y_label_m = plt.ylabel('Number of Emails', fontsize = 15)


# Day of the week
ax = df_clean.groupby(df_clean['Date'].dt.dayofweek)['content'].count().plot()
ax.set_xlabel('Day of week', fontsize=15)
_ = ax.set_ylabel('Nubmer of emails *', fontsize=16) # use _ to avoid seeing extra text
_ = plt.xticks(np.arange(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=30 , fontsize=15 )
_ = plt.yticks(fontsize=15)








