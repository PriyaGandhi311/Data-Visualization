import numpy as  np
import pandas as pd 
import json
import matplotlib.pyplot as plt
from pandas.io.json import json_normalize
import seaborn as sns

with open('data.json') as data_file:
	data= json.load(data_file)

reg = pd.json_normalize(data['data'])
#print (reg)

reg.head()
#print(reg)
amt_mean=reg.amount.sum()
# print(amt_mean)     #814070.0
category_count=reg.category.value_counts()
# print(category_count)   # Sports         10099
                        # Technology     10029
		        # Environment     9980
			# Games           9955
			# Fashion         9937
srt=reg.sort_values(by='amount',ascending=False)
#print(srt)
evnt_count=reg.event_name.value_counts()
#print(evnt_count)
reg.set_index('category')
#print(reg)
extract=reg.loc[reg['category'].isin(['Sports','Environment']) ] #all the rows with sports and environment in category column gets stored in extract variable 
#print(extract)						
ac_data=pd.DataFrame(extract) # converts extract into dataframe
ac_data.head()
ac_data.set_index("category")
ac_data=ac_data.sort_values(by='amount',ascending=False)# ac_data stores the actual data which is required for our prediction
#print(ac_data)
fig = plt.figure(figsize = (5,5))
#fig,ax =plt.subplots()
plt.xlabel("Category") 
plt.ylabel("amount")
plt.title('category more interested')
plt.bar(ac_data["category"],ac_data["amount"],color='maroon',width=0.4)#bar of sports is higher than bar of environment
fig = plt.figure(figsize = (15,5))
y_xis=ac_data.amount.value_counts()
sns.scatterplot(data=ac_data,x="location.zip_code",y=y_xis,hue='category')
ac_data.plt.line(x="location.longitude",title="longitude_relation_amount")
##sns.lineplot(data=ac_data, y="location.longitude", x="amount")
state_count=ac_data.amount.value_counts()
sns.barplot(data=ac_data, x=state_count,y="gender")#amount given by u gender is highest
plt.show()
#from the given data bicycle fund was done more from u gender and in sports category
#removed the category fashion,technology since the project needs only sports and environment category
#plotted graph between amount and other various column to conclude which combination gives maximum value
