#!/usr/bin/env python
# coding: utf-8

# In[1]:


#This code is just to visually show the race breakdown in th US to give perspective
# Data from the US Census 


# In[2]:


#This code is to make the graph
import matplotlib.pyplot as plt
import numpy as np

data = [72.0,12.8,5.7,9.5] # Type your data here
label = ['Whites','Blacks','Asians','Other'] # Type your label here

data = np.array(data)
percent = data / sum(data) * 100
labels = [[] for i in range(len(label))]
for i in range(len(label)):
    labels[i] = '{} ({:.2f}%)'.format(label[i],percent[i])

fig, axs = plt.subplots(figsize=(5, 5)) # change figure size here
pie = axs.pie(data, shadow=True, startangle=90, colors=['darkblue','mediumblue','deepskyblue','cyan'])
axs.axis('equal')
plt.legend(pie[0],labels, bbox_to_anchor=(1.4,0.5), loc="center right", fontsize=12, 
           bbox_transform=plt.gcf().transFigure)
plt.title('Race Composition in the U.S.',fontweight="bold", fontsize=15, pad=20)
plt.show()


# In[3]:


#This code is to visually show the income breakdown of the races. Data from the U.S. Census


# In[4]:


#This code is to make the graph
import matplotlib.pyplot as plt


data = [76829,49180,42037,82533,]

label = ['White', 'Black', 'Hispanic', 'Asian']


fig, axs = plt.subplots(figsize=(10,6))       
axs.bar(label, data, color=('darkviolet'))      
axs.set_title("Median Family Income by Race (2009-2013)", fontsize=20, fontweight="bold") 

axs.set_xlabel("Race", fontsize=16)

axs.set_ylabel("Median Family Income", fontsize=16)

axs.tick_params(labelsize=16)  
plt.grid(axis='y')
plt.show() 


# In[5]:


# This code is to visualize the poverty level of each race. Data from KFF.


# In[6]:


import matplotlib.pyplot as plt
import numpy as np

data = [9.0, 21.2, 17.2, 9.7, 24.2, 14.9, 12.3] # Type your data here
label = ['White', 'Black', 'Hispanic', 'Asian', 'American Indian', 'Multiple Races', 'Other'] # Type your label here

data = np.array(data)
percent = data / sum(data) * 100
labels = [[] for i in range(len(label))]
for i in range(len(label)):
    labels[i] = '{} ({:.2f}%)'.format(label[i],percent[i])

fig, axs = plt.subplots(figsize=(5, 5)) # change figure size here
pie = axs.pie(data, shadow=True, startangle=90, colors=['purple','indigo','navy','midnightblue','crimson', 'deeppink', 'mediumvioletred'])
axs.axis('equal')
plt.legend(pie[0],labels, bbox_to_anchor=(1.4,0.5), loc="center right", fontsize=12, 
           bbox_transform=plt.gcf().transFigure)
plt.title('Race Breakdown for Poverty Level',fontweight="bold", fontsize=15, pad=20)
plt.show()


# In[7]:


#This code is for a simple linear regression test to compare median income and poverty in black individuals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[8]:


df = pd.read_csv('Data/NHANES.csv')
df = df[['Race3', 'Age', 'HHIncomeMid', 'Poverty']]
df = df.dropna()


# In[9]:


# This codes is for making conditions out of the groups at hand 
adult = df['Age'] >= 18

black = df['Race3'] == 'Black'
white = df['Race3'] == 'White'

blackadult_df = df[adult & black]
whiteadult_df = df[adult & white]



# In[10]:


get_ipython().system('pip install statsmodels --user')


# In[11]:


#This code is to get the regression analysis
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

x_vals = blackadult_df['HHIncomeMid'].values
y_vals = blackadult_df['Poverty']

reg_model = OLS(y_vals, sm.add_constant(x_vals)).fit()
display(reg_model.summary())


# In[12]:


#This code is to show the correlation coefficient and p-value of the test
from scipy import stats
corr = stats.pearsonr(blackadult_df['HHIncomeMid'], blackadult_df['Poverty'])
print('Correlation coefficient:', corr[0])
print('p-value:', corr[1])


# In[13]:


b0 = reg_model.params[0]
b1 = reg_model.params[1]
x_plot = np.linspace(np.min(blackadult_df['HHIncomeMid']), np.max(blackadult_df['HHIncomeMid']), 100)


# In[14]:


#This code is to make the graph
fig, axs = plt.subplots(figsize=(12,8))
axs.scatter(blackadult_df['HHIncomeMid'], blackadult_df['Poverty'], c='mediumvioletred',
            edgecolors='none', s=30)
axs.plot(x_plot, x_plot*b1 + b0, color='red')
plt.title("Graph 1: Simple Linear Regression Test- Black Adult Median Household Income vs. Poverty Level", fontsize=20)

axs.set_xlabel("Median Household Income (US dollars)", fontsize=18)
axs.set_ylabel("Poverty", fontsize=18)
axs.tick_params(labelsize=15)
plt.show()


# In[15]:


#This code is just to show that there is practically no difference from the previous linear
#regression test with black individuals and the following one with white individuals
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

x_vals = whiteadult_df['HHIncomeMid'].values
y_vals = whiteadult_df['Poverty']

reg_model1 = OLS(y_vals, sm.add_constant(x_vals)).fit()
display(reg_model1.summary())


# In[16]:


b0_1 = reg_model1.params[0]
b1_1 = reg_model1.params[1]
x_plot_1 = np.linspace(np.min(whiteadult_df['HHIncomeMid']),
                       np.max(whiteadult_df['HHIncomeMid']), 100) 


# In[17]:


#This code is to show the correlation coefficient and p-value of the test
from scipy import stats
corr = stats.pearsonr(whiteadult_df['HHIncomeMid'],whiteadult_df['Poverty'])
print('Correlation coefficient:', corr[0])
print('p-value:', corr[1])


# In[18]:


#This code is to make the graph
fig, axs = plt.subplots(figsize=(12,8))
axs.scatter(blackadult_df['HHIncomeMid'], blackadult_df['Poverty'], c='none',
            edgecolors='mediumvioletred', s=30, label='Black Adult')
axs.scatter(whiteadult_df['HHIncomeMid'], whiteadult_df['Poverty'], c='none',
            edgecolors='pink', s=30, label='White Adult')

axs.plot(x_plot, x_plot*b1 + b0, color='purple')
axs.plot(x_plot_1, x_plot_1*b1_1 + b0_1, color='red')

plt.title("Graph 2: Simple Linear Regression Test- Comparison of Median HouseHold Income and Poverty between White and Black Individuals", fontsize=20)
axs.set_xlabel("Median Household Income (US dollars)", fontsize=18)
axs.set_ylabel("Poverty", fontsize=18)
axs.tick_params(labelsize=15)

axs.legend(prop={'size': 15})

plt.show()


# In[19]:


#This code is for a grouped histogram to show the distribution of median household income among three different races. 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[20]:


df = pd.read_csv('Data/NHANES.csv')
df = df[['Race3', 'Age', 'Poverty']]
df = df.dropna()


# In[21]:


# This code is for making conditions of the groups at hand
adult = df['Age'] >= 18

black = df['Race3'] == 'Black'
white = df['Race3'] == 'White'
hispanic = df['Race3'] == 'Hispanic'
asian = df['Race3'] == 'Asian'


blackadult_df = df[adult & black]
whiteadult_df = df[adult & white]
hispanicadult_df = df[adult & hispanic]
asianadult_df = df[adult & asian]


# In[22]:


#This code is to make the graph
fig, axs = plt.subplots(figsize=(12,8))
axs.hist(blackadult_df['Poverty'], color="b", alpha=0.3, bins=np.linspace(0,6,30), label="Black Adult")
axs.hist(whiteadult_df['Poverty'], color="r", alpha=0.3, bins=np.linspace(0,6,30), label="White Adult")
axs.hist(hispanicadult_df['Poverty'], color="g", alpha=0.3, bins=np.linspace(0,6,30), label="Hispanic Adult")
axs.hist(asianadult_df['Poverty'], color="y", alpha=0.3, bins=np.linspace(0,6,30), label="Asian Adult")

plt.title("Graph 3: Grouped Histogram- Distribution of Poverty by Race", fontsize=20)
axs.set_xlabel("Poverty", fontsize=18)
axs.set_ylabel("Frequency", fontsize=18)
axs.tick_params(labelsize=15)
axs.legend(prop={'size': 15})

plt.show()


# In[23]:


#This code is for a chi square test to see if there is an assoication between education and race. 
import pandas as pd
df = pd.read_csv("Data/NHANES.csv")


# In[24]:


data = df[['Education', 'Race3']]
data = data.dropna()
data


# In[25]:


table = pd.crosstab(index=data["Education"], columns=data["Race3"]) 
table


# In[26]:


# This code is for the chi2 and p value
from scipy import stats

chi2, p, dof, expected = stats.chi2_contingency(table)

print("chi2:", chi2)
print("p:", p)
print("dof:", dof)
print("expected:", expected)


# In[27]:


table['Asian_per'] = table['Asian'] / sum(table['Asian'])
table['Black_per'] = table['Black'] / sum(table['Black'])
table['Hispanic_per'] = table['Hispanic'] / sum(table['Hispanic'])
table['Mexican_per'] = table['Mexican'] / sum(table['Mexican'])
table['Other_per'] = table['Other'] / sum(table['Other'])
table['White_per'] = table['White'] / sum(table['White'])


# In[28]:


table


# In[29]:


#This code is to make the graph
import matplotlib.pyplot as plt
import pandas as pd

table = table.reindex(['8th Grade','9 - 11th Grade', 'College Grad', 'High School', 'Some College'])
table = table[['Black_per', 'White_per', 'Asian_per', 'Mexican_per', 'Hispanic_per','Other_per']]

label = ['Black', 'White', 'Asian', 'Mexican', 'Hispanic', 'Other']
y_value1 = table.loc['8th Grade']
y_value2 = table.loc['9 - 11th Grade']
y_value3 = table.loc['College Grad']
y_value4 = table.loc['High School']
y_value5 = table.loc['Some College']



fig, axs = plt.subplots(figsize=(10,6)) # Change the figure size here    

p1 = axs.bar(label, y_value1, color = 'indigo') # You specify the color here     
p2 = axs.bar(label, y_value2, bottom=y_value1, color = 'midnightblue')
p3 = axs.bar(label, y_value3, bottom=(y_value1+y_value2), color = 'purple')
p4 = axs.bar(label, y_value4, bottom=(y_value1+y_value2+y_value3), color = 'mediumvioletred')
p5 = axs.bar(label, y_value5, bottom=(y_value1+y_value2+y_value3+y_value4), color = 'deeppink')

axs.set_title("Graph 4: Stacked Bar Chart- Education level broken down by race", fontsize=20, fontweight="bold")   
axs.set_xlabel("Race", fontsize=14)
axs.set_ylabel("Proportion", fontsize=14)
axs.tick_params(labelsize=16)  
axs.legend((p1[0],p2[0],p3[0],p4[0],p5[0]),(table.index), bbox_to_anchor=(1.05, 1))
# make sure to add p here, if you would like to add more bars
plt.show() 


# In[30]:


def CI_prop_diff(s1, s2, n1, n2):
    p1 = s1 / n1
    p2 = s2 / n2
    se2p1 = p1 * (1 - p1) / n1
    se2p2 = p2 * (1 - p2) / n2
    se2 = se2p1 + se2p2
    se = np.sqrt(se2)
    low = (p1 - p2) - 1.96 * se
    up = (p1 - p2) + 1.96 * se
    print("Proportion difference is:", p1 - p2)
    print("95% CI is: ({}, {})".format(low, up))


# In[31]:


#These numbers are used to see the proportion of black individuals who are college grads and white individuals
#who are college grads
CI_prop_diff(s1=870, s2=76, n1=2387, n2=396)


# In[32]:


#This is code for a chi square test to see if there is an association between poverty and race.
import pandas as pd
df = pd.read_csv("Data/NHANES.csv")


# In[33]:


data = df[['Poverty', 'Race3']]
data = data.dropna()
data


# In[34]:


#This code is used to categorize the Poverty column
def function(row):
    if row['Poverty'] >= 5.0:
        return 'Rich'
    elif row['Poverty'] >= 4.0:
        return 'Above Average'
    elif row['Poverty'] >= 3.0:
        return 'Average'
    elif row['Poverty'] >= 2.0:
        return 'Below Average'
    elif row['Poverty'] <= 1.0:
        return 'Poor'
    else:
        return 'Severely Poor'


# In[35]:


data['Poverty'] = data.apply(function, axis=1)


# In[36]:


table = pd.crosstab(index=data["Poverty"], columns=data["Race3"]) 
table


# In[37]:


#This code is to get the chi2 and p value
from scipy import stats

chi2, p, dof, expected = stats.chi2_contingency(table)

print("chi2:", chi2)
print("p:", p)
print("dof:", dof)
print("expected:", expected)


# In[38]:


table['Asian_per'] = table['Asian'] / sum(table['Asian'])
table['Black_per'] = table['Black'] / sum(table['Black'])
table['Hispanic_per'] = table['Hispanic'] / sum(table['Hispanic'])
table['Mexican_per'] = table['Mexican'] / sum(table['Mexican'])
table['Other_per'] = table['Other'] / sum(table['Other'])
table['White_per'] = table['White'] / sum(table['White'])


# In[39]:


table


# In[40]:


#This code is to make the graph
import matplotlib.pyplot as plt
import pandas as pd

table = table.reindex(['Above Average', 'Average', 'Below Average', 'Poor', 'Severely Poor'])
table = table[['Asian_per','Black_per','Hispanic_per','Mexican_per','Other_per','White_per']]

label = ['Asian', 'Black', 'Hispanic', 'Mexican', 'Other', 'White']
y_value1 = table.loc['Above Average']
y_value2 = table.loc['Average']
y_value3 = table.loc['Below Average']
y_value4 = table.loc['Poor']
y_value5 = table.loc['Severely Poor']

fig, axs = plt.subplots(figsize=(10,6)) # Change the figure size here    

p1 = axs.bar(label, y_value1) # You specify the color here     
p2 = axs.bar(label, y_value2, bottom=y_value1, color = 'indigo')
p3 = axs.bar(label, y_value3, bottom=(y_value1+y_value2), color = 'midnightblue' )
p4 = axs.bar(label, y_value4, bottom=(y_value1+y_value2+y_value3), color = 'purple' )
p5 = axs.bar(label, y_value5, bottom=(y_value1+y_value2+y_value3+y_value4), color = 'mediumvioletred')

axs.set_title("Graph 5: Stacked Bar Chart- Poverty level broken down by race", fontsize=20, fontweight="bold")   
axs.set_xlabel("Race", fontsize=14)
axs.set_ylabel("Proportion", fontsize=14)
axs.tick_params(labelsize=16)  
axs.legend((p1[0],p2[0],p3[0],p4[0],p5[0]),(table.index), bbox_to_anchor=(1.05, 1))
# make sure to add p here, if you would like to add more bars
plt.show() 


# In[41]:


def CI_prop_diff(s1, s2, n1, n2):
    p1 = s1 / n1
    p2 = s2 / n2
    se2p1 = p1 * (1 - p1) / n1
    se2p2 = p2 * (1 - p2) / n2
    se2 = se2p1 + se2p2
    se = np.sqrt(se2)
    low = (p1 - p2) - 1.96 * se
    up = (p1 - p2) + 1.96 * se
    print("Proportion difference is:", p1 - p2)
    print("95% CI is: ({}, {})".format(low, up))


# In[42]:


#This code is to get the proportion difference and 95% CI
CI_prop_diff(s1=140, s2=582, n1=396, n2=2992)


# In[43]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Data/NHANES.csv')
df


# In[44]:


data = df[['Education', 'Race3']]
data = data.dropna()
data


# In[45]:


#This code is to quantify the education column.
def function(row):
    if row['Education'] == '8th Grade':
        return 1
    elif row['Education'] == '9 - 11th Grade':
        return 2
    elif row['Education'] == 'High School':
        return 3
    elif row['Education'] == 'Some College':
        return 4
    elif row['Education'] == 'College Grad':
        return 5
    else:
        return 6


# In[46]:


data['Education'] = data.apply(function, axis=1)


# In[47]:


data1 = data[data['Race3']=='Black']
data2 = data[data['Race3']=='White']


# In[48]:


data1


# In[49]:


data2


# In[50]:


def CI_mean_diff(list1, list2):
    s1 = np.var(list1)
    s2 = np.var(list2)
    n1 = len(list1)
    n2 = len(list2)
    se2 = s1/n1 + s2/n2
    se = np.sqrt(se2)

    diff = np.mean(list1) - np.mean(list2)
    low = diff - 1.96 * se
    up = diff + 1.96 * se
    print("The average difference is:", diff)
    print("The 95% CI is: ({}, {})".format(low, up))


# In[51]:


#The following four codes are used to get the individual means and variances of the two groups
np.mean(data1)


# In[52]:


np.mean(data2)


# In[53]:


np.var(data1)


# In[54]:


np.var(data2)


# In[55]:


CI_mean_diff(data2['Education'], data1['Education'])


# In[56]:


# This code is to get the test statistic and p value
from scipy import stats

t_val, p_val = stats.ttest_ind(data2['Education'], data1['Education'], equal_var=False)

print("Test statistic:", t_val)
print("p-value:", p_val)


# In[57]:


#This code is to make the graph
fig, axs = plt.subplots(figsize=(12,8))
axs.boxplot([data1['Education'],data2['Education']])
plt.title('The Education level of Black and White Individuals', fontsize=20)
axs.set_xticklabels(['Black','White'])
axs.set_ylabel('Education', fontsize=15)
axs.tick_params(labelsize=15)
plt.show()


# In[ ]:




