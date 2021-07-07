#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 10:57:40 2021

@author: gaellerojat
"""
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('/Users/gaellerojat/Desktop/ISEP/COURS A2/IA/PROJET/raw_data.csv', sep = ',')  
data = data.drop(['Unnamed: 0'], axis=1)

"""
--> To print the variables 
print(data.columns)


"""


del data['ID']



data.sex = data['sex'].astype(str)
data.sex.replace(to_replace ="male",value =0.0, inplace = True) 
data.sex.replace(to_replace ="female",value =1.0, inplace = True)
data.sex.replace("nan", data.sex.median(), inplace = True)

original_age = data.age.fillna('novalue').tolist()
age_lst = []
for age in original_age:
    if age == 'novalue':
        age_lst.append(age)  
    else:
        extract_age = re.findall(r'\d+(?:\.\d+)?', age)
        if len(extract_age)==0 or float(extract_age[0]) > 140.0:
            age_lst.append('novalue')
        elif len(extract_age)==1:
            age_lst.append(round((float(extract_age[0])),0))
        elif len(extract_age)==2:
            avg_age = round(((float(extract_age[0])+float(extract_age[1]))/2.0),0)
            age_lst.append(avg_age)

data['age_new'] = age_lst
data.age_new = data.age_new.replace('novalue',np.NaN)
data.age_new = data.age_new.fillna(round(data.age_new.mean(),0))
data = data.drop(["age"], axis=1)

data.travel_history_location.dropna(inplace = True)




original_symptoms = data.date_onset_symptoms.fillna('novalue').tolist()
symptoms_lst = []
for date in original_symptoms:
    if date == 'novalue':
        value = 2
        symptoms_lst.append(value)  
    elif "." in date:
        if not "-" in date:
            isDate = date.split(".")
            if int(isDate[0])<31 and int(isDate[1])<12:
                value = 1
                symptoms_lst.append(value)
            else:
                value = "novalue"
                symptoms_lst.append(value)
        else:
            value = "novalue"
            symptoms_lst.append(value)
    else:
        value = "novalue"
        symptoms_lst.append(value)

data['symptom_onset'] = symptoms_lst
data.symptom_onset = data.symptom_onset.replace('novalue',0)
data = data.drop(["date_onset_symptoms"], axis=1)



original_outcomes = data.outcome.fillna('novalue').tolist()
outcomes_lst = []
for status in original_outcomes:
    if status == 'novalue':
        outcomes_lst.append(status)
    elif "death" in status.lower() or "Death" in status.lower() or "dead" in status.lower() or "Dead" in status.lower() or "died" in status.lower() or 'Deceased' in status.lower():
        outcomes_lst.append(0)
    elif "critical" in status.lower() or "intubated" in status.lower() or "severe" in status.lower() or "intensive" in status.lower() or "Critical" in status.lower() or "unstable" in status.lower():
        outcomes_lst.append(1)
    elif "discharge" in status.lower() or "discharged" in status.lower() or "Discharged" in status.lower() or "not" in status.lower() or "recovered" in status.lower() or "recovering" in status.lower() or "Recovered" in status.lower() or "Alive" in status.lower() or "Stable" in status.lower() or "stable" in status.lower() :
        outcomes_lst.append(2)
    elif "Currently" in status.lower() or "Treatment" in status.lower() or "treatment" in status.lower():
        outcomes_lst.append(3)
    else:
        outcomes_lst.append('novalue')

data['outcomes'] = outcomes_lst
data.outcomes = data.outcomes.replace('novalue',np.NaN)
data.outcomes.dropna(inplace = True)
data = data.drop(["outcome"], axis=1) 
    
      

original_loc = data.travel_history_location.fillna('novalue').tolist()
loc_lst = []
for loc in original_loc:
    if loc == 'novalue':
        loc_lst.append(loc)  
    elif 'wuhan' in loc.lower():
        loc_lst.append(1)
    else:
        otherDestination = 2
        loc_lst.append(otherDestination)

data["travel_history_location_isWuhan"] = loc_lst
data.travel_history_location_isWuhan = data.travel_history_location_isWuhan.replace('novalue',0)
data = data.drop(['travel_history_location'], axis=1) 



data.fillna(data.mode().iloc[0], inplace=True)


lb_make = LabelEncoder()
for i in data.columns : 
    data[i] = lb_make.fit_transform(data[i])
    
    
    
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler

plt.figure(figsize=(12,10))
cor = data.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


cor_target = abs(cor["outcomes"])
relevant_features = cor_target[cor_target>0.03]
print(relevant_features)



x = data.copy()
del x['outcomes']
Y = data.outcomes

x = StandardScaler().fit_transform(x)

pca = decomposition.PCA(n_components=2)
principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, Y], axis = 1)
finalDf = finalDf.sample(n=2000)

targets = [0,1,2,3]
colors = ['b','g','y','r']

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

for target, color in zip(targets,colors):
    indicesToKeep = finalDf['outcomes'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

import matplotlib.pyplot as plt
import datetime

probability_symptoms = data.symptom_onset[data.travel_history_location_isWuhan == 1].value_counts(normalize = True)
probability_death = data.outcomes[data.travel_history_location_isWuhan == 1].value_counts(normalize = True)
print(probability_symptoms)
m_color = '#F8BA00'
probability_symptoms.plot(kind = 'bar', alpha = 0.5, color = m_color)
plt.title('Probability of having symptoms if the person went to Wuhan')
plt.show()

print(probability_death)
m_color = '#F8BA00'
probability_death.plot(kind = 'bar', alpha = 0.5, color = m_color)
plt.title('Probability of dying if the person went to Wuhan')
plt.show()



date_a = data.date_admission_hospital.fillna('novalue').tolist()
date_lst = []
for date in date_a:
    if "." in date:
        if not "-" in date:
            newDate = datetime.datetime.strptime(date,"%d.%m.%Y")
            date_lst.append(newDate)
        else :
            value = "novalue"
            date_lst.append(value)
    else :
        value = "novalue"
        date_lst.append(value)

data['date_admission'] = date_lst
data.date_admission = data.date_admission.replace('novalue',np.NaN)
data.date_admission.dropna(inplace = True)
data['date_admission'] = pd.to_datetime(data['date_admission'])



date_d = data.date_death_or_discharge.fillna('novalue').tolist()
dated_lst = []
for date in date_d:
    if "." in date:
        if not "-" in date:
            isDate = date.split(".")
            if int(isDate[1]) > 12:
                newDate = datetime.datetime.strptime(date,"%m.%d.%Y")
                dated_lst.append(newDate)
            else:
                newDate = datetime.datetime.strptime(date,"%d.%m.%Y")
                dated_lst.append(newDate)
        else :
            value = "novalue"
            dated_lst.append(value)
    else :
        value = "novalue"
        dated_lst.append(value)

data['date_discharge'] = dated_lst
data.date_discharge = data.date_discharge.replace('novalue',np.NaN)
data.date_discharge.dropna(inplace = True)
data['date_discharge'] = pd.to_datetime(data['date_discharge'])


data['average_recovery'] = data['date_discharge'] - data['date_admission'] 

average_recovery_interval = data.average_recovery[data.travel_history_location_isWuhan == 1].mean()
print('Average recovery interval : ')
print(average_recovery_interval)
