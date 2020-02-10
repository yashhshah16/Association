#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
# from dashapp import app as application

import pandas as pd
import plotly.graph_objs as go

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import apyori as ap
from apyori import apriori 
import mlxtend as ml
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')




def SupervisedApriori(data,consequent,min_supp,min_conf,min_lift):
    frequent_itemsets = apriori(data, min_supp, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)
    rules = rules[rules['lift']>min_lift]
    return(rules)


def rules(records,min_support=0.6,min_confidence=0.6,min_lift=1):
    te = TransactionEncoder()
    te_ary = te.fit(records).transform(records)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    sup_rules = SupervisedApriori(df,consequent = ['Attrition','No_Attrition'],min_supp=min_support, min_conf=min_confidence, min_lift=min_lift)
    fin_rules=sup_rules.sort_values(by='support',ascending=False).head(10)
    fin_rules['antecedents']=["&".join(list(x)) for x in fin_rules['antecedents']]
    fin_rules['consequents']=["&".join(list(x)) for x in fin_rules['consequents']]
    return fin_rules


emp = pd.read_csv('employee_attrition.csv')
#Cleaning
emp = emp.loc[emp['YearsWithCurrManager']<=emp['YearsAtCompany'],:]

emp.drop('EmployeeCount',axis=1,inplace=True)

emp.drop('EmployeeNumber',axis=1,inplace=True)

emp.drop('StandardHours',axis=1,inplace=True)

emp.drop('Over18',axis=1,inplace=True)


discretized_emp = emp.loc[:,['Attrition', 'BusinessTravel','Department','EducationField','Gender','JobRole', 'MaritalStatus',
                            'OverTime']]

#variables to be cut(discretized)
cols = ['Age', 'DailyRate','DistanceFromHome', 'Education','EnvironmentSatisfaction', 'HourlyRate',
    'JobInvolvement', 'JobLevel', 'JobSatisfaction','MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
        'PercentSalaryHike','RelationshipSatisfaction', 'StockOptionLevel',
        'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance','YearsAtCompany', 'YearsInCurrentRole', 
        'YearsSinceLastPromotion','YearsWithCurrManager']

def cut(variable,n):
    if n==3:
        return pd.qcut(variable, 3,labels = ['low_'+str(variable.name),'med_'+str(variable.name),'high_'+str(variable.name)])
    elif n==2:
        return pd.qcut(variable, 2,labels = ['low_'+str(variable.name),'high_'+str(variable.name)])

for i in cols:
    try:
        discretized_emp[str(i)+'_grp'] = cut(emp.loc[:,i],3)
    except:
        discretized_emp[str(i)+'_grp'] = cut(emp.loc[:,i],2)

discretized_emp['PerformanceRating_grp'] = pd.cut(emp['PerformanceRating'],2,
                                                labels =['low_PerformanceRating','high_PerformanceRating'])
discretized_emp['Attrition'] = discretized_emp['Attrition'].apply(lambda x: 'No_Attrition' if x=='No' else 'Attrition')
discretized_emp['OverTime'] = discretized_emp['OverTime'].apply(lambda x: 'No_OverTime' if x=='No' else 'OverTime')

discretized_emp = discretized_emp.dropna(axis=0,how='any')

melted_data = pd.melt(discretized_emp)
frequency = melted_data.groupby(by=['value'])['value'].count().sort_values(ascending=True)
freq_itemset = pd.DataFrame({'item':frequency.index, 'frequency':frequency.values})

records = []
for i in range(0,len(discretized_emp)):
    records.append([str(discretized_emp.values[i,j]) 
    for j in range(0, len(discretized_emp.columns))])

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(children=[html.Label('Support: '),
            dcc.Input(
        id='min_support',
        type='number',
        step=0.01,
        value=0.6
    ),html.Label('Confidence: '),dcc.Input(
        id='min_confidence',
        type='number',
        step=0.01,
        value=0.6
    ),html.Label('Lift: '),
    dcc.Input(
        id='min_lift',
        type='number',
        step=0.01,
        value=1
    ),html.H1(children='Association Rules'),
    html.Div(id='rules'),
    html.H1(children='Scatter Plot for Support vs Confidence'),
    html.Div([dcc.Graph(id='indicator-graphic')])
])

@app.callback(
    Output('rules','children'),
    [Input('min_support', 'value'),Input('min_confidence','value'),Input('min_lift','value')]
    )
def update_table(x,y,z):
    fin_rules=rules(records,min_confidence=y,min_support=x,min_lift=z)
    st=[]
    for x,y,z in zip(range(len(fin_rules['antecedents'])),fin_rules['antecedents'],fin_rules['consequents']):
        st.append(html.P(str(x+1)+". LHS="+y+" RHS="+z))
    return st

@app.callback(
    Output('indicator-graphic','figure'),
    [Input('min_support', 'value'),Input('min_confidence','value'),Input('min_lift','value')]
    )
def update_graph(x,y,z):
    fin_rules=rules(records,min_confidence=y,min_support=x,min_lift=z)
    st=[]
    for x,y,z in zip(range(len(fin_rules['antecedents'])),fin_rules['antecedents'],fin_rules['consequents']):
        st.append(str(x+1)+". LHS="+y+" RHS="+z)
    gp={
        'data': [go.Scatter(
            x=fin_rules['support'],
            y=fin_rules['confidence'],
            text=st,
            mode='markers',
            marker={
                'size': 15,
                'opacity': 0.5,
                'color':fin_rules['lift'],
                'colorscale':'Viridis',
                'line': {'width': 0.5, 'color': 'white'},
                'showscale':True
            }
        )],
        'layout': {
            'xaxis':{
                'title': 'support',
                'type': 'linear'
            },
            'yaxis':{
                'title': 'confidence',
                'type': 'linear'
            },
            'margin':{'l': 40, 'b': 40, 't': 10, 'r': 0},
            'hovermode':'closest'
        }
    }
    return gp




if __name__ == '__main__':
    app.run_server(debug=True)


# In[ ]:




