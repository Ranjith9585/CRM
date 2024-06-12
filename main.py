# main.py
import os
import base64
import io
import math
from flask import Flask, render_template, Response, redirect, request, session, abort, url_for
import mysql.connector
import hashlib
import datetime
import calendar
import random
from random import randint
from urllib.request import urlopen
import webbrowser

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import urllib.request
import urllib.parse
import csv
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  charset="utf8",
  database="crime_hotspot"

)
app = Flask(__name__)
##session key
app.secret_key = 'abcdef'
#######
UPLOAD_FOLDER = 'static/upload'
ALLOWED_EXTENSIONS = { 'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#####
@app.route('/', methods=['GET', 'POST'])
def index():
    msg=""

    return render_template('index.html',msg=msg)


@app.route('/login_admin', methods=['GET', 'POST'])
def login_admin():
    msg=""

    
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM admin WHERE username = %s AND password = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('admin'))
        else:
            msg = 'Incorrect username/password! or access not provided'
    return render_template('login_admin.html',msg=msg)

@app.route('/login', methods=['GET', 'POST'])
def login():
    msg=""

    
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM register WHERE uname = %s AND pass = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('userhome'))
        else:
            msg = 'Incorrect username/password!'
    return render_template('login.html',msg=msg)


@app.route('/register', methods=['GET', 'POST'])
def register():
    #import student
    msg=""
    if request.method=='POST':
        name=request.form['name']
        mobile=request.form['mobile']
        email=request.form['email']
        uname=request.form['uname']
        pass1=request.form['pass']
        
        
        mycursor = mydb.cursor()
        mycursor.execute("SELECT count(*) FROM register where uname=%s",(uname,))
        cnt = mycursor.fetchone()[0]
        print("ff")
        if cnt==0:
            mycursor.execute("SELECT max(id)+1 FROM register")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1
                    
            sql = "INSERT INTO register(id,name,mobile,email,uname,pass) VALUES (%s, %s, %s, %s, %s, %s)"
            val = (maxid,name,mobile,email,uname,pass1)
            mycursor.execute(sql, val)
            mydb.commit()            
            #print(mycursor.rowcount, "Registered Success")
            msg="success"
            
        else:
            msg='Already Exist'
    return render_template('register.html',msg=msg)



@app.route('/userhome', methods=['GET', 'POST'])
def userhome():
    msg=""
    uname=""
    st=""
    if 'username' in session:
        uname = session['username']
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM register where uname=%s",(uname,))
    usr = mycursor.fetchone()

    
    if request.method=='POST':
        splace=request.form['splace']
        dplace=request.form['dplace']

        loc1='%'+splace+'%'
        loc2='%'+dplace+'%'

        mycursor.execute("SELECT count(*) FROM routes where location like %s",(loc1,))
        cn = mycursor.fetchone()[0]
        if cn>0:
            mycursor.execute("SELECT * FROM routes where location like %s",(loc1,))
            dd1 = mycursor.fetchall()
            v2=""
            st="1"
            for dd in dd1:
                rid=dd[2]
                sid=dd[3]

                mycursor.execute("SELECT count(*) FROM routes where location like %s && route_id=%s",(loc2,rid))
                cn2 = mycursor.fetchone()[0]

                if cn2>0:
                    
                    
                    mycursor.execute("SELECT * FROM routes where location like %s && route_id=%s",(loc2,rid))
                    dd2 = mycursor.fetchall()
                    for dd3 in dd2:
                        rid2=dd3[2]
                        sid2=dd3[3]
                        v=""
                        if sid<sid2:
                            v=str(sid)+","+str(sid2)
                        else:
                            v=str(sid2)+","+str(sid)

                        v2=str(rid2)+","+v+"|"

            ff=open("static/route.txt","w")
            ff.write(v2)
            ff.close()
            msg="Location"
                    
        else:
            st="2"
            msg="No Location Found!"
   
    return render_template('userhome.html',msg=msg,usr=usr,st=st)



@app.route('/view_route', methods=['GET', 'POST'])
def view_route():
    msg=""
    uname=""
    st=""
    data=[]
    if 'username' in session:
        uname = session['username']
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM register where uname=%s",(uname,))
    usr = mycursor.fetchone()

    ff=open("static/route.txt","r")
    vv=ff.read()
    ff.close()

    value=vv.split("|")
    
    n=len(value)
    if n>0:
        i=0
        n1=n-1
        #while i<n1:
        v1=value[0]
        v2=v1.split(",")
        mycursor.execute("SELECT * FROM routes where route_id=%s && sid between %s and %s",(v2[0],v2[1],v2[2]))
        dd2 = mycursor.fetchall()
        
        dd4=[]
        for dd3 in dd2:
            dt=[]
            print(dd3[1])
            dt.append(dd3[1])
            loc='%'+dd3[1]+'%'
            
            
            
            mycursor.execute("SELECT * FROM crime_data where district like %s limit 0,5",(loc,))
            dd4 = mycursor.fetchall()
            dt2=[]
            for dd5 in dd4:
                
                dt1=[]
                s=dd5[4]+dd5[5]+dd5[6]+dd5[7]+dd5[8]+dd5[9]
                sf=""
                per=0
                if s>500:
                    sf="Not Safety"
                    s1=randint(510,580)
                    s11=(s1/600)*100
                    per=100-s11
                    
                elif s>400:
                    sf="Low Safety"
                    s1=(s/600)*100
                    per=100-s1
                elif s>250:
                    sf="Medium Safety"
                    s1=(s/600)*100
                    per=100-s1
                else:
                    
                    sf="Safety"
                    s1=(s/600)*100
                    per=100-s1

                per1=round(per,2)
                dt1.append(dd5[2])
                dt1.append(sf)
                dt1.append(per1)
                    
                dt2.append(dt1)
                
            dt.append(dt2)
            data.append(dt)
        #i+=1
            
    
    #print(data)

    return render_template('view_route.html',msg=msg,usr=usr,st=st,data=data)


@app.route('/map', methods=['GET', 'POST'])
def map():
    msg=""
    uname=""
    st=""
    data=[]
    if 'username' in session:
        uname = session['username']
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM register where uname=%s",(uname,))
    usr = mycursor.fetchone()

    ff=open("static/route.txt","r")
    vv=ff.read()
    ff.close()

    value=vv.split("|")
    
    n=len(value)
    if n>0:
        i=0
        n1=n-1
        #while i<n1:
        v1=value[0]
        v2=v1.split(",")
        mycursor.execute("SELECT * FROM routes where route_id=%s && sid between %s and %s",(v2[0],v2[1],v2[2]))
        dd2 = mycursor.fetchall()
        
        dd4=[]
        for dd3 in dd2:
            dt=[]
            print(dd3[1])
            dt.append(dd3[1])
            loc='%'+dd3[1]+'%'
            
            
            
            mycursor.execute("SELECT * FROM crime_data where district like %s limit 0,5",(loc,))
            dd4 = mycursor.fetchall()
            dt2=[]
            for dd5 in dd4:
                
                dt1=[]
                s=dd5[4]+dd5[5]+dd5[6]+dd5[7]+dd5[8]+dd5[9]
                sf=""
                per=0
                if s>500:
                    sf="Not Safety"
                    s1=randint(510,580)
                    s11=(s1/600)*100
                    per=100-s11
                    
                elif s>400:
                    sf="Low Safety"
                    s1=(s/600)*100
                    per=100-s1
                elif s>250:
                    sf="Medium Safety"
                    s1=(s/600)*100
                    per=100-s1
                else:
                    
                    sf="Safety"
                    s1=(s/600)*100
                    per=100-s1

                per1=round(per,2)
                dt1.append(dd5[2])
                dt1.append(sf)
                dt1.append(per1)
                dt1.append(dd5[10])
                dt1.append(dd5[11])
                    
                dt2.append(dt1)
                
            dt.append(dt2)
            data.append(dt)
        #i+=1
            
    
    #print(data)

    return render_template('map.html',msg=msg,usr=usr,st=st,data=data)


@app.route('/admin', methods=['GET', 'POST'])
def admin():
    msg=""

    df=pd.read_csv('static/dataset/crimes.csv')
    df.head()
        
    return render_template('admin.html',msg=msg)

@app.route('/process1', methods=['GET', 'POST'])
def process1():
    msg=""
    data=[]
    df=pd.read_csv('static/dataset/crimes.csv')
    dat=df.head(200)

    for ss in dat.values:
        data.append(ss)
    
    return render_template('process1.html',data=data)

@app.route('/process2', methods=['GET', 'POST'])
def process2():
    msg=""
    mem=0
    cnt=0
    cols=0
    filename = 'static/dataset/crimes.csv'
    data1 = pd.read_csv(filename, header=0)
    data2 = list(data1.values.flatten())
    cname=[]
    data=[]
    dtype=[]
    dtt=[]
    nv=[]
    i=0
    
    sd=len(data1)
    rows=len(data1.values)
    
    #print(data1.columns)
    col=data1.columns
    #print(data1[0])
    for ss in data1.values:
        cnt=len(ss)
        

    i=0
    while i<cnt:
        j=0
        x=0
        for rr in data1.values:
            dt=type(rr[i])
            if rr[i]!="":
                x+=1
            
            j+=1
        dtt.append(dt)
        nv.append(str(x))
        
        i+=1

    arr1=np.array(col)
    arr2=np.array(nv)
    data3=np.vstack((arr1, arr2))


    arr3=np.array(data3)
    arr4=np.array(dtt)
    
    data=np.vstack((arr3, arr4))
   
    print(data)
    cols=cnt
    mem=float(rows)*0.75

    return render_template('process2.html',data=data, msg=msg, rows=rows, cols=cols, dtype=dtype, mem=mem)


@app.route('/process3', methods=['GET', 'POST'])
def process3():
    df=pd.read_csv('static/dataset/crimes.csv')
    df.head()
    df.drop('Unnamed: 0',axis=1,inplace=True)
    df['STATE/UT'].unique()
    df.loc[df['STATE/UT'] == 'A&N Islands', 'STATE/UT'] = 'A & N ISLANDS'
    df.loc[df['STATE/UT'] == 'D&N Haveli', 'STATE/UT'] = 'D & N HAVELI'
    df.loc[df['STATE/UT'] == 'Delhi UT', 'STATE/UT'] = 'DELHI'

    #converting all the state names to capitals
    df['STATE/UT'] = pd.Series(str.upper(i) for i in df['STATE/UT'])
    df['DISTRICT'] = pd.Series(str.upper(i) for i in df['DISTRICT'])
    #stroring the sum of all crimes comitted within a state statewise
    state_all_crimes = df.groupby('STATE/UT').sum()

    #droping the sum of year column
    state_all_crimes.drop('Year',axis=1,inplace=True)

    #adding a column containig the total crime against women in that state
    col_list= list(state_all_crimes)
    state_all_crimes['Total']=state_all_crimes[col_list].sum(axis=1)
    all_crimes = state_all_crimes

    #sorting the statewise crime from highest to lowest
    state_all_crimes.sort_values('Total',ascending=False)

    state_all_crimes=state_all_crimes.reset_index()
    total_df=state_all_crimes.sum(axis=0).reset_index()
    tf=pd.DataFrame(total_df)

    tf=tf.drop([0])
    tf=tf.drop([8])

    sorted_df = state_all_crimes.sort_values('Total',ascending=False)
    '''fig = px.bar( x=tf["index"],y=tf[0], color=tf[0], 
                 labels={'x': "Crimes", 'y': "Count"}, title="Total Cases", 
                 color_continuous_scale='burg')
    fig.show()'''

    #sates v/s total crimes
    '''sorted_df = state_all_crimes.sort_values('Total',ascending=False)
    fig = px.bar( x=sorted_df['STATE/UT'],y=sorted_df["Total"], color=sorted_df["Total"], 
                 labels={'x': "States", 'y': "Count"}, title="Total Cases", 
                 color_continuous_scale='burg')
    fig.show()'''

    '''fig = px.bar( x=state_all_crimes['STATE/UT'],y=state_all_crimes["Rape"], color=state_all_crimes["Rape"], 
             labels={'x': "States", 'y': "Count"}, title="Rape Cases", 
             color_continuous_scale='burg')
    fig.show()'''

    #states v/s  kidnapping and abduction

    '''fig = px.bar( x=state_all_crimes['STATE/UT'],y=state_all_crimes["Kidnapping and Abduction"], color=state_all_crimes["Kidnapping and Abduction"], 
                 labels={'x': "States", 'y': "Count"}, title="Kidnapping and Abduction Cases", 
                 color_continuous_scale='burg')
    fig.show()'''

    #states v/s Importation of Girls

    '''importation_df = state_all_crimes.copy()
    importation_df.loc[importation_df['Importation of Girls'] <= 50, 'STATE/UT'] = 'Others' # Represent only large countries
    fig = px.pie(importation_df, values='Importation of Girls', names='STATE/UT', title="Importation of Girls", 
                color_discrete_sequence=px.colors.sequential.Teal_r)
    fig.update_traces(textposition='inside', textinfo='label+value',
                    marker=dict(line=dict(color='#000000', width=2)))
    #fig.update_layout(annotations=[dict(text='count', x=0.5, y=0.5, font_size=20, showarrow=False)])
    fig.show()'''

    return render_template('process3.html')

@app.route('/process4', methods=['GET', 'POST'])
def process4():
    data=[]
    data2=[]
    df=pd.read_csv('static/dataset/crimes.csv')
    df.head()
    df.drop('Unnamed: 0',axis=1,inplace=True)
    df['STATE/UT'].unique()
    df.loc[df['STATE/UT'] == 'A&N Islands', 'STATE/UT'] = 'A & N ISLANDS'
    df.loc[df['STATE/UT'] == 'D&N Haveli', 'STATE/UT'] = 'D & N HAVELI'
    df.loc[df['STATE/UT'] == 'Delhi UT', 'STATE/UT'] = 'DELHI'

    #converting all the state names to capitals
    df['STATE/UT'] = pd.Series(str.upper(i) for i in df['STATE/UT'])
    df['DISTRICT'] = pd.Series(str.upper(i) for i in df['DISTRICT'])
    #stroring the sum of all crimes comitted within a state statewise
    state_all_crimes = df.groupby('STATE/UT').sum()

    #droping the sum of year column
    state_all_crimes.drop('Year',axis=1,inplace=True)

    #adding a column containig the total crime against women in that state
    col_list= list(state_all_crimes)
    state_all_crimes['Total']=state_all_crimes[col_list].sum(axis=1)
    all_crimes = state_all_crimes

    #sorting the statewise crime from highest to lowest
    state_all_crimes.sort_values('Total',ascending=False)

    state_all_crimes=state_all_crimes.reset_index()
    total_df=state_all_crimes.sum(axis=0).reset_index()
    tf=pd.DataFrame(total_df)

    tf=tf.drop([0])
    tf=tf.drop([8])

    sorted_df = state_all_crimes.sort_values('Total',ascending=False)
    ##
    dat=all_crimes = all_crimes.reset_index()
    for ss in dat.values:
        data.append(ss)

    all_crimes.shape
    #finding the mean number of crimes
    m=all_crimes['Total'].mean()
    print('mean=',m)

    #finding the quantiles 
    q = np.quantile(all_crimes['Total'],[0.25,0.75])
    print(q)
    l=q[0]
    u=q[1]

    #copying the state_all_crimes to a new dataframe to normalise values and predict
    df_kmeans = all_crimes.loc[:,all_crimes.columns!="STATE/UT"]

    #adding an additional column called output
    output=[]
    for i in df_kmeans['Total']:
        if i >= m:
            output.append(1)#redzone
        elif m > i:
            output.append(0)#safe

    all_crimes['output']=output
    df_kmeans_y=all_crimes['output']

    #feature scaling
    from sklearn.preprocessing import MinMaxScaler
    cols = df_kmeans.columns

    ms=MinMaxScaler()

    df_kmeans = ms.fit_transform(df_kmeans)
    df_kmeans = pd.DataFrame(df_kmeans,columns=[cols])
    df_kmeans.head()
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=0) 
    kmeans.fit(df_kmeans)
    kmeans.inertia_
    #checking the accuracy

    labels = kmeans.labels_

    # check how many of the samples were correctly labeled
    correct_labels = sum(df_kmeans_y == labels)

    print('labels:',labels)
    print('df_kmeans output:',df_kmeans_y)
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, df_kmeans_y.size))

    #based on the prediction of the k means algorithm classifying the states 
    #as safe or unsafe for women
    final=[]
    for i in range(len(labels)):
        state=all_crimes['STATE/UT'][i]
        label = labels[i]
        if label == 1:
            final.append([state,'unsafe'])
            
        else:
            final.append([state,'safe'])
            

    dat2=final_df = pd.DataFrame(final, columns=['STATES/UT', 'SAFE/UNSAFE'])
    
    #final_df
    ##
    for ss2 in dat2.values:
        data2.append(ss2)

    return render_template('process4.html',data=data,data2=data2)

#SVM Classification
class SVM:
    def fit(self, X, y):
        n_samples, n_features = X.shape# P = X^T X
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = np.dot(X[i], X[j])
                P = cvxopt.matrix(np.outer(y, y) * K)# q = -1 (1xN)
        q = cvxopt.matrix(np.ones(n_samples) * -1)# A = y^T 
        A = cvxopt.matrix(y, (1, n_samples))# b = 0 
        b = cvxopt.matrix(0.0)# -1 (NxN)
        G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))# 0 (1xN)
        h = cvxopt.matrix(np.zeros(n_samples))
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)# Lagrange multipliers
        a = np.ravel(solution['x'])# Lagrange have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]# Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
        self.b /= len(self.a)# Weights
        self.w = np.zeros(n_features)
        for n in range(len(self.a)):
            self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        
    def project(self, X):
        return np.dot(X, self.w) + self.b
    
    
    def predict(self, X):
        return np.sign(self.project(X))
    
@app.route('/process5', methods=['GET', 'POST'])
def process5():
    data=[]
    data2=[]
    df=pd.read_csv('static/dataset/crimes.csv')
    df.head()
    df.drop('Unnamed: 0',axis=1,inplace=True)
    df['STATE/UT'].unique()
    df.loc[df['STATE/UT'] == 'A&N Islands', 'STATE/UT'] = 'A & N ISLANDS'
    df.loc[df['STATE/UT'] == 'D&N Haveli', 'STATE/UT'] = 'D & N HAVELI'
    df.loc[df['STATE/UT'] == 'Delhi UT', 'STATE/UT'] = 'DELHI'

    #converting all the state names to capitals
    df['STATE/UT'] = pd.Series(str.upper(i) for i in df['STATE/UT'])
    df['DISTRICT'] = pd.Series(str.upper(i) for i in df['DISTRICT'])
    #stroring the sum of all crimes comitted within a state statewise
    state_all_crimes = df.groupby('STATE/UT').sum()

    #droping the sum of year column
    state_all_crimes.drop('Year',axis=1,inplace=True)

    #adding a column containig the total crime against women in that state
    col_list= list(state_all_crimes)
    state_all_crimes['Total']=state_all_crimes[col_list].sum(axis=1)
    all_crimes = state_all_crimes

    #sorting the statewise crime from highest to lowest
    state_all_crimes.sort_values('Total',ascending=False)

    state_all_crimes=state_all_crimes.reset_index()
    total_df=state_all_crimes.sum(axis=0).reset_index()
    tf=pd.DataFrame(total_df)

    tf=tf.drop([0])
    tf=tf.drop([8])

    sorted_df = state_all_crimes.sort_values('Total',ascending=False)
    ##
    dat=all_crimes = all_crimes.reset_index()
    for ss in dat.values:
        dt=[]
        dt.append(ss[0])
      
        f1=float(ss[8])
        if f1>200000:
            dt.append("Not Safety")
        elif f1>100000:
            dt.append("Medium Safety")

        elif f1>50000:
            dt.append("Low Safety")
        else:
            dt.append("Safety")        
     
        data.append(dt)

    all_crimes.shape
    #finding the mean number of crimes
    m=all_crimes['Total'].mean()
    print('mean=',m)

    #finding the quantiles 
    q = np.quantile(all_crimes['Total'],[0.25,0.75])
    print(q)
    l=q[0]
    u=q[1]

    #copying the state_all_crimes to a new dataframe to normalise values and predict
    df_kmeans = all_crimes.loc[:,all_crimes.columns!="STATE/UT"]

    #adding an additional column called output
    output=[]
    for i in df_kmeans['Total']:
        if i >= m:
            output.append(1)#redzone
        elif m > i:
            output.append(0)#safe

    all_crimes['output']=output
    df_kmeans_y=all_crimes['output']

    #feature scaling
    from sklearn.preprocessing import MinMaxScaler
    cols = df_kmeans.columns

    ms=MinMaxScaler()

    df_kmeans = ms.fit_transform(df_kmeans)
    df_kmeans = pd.DataFrame(df_kmeans,columns=[cols])
    df_kmeans.head()
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=0) 
    kmeans.fit(df_kmeans)
    kmeans.inertia_
    #checking the accuracy

    labels = kmeans.labels_

    # check how many of the samples were correctly labeled
    correct_labels = sum(df_kmeans_y == labels)

    print('labels:',labels)
    print('df_kmeans output:',df_kmeans_y)
    print("Result: %d out of %d samples were correctly labeled." % (correct_labels, df_kmeans_y.size))

    #based on the prediction of the k means algorithm classifying the states 
    #as safe or unsafe for women
    final=[]
    for i in range(len(labels)):
        state=all_crimes['STATE/UT'][i]
        label = labels[i]
        if label == 1:
            final.append([state,'unsafe'])
            
        else:
            final.append([state,'safe'])
            

    dat2=final_df = pd.DataFrame(final, columns=['STATES/UT', 'SAFE/UNSAFE'])
    
    #final_df
    ##
    for ss2 in dat2.values:
        data2.append(ss2)

    return render_template('process5.html',data=data,data2=data2)
  



##########################
@app.route('/logout')
def logout():
    # remove the username from the session if it is there
    session.pop('username', None)
    return redirect(url_for('index'))



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)


