#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd


# # 1st i'm going to do DATA CLEANING
#         

# In[5]:


df.info()


# In[4]:


df= pd.read_csv('spam.csv', encoding= 'ISO-8859-1')


# In[6]:


df.sample(5)


# In[7]:


df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)


# In[8]:


# renaming the cols
df.rename(columns={'v1':'target','v2':'text'},inplace=True)
df.sample(5)


# In[9]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[10]:


df['target'] = encoder.fit_transform(df['target'])


# In[11]:


df.head()


# In[12]:


#checking missing value 
df.isnull().sum()


# In[13]:



# check for duplicate values
df.duplicated().sum()


# In[14]:


# remove duplicates
df = df.drop_duplicates(keep='first')


# In[15]:


df.duplicated().sum()


# In[16]:


df


# # EDA

# In[17]:


df['target'].value_counts()


# In[18]:


import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(), labels=['ham','spam'],autopct="%0.2f")
plt.show()


# In[19]:


#here we se only 88% of emails are ham rest of remain are spam


# In[20]:


import nltk
# here we are dividing the segments in 3 part for deeper analysis 
# 1st; charater
#2nd ; words used
#3rd:  sentence


# In[21]:


get_ipython().system('pip install nltk')


# In[22]:


nltk.download('punkt')


# In[23]:


df['num_characters']=df['text'].apply(len)
df.head()


# In[24]:


# num of words
df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))
#we are breaking every sms in terms of words


# In[25]:


#we are breaking sms in terms of sentences of we can say tokenize the sentences
df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[26]:


df.head()


# In[27]:


df[['num_characters','num_words','num_sentences']].describe()


# In[28]:


df[df['target']==0][['num_characters','num_words','num_sentences']].describe()
#this is for ham msg


# In[29]:


# here we clearing the diff. btw ham and spanm msgs,spam msg classification
df[df['target']==1][['num_characters','num_words','num_sentences']].describe()


# In[30]:


import seaborn as sns
plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_characters'])#histogram for ham msgs
sns.histplot(df[df['target'] == 1]['num_characters'],color='red')# histogram for spam msgs


# In[31]:


# here we are cheking the relation btw charracter,words and sentences 
sns.pairplot(df,hue='target')


# In[32]:


df.corr()#corelation coeff


# In[33]:


sns.heatmap(df.corr(),annot=True)# heatmap


# # DATA PREPROCESSING
# ###  LOWER CASE
# ### TOKENIZATION
# ### REMOVING SPECIAL CHARACTER
# ### REMOVING STOP WORDS AND PUCTUATION
# ### STEMMING

# In[41]:


import string
from nltk.corpus import stopwords
def transform_text(text):
    text = text.lower()#here we making everything in lower case
    text = nltk.word_tokenize(text)#HERE we break or tokenize the sentence/words/phrase
    
    y = []
    for i in text:
        if i.isalnum():# here we removing the alpha numeric character/special characters
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)


# In[42]:


#from nltk.corpus import stopwords
#stopwords.words('english')
# import string
#string.punctuation 
#these are the code for the seeing the stopwords and punctuations
transform_text("i'm Making this project named as spam filter futher i'll make it 90% accurate.")


# In[43]:


df['text'][12]


# In[44]:


df['transform_text']=df['text'].apply(transform_text)#here we succesfully made the above transformations and sotre it into a func


# In[45]:


df.head()


# In[47]:


from wordcloud import WordCloud
wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')


# In[51]:



spam_wc = wc.generate(df[df['target'] == 1]['transform_text'].str.cat(sep=" "))
plt.figure(figsize=(15,6))#define the size of figure for the spam
plt.imshow(spam_wc)


# In[52]:


spam_wc = wc.generate(df[df['target'] == 0]['transform_text'].str.cat(sep=" "))
plt.figure(figsize=(15,6))#define the size of figure for the ham
plt.imshow(spam_wc)


# In[53]:


df[df['target']==1]#extract particular msg in spam mail,so we can say every msg is an item or list of strings


# In[55]:


spam_corpus = []#an empty list
for msg in df[df['target'] == 1]['transform_text'].tolist():
    for word in msg.split(): # split the words
        spam_corpus.append(word)   # here we print all the spam msgs


# In[56]:


len (spam_corpus)


# In[61]:


from collections import Counter
Counter(spam_corpus).most_common(30)#Counter will create a dictionary


# In[62]:


from collections import Counter
sns.barplot(pd.DataFrame(Counter(spam_corpus).most_common(30))[0],pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# In[64]:


ham_corpus = []
for msg in df[df['target'] == 0]['transform_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)


# In[65]:


from collections import Counter
sns.barplot(pd.DataFrame(Counter(ham_corpus).most_common(30))[0],pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# # MODEL BUILDING

# ### HERE WE USE NAVIE'S BAYES

# In[66]:


#here we going to vectorize the text
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)


# In[70]:


X = tfidf.fit_transform(df['transform_text']).toarray()
X.shape


# In[71]:


from sklearn.model_selection import train_test_split


# In[73]:


y = df['target'].values


# In[74]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# In[75]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[76]:


gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()


# In[77]:


gnb.fit(X_train,y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[ ]:


#here it showing the low precisition score is low
#though the data is imbalanced so precesion is matter


# In[78]:


mnb.fit(X_train,y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


# In[81]:


bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# In[82]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
# here we running different model/ algorithms for better accuracy or well rounded result


# In[85]:


svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)


# In[86]:


clfs = {
    'SVC' : svc,
    'KN' : knc, 
    'NB': mnb, 
    'DT': dtc, 
    'LR': lrc, 
    'RF': rfc, }


# In[95]:


def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    
    return accuracy,precision


# In[96]:


performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)


# In[87]:


train_classifier(svc,X_train,y_train,X_test,y_test)


# In[88]:


accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    
    current_accuracy,current_precision = train_classifier(clf, X_train,y_train,X_test,y_test)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)


# In[89]:


# model improve
# 1. Change the max_features parameter of TfIdf


# In[90]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000':accuracy_scores,'Precision_max_ft_3000':precision_scores}).sort_values('Precision_max_ft_3000',ascending=False)


# In[91]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_scaling':accuracy_scores,'Precision_scaling':precision_scores}).sort_values('Precision_scaling',ascending=False)


# In[97]:


new_df = performance_df.merge(temp_df,on='Algorithm')


# In[98]:


new_df_scaled = new_df.merge(temp_df,on='Algorithm')


# In[99]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_num_chars':accuracy_scores,'Precision_num_chars':precision_scores}).sort_values('Precision_num_chars',ascending=False)


# In[100]:


new_df_scaled.merge(temp_df,on='Algorithm')


# In[101]:


#we also can do scaling


# In[ ]:




