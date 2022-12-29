from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AffinityPropagation
import matplotlib.pyplot as plt
import matplotlib
from sklearn import svm
from sklearn.metrics import f1_score
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import pyswarms as ps
from SwarmPackagePy import testFunctions as tf
from sklearn import linear_model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout

main = tkinter.Tk()
main.title("Detecting Mental Disorders in Social Media Through Emotional Patterns - The case of Anorexia and Depression")
main.geometry("1200x1200")

global dataset, emotion_lex
global texts,label,subject
global mask, users, BOS, user_label
global tfidf_vectorizer, X, Y, classifier

pso_classifier = linear_model.LogisticRegression(max_iter=1000)

#module to get subemotion of given words
def getEmotion(query):
    sub_emotion = None
    for i in range(len(emotion_lex)):
        word = emotion_lex[i,0]
        emotion = emotion_lex[i,1]
        association = emotion_lex[i,2]
        if word == query and association == 1:
            sub_emotion = emotion
            break
    return sub_emotion #return sub emotion of given word

    
def uploadDataset():
    global dataset
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.insert(END,str(filename)+" Dataset Loaded\n\n")
    pathlabel.config(text=str(filename)+" Dataset Loaded\n\n")
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)
    text.insert(END,str(dataset.head()))

def preprocessDataset():
    global dataset, emotion_lex
    global texts,label,subject
    texts = []
    label = []
    subject = []
    text.delete('1.0', END)
    emotion_lex = pd.read_csv("Dataset/NRC-emotion-lexicon.txt",names=["word", "emotion", "association"], sep='\t', keep_default_na=False)
    text.insert(END,str(emotion_lex.head())+"\n\n")
    emotion_lex = emotion_lex.values
    text.insert(END,"Lex emotion loaded & its size: "+str(len(emotion_lex))+"\n\n")

    for i in range(len(dataset)):
        textData = dataset.get_value(i, 'Text')
        lbl = dataset.get_value(i, 'Labels')
        subject_id = dataset.get_value(i, 'ID')
        textData = textData.strip().lower()
        textData = textData[0:300]
        texts.append(textData)
        label.append(lbl)
        subject.append(subject_id)
    text.insert(END,"Total users post found in dataset: "+str(len(texts)))

def maskFeatures():
    text.delete('1.0', END)
    global mask, users, BOS, user_label
    global texts,label,subject
    mask = []
    users = []
    user_label = []
    BOS = []
    if os.path.exists('model/bos.txt'):
        with open('model/bos.txt', 'rb') as file:
            BOS = pickle.load(file)
        file.close()
        mask = BOS[0]
        users = BOS[1]
        user_label = BOS[2]
    else:
        for i in range(len(texts)):
            textData = texts[i].strip().lower()
            tokens = textData.split(" ")
            emotion = ''
            for token in tokens:
                sub_emotion = getEmotion(token.strip())
                if sub_emotion is not None:
                    emotion += sub_emotion+" "
            if len(emotion.strip()) > 0:
                mask.append(emotion)
                users.append(subject[i])
                user_label.append(label[i])
        BOS = [mask,users,user_label]
        with open('model/bos.txt', 'wb') as file:
            pickle.dump(BOS, file)
        file.close()  
    temp = np.asarray(mask)
    text.insert(END,"BOS features\n\n")
    text.insert(END,str(temp))

def tfidf():
    global tfidf_vectorizer, X, Y, mask
    text.delete('1.0', END)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf = tfidf_vectorizer.fit_transform(mask).toarray()        
    df = pd.DataFrame(tfidf, columns=tfidf_vectorizer.get_feature_names())
    text.insert(END,"TF-IDF Weight values\n\n")
    text.insert(END,str(df.head())+"\n\n")
    df = df.values
    X = df[:, 0:df.shape[1]]
    Y = np.asarray(user_label)
    clustering = AffinityPropagation().fit(X) #peffroming affinity alustering
    centroid = clustering.cluster_centers_
    predict = clustering.predict(X) #predicting control and depress user
    sc = plt.scatter(X[0:70,0], X[0:70,1], c = predict[0:70], cmap = matplotlib.colors.ListedColormap(['blue','red']))
    plt.title("Depression & Control Affinity Clustering Graph")
    sc = plt.scatter(X[70:149,0], X[70:149,1], c = predict[70:149], cmap = matplotlib.colors.ListedColormap(['blue','red']),label=['Control','Anorexia'])
    plt.title("Anorexia & ControlAffinity Clustering Graph")
    plt.show()

def f_per_particle(m, alpha):
    global X
    global Y
    global pso_classifier
    total_features = X.shape[1]
    if np.count_nonzero(m) == 0:
        X_subset = X
    else:
        X_subset = X[:,m==1]
    pso_classifier.fit(X_subset, Y)
    P = (pso_classifier.predict(X_subset) == Y).mean()
    j = (alpha * (1.0 - P) + (1.0 - alpha) * (1 - (X_subset.shape[1] / total_features)))
    return j

def f(x, alpha=0.88):
    n_particles = x.shape[0]
    j = [f_per_particle(x[i], alpha) for i in range(n_particles)]
    return np.array(j)

def extensionSVMCNN():
    global tfidf_vectorizer, X, Y
    options = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 5, 'p':2}
    dimensions = X.shape[1] # dimensions should be the number of features
    optimizer = ps.discrete.BinaryPSO(n_particles=10, dimensions=dimensions, options=options) #CREATING PSO OBJECTS 
    cost, pos = optimizer.optimize(f, iters=10)#OPTIMIZING FEATURES
    text.insert(END,"\nFeatures found in dataset before applying PSO : "+str(X.shape[1])+"\n\n")
    X_selected_features = X[:,pos==1]  # PSO WILL SELECT IMPORTANT FEATURES WHERE VALUE IS 1
    text.insert(END,"Features found in dataset after applying PSO : "+str(X_selected_features.shape[1])+"\n\n")

    X_train, X_test, y_train, y_test = train_test_split(X_selected_features, Y, test_size=0.2)
    svm_cls = svm.SVC() #training with SVM
    svm_cls.fit(X_selected_features, Y)
    svm_predict = svm_cls.predict(X_test) 
    F1 = f1_score(y_test,svm_predict,average='macro') * 100
    text.insert(END,"\nExtension PSO Optimized SVM F1-Score: "+str(F1)+"\n\n")

    Y_train = to_categorical(y_train)
    Y1 = to_categorical(Y)
    XX = X_selected_features.reshape(X_selected_features.shape[0],X_selected_features.shape[1],1,1)
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1,1)
    X_test = X_test.reshape(X_test.shape[0],X_train.shape[1],1,1)
    classifier = Sequential()
    classifier.add(InputLayer(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])))
    classifier.add(Conv2D(25, (5, 5), activation='relu', strides=(1, 1), padding='same'))
    classifier.add(MaxPool2D(pool_size=(1, 1), padding='same'))
    classifier.add(Conv2D(50, (5, 5), activation='relu', strides=(2, 2), padding='same'))
    classifier.add(MaxPool2D(pool_size=(1, 1), padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(Conv2D(70, (3, 3), activation='relu', strides=(2, 2), padding='same'))
    classifier.add(MaxPool2D(pool_size=(1, 1), padding='valid'))
    classifier.add(BatchNormalization())
    classifier.add(Flatten())
    classifier.add(Dense(units=100, activation='relu'))
    classifier.add(Dense(units=100, activation='relu'))
    classifier.add(Dropout(0.25))
    classifier.add(Dense(units=Y1.shape[1], activation='softmax'))
    classifier.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    classifier.fit(XX, Y1,  epochs=10, shuffle=True, verbose=2)
    predict = classifier.predict(X_test) #perform prediction
    predict = np.argmax(predict, axis=1)
    testLabel = y_test
    fscore = f1_score(testLabel,predict,average='macro') * 100      
    text.insert(END,"Extension PSO Optimized CNN F1-Score: "+str(fscore)+"\n\n")
    

def runSVMCNN():
    text.delete('1.0', END)
    global tfidf_vectorizer, X, Y, classifier
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    svm_cls = svm.SVC() #training with SVM
    svm_cls.fit(X, Y)
    svm_predict = svm_cls.predict(X_test) 
    F1 = f1_score(y_test,svm_predict,average='macro') * 100
    text.insert(END,"\nSVM F1-Score: "+str(F1)+"\n\n")

    Y_train = to_categorical(y_train)
    Y1 = to_categorical(Y)
    XX = X.reshape(X.shape[0],X.shape[1],1,1)
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1,1)
    X_test = X_test.reshape(X_test.shape[0],X_train.shape[1],1,1)
    classifier = Sequential() #Training with CNN
    classifier.add(Convolution2D(32, 1, 1, input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (1, 1)))
    classifier.add(Convolution2D(32, 1, 1, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (1, 1)))
    classifier.add(Flatten())
    classifier.add(Dense(output_dim = 256, activation = 'relu'))
    classifier.add(Dense(output_dim = Y_train.shape[1], activation = 'softmax'))                             
    print(classifier.summary())
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    classifier.fit(XX, Y1, batch_size=8, epochs=10, shuffle=True, verbose=2)
    predict = classifier.predict(X_test) #perform prediction
    predict = np.argmax(predict, axis=1)
    testLabel = y_test
    fscore = f1_score(testLabel,predict,average='macro') * 100      
    text.insert(END,"CNN F1-Score: "+str(fscore)+"\n\n")
    
def getCount(mask_list, emotion):
    #'negative , anticipation, anger disgust trust 
    arr = mask_list.split(" ")
    count = 0
    for i in range(len(arr)):
        if arr[i] == emotion:
            count += 1
    return count   

def graph(title,control,emotion):
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Chunk Time')
    plt.ylabel('Average Emotion Value')
    plt.plot(control, 'ro-', color = 'blue')
    plt.plot(emotion, 'ro-', color = 'red')
    plt.legend(['Control', 'Depression'], loc='upper left')
    plt.title(title)
    plt.show()     
    
def emotionalSignal():
    text.delete('1.0', END)
    chunk_mask = []
    chunk_users = []
    chunk_user_label = []
    dynamic_BOS = []
    if os.path.exists('model/dynamic_bos.txt'):
        with open('model/dynamic_bos.txt', 'rb') as file:
            dynamic_BOS = pickle.load(file)
        file.close()
        chunk_mask = dynamic_BOS[0]
        chunk_users = dynamic_BOS[1]
        chunk_user_label = dynamic_BOS[2]
    else:
        for i in range(len(chunk_text)):
            textData = chunk_text[i].strip().lower()
            tokens = textData.split(" ")
            emotion = ''
            for token in tokens:
                sub_emotion = getEmotion(token.strip())
                if sub_emotion is not None:
                    emotion += sub_emotion+" "
            if len(emotion.strip()) > 0:
                chunk_mask.append(emotion)
                chunk_users.append(chunk_subject[i])
                chunk_user_label.append(chunk_label[i])
        dynamic_BOS = [chunk_mask,chunk_users,chunk_user_label]
        with open('model/dynamic_bos.txt', 'wb') as file:
            pickle.dump(dynamic_BOS, file)
        file.close()

    chunk_users = np.asarray(chunk_users)
    group,count = np.unique(chunk_users,return_counts=True)
    groups = len(chunk_users)-40
    group_index = 100 
    start = 0
    max_group = 0
    control_anger_group = []
    control_negative_group = []
    control_trust_group = []
    control_anti_group = []
    control_disgust_group = []

    depression_anger_group = []
    depression_negative_group = []
    depression_trust_group = []
    depression_anti_group = []
    depression_disgust_group = []

    ano_anger_group = []
    ano_negative_group = []
    ano_trust_group = []
    ano_anti_group = []
    ano_disgust_group = []

    while group_index < groups:
        control_anger  = 0
        control_trust = 0
        control_disgust = 0
        control_anticipation = 0
        control_negative = 0
        depression_anger  = 0
        depression_trust = 0
        depression_disgust = 0
        depression_anticipation = 0
        depression_negative = 0
        ano_anger  = 0
        ano_trust = 0
        ano_disgust = 0
        ano_anticipation = 0
        ano_negative = 0
        for i in range(start,group_index):
            for j in range(len(chunk_mask)):
                if max_group < 5:
                    if chunk_users[j] == group[i] and chunk_user_label[j] == 0 or chunk_user_label[j] == 1:
                        control_anger  = control_anger + getCount(chunk_mask[j],"anger")/100
                        control_trust = control_trust + getCount(chunk_mask[j],"trust")/100
                        control_disgust = control_disgust + getCount(chunk_mask[j],"disgust")/100
                        control_anticipation = control_anticipation + getCount(chunk_mask[j],"anticipation")/100
                        control_negative = control_negative + getCount(chunk_mask[j],"negative")/100
                    if chunk_users[j] == group[i] and chunk_user_label[j] == 0:
                        ano_anger  = ano_anger + getCount(chunk_mask[j],"anger")
                        ano_trust = ano_trust + getCount(chunk_mask[j],"trust")
                        ano_disgust = ano_disgust + getCount(chunk_mask[j],"disgust")
                        ano_anticipation = ano_anticipation + getCount(chunk_mask[j],"anticipation")
                        ano_negative = ano_negative + getCount(chunk_mask[j],"negative")    
                    if chunk_users[j] == group[i] and chunk_user_label[j] == 1:
                        depression_anger  = depression_anger + getCount(chunk_mask[j],"anger")
                        depression_trust = depression_trust + getCount(chunk_mask[j],"trust")
                        depression_disgust = depression_disgust + getCount(chunk_mask[j],"disgust")
                        depression_anticipation = depression_anticipation + getCount(chunk_mask[j],"anticipation")
                        depression_negative = depression_negative + getCount(chunk_mask[j],"negative")    
        if max_group < 5:
            text.insert(END,"Control: "+str(control_anger)+" "+str(control_trust)+" "+str(control_disgust)+" "+str(control_anticipation)+" "+str(control_negative)+"\n")
            text.insert(END,"Depression: "+str(depression_anger)+" "+str(depression_trust)+" "+str(depression_disgust)+" "+str(depression_anticipation)+" "+str(depression_negative)+"\n\n")
            control_anger_group.append(control_anger/100)
            control_negative_group.append(control_negative/100)
            control_trust_group.append(control_trust/100)
            control_anti_group.append(control_anticipation/100)
            control_disgust_group.append(control_disgust/100)
            depression_anger_group.append(depression_anger/100)
            depression_negative_group.append(depression_negative)
            depression_trust_group.append(depression_trust/100)
            depression_anti_group.append(depression_anticipation/100)
            depression_disgust_group.append(depression_disgust/100)
            ano_anger_group.append(ano_anger/100)
            ano_negative_group.append(ano_negative/100)
            ano_trust_group.append(ano_trust/100)
            ano_anti_group.append(ano_anticipation/100)
            ano_disgust_group.append(ano_disgust/100)
        start = group_index - 28
        group_index = group_index + 10
        max_group += 1
    graph("Depression Anger",control_anger_group,depression_anger_group)
    graph("Depression Negative",control_negative_group,depression_negative_group)
    graph("Depression Anticipation",control_anti_group,depression_anti_group)
    graph("Depression Trust",control_trust_group,depression_trust_group)
    graph("Depression Disgust",control_disgust_group,depression_disgust_group)        

def predict():
    global classifier, tfidf_vectorizer
    text.delete('1.0', END)
    textData = tf1.get()
    testReview = tfidf_vectorizer.transform([textData]).toarray()
    testReview = testReview.reshape(testReview.shape[0],testReview.shape[1],1,1)
    predict = classifier.predict(testReview)
    predict = np.argmax(predict)
    result = "No Depression Detected"
    if predict == 1:
        result = "Depression Detected"
    tokens = textData.split(" ")
    emotion = ''
    for token in tokens:
        sub_emotion = getEmotion(token.strip())
        if sub_emotion is not None:
            emotion += sub_emotion+" "  
            if sub_emotion == "negative" or sub_emotion == 'anger':
                result = "Depression Detected"            
    text.insert(END,"Given Post : "+tf1.get()+"\n\n")    
    text.insert(END,"For given post prediction output is : ====> "+result+"\n\n")
    text.insert(END,"Emotion Detected as : "+str(emotion))
    
    
font = ('times', 14, 'bold')
title = Label(main, text='Detecting Mental Disorders in Social Media Through Emotional Patterns - The case of Anorexia and Depression')
title.config(bg='DarkGoldenrod1', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=5,y=5)

font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Depression Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=560,y=100)

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocessDataset)
preprocessButton.place(x=50,y=150)
preprocessButton.config(font=font1)

hybridMLButton = Button(main, text="Generate Mask Features", command=maskFeatures)
hybridMLButton.place(x=50,y=200)
hybridMLButton.config(font=font1)

snButton = Button(main, text="Mask Features into TF-IDF Weight", command=tfidf)
snButton.place(x=50,y=250)
snButton.config(font=font1)

snButton = Button(main, text="Train SVM & CNN", command=runSVMCNN)
snButton.place(x=50,y=300)
snButton.config(font=font1)

extensionButton = Button(main, text="PSO Optimized SVM & CNN", command=extensionSVMCNN)
extensionButton.place(x=50,y=350)
extensionButton.config(font=font1)

graphButton = Button(main, text="Emotional Signals Comparison", command=emotionalSignal)
graphButton.place(x=50,y=400)
graphButton.config(font=font1)

l1 = Label(main, text='Input Your Post')
l1.config(font=font1)
l1.place(x=50,y=450)

tf1 = Entry(main,width=50)
tf1.config(font=font1)
tf1.place(x=10,y=500)

exitButton = Button(main, text="Predict Disorder from Post", command=predict)
exitButton.place(x=50,y=550)
exitButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=400,y=150)
text.config(font=font1)


main.config(bg='LightSteelBlue1')
main.mainloop()
