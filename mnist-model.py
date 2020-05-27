from keras.layers import Dense, Convolution2D, MaxPooling2D, ZeroPadding2D, Dropout, InputLayer
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten, Activation
from keras.models import Sequential,load_model
import random
from keras.optimizers import RMSprop, Adam, Adamax, SGD, Nadam

from keras.datasets import mnist

dataset = mnist.load_data('mnist.db')

#Splitting and reshaping the dataset
img_rows=28
img_cols=28

train , test = dataset
X_train , y_train = train
X_test , y_test = test

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#Converting y to categorical data 
from keras.utils.np_utils import to_categorical
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)


model = Sequential()

#CRP1
model.add(Convolution2D(filters=random.choice((16,32,64)),  
                        kernel_size=random.choice(((3,3),(5,5))),  
                        activation='relu', 
                        input_shape=input_shape ))
model.add(MaxPooling2D(pool_size=(2,2)))

#CRP2
model.add(Convolution2D(filters=random.choice((16,32,64)),  
                        kernel_size=random.choice(((3,3),(5,5))),  
                        activation='relu' ))
model.add(MaxPooling2D(pool_size=(2,2)))      


#Flatten
model.add(Flatten())

#Dense1
model.add(Dense(units=50, activation='relu'))

def addCRP(model, no_CRP):
    for i in range(no_CRP):
        model.add(Convolution2D(filters=random.randint(8,32), 
                        kernel_size=random.choice(((3,3),(5,5),(7,7))), 
                        activation='relu' ))
        model.add(MaxPooling2D(pool_size=random.choice((2,2),(4,4))))
        model.add(BatchNormalization())
        model.add(Flatten())

def addDense(model, no_Dense):
    for i in range(no_Dense):
        model.add(Dense(units=random.choice((32,64)), activation='relu'))

i=1
while(i!=5):
    #Output layer - manually added - we have 10 classes
    model.add(Dense(units=10, activation='softmax'))
    
    #Compiling the model
    model.compile(optimizer=random.choice((RMSprop(lr=0.001), Adam(lr=0.001), Adamax(lr=0.001), SGD(lr=0.001), Nadam(lr=0.001))),
        loss='categorical_crossentropy',
        metrics=['accuracy']
              )
    
    #Model summary
    model.summary()
    
    out = model.fit(
        X_train,y_train_cat,
        epochs=random.randint(10,30),
        validation_data=(X_test, y_test_cat))
    
    print(out.history, end='\n\n\n')

    print(out.history['accuracy'][0])

    model.save('MNISTClassifier.h5')

    mod =str(model.summary())
    accuracy = str(out.history['accuracy'][0])
    
    print(accuracy , file = open("/root/mnist-model/mnistaccuracy.txt","a"))

    if out.history['accuracy'][0] >= .80:
        import smtplib
        # creates SMTP session 
        s = smtplib.SMTP('smtp.gmail.com', 587)
        # start TLS for security 
        s.starttls()

        # Authentication 
        s.login("sender@gmail.com", "receiver@gmail.com")


        # message to be sent 
        message1 = accuracy
        message2 = mod


        # sending the mail 
        s.sendmail("sender@gmail.com", "sender@gmail.com", message1)
        s.sendmail("receiver@gmail.com", "receiver@gmail.com", message2)

        # terminating the session 
        s.quit()
        print("Model trained to an accuracy of: " + accuracy)
        break
    '''elif accuracy>.50:
        #model=load_model('MNISTClassifier.h5')
        popped_output=model.pop()
        addDense(model, random.randint(1,3))
        print("Training the model for the "+ i +"th time by adding only Dense layers!")'''
    else:
        popped_output=model.pop()
        #addCRP(model, random.randint(1,2))
        addDense(model, random.randint(1,3))
        print("Training the model for the "+ i +"th time by adding CRP and Dense layers!")
    i=i+1