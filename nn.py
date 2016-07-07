import numpy as np
import pandas as pd
import sklearn.datasets as data

import base

class NN():
    def __init__(self):
        pass

    def fit(self,X,y):
        self.X=X
        self.y=y
        cy=pd.Categorical(y)
        self.K=len(cy.categories)
        M=10**5

        #self.Omega=[np.array([0.5]*(X.shape[1])) for i in range(self.K)]
        n_hidden=6
        n_input=X.shape[1]
        n_output=self.K
        self.hidden_weight=np.random.random_sample((n_hidden, n_input + 1))
        self.output_weight = np.random.random_sample((n_output, n_hidden + 1))
        self.hidden_momentum = np.zeros((n_hidden, n_input + 1))
        self.output_momentum = np.zeros((n_output, n_hidden + 1))
        self.k=0
        for i in range(M):
            self.optim_1step()
            #print(self.hidden_weight)
            #print(self.output_weight)
            #print(i)
            #print("ceroor",self.calc_error())
            self.k+=1
            self.k%=self.X.shape[0]

    def loss1(self,y,x):
        p=[]
        hoge=np.dot(x,self.Omega[y])
        p=max(hoge/(1+hoge),0.001)
        ll=-np.log(p)

        return ll

    def softmax(self,x):
        p=[]
        for omega in self.Omega:
            hoge=np.dot(x,omega.T)
            hoge=np.exp(hoge)
            p.append(hoge)

        p=np.array(p)
        p=p/np.sum(p)
        #print(p)
        #input()
        return p

    def sigmoid(self,x):
        return np.vectorize(lambda xx:1/(1+np.exp(-xx)))(x)

    def forward(self,x):
        z = self.sigmoid(self.hidden_weight.dot(np.r_[x,1]))
        #z = self.__sigmoid(self.hidden_weight.dot(numpy.r_[numpy.array([1]), x]))
        y = self.sigmoid(self.output_weight.dot(np.r_[z,1]))
        return (z,y)

    def optim_1step(self,prob=True):

        if prob:
            rn=np.random.choice(self.X.shape[0],1)
            x=self.X[rn,:][0]
            tk=self.y[rn]
        else:
            x=self.X[self.k,:]
            tk=self.y[self.k]

        t=np.array([0]*self.K) #正解データ
        t[tk]=1
        z,y=self.forward(x)
        h=0.1
        mu=0.8

        _output_weight = self.output_weight
        opdelta=(t-y)*y*(1.0-y)
        #self.output_weight += h * opdelta.reshape((-1, 1)) * np.r_[z,np.array([1])]- mu * self.output_momentum
        self.output_weight += h *np.outer(opdelta,np.r_[z,np.array([1])] )- mu * self.output_momentum
        self.output_momentum = self.output_weight - _output_weight
        #input()
        _hidden_weight = self.hidden_weight
        hidden_delta = (self.output_weight[:,:(-1)].T.dot(opdelta)).T * z * (1.0 - z)
        #self.hidden_weight -= h * hidden_delta.reshape((-1, 1)) * np.r_[x,np.array([1])]
        #import pdb;pdb.set_trace()
        self.hidden_weight += h *np.outer(hidden_delta,np.r_[x,np.array([1])] )
        self.hidden_momentum = self.hidden_weight - _hidden_weight

    def predict(self,X):
        N=X.shape[0]
        Y=[]
        Y_C=[]
        for i in range(N):
            x=X[i,:]
            z,y=self.forward(x)
            Y.append(y)
            Y_C.append(y.argmax())
        return Y_C

    def pred_evaluate(self,X,Y):
        prd=self.predict(X)
        def n_collect(x,y):
            return sum(map(lambda x:int(x[0]==x[1]),zip(x,y)))

        hoge=n_collect(prd,Y)
        print("n_correct",hoge)
        print("n_sample",len(Y))
        print("accuracy",hoge/len(Y))
        print(prd)
        print(Y)



    def calc_error(self):
        N = self.X.shape[0]
        err = 0.0
        for i in range(N):
            x = X[i, :]
            t = self.y[i]

            z, y = self.forward(x)
            err += (y - t).dot((y - t).reshape((-1, 1))) / 2.0

        return err

    def __optim_1step(self):
        rn=np.random.choice(self.X.shape[0],1)
        x=self.X[rn,:]
        k=self.y[rn]
        domega=0.1
        h=0.1
        #loss1=self.loss1

        for i,omega in enumerate(self.Omega):
            y=self.sigmoid(x,omega)
            #print(y)
            #print(x)
            #input()
            if i==k:
                s=1
            else:
                s=0

            self.Omega[i]=omega-h*(y-s)*x

        '''
        for i,om in enumerate(omega):
            l_0=loss1(k,x)
            #print(self.Omega[k])
            self.Omega[k][i]+= domega
            #print(self.Omega[k])

            dL=loss1(k,x)-l_0
            #print("dl",dL)
            #input()
            self.Omega[k][i]-= domega
            y=self.sigmoid()
            self.Omega[k][i]=self.Omega[k][i]-h*(y-k)*x[i]
            #self.Omega[k][i]=self.Omega[k][i]-h*dL/domega
        '''

        return self.Omega

    def __predict(self,X):
        res=[]
        for xx in X:
            res.append(np.array(self.softmax(xx)))
        return res

dat=data.load_iris()
Y=dat.target
X=dat.data
#X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#y=np.array([1,0,0,1,0,1,1,0])
#X=pd.concat([pd.DataFrame([1]*X.shape[0]),pd.DataFrame(X)],axis=1)
#df=pd.read_csv("/Users/iijimasatoshi/dataset/xor_pattern")
#y=df['yy']
#X=df.iloc[:,1:]
X=np.array(X)
import sklearn.cross_validation as cv
X,X_test,Y,Y_test=cv.train_test_split(X,Y,random_state=42)

nn=NN()
nn.fit(X,Y)
prd=nn.predict(X_test)
a="sss"
nn.pred_evaluate(X_test,Y_test)
