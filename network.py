import cv2 as cv
import numpy as np
import sys
from mnist import MNIST

class network:

    def __init__(self, N, S, M): #N, S, M - кол-во нейронов на входном, скрытом и выходном слоях
         
         self.w1 = 0.01 * np.random.default_rng().standard_normal((S, N)).astype(np.float64)
         self.w2 = 0.01 * np.random.default_rng().standard_normal((M, S)).astype(np.float64)
         
         self.b1 = 0.01 * np.random.default_rng().standard_normal((S)).astype(np.float64)
         self.b2 = 0.01 * np.random.default_rng().standard_normal((M)).astype(np.float64)
         
         self.shape = [N, S, M]
    
    def Sactivation(self, Slayer):
        return np.clip(Slayer, 0.0, np.max(Slayer))
    
    def Mactivation(self, Mlayer):
        Mlayer = np.exp(Mlayer - np.amax(Mlayer))
        Mlayer /= sum(Mlayer)
        return Mlayer

    def calcSlayer(self, data):
        return np.matmul(self.w1, data) + self.b1

    def calcMlayer(self, Slayer):
        return np.matmul(self.w2, Slayer) + self.b2
        

    def decide(self, data):
        return self.Mactivation(self.calcMlayer(self.Sactivation(self.calcSlayer(data))))
        
    def gradient(self, data, labels):
        idealM = np.zeros((self.shape[2]), np.float64)
        idealM[labels] = 1.0
        
        fs = self.Sactivation(self.calcSlayer(data)) 
        fm = self.Mactivation(self.calcMlayer(fs))

        dEdb2 = idealM * (1 - fm)

        dEdw2 = np.outer(dEdb2, fs)
        
        dEdb1 = (1 - fm[labels]) * self.w2[labels,:] * fs
        
        dEdw1 = np.outer(dEdb1, data) 
        
        return dEdw2, dEdb2, dEdw1, dEdb1    
   

class traningClassificationsNetwork: 
    
    def __init__(self, pack_sz, number_of_epochs, net):
         self.pack_sz = pack_sz
         self.number_of_epochs = number_of_epochs
         self.net = net
         
    def packTrainingNetwork(self, data_pack, labels_pack):
        dEdw2 = np.zeros_like(self.net.w2, np.float64)
        dEdb2 = np.zeros_like(self.net.b2, np.float64)
        dEdw1 = np.zeros_like(self.net.w1, np.float64)
        dEdb1 = np.zeros_like(self.net.b1, np.float64)
        
        for k in range(self.pack_sz):
            t = self.net.gradient(data_pack[k], labels_pack[k])
            dEdw2 += t[0]
            dEdb2 += t[1]
            dEdw1 += t[2]
            dEdb1 += t[3]
        

        self.net.w2 += 0.1 * (1 / self.pack_sz) * dEdw2
        self.net.b2 += 0.1 * (1 / self.pack_sz) * dEdb2
        self.net.w1 += 0.1 * (1 / self.pack_sz) * dEdw1
        self.net.b1 += 0.1 * (1 / self.pack_sz) * dEdb1

    def training(self, data, labels):
        
        meanData = np.zeros_like(data[0], np.float64)
        
        for i in range(len(data)):
            meanData += data[i]
        meanData /= len(data)
        for i in range(len(data)):
            data[i] -= meanData
            
        for epoch in range(self.number_of_epochs):
            for i in range(len(data) // self.pack_sz):
            
                data_pack = np.array(data[i *  self.pack_sz : (i + 1) *  self.pack_sz]).reshape(self.pack_sz, 784).astype(np.float64)
                labels_pack = np.array(labels[i *  self.pack_sz : (i + 1) *  self.pack_sz]).reshape(self.pack_sz).astype(np.uint8)
                
                self.packTrainingNetwork(data_pack, labels_pack)
                
            print(epoch)
    
    def testTopK1(self, data, labels):
        countErr = 0
        for i in range(len(data)):
            indmx = np.argmax(self.net.decide(np.array(data[i]).astype(np.float64)))
            countErr += indmx != labels[i];
        
        return (len(data) - countErr) / len(data)
        


def main():
    mndata = MNIST('data/')
    mndata.gz = False
    images, labels = mndata.load_training()
    
    net = network(784, 300, 10)
    tr = traningClassificationsNetwork(100, 20, net)
   
    tr.training(images, labels)
    
    images, labels = mndata.load_testing()
    print(tr.testTopK1(images, labels))

if __name__ == '__main__':
     sys.exit(main() or 0)