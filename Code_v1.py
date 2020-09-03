import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import math
import random
#!pip install pythermalcomfort
from pythermalcomfort.models import pmv_ppd
from sklearn.preprocessing import KBinsDiscretizer
tempBins = KBinsDiscretizer(n_bins=31, encode='ordinal', strategy='uniform')
tempBins.fit(np.linspace(15, 30, 30).reshape(-1,1))
pmvBins = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
pmvBins.fit(np.linspace(-3.5, 3.5, 15).reshape(-1,1))

def logPlot():
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(20, 5)
    x0 = np.linspace(0, len(score0), len(score0))
    sns.lineplot(x0, score0, ax=ax[0], color='blue', label='Human')
    sns.lineplot(x0, score1, ax=ax[0], color='black', label = 'SmartH')
    ax[0].set(xlabel='Episodes', ylabel='Reward')
    task = []
    reward = []
    r = 0
    for i in H.log_task[:5000]:
        if i[0] in [3, 4, 5]:
            task.append(3)
            reward.append(i[5])
        else:
            task.append(i[0])
            reward.append(i[2])
    x = np.linspace(0, len(task), len(task))
    ax[1].grid(False)
    sns.lineplot(x, task, ax=ax[1], linewidth=2, color='black')
    ax2 = ax[1].twinx()
    ax2.grid(True)
    sns.lineplot(x, reward, ax=ax2, linewidth=2, color='green')
    ax[1].set_xlabel('States')
    ticks = [3, 2, 1, 0]
    labels = ['Change Temp','Task 2','Task 1','Task 0']
    ax[1].set_yticks(ticks)
    ax[1].set_yticklabels(labels)
    ax2.set(ylabel='Reward/ CostL In Green')
	
	
statemap = []
j = -3
k = 0
while True:
    for i in range(40):
        #tempState = tempBins.transform(np.array([j]).reshape(-1, 1))
        pairState = str(int(j*10)) + str(i)
        #print(j, i, pairState)
        k += 1
        statemap.append(int(pairState))
    j += 0.5
    if j==3.5:
        break
#print(len(statemap), k)

class Human:
    def __init__(self, statemap, lr=0.01, gamma=0.99, r_lr=0.8):
        self.temp = 0
        self.pA_npA = nA = 16 #Root + Phy + Watch + P_Cont + P_Leave + W_Cont + W_Leav
        self.statespace = SS = 2000
        self.Qr = np.zeros((nA, SS))
        self.Qc = np.zeros((nA, SS, nA))
        self.Qe = np.zeros((nA, SS, nA))
        self.epsilon = 0.1
        self.valid_actions = []
        self.alpha = lr
        self.alpha_0 = lr
        self.r_lr = r_lr
        self.gamma = gamma
        self.sum = 0
        self.done = False
        self.statemap = statemap
        self.time = [0, 0, 0]
        self.nState = [0, 0, 0, 0, 0, 0]
        self.reward = 0
        self.log_task = []
        self.QrCopy = self.Qr.copy()
        self.taskStack = []
        self.actionStack = []
        self.pmv = [-1, -1, -1]
        self.ticks = 0
        self.prev = 0
        self.act0 = []
        self.act1 = []
        self.act2 = []
        self.taskMemory = set()
        self.hum = 50
        self.countdown = 170
    
        activity0 = self.task0 = 0  # Non Primitive Action
        activity1 = self.task1 = 1  # Non Primitive Action
        activity2 = self.task2 = 2  # Non Primitive Action
        set0 = self.set0 = 3    # Non Primitive Action
        set1 = self.set1 = 4    # Non Primitive Action
        set2 = self.set2 = 5    # Non Primitive Action
        root = self.root = 6    # Non Primitive Action
        t0h0 = self.t0h0 = 7    # Primitive Action
        t0h1 = self.t0h1 = 8    # Primitive Action
        t1h0 = self.t1h0 = 9    # Primitive Action
        t1h1 = self.t1h1 = 10   # Primitive Action
        cont = self.cont = 11   # Primitive Action
        leave = self.leave = 12 # Primitive Action

        self.graph = [
            (cont, leave, set0), # Activity 0
            (cont, leave, set1), # Activity 1
            (cont, leave, set2), # Activity 2
            (t0h0, t0h1, t1h0, t1h1), # Set Temperature First then run on the treadmill
            (t0h0, t0h1, t1h0, t1h1),
            (t0h0, t0h1, t1h0, t1h1),
            (activity0, activity1, activity2), #Root --> Physical Activity, Watching TV
            set(), #Leave
            set(), #Cont
            set(), # Temp1
            set(), # Temp2
            set(),
            set()
        ]

    def reset(self, temp):
        self.temp = random.randint(20, 25)
        self.hum = int(random.choice([i for i in range(50, 60, 5)]))
        self.time = [0, 0, 0]
        self.nState = [0, 0, 0, 0, 0, 0]
        self.prev = 0
        self.sum = 0
        self.done = False
        self.log_task = []
        self.taskStack = []
        self.actionStack = []
        self.pmv = [-1, -1, -1]
        self.ticks = 0
        self.act0 = []
        self.act1 = []
        self.act2 = []
        self.taskMemory = set()
        self.countdown = 170

    def getPmv(self, temp, hum, task):
        metabolic = [1, 2, 1.2]
        pmv =  pmv_ppd(tdb=temp, tr=25, vr=0.0, rh=hum, met=metabolic[task], clo=0.5, wme=0, standard="ASHRAE")['pmv']
        terminal = 0 if -0.5<=pmv<=0.5 else -1 if (0.5<pmv<=1 or -1<=pmv<-0.5) else -2 if(1<pmv<=1.5 or -1.5<=pmv<-1) else -3
        return terminal, pmv

    def getState(self, time, task):
        metabolic = [1, 2, 1.4]
        pmv =  pmv_ppd(tdb=self.temp, tr=25, vr=0.0, rh=self.hum, met=metabolic[task], clo=0.5, wme=0, standard="ASHRAE")['pmv']
        pmv_state = round(pmv*2)/2
        return statemap.index(int(str(int(pmv_state*10)) + str(time)))

        
    def cost(self, time, task):
        p0 = [2 if 0<=i<7 else 6 if 7<i<22 else 15 if 22<i<39 else 0 for i in range(40)]
        p1 = [1.5 if(0<=i<8 or 18<i<26) else 6.5 if(25<i<33) else 0 if (i==18 or i==33 or i==39) else 0.5 if i>33 else 5 for i in range(40)]
        p2 = [2 if 0<=i<7 else 6 if 7<i<22 else 15 if 22<i<39 else 0 for i in range(40)]
        penalty = [p0, p1, p2]
        return -penalty[task][time]
    
    def task_0(self, action):
        reward = [0 for i in range(40)]
        penalty = [2 if 0<=i<7 else 15 if 7<i<22 else 6 if 22<i<39 else 0 for i in range(40)]
        reward[5], reward[6], reward[20], reward[21], reward[37], reward[38], reward[39] = 7, 7, 30, 30, 15, 15, 0

        if action==self.cont:
            if self.time[0] == 39:
                self.done = True
                self.taskMemory.add(0)
            else:
                self.time[0] += 1
                self.nState[0] = self.getState(self.time[0], 0)
                self.reward = reward[self.time[0]]+self.getPmv(self.temp, self.hum, 0)[0]
                self.done = False
        elif action==self.leave:
            self.reward = -penalty[self.time[0]]
            self.done = True
    
    def task_1(self, action):
        rew = 25
        reward = []
        for i in range(40):
            if ((i-1)%15==0 or (i-1)%15==1):
                reward.append(rew)
                if (i-1)%15==1:
                    rew = max(rew-10, 0)
            else:
                reward.append(0)
        penalty = [1.5 if(0<=i<8 or 18<i<26) else 6.5 if(25<i<33) else 0 if (i==18 or i==33 or i==39) else 0.5 if i>33 else 5 for i in range(40)]

        if action==self.cont:
            if self.time[1] == 39:
                self.taskMemory.add(1)
                self.done = True
            else:
                self.time[1] += 1
                self.nState[1] = 520 + self.getState(self.time[1], 1)
                self.reward = reward[self.time[1]] + self.getPmv(self.temp, self.hum, 1)[0]
                self.done = False
        elif action==self.leave:
            self.reward = -penalty[self.time[1]]
            self.done = True

    def task_2(self, action):

        reward = [0 for i in range(40)]
        penalty = [2 if 0<=i<7 else 6 if 7<i<22 else 15 if 22<i<39 else 0 for i in range(40)]
        reward[5], reward[6], reward[20], reward[21], reward[37], reward[38], reward[39] = 7, 7, 15, 15, 30, 30, 0
        
        if action==self.cont:
            if self.time[2] == 39:
                self.taskMemory.add(2)
                self.done = True
            else:
                self.time[2] += 1
                self.nState[2] = 1040 + self.getState(self.time[2], 2)
                self.reward = reward[self.time[2]] + self.getPmv(self.temp, self.hum, 2)[0]
                self.done = False
                if self.time[2] == 39:
                    self.taskMemory.add(2)
        elif action==self.leave:
            self.reward = -penalty[self.time[2]]
            self.done = True  

    def changeTH(self, action):
        activity = self.taskStack[-1]
        #prev_D_pmv,_ = self.getPmv(self.temp, self.hum, activity)
        map = 0 if activity==0 else 520 if activity==1 else 1040
        
        if action==self.t0h0: #Increase temp by 0.5
            self.hum = max(30, self.hum - 5)
        elif action==self.t0h1:
            self.hum = min(70, self.hum + 5)
        elif action==self.t1h0:
            self.temp = min(29, self.temp + 0.5)
        elif action==self.t1h1:
            self.temp = max(15, self.temp - 0.5)

        self.nState[activity+3] = map + self.getState(self.time[activity], activity)
        D_pmv, pmv = self.getPmv(self.temp, self.hum, activity)
        self.ticks = -abs(pmv)
        
        if self.ticks-self.prev > 0:
            self.reward = 0
        else:
            self.reward = -1

        #self.pmv[0] = discrete_pmv
        if D_pmv==0:
            self.taskMemory.add(activity+3)
            self.prev = 0 
        self.nState[activity] = map + self.getState(self.time[activity], activity)
        self.done = True
        #else:
        self.prev = self.ticks
        #self.done = False

    def isPrimitive(self, i):
        return True if i > 6 else False
    
    def step(self, action, act):
        self.task_0(action) if act==0 else self.task_1(action) if act==1 else self.task_2(action) if act==2 else self.changeTH(action)
        
    def exitState(self, i):
        if i in [0, 1, 2]:
            return True if self.time[i] == 39 else True if self.done else False
        elif i in [3, 4, 5]:
            return self.done
        
    def isTerminal(self, i):
        if i == self.root:
            return True if np.mean(self.time) == 39 else False
        elif i in [self.task0, self.task1, self.task2]:
            return True if self.time[i]==39 else True if self.done else False
        elif i in [self.set0, self.set1, self.set2]:
            return True if self.done else False
        elif self.isPrimitive(i):
            return True
        else:
            return False
            
    def getAction(self, i, s):
        Q = np.arange(0)
        valid_actions = []
        for act in self.graph[i]:
            if i==self.root:
                if act in self.taskMemory:
                    continue
                state = s#self.mapState(s, act)
                Q = np.concatenate((Q, [self.Qr[act, state] + self.Qc[i, state, act]]))
                valid_actions = np.concatenate((valid_actions, [act])) 
            else:       
                #if act in self.taskMemory:
                #    continue         
                Q = np.concatenate((Q, [self.Qr[act, s] + self.Qc[i, s, act] + self.Qe[i, s, act]]))
                valid_actions = np.concatenate((valid_actions, [act])) 

        if i==self.root and self.taskStack:
            self.taskStack.pop()
        if random.random() <= self.epsilon:
            a = int(np.random.choice(valid_actions))
            return a
        else:
            a = int(valid_actions[np.argmax(Q)])
            return a
        
    def decay(self, x, eps):
      q = 10*x
      self.epsilon = max(0.005, 0.2*math.exp(-q/eps))
      self.alpha = self.alpha_0*math.exp(-q/eps)
    
    def eval(self, i, s): #Evalue the Value Function for the Root
      if self.isPrimitive(i):
        return self.Qr[i, s]
      else:
        for action in self.graph[i]:
            self.QrCopy[action, s] = self.eval(action, s)
        Q = np.arange(0)
        for action in self.graph[i]:
            #a = np.argmax(self.Qc[i, s, :])
            Q = np.concatenate((Q, [self.QrCopy[action, s] + self.Qc[i, s, action]]))
        a = np.argmax(Q)
        return self.QrCopy[a, s]
    
    def parent(self, i):
        if i in [0, 1, 2]:
            return 6
        elif i in [3, 4, 5]:
            return self.taskStack[-1]
    
    def argmax(self, i, s):
        Q = np.arange(0)
        actions = []
        for act in self.graph[i]:
            if i != self.root:
                Q = np.concatenate((Q, [self.Qr[act, s] + self.Qc[i, s, act] + self.Qe[i, s, act]]))
            else:
                Q = np.concatenate((Q, [self.Qr[act, s] + self.Qc[i, s, act]]))
            actions = np.concatenate((actions, [act]))
        return int(actions[np.argmax(Q)])

    def set_lr(self):
        self.alpha = 0

    def MAXQ_HO(self, i, s, task):
        visitedStates = []
        if self.isPrimitive(i):
            ### AI ###
            sh.h_action = i
            sh.temp = self.temp
            sh.hum = self.hum   
            sh.activity = self.taskStack[-1]
            sh_action = sh.getAction()
            if i in [7, 8, 9, 10]:
                award = 0 if (i-7)==sh_action else -1
            elif i in [0, 1, 2, 3, 4, 5, 11, 12]:
                award = 0 if sh_action==4 else -1
            ##########
            
            ### Human ###
            self.step(i, task)
            self.Qr[i, s] = (1-self.alpha)*self.Qr[i, s] + self.alpha*self.reward
            visitedStates.insert(0, s) #push s into the beginning of seq
            self.log_task.append([task, i, self.reward, self.temp, self.time.copy(), self.ticks, self.hum, self.taskMemory.copy()])
            self.sum = max(self.sum+self.reward, -300) 
            self.countdown = max(0, self.countdown-1)
            #############
            
            ### AI ###
            #if i in [7, 8, 9, 10]:
            sh.temp = self.temp
            sh.hum = self.hum
            sh.step(award, self.done)
            ##########
        else:
            while not self.isTerminal(i):
                a = self.getAction(i, s)
                if i==self.root:
                    self.taskStack.append(a)
                childSeq = self.MAXQ_HO(a, s, i)
                if a==self.cont:
                    self.actionStack.insert(0, childSeq[0])
                    s = self.nState[i]
                    if self.exitState(i):
                        childSeq = self.actionStack.copy()
                        self.actionStack = []
                    else:
                        continue
                elif a==self.leave:#self.leave:
                    self.actionStack.insert(0, childSeq[0])
                    childSeq = self.actionStack.copy()
                    self.actionStack = []
                N = 1
                self.QrCopy = self.Qr.copy()
                for vState in childSeq:
                    if i==self.root:
                        cS, nS = vState, self.nState[a] 
                        A = np.argmax(self.Qr[0:3, nS] + self.Qc[i, nS, 0:3])
                        self.Qr[a, vState] = (1-self.alpha)*self.Qr[a, cS] + self.alpha*(self.cost(self.time[a], a) + self.eval(A, vState))
                        #self.Qc[self.root, cS, a] += self.alpha*(-self.Qc[self.root, cS, a] + 0.9**N*(self.eval(A, nS) + self.Qc[self.root, nS, A]))
                        self.Qc[i, vState, a] += self.alpha*(-self.Qc[i, vState, a] + 0.9**N*(self.Qc[i, nS, A] + self.Qr[A, i])) 
                    else:
                        cS, nS = vState, self.nState[i]  
                        aStar = self.argmax(i, nS)
                        if a in [self.set0, self.set1, self.set2]:
                            self.Qr[a, vState] += self.alpha*(-self.Qr[a, vState] + self.eval(a, vState))
                        if self.exitState(i):
                            A = self.argmax(self.parent(i), nS)
                            self.Qe[i, vState, a] = (1-self.alpha)*self.Qe[i, vState, a] + self.alpha*(self.gamma**N*(self.Qr[A, nS] + self.Qc[self.parent(i), nS, A]))
                        else:
                            self.Qe[i, vState, a] = (1-self.alpha)*self.Qe[i, vState, a] + self.alpha*(self.gamma**N*self.Qe[i, nS, aStar])  
                        self.Qc[i, vState, a] += self.alpha*(-self.Qc[i, vState, a] + self.gamma**N*(self.Qc[i, nS, aStar] + self.Qr[aStar, i] + self.Qe[i, nS, aStar]))  
                    N += 1
                for vState in reversed(childSeq):
                    visitedStates.insert(0, vState)
                s = self.nState[i] if i!=self.root else self.nState[a]
            self.done = False
        return visitedStates       

		
		
random.seed(10)
#H = Human(statemap, lr=0.01, gamma=0.9, r_lr=0.8)
sh = AI(states, gamma=0.9, lr=0.001, train=True)
score0 = []
score1 = []
trigger = 0
eps = 150
threshold = 290
sns.set(palette='bright')
for i in range(eps):
    H.reset(15)
    sh.reset()
    H.MAXQ_HO(6, 0, None)
    #H.decay(i, eps)
    sh.decay(i, eps)
    score1.append(sh.score)
    score0.append(H.sum)
    if H.sum>threshold:
        trigger += 1
        if trigger > 20:
            break
    if i%5==0:
        print(i, H.sum, sh.score) 
#H.set_lr()
#sh.freeze()
logPlot()

states = []
for task in range(3):
    t = 15
    while True:
        for i in range(30, 71, 5):
            pairState = int(str(int(t*2))+str(i))+(task)
            #print(t, i, pairState, k)
            states.append(int(pairState))
            q.add(int(pairState))
            k += 1
        t += 0.5
        if t > 30:
            break
            
class AI:
    def __init__(self, states, gamma=0.99, lr=0.01, train = True):
        self.temp = 0
        self.hum = 50
        self.activity = 0
        self.Q = np.zeros([900, 5])
        self.alpha = lr
        self.alpha_0 = lr
        self.epsilon = 0.5
        self.gamma = gamma
        self.reward = 0
        self.states = states
        self.h_action = 0
        self.cState = 0
        self.nState = 0
        self.score = 0
        self.rew = []
        self.train = train
    
    def reset(self):
        self.temp = 15
        self.hum = 50
        self.score = 0
    
    def freeze(self):
        self.alpha = 0
        
    def getState(self):
        #val = int(str(self.temp)+str(self.hum))+self.activity
        val = int(str(int(self.temp*2))+str(self.hum))+self.activity
        return self.states.index(val)
        
    def step(self, reward, done):
        self.score += reward
        nState = self.getState()
        cState = self.cState
        if done:
            val = np.array((reward))
        else:
            val = np.array((reward + self.gamma*np.max(self.Q[nState, :])))
        #print(cState, self.action)
        self.Q[cState, self.action] = (1-self.alpha)*self.Q[cState, self.action] + self.alpha*val
        
    def decay(self, x, eps):
        q = 10*x
        self.alpha = self.alpha_0*math.exp(-q/eps)
        self.epsilon = max(0.001, 0.1*math.exp(-q/eps))     
    
    def getAction(self):
        self.cState = self.getState()
        if self.train:
            if random.random() < self.epsilon:
                self.action = random.randint(0, 3)
                return self.action
            else:
                self.action =  np.argmax(self.Q[self.cState, :])
                return self.action
        else:
            state = self.getState()
            self.action =  np.argmax(self.Q[state, :])
            return self.action

			
