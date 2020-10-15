statemap = []
j = -3
k = 0
while True:
    for i in range(40):
        pairState = str(int(j*100)) + str(i)
        statemap.append(int(pairState))
    j = round(j+0.2, 1)
    if j>3:
        break

class HumanA:
    def __init__(self, learn, statemap, lr=0.01, gamma=0.99, pmv = [1, 1, 1]):
        self.temp = 0
        self.pA_npA = nA = 16 #Root + Phy + Watch + P_Cont + P_Leave + W_Cont + W_Leav
        self.statespace = SS = 3720
        self.Qr = np.zeros((nA, SS))
        self.Qc = np.zeros((nA, SS, nA))
        self.Qe = np.zeros((nA, SS, nA))
        self.epsilon = 0.1
        self.valid_actions = []
        self.alpha = lr
        self.alpha_0 = lr
        self.gamma = gamma
        self.sum = 0
        self.done = False
        self.statemap = statemap
        self.time = [0, 0, 0]
        self.nState = [0, 1240, 2480, 0, 1240, 2480]
        self.reward = 0
        self.QrCopy = self.Qr.copy()
        self.taskStack = []
        self.actionStack = []
        self.pmv = []
        self.pmvSet = pmv
        self.ticks = 0
        self.taskMemory = set()
        self.hum = 50
        self.countdown = 110
        self.learn = learn
        self.pmv_time = 0
        self.states = []
        self.prev = 0
        self.log = []
        self.PlotGraph = []
        self.wait = 5
        self.tempStack = []
    
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
            (leave, cont, set0), # Activity 0
            (leave, cont, set1), # Activity 1
            (leave, cont, set2), # Activity 2
            (t0h0, t0h1, t1h0, t1h1), # Set Temperature First then run on the treadmill
            (t0h0, t0h1, t1h0, t1h1),
            (t0h0, t0h1, t1h0, t1h1),
            (activity0, activity1, activity2), #Root --> Physical Activity, Watching TV
            set(), # Leave
            set(), # Cont
            set(), # Temp1
            set(), # Temp2
            set(),
            set()
        ]

    def reset(self, temp):
        self.temp = 30#random.randint(20, 25)
        self.hum = 70#int(random.choice([i for i in range(40, 60, 5)]))
        self.time = [0, 0, 0]
        self.nState = [0, 1240, 2480, 0, 1240, 2480]
        self.prev = 0
        self.sum = 0
        self.done = False
        self.log_task = []
        self.taskStack = []
        self.actionStack = []
        self.pmv = []
        self.ticks = 0
        self.taskMemory = set()
        self.countdown = 120
        self.pmv_time = 0
        self.states = []
        self.prev = 0
        self.log = []
        self.PlotGraph = []
        self.wait = 0
        self.tempStack = []
        

    def getPmv(self, temp, hum, task):
        metabolic = self.pmvSet
        pmv =  pmv_ppd(tdb=temp, tr=25, vr=0.0, rh=hum, met=metabolic[task], clo=0.5, wme=0, standard="ASHRAE")['pmv']
        terminal = 0 if -0.5<=pmv<=0.5 else -1 if (0.5<pmv<=1 or -1<=pmv<-0.5) else -2 if(1<pmv<=1.5 or -1.5<=pmv<-1) else -3
        return terminal, pmv

    def getState(self, time, task):
        metabolic = self.pmvSet
        pmv =  round(pmv_ppd(tdb=self.temp, tr=25, vr=0.0, rh=self.hum, met=metabolic[task], clo=0.5, wme=0, standard="ASHRAE")['pmv'], 1)
        pmv_state = round(((pmv*10)%2)/10 + pmv, 1)
        state = statemap.index(int(str(int(pmv_state*100)) + str(time)))
        self.states.append(state)
        return state

        
    def cost(self, time, task):
        p0 = [3 if 0<=i<7 else 2 if 7<i<22 else 10 if 22<i<39 else 0 for i in range(40)]
        p1 = [10 if(0<=i<3) else 1.5 if (3<i<10 or 18<i<30) else 8 if (9<i<18) else 3.5 if(30<=i<39) else 0 if (i==18 or i==33 or i==39) else 0  for i in range(40)]
        p2 = [6 if 0<=i<7 else 2 if 7<i<22 else 15 if 22<i<39 else 0 for i in range(40)]
        penalty = [p0, p1, p2]
        return -penalty[task][time]
    
    def task_0(self, action):
        reward = [0 for i in range(40)]
        penalty = [3 if 0<=i<7 else 2 if 7<i<22 else 10 if 22<i<39 else 0 for i in range(40)]
        reward[5], reward[6], reward[20], reward[21], reward[37], reward[38], reward[39] = 10, 10, 5, 5, 15, 15, 0

        self.taskMemory.discard(5)
        self.taskMemory.discard(4)
        if action==self.cont:
            self.wait = max(0, self.wait-1)
            self.countdown = max(0, self.countdown-1)
            if self.time[0] == 39:
                self.reward = reward[self.time[0]] 
                self.done = True
                self.wait = 0
            else:
                self.time[0] += 1
                self.nState[0] = self.getState(self.time[0], 0)
                self.reward = reward[self.time[0]] + self.getPmv(self.temp, self.hum, 0)[0]
                self.done = False
        elif action==self.leave:
            self.wait = 0
            self.pmv_time = 0
            self.countdown = max(0, self.countdown-1)
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
        reward[31] = reward[32] = 0
        reward[37] = reward[38] = 5
        penalty = [10 if(0<=i<3) else 1.5 if (3<i<10 or 18<i<30) else 8 if (9<i<18) else 3.5 if(30<=i<39) else 0 if (i==18 or i==33 or i==39) else 0  for i in range(40)]

        self.taskMemory.discard(3)
        self.taskMemory.discard(5)

        if action==self.cont:
            self.wait = max(0, self.wait-1)
            self.countdown = max(0, self.countdown-1)
            if self.time[1] == 39:
                self.reward = reward[self.time[1]]
                self.done = True
                self.wait = 0
            else:
                self.time[1] += 1
                self.nState[1] = 1240 + self.getState(self.time[1], 1)
                self.reward = reward[self.time[1]] + self.getPmv(self.temp, self.hum, 1)[0]
                self.done = False
        elif action==self.leave:
            self.wait = 0
            self.pmv_time = 0
            self.countdown = max(0, self.countdown-1)
            self.reward = -penalty[self.time[1]]
            self.done = True

    def task_2(self, action):

        reward = [0 for i in range(40)]
        penalty = [6 if 0<=i<7 else 2 if 7<i<22 else 15 if 22<i<39 else 0 for i in range(40)]
        reward[5], reward[6], reward[20], reward[21], reward[37], reward[38], reward[39] = 15, 15, 7, 7, 30, 30, 0
        
        self.taskMemory.discard(3)
        self.taskMemory.discard(4)
        if action==self.cont:
            self.wait = max(0, self.wait-1)
            self.countdown = max(0, self.countdown-1)
            if self.time[2] == 39:
                self.reward = reward[self.time[2]]
                self.done = True
                self.wait = 5
            else:
                self.time[2] += 1
                self.nState[2] = 2480 + self.getState(self.time[2], 2)
                self.reward = reward[self.time[2]] + self.getPmv(self.temp, self.hum, 2)[0]
                self.done = False
        elif action==self.leave:
            self.wait = 0
            self.pmv_time = 0
            self.countdown = max(0, self.countdown-1)
            self.reward = -penalty[self.time[2]]
            self.done = True

    def changeTH(self, action):
        activity = self.taskStack[-1]
        map = 0 if activity==0 else 1240 if activity==1 else 2480
        D_pmv, pmv = self.getPmv(self.temp, self.hum, activity)
        self.countdown = max(0, self.countdown-1)
        if D_pmv == 0:
            self.time[activity] += 1
            self.reward = -5
            self.done = True
            self.taskMemory.add(activity+3)
            self.nState[activity+3] = map + self.getState(0, activity)
        else:
            self.pmv_time += 1
            if action==self.t0h0:
                self.hum = max(30, self.hum - 5)
            elif action==self.t0h1:
                self.hum = min(70, self.hum + 5)
            elif action==self.t1h0:
                self.temp = min(29, self.temp + 1)
            elif action==self.t1h1:
                self.temp = max(15, self.temp - 1)

            self.nState[activity+3] = map + self.getState(0, activity)
            D_pmv, pmv = self.getPmv(self.temp, self.hum, activity)
            self.pmv.append(pmv)
            self.ticks = -abs(pmv)

            if self.ticks-self.prev > 0:
                self.reward = 0
            else:
                self.reward = -1
            if D_pmv==0 or self.countdown==0:
                self.reward = 10*(40 - self.time[activity])/10
                self.taskMemory.add(activity+3)
                self.prev = self.ticks
                self.nState[activity] = map + self.getState(self.time[activity], activity)
                self.done = True
            else:
                self.prev = self.ticks
                self.done = False

    def isPrimitive(self, i):
        return True if i > 6 else False
    
    def step(self, action, act):
        self.task_0(action) if act==0 else self.task_1(action) if act==1 else self.task_2(action) if act==2 else self.changeTH(action)
                
    def isTerminal(self, i):
        if self.countdown==0:
            return True
        if i == self.root:
            return True if np.mean(self.time) == 39 else False
        elif i in [self.task0, self.task1, self.task2]:
            if self.time[i] == 39:
                self.wait = 5
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
        leftTask = len((np.array([39, 39, 39]) - np.array(self.time))[np.array(([39, 39, 39]) - np.array(self.time))>0])
        for act in self.graph[i]:
            if i==self.root:
                if act in self.taskStack:
                    if self.time[act]==39 or leftTask > 1:
                        continue

                Q = np.concatenate((Q, [self.Qr[act, s] + self.Qc[i, s, act]]))
                valid_actions = np.concatenate((valid_actions, [act])) 
            else:
                if act in self.taskMemory:
                    continue
                Q = np.concatenate((Q, [self.Qr[act, s] + self.Qc[i, s, act] + self.Qe[i, s, act]]))
                valid_actions = np.concatenate((valid_actions, [act])) 

        if i==self.root and self.taskStack:
            for task in self.taskStack:
                if self.time[task]!=39:
                    self.taskStack.remove(task)
        #random.seed(506)
        if random.random() <= self.epsilon and self.learn:
            a = int(np.random.choice(valid_actions))
            self.log.append([i, s, a, Q, valid_actions, 'R', self.time.copy()])
            return a
        else:
            a = int(valid_actions[np.argmax(Q)])
            self.log.append([i, s, a, Q, valid_actions,'NR', self.time.copy()])
            return a
        
    def decay(self, x, eps):
        q = 6*x
        self.epsilon = max(0.001, 0.6*math.exp(-q/eps))
        self.alpha = self.alpha_0*math.exp(-q/eps)
    
    def eval(self, i, s): #Evalue the Value Function for the Root
      if self.isPrimitive(i):
        return self.Qr[i, s]
      else:
        for action in self.graph[i]:
            self.QrCopy[action, s] = self.eval(action, s)
        Q = np.arange(0)
        temp_action = []
        for action in self.graph[i]:
            A = self.argmax(i, s)
            Q = np.concatenate((Q, [self.QrCopy[action, s] + self.Qc[i, s, A]]))
            temp_action.append(action)
        a = temp_action[np.argmax(Q)]
        return self.QrCopy[a, s]
    
   
    def argmax(self, i, s):
        Q = np.arange(0)
        actions = []
        leftTask = len((np.array([39, 39, 39]) - np.array(self.time))[np.array(([39, 39, 39]) - np.array(self.time))>0])
        for act in self.graph[i]:
            if i != self.root:
                Q = np.concatenate((Q, [self.Qr[act, s] + self.Qc[i, s, act] + self.Qe[i, s, act]]))
            else:
                if act in self.taskStack:
                    #continue
                    if self.time[act]==39 or leftTask > 1:
                        continue
                if s==None:
                    Q = np.concatenate((Q, [self.Qr[act, self.nState[act]] + self.Qc[i, self.nState[act], act]]))
                else:
                    Q = np.concatenate((Q, [self.Qr[act, s] + self.Qc[i, s, act]]))
            actions = np.concatenate((actions, [act]))
        return int(actions[np.argmax(Q)])

    def set_lr(self):
        self.alpha = 0
    
    def exReward(self, a, s):
        Q = np.arange(0)
        actions = []
        for act in self.graph[self.root]:
            #if act in self.taskStack:
            if act==a:
                continue
            else:
                Q, actions = np.concatenate((Q, [self.Qr[act, s] + self.Qc[self.root, s, act]])), np.concatenate((actions, [act]))
        if len(actions)==0:
            return a
        else:
            return int(actions[np.argmax(Q)])

    def MAXQ_HO(self, i, s, task):
        visitedStates = []
        if self.isPrimitive(i):
            if sh.train:
                action = sh.getAction()
                if i in [7, 8, 9, 10]:
                    award = 0 if i==action+7 else -1
                else:
                    award = 0 if action==4 else -1
                #self.PlotGraph.append(['SH', 30, award, self.ticks]) if action==4 else self.PlotGraph.append(['SH', 25, award, self.ticks])
                sh.step(award, np.mean(self.time)==39)
            else:
                action = sh.getAction()
                if action==0: #Increase temp by 0.5
                    self.hum = max(30, self.hum - 10)
                elif action==1:
                    self.hum = min(70, self.hum + 10)
                elif action==2:
                    self.temp = min(29, self.temp + 1.5)
                elif action==3:
                    self.temp = max(15, self.temp - 1.5)
                if i in [7, 8, 9, 10]:
                    award = 0 if i==action+7 else -1
                    sh.step(award, np.mean(self.time)==39)
                else:
                    award = 0 if action==4 else -1
                    sh.step(award, np.mean(self.time)==39)
                self.tempStack.append([self.temp, self.hum])
            #self.PlotGraph.append(['SH', 30, award, self.ticks]) if action==4 else self.PlotGraph.append(['SH', 25, award, self.ticks])
                
            self.step(i, task)
            self.Qr[i, s] = (1-self.alpha)*self.Qr[i, s] + self.alpha*self.reward
            visitedStates.insert(0, s) #push s into the beginning of seq
            self.PlotGraph.append([task, i, self.reward, self.temp, self.time.copy(), self.ticks, self.countdown])
            self.sum = max(self.sum+self.reward, -300) 
            self.tempStack.append([self.temp, self.hum])
        else:
            while not self.isTerminal(i):
                a = self.getAction(i, s)

                if i==self.root:
                    self.taskStack.append(a)
                    self.actionStack = []
                    root_state = s
                    s = self.nState[a]
                elif i in [0, 1, 2] and len(self.actionStack)==0:
                    a = self.cont
                if a==self.cont:
                    while True:
                        if self.countdown != 0:     
                            self.actionStack.insert(0, self.MAXQ_HO(self.cont, s, i)[0])
                        s = self.nState[i]
                        if self.getAction(i, s)!=self.cont or self.done or self.countdown==0:
                            childSeq = self.actionStack.copy()
                            break
                        else:
                            continue
                else:
                    childSeq = self.MAXQ_HO(a, s, i)
                N = 1
                self.QrCopy = self.Qr.copy()
                for vState in childSeq:
                    if i==self.root:
                        # DO NOT TOUCH #################################################################################################
                        if np.mean(self.time) != 39:
                            nA = self.argmax(self.root, None)
                            cS, nS = vState, self.nState[nA]
                            A = self.argmax(self.root, nS)#np.argmax(self.Qr[0:3, nS] + self.Qc[self.root, nS, 0:3])                   
                            QRoot = self.eval(A, nS) + self.Qc[self.root, nS, A]
                            self.Qc[i, cS, a] = (1-self.alpha)*self.Qc[i, cS, a] + self.alpha*0.9**N*QRoot
                        else:
                            self.Qc[i, cS, a] = 0  #(1-self.alpha)*self.Qc[i, vState, a] + self.alpha*0.95**N*QRoot
                        #if np.mean(self.time) != 39:
                        self.Qr[a, vState] =  (1-self.alpha)*self.Qr[a, vState] + self.alpha*(self.cost(self.time[a], a) + self.eval(a, vState))
                        nA = self.exReward(a, nS)
                        QRoot = self.Qr[nA, nS] + self.Qc[self.root, nS, nA]
                        self.Qe[a, cS, 12] = (1-self.alpha)*self.Qe[a, cS, 12] + self.alpha*(1-self.gamma)**N*QRoot
                        #################################################################################################################

                    elif i!=self.root:
                        ### DO NOT TOUCH THIS EITHER #########################################################################
                        cS, nS = vState, self.nState[i]  
                        aStar = self.argmax(i, nS)
                        self.Qr[a, cS] = self.eval(a, cS)
                        Qtype = self.eval(aStar, nS) + self.Qc[i, nS, aStar] + self.Qe[i, nS, aStar]
                        if (i in [self.task0, self.task1, self.task2] and self.time[i]==39) or a==self.leave:
                            #self.Qc[i, cS, a] = (1-self.alpha)*self.Qc[i, cS, a] + self.alpha*self.gamma**N*Qtype
                            self.Qc[i, childSeq[0], a] = 0
                        else:
                            self.Qc[i, cS, a] = (1-self.alpha)*self.Qc[i, cS, a] + self.alpha*self.gamma**N*Qtype   
                        ######################################################################################################
                    N += 1

                for vState in reversed(childSeq):
                    visitedStates.insert(0, vState)
                s = self.nState[i] if i!=self.root else nS
            self.done = False
        return visitedStates       
