rootState = []
for task in range(3):
    for time in range(40):
        var = (task << 10) | time
        rootState.append(var)
        
taskState = []
for task in range(3):
    for time in range(40):
        for temp in range(15, 31):
            for hum in range(30, 71, 5):
                x = (task << 40) | (time << 20) | (temp << 10) | hum
                taskState.append(x)

PMVstate = []
for task in range(3):
    for temp in range(15, 31):
        for hum in range(30, 71, 5):
            PMVstate.append((task << 20) | (temp << 10) | hum)


class HumanA:
    def __init__(self, statemap, lr=0.01, gamma=0.99, pmv = [1, 1, 1]):
        self.temp = 0
        self.prevStates = []
        self.pA_npA = nA = 16 #Root + Phy + Watch + P_Cont + P_Leave + W_Cont + W_Leav
        self.statespace = SS = 66000#3720
        self.Qr = np.zeros((nA, SS))
        self.Qc = np.zeros((nA, SS, nA))
        self.Qe = np.zeros((nA, SS, nA))
        self.epsilon = 0.9
        self.valid_actions = []
        self.alpha = lr
        self.alpha_0 = lr
        self.gamma = gamma
        self.sum = 0
        self.done = False
        self.statemap = statemap
        self.time = [0, 0, 0]
        self.nState = [30790, 0, 0, 0, 1240, 2480]
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
        self.NextRootState = 0
        self.pmv_time = 0
        self.states = []
        self.prev = 0
        self.log = []
        self.PlotGraph = []
        self.wait = 5
        self.tempStack = []
        self.visitedRootState = []
        self.pAction = []
    
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
        self.pAction = []
        self.prevStates = []
        self.temp = 30 #random.randint(20, 25)
        self.hum = 70 #int(random.choice([i for i in range(40, 60, 5)]))
        self.time = [0, 0, 0]
        self.cTime = []
        self.nState = [0, 0, 0, 0, 0, 0]
        self.prev = 0
        self.sum = 0
        self.done = False
        self.log_task = []
        self.taskStack = []
        self.actionStack = []
        self.pmv = []
        self.ticks = 0
        self.taskMemory = set()
        self.countdown = 130
        self.pmv_time = 0
        self.states = []
        self.prev = 0
        self.log = []
        self.PlotGraph = []
        self.wait = 0
        self.tempStack = []
        self.visitedRootState = []
        self.NextRootState = 0
        
    def rootState(self, task):
        var = (task << 10) | self.time[task]
        return rootState.index(var)
    
    def mapState(self, level, time, task):
        if level in [self.task0, self.task1, self.task2]:
            state

        if level == self.root:
            var = (time << 40 ) | (self.temp << 20) | (self.hum << 10) | task
            return taskState.index(var)

    def getPmv(self, temp, hum, task):
        metabolic = self.pmvSet
        pmv =  pmv_ppd(tdb=temp, tr=25, vr=0.0, rh=hum, met=metabolic[task], clo=0.5, wme=0, standard="ASHRAE")['pmv']
        terminal = 0 if -0.5<=pmv<=0.5 else -1 if (0.5<pmv<=1 or -1<=pmv<-0.5) else -2 if(1<pmv<=1.5 or -1.5<=pmv<-1) else -3
        return terminal, pmv

    def getState(self, time, task):
        x = (task << 40) | (time << 20) | (self.temp << 10) | self.hum
        state = taskState.index(x)
        #self.states.append(state)
        return state

        
    def cost(self, time, task):
        p0 = [3 if 0<=i<7 else 2 if 7<i<22 else 10 if 22<i<39 else 0 for i in range(40)]
        p1 = [10 if(0<=i<3) else 1.5 if (3<i<10 or 18<i<30) else 8 if (9<i<18) else 3.5 if(30<=i<39) else 0 if (i==18 or i==33 or i==39) else 0  for i in range(40)]
        p2 = [6 if 0<=i<7 else 2 if 7<i<22 else 15 if 22<i<39 else 0 for i in range(40)]
        penalty = [p0, p1, p2]
        return -penalty[task][time]
    
    def task_0(self, action):
        reward = [0 for i in range(40)]
        penalty = [6 if 0<=i<7 else 2 if 7<i<22 else 15 if 22<i<39 else 0 for i in range(40)]
        reward[5], reward[6], reward[20], reward[21], reward[37], reward[38], reward[39] = 15, 15, 7, 7, 30, 30, 0
        
        self.taskMemory.discard(5)
        self.taskMemory.discard(4)

        if action==self.cont:
            self.countdown = max(0, self.countdown-1)
            if self.time[0] == 39:
                XX = reward[self.time[0]] 
                self.done = True
            else:
                self.visitedRootState.insert(0, self.rootState(0))
                self.time[0] += 1
                self.nState[0] = self.getState(self.time[0], 0)
                XX = reward[self.time[0]] + self.getPmv(self.temp, self.hum, 0)[0]
                self.done = False
        elif action==self.leave:
            self.visitedRootState.insert(0, self.rootState(0))
            self.pmv_time = 0
            self.countdown = max(0, self.countdown-1)
            self.done = True
            XX = -penalty[self.time[0]]
        return XX
            
    
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
            self.countdown = max(0, self.countdown-1)
            if self.time[1] == 39:
                XX = reward[self.time[1]]
                self.wait = 0
            else:
                self.visitedRootState.insert(0, self.rootState(1))
                self.time[1] += 1
                self.nState[1] = self.getState(self.time[1], 1) # + 1240
                XX = reward[self.time[1]] + self.getPmv(self.temp, self.hum, 1)[0]
                self.done = False
        elif action==self.leave:
            self.visitedRootState.insert(0, self.rootState(1))
            self.pmv_time = 0
            self.countdown = max(0, self.countdown-1)
            XX = -penalty[self.time[1]]
            self.done = True
        
        return XX

    def task_2(self, action):

       
        reward = [0 for i in range(40)]
        penalty = [3 if 0<=i<7 else 2 if 7<i<22 else 10 if 22<i<39 else 0 for i in range(40)]
        reward[5], reward[6], reward[20], reward[21], reward[37], reward[38], reward[39] = 10, 10, 5, 5, 15, 15, 0

        self.taskMemory.discard(3)
        self.taskMemory.discard(4)
        if action==self.cont:
            self.countdown = max(0, self.countdown-1)
            if self.time[2] == 39:
                XX = reward[self.time[2]]
                self.done = True
            else:
                self.visitedRootState.insert(0, self.rootState(2))
                self.time[2] += 1
                self.nState[2] = self.getState(self.time[2], 2) # + 2480
                XX = reward[self.time[2]] + self.getPmv(self.temp, self.hum, 2)[0]
                self.done = False
        elif action==self.leave:
            self.visitedRootState.insert(0, self.rootState(2))
            self.pmv_time = 0
            self.countdown = max(0, self.countdown-1)
            XX = -penalty[self.time[2]]
            self.done = True
        
        return XX

    def changeTH(self, action):
        task = self.taskStack[-1]
        D_pmv, pmv = self.getPmv(self.temp, self.hum, task)
        self.countdown = max(0, self.countdown-1)
        if D_pmv == 0:
            reward, self.done = -5, True
            self.taskMemory.add(task+3)
            #self.nState[task+3] = map + self.getState(self.time[task], task)
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

            self.nState[task+3] = PMVstate.index((task << 20) | (self.temp << 10) | self.hum)
            D_pmv, pmv = self.getPmv(self.temp, self.hum, task)
            self.pmv.append(pmv)
            self.ticks = -abs(pmv)

            if self.ticks-self.prev > 0:
                reward =  1
            else:
                reward = -1
            if D_pmv==0:
                reward = 10 #Pseudo Reward
                self.taskMemory.add(task+3)
                self.prev = self.ticks
                self.nState[task] = self.getState(self.time[task], task)
                self.done = True
            else:
                self.prev = self.ticks
                self.done = False
        return reward

    def isPrimitive(self, i):
        return True if i > 6 else False
    
    def step(self, action, act):
        return self.task_0(action) if act==0 else self.task_1(action) if act==1 else self.task_2(action) if act==2 else self.changeTH(action)
                
    def isTerminal(self, i):
        if self.countdown==0:
            return True
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
        K = np.arange(0)
        valid_actions = []
        leftTask = len((np.array([39, 39, 39]) - np.array(self.time))[np.array(([39, 39, 39]) - np.array(self.time))>0])
        for act in self.graph[i]:
            if i==self.root:
                K = np.concatenate((K, [self.Qr[act, s] + self.Qc[self.root, s, act]]))
                if act in self.taskStack:
                    if self.time[act]==39 or leftTask > 1:
                        continue
                Q = np.concatenate((Q, [self.Qr[act, s] + self.Qc[self.root, s, act]]))
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

        if random.random() <= self.epsilon:
            a = int(np.random.choice(valid_actions))
            self.log.append("__".join(str(var) for var in [i, s, a, Q, valid_actions, 'R', K]))
            return a
        else:
            a = int(valid_actions[np.argmax(Q)])
            self.log.append("__".join(str(var) for var in [i, s, a, Q, valid_actions, 'NR', K]))
            return a
        
    def decay(self, x, eps):
        q = 5*x
        self.epsilon = max(0.001, math.exp(-q/eps))
        self.alpha = self.alpha_0*math.exp(-q/eps)
    
    def eval(self, i, s): #Evalue the Value Function for the Root
      if self.isPrimitive(i):
        return self.QrCopy[i, s]
      else:
        for action in self.graph[i]:
            state = self.mapState(action, s)
            self.QrCopy[action, state] = self.eval(action, state)
        if i in [0, 1, 2]:
            potAct = [i+3, 11, 12]
        elif i in [3, 4, 5]:
            potAct = [7, 8, 9, 10]
        a_ = np.argmax([self.QrCopy[x, state] + self.Qc[i, state, x] for x in potAct])
        return self.QrCopy[potAct[a_], state]

    def mapState(self, action, state):
        if action in [0, 1, 2]:
            state = rootState[state]
            t = state & 0xff
            var = (action << 40) | (t << 20) | (self.temp << 10) | self.hum
            return taskState.index(var)
        elif action in [3, 4, 5]:
            cstate = taskState[state]
            ctask = (cstate >> 40) & 0xff
            ctemp = (cstate >> 10) & 0xff
            chum = cstate & 0xff
            var = (ctask << 20) | (temp << 10) | hum
            return PMVstate.index(var)
        
        elif self.isPrimitive(action):
            return state
    
   
    def argmax(self, i, s):
        Q = np.arange(0)
        actions = []
        leftTask = len((np.array([39, 39, 39]) - np.array(self.time))[np.array(([39, 39, 39]) - np.array(self.time))>0])
        for act in self.graph[i]:
            if i != self.root:
                Q = np.concatenate((Q, [self.Qr[act, s] + self.Qc[i, s, act] + self.Qe[i, s, act]]))
            else:
                if act in self.taskStack:
                    if self.time[act]==39 or leftTask > 1:
                        continue
                #if s==None:
                Q = np.concatenate((Q, [self.Qr[act, s] + self.Qc[i, s, act]]))
                #else:
                #    Q = np.concatenate((Q, [self.Qr[act, s] + self.Qc[i, s, act]]))
            actions = np.concatenate((actions, [act]))
        return int(actions[np.argmax(Q)])

    
    def exReward(self, a, s):
        Q = np.arange(0)
        actions = []
        for act in self.graph[self.root]:
            if act!=a:
                Q, actions = np.concatenate((Q, [self.Qr[act, s] + self.Qc[self.root, s, act]])), np.concatenate((actions, [act]))
        return int(actions[np.argmax(Q)])
    

    def SHS(self, i):
        if sh.train:
            action = sh.getAction()
            if i in [7, 8, 9, 10]:
                award = 0 if i==action+7 else -1
            else:
                award = 0 if action==4 else -1
            sh.step(award, np.mean(self.time)==39)
        else:
            action = sh.getAction()
            if action==0: #Increase temp by 0.5
                self.hum = max(30, self.hum - 5)
            elif action==1:
                self.hum = min(70, self.hum + 5)
            elif action==2:
                self.temp = min(29, self.temp + 1)
            elif action==3:
                self.temp = max(15, self.temp - 1)
            if i in [7, 8, 9, 10]:
                award = 0 if i==action+7 else -1
                sh.step(award, np.mean(self.time)==39)
            else:
                award = 0 if action==4 else -1
                sh.step(award, np.mean(self.time)==39)
                
    def MAXQ_HO(self, i, s, task):
        visitedStates = []
        if self.isPrimitive(i):
            self.SHS(i)  
            reward = self.step(i, task)
            self.Qr[i, s] = (1-self.alpha)*self.Qr[i, s] + self.alpha*reward
            visitedStates.insert(0, s)
            self.PlotGraph.append([task, i, reward, self.Qr[i, s], self.time.copy(), self.ticks, s])
            self.sum += reward

        else:
            while not self.isTerminal(i):
                a = self.getAction(i, s)
                if len(self.log)==1 and i==self.root:
                    a = 1
                if i==self.root:
                    #a = 1 if len(self.taskStack) == 0 else a
                    self.taskStack.append(a)
                    self.actionStack, self.visitedRootState, self.tempStack = [], [], []
                    s = self.nState[a]
                    
                elif i in [0, 1, 2] and len(self.actionStack)==0:
                    a = self.cont
                elif i in [0, 1, 2] and a in [3, 4, 5]:
                    s = self.nState[a]

                
                childSeq = self.MAXQ_HO(a, s, i)
                if a==self.cont:
                    self.actionStack.insert(0, childSeq[0])         
                    #self.visitedRootState.insert(0, self.rootState(i))
                    s = self.nState[i]
                    continue
                #if a==self.leave:
                    #self.visitedRootState.insert(0, self.rootState(i))
                if a in [7, 8, 9, 10]:
                    self.tempStack.insert(0, childSeq[0])
                    s = self.nState[i]
                    visitedStates = self.tempStack.copy()
                    continue

                self.QrCopy = self.Qr.copy()

                N = 1
                for cS in self.visitedRootState:
                    if i!=self.root:
                        break
                    if np.mean(self.time) != 39:
                        A = self.argmax(self.root, self.visitedRootState[0])
                        nRS = self.rootState(A)
                        QRoot = self.eval(A, nRS) + self.Qc[self.root, nRS, A]
                        self.Qc[self.root, cS, a] = (1-self.alpha)*self.Qc[self.root, cS, a] + 0.1*0.25**N*QRoot
                    else:
                        self.Qc[self.root, cS, a] = 0  #(1-self.alpha)*self.Qc[i, vState, a] + self.alpha*0.95**N*QRoot
                    
                    self.Qr[a, cS] = self.eval(a, nRS)# + self.cost(self.time[a], a)
                    N += 1

                N = 1
                for cS in self.actionStack:
                    if i not in [0, 1, 2]:
                        break
                    #indices = [0, 1, 2]
                    #indices.remove(i)
                    nS = self.nState[i]
                    aStar = self.argmax(i, nS) 
                    #self.Qr[self.cont, cS] = self.eval(self.cont, cS)
                    #self.pAction.append([self.Qr[self.cont, cS], self.eval(a, cS), self.cont])
                    self.Qc[i, cS, self.cont] = (1-self.alpha)*self.Qc[i, cS, self.cont] + self.alpha*(self.gamma**N)*(self.Qr[aStar, nS] + self.Qc[i, nS, aStar])
                    nS = rootState.index((i << 10 ) | self.time[i])
                    
                    tempState = rootState.index((i << 10) | (self.time[i] - (len(self.actionStack) - N + 1)))
                    #A = self.argmax(self.root, self.visitedRootState[0])
                    A = self.argmax(self.root, tempState)
                    nS = self.rootState(A)
                    #nS = rootState.index((i << 10) | (self.time[i] - (len(self.actionStack) - N + 1)))
                    QRoot = self.Qr[A, nS] + self.Qc[self.root, nS, A]
                    self.Qe[i, cS, self.leave] = (1-self.alpha)*self.Qe[i, cS, self.leave] + self.alpha*(1-self.gamma)**N*QRoot
                    N += 1
                
                N = 1
                ## For TH
                if i in [3, 4, 5]:
                    childSeq = self.tempStack.copy()
                for state in childSeq:
                    if i in [3, 4, 5]:
                        cS, nS = state, self.nState[i]  
                        aStar = self.argmax(i, nS)
                        #self.Qr[a, cS] = self.eval(i, cS)
                        #Qtype = self.eval(i, nS) + self.Qc[i, nS, aStar]
                        Qtype = self.Qr[aStar, nS] + self.Qc[i, nS, aStar]
                        self.Qc[i, cS, a] = (1-self.alpha)*self.Qc[i, cS, a] + self.alpha*self.gamma**N*Qtype 

                    elif i in [0, 1, 2] and a in [self.set0, self.set1, self.set2]:
                        cS = state
                        nS = self.nState[i]
                        temp_state = PMVstate[cS]
                        temperature, humidity = (temp_state >> 10) & 0xff, temp_state & 0xff
                        temp_state = (i << 40) | (self.time[i] << 20) | (temperature << 10) | humidity
                        cS = taskState.index(temp_state)
                        A = self.argmax(i, nS)
                        self.Qr[i, cS] = self.eval(i, cS)
                        Qtype = self.Qr[A, nS] + self.Qc[i, nS, A]
                        self.Qc[i, cS, a] = (1-self.alpha)*self.Qc[i, cS, a] + self.alpha*self.gamma**N*Qtype
                    elif i in [0, 1, 2] and a==self.leave:
                        #self.Qr[i, state] = self.eval(i, state)
                        self.Qc[i, state, self.leave] = 0
                    N += 1

                if len(self.actionStack) > 0 and a==self.leave:
                    self.actionStack.insert(0, childSeq[0])
                    childSeq = self.actionStack.copy()

                for vState in reversed(childSeq):
                    visitedStates.insert(0, vState)
                #if i!=self.root:
                s = self.nState[i] if i!=self.root else nRS #rootState.index((i << 10) | self.time[i])
            self.done = False
        return visitedStates       
