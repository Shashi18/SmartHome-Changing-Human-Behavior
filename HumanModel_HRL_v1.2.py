np.set_printoptions(suppress=True, precision=6)
rootState = []
for t0 in range(40):
    for t1 in range(40):
        for t2 in range(40):
            var = (t0 << 20) | (t1 << 10) | t2
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


class HumanModel:
    def __init__(self, flag, lr=0.01, gamma=[0.9, 0.9, 0.9], pmv = [1, 1, 1]):
        self.flag = flag
        self.temp, self.hum = env.getTH(0)
        self.pA_npA = nA = 13 # All Primitive + Non- Primiitive actions
        self.statespace = SS = 64000
        self.Qr = np.zeros((nA, SS))               # Value Function
        self.Qc = np.zeros((nA, SS, nA))           # Completion Rewards Qc
        self.Qe = np.zeros((nA, SS, nA))           # Exit Rewards Qe
        self.epsilon = 0.9
        self.alpha = lr
        self.alpha_0 = lr
        self.gamma = gamma
        self.sum = 0        
        self.done = False                           # Termination Indicator
        self.cTime = [0, 0, 0]                      # Current Time
        self.time = [0, 0, 0]                       # Next Time
        self.nState = [0, 0, 0, 0, 0, 0]            # Stores next states
        self.QrCopy = self.Qr.copy()                # Copy Value Function for Recursive updates of non-primitive actions
        self.taskStack = []                         # Stores completed tasks and ongoing tasks
        self.pmv = []                               # Stores PMV variation
        self.metabolism = pmv                       # Metabolism indices
        self.log = []                               # For LOG purpose
        self.PlotGraph = []                         # For Plotting
        self.trash = []                             # Store initial temperatures
        self.bin = []                               # For LOG purpose
        self.discrete_pmv = -1                      # Stores discrete PMV
        self.curr_pmv, self.prev_pmv, self.seed = 0, 0, 0 # Stores current, previous PMV (continuous values)

        activity0 = self.task0 = 0  # Non Primitive Action
        activity1 = self.task1 = 1  # Non Primitive Action
        activity2 = self.task2 = 2  # Non Primitive Action
        setTH0 = self.adjustTH0 = 3    # Non Primitive Action
        setTH1 = self.adjustTH1 = 4
        setTH2 = self.adjustTH2 = 5
        root = self.root = 6    # Primitive Action
        incT = self.incT = 7
        decT = self.decT = 8
        incH = self.incH = 9
        decH = self.decH = 10
        cont = self.cont = 11   # Primitive Action
        leave = self.leave = 12   # Primitive Action



        self.graph = [
            (cont, leave, setTH0), # Activity 0
            (cont, leave, setTH1), # Activity 1
            (cont, leave, setTH2), # Activity 2
            (incT, decT, incH, decH),
            (incT, decT, incH, decH),
            (incT, decT, incH, decH),
            (activity1, activity0, activity2), #Root --> Physical Activity, Watching TV
            set(), # Leave
            set(), # Cont
            set(), # Temp1
            set(), # Temp2
            set(),
            set()
        ]

    def SHS(self, i):
        if sh.train:
            action = sh.getAction(self.flag)
            award = 0 if i in [7, 8, 9, 10] and i == action + 7 else 0 if i not in [7, 8, 9, 10] and action == 4 else -1
            sh.step(award, np.mean(self.time)==39)
        else:
            action = sh.getAction(self.flag)
            self.temp, self.hum = env.getTH(action+7)
            if i in [7, 8, 9, 10]:
                award = 0 if i==action else -1
                sh.step(award, np.mean(self.time)==39)
            else:
                award = 0 if action==4 else -1
                sh.step(award, np.mean(self.time)==39)
                
    def reset(self, temp):
        self.temp, self.hum = env.getTH(0) #random.randint(20, 25)
        self.time = [0, 0, 0]
        self.cTime = [0, 0, 0]
        self.nState = [0, 0, 0, 0, 0, 0]
        self.sum = 0
        self.done = False
        self.log_task = []
        self.taskStack = []
        self.pmv = []
        self.taskMemory = set()
        self.countdown = 120
        self.pmv_time = 0
        self.log = []
        self.PlotGraph = []
        self.trash = []
        self.bin = []
        self.discrete_pmv = -1
        self.graph[6] = tuple(sorted([1, 0, 2], key = lambda x: random.random()))
        self.curr_pmv, self.prev_pmv = -1, -1

    def rootState(self, task):
        var = (self.time[0] << 20) | (self.time[1] << 10) | self.time[2]
        return rootState.index(var)

    def getPmv(self, task):
        temp, hum = env.getTH(0)
        pmv =  pmv_ppd(tdb=temp, tr=25, vr=0.0, rh=hum, met=self.metabolism[task], clo=0.35, wme=0, standard="ASHRAE")['pmv']
        if self.flag != 3: #3 Means Human Model C with different reward structure
            terminal = 0 if -0.5<=pmv<=0.5 else -1 if (0.5<pmv<=1 or -1<=pmv<-0.5) else -2 if(1<pmv<=1.5 or -1.5<=pmv<-1) else -3
        else: # Human Model C
            terminal = 0 if -0.1<=pmv<=0.1 else -1 if (0.1<pmv<=1 or -1<=pmv<-0.1) else -2 if(1<pmv<=1.5 or -1.5<=pmv<-1) else -3
        return terminal, pmv

    def getState(self, time, task):
        temp, hum = env.getTH(0)
        encoded_state = (task << 40) | (time << 20) | (temp << 10) | hum
        state = taskState.index(encoded_state)
        return state
    
    # Maps child level state to Root level state 
    def trueStates(self, i, statesVisited, a):
        if i != self.root:
            return statesVisited
        root_states = []
        for S in statesVisited:
            root_states.append(self.mapUP(self.root, S, a))
        return list(dict.fromkeys(root_states))

    def task_reward(self, task, time, action):
        if action not in [self.cont, self.leave]:
            raise ValueError('Wrong action passed')
            return
        if task == 0:
            reward = [0 for i in range(40)]
            left = [i if i%5==0 else 0 for i in range(40)]
            right = [40-i if i%5==0 else 0 for i in range(40)]
            reward = [min(left[i], right[i]) for i in range(40)]
            penalty = [10 if 0<=i<6 else 8 if 6<i<11 else 6 if 11<i<16 else 4 if 16<i<21 else 2 if 21<i<26 else 1 if (26<i<31 or 31<i<39) else 0 for i in range(40)]
            return reward[time] if action == self.cont else -penalty[time] 
        elif task == 1:
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
            return reward[time] if action == self.cont else -penalty[time] 
        else:
            reward = [0 for i in range(40)]
            penalty = [2 if 0<=i<7 else 4 if 7<i<14 else 2.5 if 14<i<22 else 1 if 22<i<27 else 0.25 if 27<i<39 else 0 for i in range(40)]
            reward[5], reward[6], reward[12], reward[13], reward[20], reward[21], reward[25], reward[26], reward[37], reward[38], reward[39] = 8, 8, 16, 16, 10, 10, 4, 4, 24, 24, 0
            return reward[time] if action == self.cont else -penalty[time]

    # This is Task 0 #
    def task_0(self, action):
        # For continuing the task
        if action==self.cont:
            if self.time[0] == 39:
                self.cTime = self.time.copy()
                net_reward = self.task_reward(0, self.time[0], self.cont) + self.getPmv(0)[0]
                self.done = True
            else:
                self.cTime = self.time.copy()
                net_reward = self.task_reward(0, self.time[0], self.cont) + self.getPmv(0)[0]
                self.time[0] += 1
                self.nState[0] = self.getState(self.time[0], 0)
                self.done = False
        # For Leaving the task
        elif action==self.leave:
            if self.flag == 3: # Human Model C : Leaving 
                net_reward = self.task_reward(0, self.time[0], self.leave) - self.getPmv(0)[0]
            else:
                net_reward = self.task_reward(0, self.time[0], self.leave)
            self.time[0] += 1
            self.nState[0] = self.getState(self.time[0], 0)
            self.done = True

        return net_reward


    # This is Task 1
    def task_1(self, action):
        # For continuing the task
        if action==self.cont:
            if self.time[1] == 39:
                self.cTime = self.time.copy()
                net_reward = self.task_reward(1, self.time[1], self.cont) + self.getPmv(1)[0]
                self.done = True
            else:
                self.cTime = self.time.copy()
                net_reward = self.task_reward(1, self.time[1], self.cont) + self.getPmv(1)[0]
                self.time[1] += 1
                self.nState[1] = self.getState(self.time[1], 1) # + 1240

                self.done = False
        # For Leaving the task
        elif action==self.leave:
            if self.flag == 3: # Human Model C 
                net_reward = self.task_reward(1, self.time[1], self.leave) - self.getPmv(1)[0]
            else:
                net_reward = self.task_reward(1, self.time[1], self.leave)
            self.time[1] += 1
            self.nState[1] = self.getState(self.time[1], 1)
            self.done = True

        return net_reward

    # This is Task 2
    def task_2(self, action):
        #For continuing the task
        if action==self.cont:
            if self.time[2] == 39:
                self.cTime = self.time.copy()
                net_reward = self.task_reward(2, self.time[2], self.cont) + self.getPmv(2)[0]
                self.done = True
            else:
                self.cTime = self.time.copy()
                net_reward = self.task_reward(2, self.time[2], self.cont) + self.getPmv(2)[0]
                self.time[2] += 1
                self.nState[2] = self.getState(self.time[2], 2)
                self.done = False
        
        # For leaving the task
        elif action==self.leave:
            if self.flag == 3:  # Human Model C 
                net_reward = self.task_reward(2, self.time[2], self.leave) - self.getPmv(2)[0]
            else:
                net_reward = self.task_reward(2, self.time[2], self.leave)
            self.time[2] += 1
            self.nState[2] = self.getState(self.time[2], 2)
            self.done = True

        return net_reward

    # This function sets the desired temperature and humidity level
    def changeTH(self, action):
        currentTask = self.taskStack[-1]
        discrete_pmv, pmv = self.getPmv(currentTask)
        if discrete_pmv == 0:
            reward, self.done = -5, True
            self.time[currentTask] += 1
            self.nState[currentTask] = self.getState(self.time[currentTask], currentTask)
        else:
            self.temp, self.hum = env.getTH(action)
            self.nState[currentTask+3] = PMVstate.index((currentTask << 20) | (self.temp << 10) | self.hum)
            self.discrete_pmv, cont_pmv = self.getPmv(currentTask)
            self.pmv.append(cont_pmv)
            self.curr_pmv = -abs(cont_pmv)
            
            reward = 0 if (self.curr_pmv - self.prev_pmv) > 0 else -1
            self.prev_pmv = self.curr_pmv
            if self.discrete_pmv==0:
                self.nState[currentTask], self.done, reward = self.getState(self.time[currentTask], currentTask), True, 10
            else:
                self.done = True
        return reward

    # Decay Rate of Epsilon and Learning Rate
    def decay(self, x, eps):
        
        q = 4*x
        self.epsilon = max(0.001, math.exp(-5*x/eps))
        self.alpha = self.alpha_0*math.exp(-q/eps)

    # Check whether the node is primitive(non-SMDP) or composite(SMDP)
    def isPrimitive(self, i):
        return True if i > 6 else False

    # Take action
    def step(self, action, act):
        return self.task_0(action) if act==0 else self.task_1(action) if act==1 else self.task_2(action) if act==2 else self.changeTH(action)

    # Checks whether the current SMDP is terminated
    def isTerminal(self, i):
        if i == self.root:
            return True if np.mean(self.time)==39  else False
        elif i in [self.task0, self.task1, self.task2]:
            return True if self.time[i]==39 else True if self.done else False
        elif i in [3, 4, 5]:
            return self.done
        elif self.isPrimitive(i):
            return True
        else:
            return False

    # Get the current action based on the state for a particular non-primitive node
    def getAction(self, i, s):
        Q = np.arange(0)
        valid_actions = []
        leftTask = len((np.array([39, 39, 39]) - np.array(self.time))[np.array(([39, 39, 39]) - np.array(self.time))>0])
        for act in self.graph[i]:
            if i==self.root:
                #K = np.concatenate((K, [self.Qr[act, s] + self.Qc[self.root, s, act]]))
                if act in self.taskStack:
                    if self.time[act]==39 or leftTask > 1:
                        continue
                Q = np.concatenate((Q, [self.Qr[act, s] + self.Qc[self.root, s, act]]))
                valid_actions = np.concatenate((valid_actions, [act])) 
            else:
                Q = np.concatenate((Q, [self.Qr[act, s] + self.Qc[i, s, act] + self.Qe[i, s, act]]))
                valid_actions = np.concatenate((valid_actions, [act])) 

        if i==self.root and self.taskStack:
            for task in self.taskStack:
                if self.time[task]!=39:
                    self.taskStack.remove(task)

        if random.random() < self.epsilon:
            a = int(np.random.choice(valid_actions))
            self.log.append("".join(str(var).ljust(6) for var in [i, s, a, Q, valid_actions, self.Qr[a, s], self.time.copy(), 'R']))
            if i==self.root: self.taskStack.append(a)
            return a
        else:
            a = int(valid_actions[np.argmax(Q)])
            self.log.append("".join(str(var).ljust(6) for var in [i, s, a, Q, valid_actions, self.Qr[a, s], self.time.copy(), 'NR']))
            if i==self.root: self.taskStack.append(a)
            return a

    # This function gives the next possible action for the NEXT state thus it is not entered in the LOG
    def argmax(self, i, s):
        Q = np.arange(0)
        actions = []
        leftTask = len((np.array([39, 39, 39]) - np.array(self.time))[np.array(([39, 39, 39]) - np.array(self.time))>0])
        for act in self.graph[i]:
            if i != self.root:
                Q = np.concatenate((Q, [self.Qr[act, s] + self.Qc[i, s, act] + self.Qe[i, s, act]]))
            else:
                if act in self.taskStack:
                    if self.cTime[act]==39 or leftTask > 1:
                        continue
                Q = np.concatenate((Q, [self.Qr[act, s] + self.Qc[i, s, act]]))
            actions = np.concatenate((actions, [act]))
        return int(actions[np.argmax(Q)])

    #Evaluate the Value Decomposition (Qr) of a non-primitive SMDP node. 
    def EvaluateRecursive(self, i, s):
        if self.isPrimitive(i):
            return self.QrCopy[i, s]
        else:
            valid_action = []
            task_state = self.mapDown(i, s)
            for action in self.graph[i]:
                self.QrCopy[action, task_state] = self.EvaluateRecursive(action, task_state)
                valid_action.append(action)
            a = np.argmax([self.QrCopy[action, task_state] + self.Qc[i, task_state, action] for action in valid_action])
            return self.QrCopy[valid_action[a], task_state]

    # Maps Parent Node state to Child Node state down the Hierarchy
    def mapDown(self, action, index):
        if action in [0, 1, 2]:
            self.trash = [self.temp, self.hum] if len(self.trash) == 0 else self.trash
            state = rootState[index]
            if action == 0:
                return taskState.index((action << 40) | (state >> 20 & 0xff) << 20 | self.temp << 10 | self.hum)
            elif action == 1:
                return taskState.index(action << 40 | (state >> 10 & 0xff) << 20 | self.temp << 10 | self.hum)
            else:
                return taskState.index(action << 40 | (state & 0xff) << 20 | self.trash[0] << 10 | self.trash[1])
        elif action in [3, 4, 5]:
            state = taskState[index]
            task, temp, hum = (state >> 40) & 0xff, (state >> 10) & 0xff, state & 0xff
            var = (task << 20) | (temp << 10) | hum
            return PMVstate.index(var)

        elif self.isPrimitive(action):
            return index


    # This function updates the Q table for all Tasks when the action was to continue
    # This is separate and not in MAXQ_HO to follow the SMDP routine i.e. updates all visited states at once rather than one by one.
    def update(self, flag, childSeq, i, s_, a_):
        if flag == 1:
            for vS in childSeq:
                state = self.mapUP(i, vS, a_)
                self.Qr[a_, state] = self.EvaluateRecursive(a_, state)
            return

        if len(childSeq) == 0:
            return
        root_s, N = self.mapUP(self.root, childSeq[0], i), 1
        root_a = self.argmax(self.root, root_s)
        for vS in childSeq:
            if vS != s_:
                self.Qc[i, vS, self.cont] = (1-self.alpha)*self.Qc[i, vS, self.cont] + self.alpha*self.gamma[i]**N*(self.Qr[a_, s_] + self.Qc[i, s_, a_])
            N += 1

    # Maps Child Node state to Parent Node state down the Hierarchy
    def mapUP(self, i, s, a):
        if i in [3, 4, 5] or (i in [0, 1, 2] and self.isPrimitive(a)):
            return s
        elif i in [0, 1, 2] and not self.isPrimitive(a):#a in [3, 4, 5]:       
            temp, hum = PMVstate[s] >> 10 & 0xff, PMVstate[s] & 0xff
            return taskState.index((i << 40) | (self.time[i] << 20) | (temp << 10) | hum)
        else:
            time = (taskState[s] >> 20) & 0xff
            if a == 0:
                return rootState.index(time << 20 | self.time[1] << 10 | self.time[2])
            elif a == 1:
                return rootState.index(self.time[0] << 20 | time << 10 | self.time[2])
            else:
                return rootState.index(self.time[0] << 20 | self.time[1] << 10 | time)

    # Trains the agent to set correct TH to obtain maximum confort
    def trainPMV(self, i, s, a):
        self.trash = [self.temp, self.hum]
        retStates = []
        for episode in range(1):
            alpha = 0.01
            while not self.isTerminal(i):
                self.SHS(a)
                if not sh.train:
                    temp, hum = env.getTH(0)
                    s = PMVstate.index((self.taskStack[-1] << 20) | (temp << 10) | hum)
                    a = self.getAction(i, s)
                retStates.insert(0, s)
                score = self.changeTH(a)
                self.Qr[a, s] = (1-self.alpha)*self.Qr[a, s] + alpha*score
                if episode == 0:
                    self.PlotGraph.append([i, a, score, self.Qr[a, s], 0, self.curr_pmv, s])
                s = self.nState[i]
                a_ = self.getAction(i, s)
                self.Qc[i, s, a] = (1-self.alpha)*self.Qc[i, s, a] + alpha*0.99*(self.Qr[a_, s] + self.Qc[i, s, a_])
#                 valid_actions = [3, 4, 5].remove(i)
                a = a_
            self.done = False
        self.sum += score
        return retStates
                
    # This is the main function that is called. It is recursive and calls itself while travelling in the hierarchy.
    def MAXQ_HO(self, i, s, task):
        visitedStates = []
        ################################# PRIMITIVE ACTIONS ###############################################
        if self.isPrimitive(i):
            self.SHS(i)
            reward = self.step(i, task)
            self.Qr[i, s] = (1-self.alpha)*self.Qr[i, s] + self.alpha*reward
            visitedStates.insert(0, s)
            self.PlotGraph.append([task, i, reward, self.Qr[i, s], 0, self.curr_pmv, s])
            self.sum += reward

        else:
            ######################### COMPOSITE ACTIONS ##################################################
            N = 1
            while not self.isTerminal(i):
                a = self.getAction(i, s)
                s = self.nState[a] if i == self.root or a in [3, 4, 5] else s
                if i in [3, 4, 5]:
                    visitedStates = self.trainPMV(i, s, a)
                    self.bin.append((i, a, visitedStates, self.time.copy()))
                    break
                childSeq = self.MAXQ_HO(a, s, i)
                self.bin.append((i, a, childSeq, self.time.copy()))
                s_ = self.nState[i] if i in [0, 1, 2, 3, 4, 5] else self.rootState(a)
                a_ = self.argmax(i, s_)
                self.QrCopy = self.Qr.copy()

                if a == self.cont:
                    visitedStates, s = childSeq + visitedStates, s_
                    if self.time[i] == 39:
                        visitedStates.insert(0, s)
                        self.update(0, visitedStates, i, s_, a_) 
                        break
                    continue
                elif a == self.leave:    
                    self.update(0, visitedStates, i, s_, a_) 
                    s = s_
                N = 1# if i == self.root else N
                gamma = 0.9 if i in [self.root, self.adjustTH0, self.adjustTH1, self.adjustTH2] else self.gamma[i]
                self.QrCopy = self.Qr.copy()
                
                for vS in self.trueStates(i, childSeq, a):
                    state = self.mapUP(i, vS, a) if i != self.root else vS # Child Node to Parent Node Mapping
                    if i == self.root or a in [3, 4, 5]:
                        self.Qr[a, state] = self.EvaluateRecursive(a, state)
                    if a != self.leave:
                        self.Qc[i, state, a] = (1-self.alpha)*self.Qc[i, state, a] + self.alpha*gamma**N*(self.Qr[a_, s_] + self.Qc[i, s_, a_] + self.Qe[i, s_, a_])# + self.Qc[i, s_, a_])
                    else:
                        self.Qc[i, state, a] = 0
                    if a != a_ and i == self.root:   
                        Qroot = self.Qr[a_, s_] + self.Qc[i, s_, a_]
                        exitState = self.mapDown(a, vS)
                        self.Qe[a, exitState, self.leave] = (1-self.alpha)*self.Qe[a, exitState, self.leave] + self.alpha*(1-self.gamma[a])**N*(Qroot)# + self.Qc[i, s_, a_])
                    N = N + 1 if i not in [3, 4, 5] else 1
                    visitedStates.insert(0, state)
                s = s_
            self.done = False
        return visitedStates
