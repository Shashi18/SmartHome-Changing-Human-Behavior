##########################################################
# Human Model: Hierarchical Reinforcement Learning       #
# Code Author: Shashi Suman                              #
# Affiliations: Queen's University, Kingston             #
# Date: 20th August 2020                                 #
# Code used in: AAAI 21 Workshop, IJCAI 21               #
##########################################################

# 					                Root
#                                             		  | 
#                    _____________________________________|____________________________________
#                    |	 				  |                                   |
# 	      [Activity 0]		             [Activity 1]                        [Activity 2]
#      ______________|_____________        _______________|______________	  ____________|__________
#      |             |	          |        | 		  |	        |	  |	      |		|
#  [Continue]	  [Leave]      	  |    [Continue]         |          [Leave]      |       [Continue]  Leave]
# 				  |			  |			  |
# 				  |_______________________|_______________________|
# 				  			  |
# 						     [Set Point]
# 				         _________________|____________________
# 				         |            |           |            |
#                                      [I Temp]    [D Temp]     [I Hum%]     [D Hum%]

# State space for the Root node (Node 6) #
rootState = [(task << 10) | time for time in range(40) for task in range(3)]

# State space for the Task nodes (Nodes 0, 1, 2) #
taskState = [(task << 40) | (time << 20) | (temp << 10) | hum for hum in range(30, 71, 5) for temp in range(15, 31) for time in range(40) for task in range(3)]

# State space for the SetPoint nodes (Nodes 3, 4, 5, 6) #
PMVstate = [(temp << 10) | hum for hum in range(30, 71, 5) for temp in range(15, 31)]


# Human HRL Model #
class HumanA:
    def __init__(self, lr=0.01, gamma=0.99, metabolism = [1, 1, 1]):
        self.temp = 15
        self.pA_npA = nA = 13 #Root + Phy + Watch + P_Cont + P_Leave + W_Cont + W_Leav
        self.statespace = SS = np.max(len(rootState), len(taskState), len(PMVstate)) #Should be 16200
        self.Qr = np.zeros((nA, SS))
        self.Qc = np.zeros((nA, SS, nA))
        self.Qe = np.zeros((nA, SS, nA))
        self.epsilon = 0.1
        self.alpha = lr
        self.alpha_0 = lr
        self.gamma = gamma
        self.done = False
        self.time = [0, 0, 0]
        self.nextState = [0, 5760, 11520, 0, 0, 0]
        self.QrCopy = self.Qr.copy()
        self.taskStack = []
        self.continueStack = []
        self.setPointStack = set()
        self.MetSet = metabolism
        self.absPMV = 0
        self.hum = 50
        self.window = 110
        self.prev = 0
        self.visitedRootState = []
    
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
		
	self.taskSet = [self.task0, self.task1, self.task2]
	self.setPointSet = [self.set0, self.set1, self.set2]

        self.graph = [
            (leave, cont, set0), # Node Activity 0
            (leave, cont, set1), # Node Activity 1
            (leave, cont, set2), # Node Activity 2
            (t0h0, t0h1, t1h0, t1h1), # Node Increase/Decrease Temperarure, Increase/Decrease humidity setpoint
            (t0h0, t0h1, t1h0, t1h1), # Node Increase/Decrease Temperarure, Increase/Decrease humidity setpoint
            (t0h0, t0h1, t1h0, t1h1), # Node Increase/Decrease Temperarure, Increase/Decrease humidity setpoint
            (activity0, activity1, activity2), # Node Root --> Activity 0, Activity 1, Activity 2
            set(), # Increase Temperature Setpoint
            set(), # Decrease Temperature Setpoint
            set(), # Increase Humidity Setpoint
            set(), # Decrease HUmidty Setpoint
            set(), # Continue Task
            set()  # Leave Task
        ]

    def reset(self, temp):
        self.temp = random.randint(20, 25)
        self.hum = int(random.choice([i for i in range(40, 60, 5)]))
        self.time = [0, 0, 0]
        self.cTime = []
        self.nextState = [0, 5760, 11520, 0, 0, 0]
        self.prev = 0
        self.done = False
        self.taskStack = []
        self.continueStack = []
        self.setPointStack = set()
        self.absPMV = 0
        self.window = 130
        self.prev = 0
        self.visitedRootState = []
        self.NextRootState = 0
      
    # Get the Root node state
    def rootState(self, task):
        var = (task << 10) | self.time[task]
        return rootState.index(var)

    # Get the PMV(Predicted Mean Vote) for current Temperature and Humidity
    def getPmv(self, task):
        # Get current temperature/ humidity from environment 
        temp = env.getTemperature()
        hum = env.getHumidity()
        metabolic = self.MetSet[task]
        pmv =  pmv_ppd(tdb=temp, tr=25, vr=0.0, rh=hum, met=metabolic, clo=0.5, wme=0, standard="ASHRAE")['pmv']
        terminal = 0 if -0.5<=pmv<=0.5 else -1 if (0.5<pmv<=1 or -1<=pmv<-0.5) else -2 if(1<pmv<=1.5 or -1.5<=pmv<-1) else -3
        return terminal, pmv

    # Get the current state of Task Nodes
    def getTaskState(self, time, task):
        x = (task << 40) | (time << 20) | (self.temp << 10) | self.hum
        state = taskState.index(x)
        return state

    # Penalty for Root Node #    
    def cost(self, time, task):
        p0 = [3 if 0<=i<7 else 2 if 7<i<22 else 10 if 22<i<39 else 0 for i in range(40)]
        p1 = [10 if(0<=i<3) else 1.5 if (3<i<10 or 18<i<30) else 8 if (9<i<18) else 3.5 if(30<=i<39) else 0 if (i==18 or i==33 or i==39) else 0  for i in range(40)]
        p2 = [6 if 0<=i<7 else 2 if 7<i<22 else 15 if 22<i<39 else 0 for i in range(40)]
        penalty = [p0, p1, p2]
        return -penalty[task][time]
    
    # Task/Node 0 #
    def task_0(self, action):
        reward = [0 for i in range(40)]
        penalty = [3 if 0<=i<7 else 2 if 7<i<22 else 10 if 22<i<39 else 0 for i in range(40)]
        
        reward[5], reward[6], reward[20], reward[21], reward[37], reward[38], reward[39] = 10, 10, 5, 5, 15, 15, 0

        self.setPointStack.discard(self.set1)
        self.setPointStack.discard(self.set2)

        if action==self.cont:
            self.window = max(0, self.window-1)
            if self.time[0] == 39:
                score = reward[self.time[0]] 
                self.done = True
            else:
                self.time[0] += 1
                self.nextState[0] = self.getTaskState(self.time[0], 0)
                score = reward[self.time[0]] + self.getPmv(0)[0]
                self.done = False
        elif action==self.leave:
            self.pmv_time = 0
            self.window = max(0, self.window-1)
            self.done = True
            score = -penalty[self.time[0]]
        return score
            
    # Task/Node 1 #
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

        self.setPointStack.discard(self.set0)
        self.setPointStack.discard(self.set2)

        if action==self.cont:
            self.window = max(0, self.window-1)
            if self.time[1] == 39:
                score = reward[self.time[1]]
                self.done = True
            else:
                self.time[1] += 1
                self.nextState[1] = self.getTaskState(self.time[1], 1)
                score = reward[self.time[1]] + self.getPmv(1)[0]
                self.done = False
        elif action==self.leave:
            self.pmv_time = 0
            self.window = max(0, self.window-1)
            score = -penalty[self.time[1]]
            self.done = True
        
        return score

    # Task/Node 2 #
    def task_2(self, action):

        reward = [0 for i in range(40)]
        penalty = [6 if 0<=i<7 else 2 if 7<i<22 else 15 if 22<i<39 else 0 for i in range(40)]
        reward[5], reward[6], reward[20], reward[21], reward[37], reward[38], reward[39] = 15, 15, 7, 7, 30, 30, 0
        
        self.setPointStack.discard(self.set0)
        self.setPointStack.discard(self.set1)
        if action==self.cont:
            self.window = max(0, self.window-1)
            if self.time[2] == 39:
                score = reward[self.time[2]]
                self.done = True
            else:
                self.time[2] += 1
                self.nextState[2] = self.getTaskState(self.time[2], 2)
                score = reward[self.time[2]] + self.getPmv(2)[0]
                self.done = False
        elif action==self.leave:
            self.pmv_time = 0
            self.window = max(0, self.window-1)
            score = -penalty[self.time[2]]
            self.done = True
        
        return score

    # SetPoint/ Node 3, 4, 5, 6
    def changeTH(self, action):
        task = self.taskStack[-1]
        # D_pmv: Discrete PMV, C_pmv: Continuous PMV
        D_pmv, C_pmv = self.getPmv(task) 
        self.window = max(0, self.window-1)
        if D_pmv == 0:
            reward, self.done = -5, True
            self.setPointStack.add(task+3)
        else:
            if action == self.t0h0:
                self.hum = max(30, self.hum - 5)
            elif action == self.t0h1:
                self.hum = min(70, self.hum + 5)
            elif action == self.t1h0:
                self.temp = min(29, self.temp + 1)
            elif action == self.t1h1:
                self.temp = max(15, self.temp - 1)

            # Send the setpoints to the Environment
            env.step(self.temp, self.hum)
            self.nextState[task+3] = PMVstate.index((self.temp << 10) | self.hum)
            
            D_pmv, C_pmv = self.getPmv(task)
            self.absPMV = -abs(C_pmv)
            reward = 1 if self.absPMV-self.prev > 0 else 0
            
            if D_pmv == 0:
                self.setPointStack.add(task+3)
                self.prev = 0
                self.nextState[task] = self.getState(self.time[task], task)
                self.done = True
            else:
                self.prev = self.absPMV
                self.done = False
        return reward

    # Check if the action is a primitive action ? #
    def isPrimitive(self, i):
        return True if i > 6 else False
    
    # Take action for Node X #
    def step(self, action, node):
        return self.task_0(action) if node==0 else self.task_1(action) if node==1 else self.task_2(action) if node==2 else self.changeTH(action)
                
    # Check if we are in the terminal state of the Node X #
    def HasNodeTerminated(self, i):
        if self.window==0:
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
            
    # Get action for state s in Node i #
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
#                 if act in self.setPointStack:
#                     continue
                Q = np.concatenate((Q, [self.Qr[act, s] + self.Qc[i, s, act] + self.Qe[i, s, act]]))
                valid_actions = np.concatenate((valid_actions, [act])) 

        # Keep only finished tasks in TaskStack #
        if i==self.root and self.taskStack:
            for task in self.taskStack:
                if self.time[task]!=39:
                    self.taskStack.remove(task)
                    
        if random.random() <= self.epsilon:
            return int(np.random.choice(valid_actions))
        else:
            return int(valid_actions[np.argmax(Q)])
        
    # Learning Rate/ Epsilon Episodic Decay #
    def decay(self, x, eps):
        q = 6*x
        self.epsilon = max(0.001, math.exp(-q/eps))
        self.alpha = self.alpha_0*math.exp(-q/eps)
    
    # Recursively Evaluate the Value Function -> Qr only for Non Primitive actions # 
    def evalValueFunct(self, i, s):
        if self.isPrimitive(i):
            return self.QrCopy[i, s]
        else:
            for action in self.graph[i]:
                state = self.mapState(action, s)
                self.QrCopy[action, state] = self.evalValueFunc(action, state)
            a = np.argmax(self.QrCopy[:, state] + self.Qc[i, state, :])
            return self.QrCopy[a, state]

    
    # Map Child state to Parent state #
    def mapState(self, action, state):
        if action in [0, 1, 2]:
            encodedState = rootState[state]
            time = encodedState & 0xff
            var = (action << 40) | (time << 20) | (self.temp << 10) | self.hum
            return taskState.index(var)
        elif action in [3, 4, 5]:
            encodedState = taskState[state]
            temp = (encodedState >> 10) & 0xff
            hum = encodedState & 0xff
            var = (temp << 10) | hum
            return PMVstate.index(var)
        elif self.isPrimitive(action):
            return state
   
    # Next Possible action for state S for Node i #
    def possibleAction(self, i, s):
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
                Q = np.concatenate((Q, [self.Qr[act, s] + self.Qc[i, s, act]]))
            actions = np.concatenate((actions, [act]))
        return int(actions[np.argmax(Q)])

    

    # MAX Q: Hierarchical Optimal Algorithm #                                                      
    def MAXQ_HO(self, currentNode, state, task):                                                       # #
        visitedStates = []                                                                                 #
        if self.isNodePrimitive(currNode):                                                                 #
            reward = self.step(currentNode, task)                                                           # # If i is a primitive action
            # Update the Value Function #                                                                  #
            self.Qr[currentNode, state] = (1-self.alpha)*self.Qr[currentNode, state] + self.alpha*reward   #
            visitedStates.insert(0, state)                                                             # #                                                     
																								   # #
        else:
            # Check if the Node been terminated #
            while not self.HasNodeTerminated(currentNode):
                action = self.getAction(currentNode, state)
                if currentNode==self.root:
                    self.taskStack.append(action)
                    self.continueStack = []
                    self.visitedRootState = []
                    state = self.nextState[action]
                    
                elif currentNode in self.taskSet and len(self.continueStack)==0:
                    action = self.cont
                elif currentNode in self.taskSet and action in self.setPointSet:
                    state = self.nextState[action]

                
                childSeq = self.MAXQ_HO(action, state, currentNode)
                if action==self.cont:
                    #Insert states if action is continue
                    self.continueStack.insert(0, childSeq[0])
                    #Insert mapped root states in root stack
                    self.visitedRootState.insert(0, self.rootState(i))
                    state = self.nextState[currentNode]
                    continue
                if action==self.leave:
                    #Insert mapped root states in root stack
                    self.visitedRootState.insert(0, self.rootState(currentNode))
                if action in [self.t0h0, self.t0h1, self.t1h0, self.t1h1]:
                    self.tempStack.insert(0, childSeq[0])
                    state = self.nextState[currentNode]
                    visitedStates = self.tempStack.copy()
                    continue

                N = 1

                self.QrCopy = self.Qr.copy()

                # Update Q Matrix for Root Node #
                for currentState in self.visitedRootState:
                    if currentNode!=self.root:
                        break
                    if np.mean(self.time) != 39:
                        nextAction = self.possibleAction(self.root, self.visitedRootState[0])
                        nextState = self.rootState(nextAction)
                        QRoot = self.evalValueFunc(self.root, currentState) + self.Qc[self.root, nextState, nextAction]
                        self.Qc[self.root, cS, a] = (1-self.alpha)*self.Qc[self.root, cS, a] + self.alpha*0.98**N*QRoot
                    else:
                        self.Qc[self.root, cS, a] = 0
                    
                    self.Qr[self.root, cS] = self.cost(self.time[action], action) + self.evalValueFunc(self.root, currentState)
                    N += 1

                # Update Q Matrix for Non-Root Node #
                for currentState in childSeq:
                    # Update Q Matrix for SetPoint node #
                    if i in [self.set0, self.set1, self.set2]:
                        cS, nS = state, self.nextState[i]  
                        A = self.possibleAction(i, nS)
                        self.Qr[i, cS] = self.evalValueFunc(i, cS)
                        self.Qc[i, cS, a] = (1-self.alpha)*self.Qc[i, cS, a] + self.alpha*self.gamma**N*(self.Qr[A, nS] + self.Qc[i, nS, A]) 

                    # Update Q Matrix for Task Nodes with SetPoint(Non Primitive) Actions #
                    elif i in [self.task0, self.task1, self.task2] and a in [self.set0, self.set1, self.set2]:
                        cS = state
                        nS = self.nextState[i]
                        
                        # Map SetPoint Node states to its parent node states
                        temp_state = PMVstate[cS]
                        temperature, humidity = (temp_state >> 10) & 0xff, temp_state & 0xff
                        temp_state = (i << 40) | (self.time[i] << 20) | (temperature << 10) | humidity
                        cS = taskState.index(temp_state)
                        
                        A = self.possibleAction(i, nS)
                        Qtype = self.Qr[A, nS] + self.Qc[i, nS, A]
                        self.Qr[i, cS] = self.evalValueFunc(i, cS)
                        self.Qc[i, cS, a] = (1-self.alpha)*self.Qc[i, cS, a] + self.alpha*self.gamma**N*Qtype
                    
                    # Update Q Matrix for Task Nodes with Leave(Primitive) Action #
                    elif i in [self.task0, self.task1, self.task2] and a==self.leave:
                        self.Qr[i, state] = self.evalValueFunc(i, state)
                        self.Qc[i, state, self.leave] = 0
                    N += 1

                # Update Q Matrix for ONLY Task Nodes with Continue action #
                N = 1
                for cS in self.continueStack:
                    if currentNode not in [self.task0, self.task1, self.task2]: # Prevent Double Update
                        break
                    indices = [0, 1, 2]
                    indices.remove(i)
                    nS = self.nextState[i]
                    a_ = self.possibleAction(i, nS) 
                    self.Qr[currentNode, cS] = self.evalValueFunc(currentNode, cS)
                    self.Qc[currentNode, cS, self.cont] = (1-self.alpha)*self.Qc[currentNode, cS, self.cont] + self.alpha*(self.gamma)**N*(self.Qr[a_, nS] + self.Qc[currentNode, nS, a_])

                    # Qe Update if Exit state is found: All states are exit states#
                    A = self.possibleAction(self.root, self.visitedRootState[0])
                    nS = self.rootState(A)
                    QRoot = self.Qr[A, nS] + self.Qc[self.root, nS, A]
                    self.Qe[currentNode, cS, self.leave] = (1-self.alpha)*self.Qe[currentNode, cS, self.leave] + self.alpha*(1-self.gamma)**N*QRoot
                    N += 1

                # Merge Continue and Leave State #
                if len(self.continueStack) > 0 and a==self.leave:
                    self.continueStack.insert(0, childSeq[0])
                    childSeq = self.continueStack.copy()

                # Return the visited states to the parent node
                for state in reversed(childSeq):
                    visitedStates.insert(0, state)
                s = self.nextState[currentNode] if currentNode!=self.root else nS
            self.done = False
        return visitedStates       
