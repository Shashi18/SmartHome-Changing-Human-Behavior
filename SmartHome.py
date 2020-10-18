states = []
for task in range(3):
    t = 15
    while True:
        for i in range(30, 71, 5):
            pairState = int(str(int(t*2))+str(i))+(task)
            #print(t, i, pairState, k)
            states.append(int(pairState))
            k += 1
        t += 0.5
        if t > 40:
            break
            
class AI:
    def __init__(self, Human, states, gamma=0.99, lr=0.01, train = True):
        self.temp = Human.temp
        self.hum = Human.hum
        self.activity = 0
        self.Q = np.zeros([1900, 5])
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
        self.Human = Human
    
    def reset(self):
        self.temp = 15
        self.hum = 50
        self.score = 0
    
    def freeze(self):
        self.alpha = 0
        self.train = False
        
    def getState(self):
        #val = int(str(self.temp)+str(self.hum))+self.activity
        self.temp = self.Human.temp
        self.hum = self.Human.hum
        self.activity = self.Human.taskStack[-1]
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
        q = 4*x
        self.alpha = self.alpha_0*math.exp(-q/eps)
        self.epsilon = max(0.001, 0.5*math.exp(-q/eps))     
    
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
