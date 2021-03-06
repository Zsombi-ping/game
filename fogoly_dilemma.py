class player:
    
    def __init__(self,name,order,strategySpace,payoffs,choice,suboptimal,strategies,state,gameplay):
        self.name = name
        self.order = order
        self.strategySpace = strategySpace
        self.payoffs = payoffs
        self.choice = choice
        self.suboptimal = suboptimal
        self.strategies = strategies
        self.state = state
        self.gameplay = gameplay


    def processGame(self,G):
        for i in range(0,len(G)):
            X = G[i]
            if X[0] == self.name:
                for j in range(1,len(X)):
                    Branch = X[j]
                    Alternative = list(Branch)
                    del Alternative[len(Alternative) - 1]
                    self.strategySpace = self.strategySpace + [tuple(Alternative)]
                    self.payoffs = self.payoffs + [Branch[len(Branch) - 1]]
        
    def evaluate(self):
        X = []
        for i in range(0,len(self.strategySpace)):            
            Alternative1 = self.strategySpace[i]
            for j in range(0,len(self.strategySpace)):
                Alternative2 = self.strategySpace[j]
                if Alternative1 != Alternative2:
                    if len(Alternative1) == len(Alternative2):
                        Compare = 0
                        for k in range(0,len(Alternative1) - 1):
                            if Alternative1[k] == Alternative2[k]:
                                Compare = Compare + 0
                            else:
                                Compare = Compare + 1
                        if Compare == 0:
                            PayoffCompare = [self.payoffs[i],self.payoffs[j]]
                            M = max(PayoffCompare)
                            if self.payoffs[i] == M:
                                self.choice = Alternative1
                                X = X + [self.choice]
                            else:
                                self.suboptimal = self.suboptimal + [Alternative1]
                            if self.payoffs[j] == M:
                                self.choice = Alternative2
                                X = X + [self.choice]
                            else:
                                self.suboptimal = self.suboptimal + [Alternative2]
            
        X = set(X)
        self.suboptimal = set(self.suboptimal)
        self.strategies = list(X - self.suboptimal)
        print ("\nStrategies selected by ", self.name," : ")
        print (self.strategies)
        for l in range(0,len(self.strategies)):
            strategy = self.strategies[l]
            for m in range(0,len(strategy)):
                O = self.order[m]
                self.state[O] = strategy[m]
            self.gameplay = self.gameplay + [tuple(self.state)]
                    
        
        

class game:
    def __init__(self,players,structure,optimal):
        self.players = players
        self.structure = structure
        self.optimal = optimal

    def Nash(self,GP):
        Y = set(GP[0])
        for i in range(0,len(GP)):
            X = set(GP[i])
            Y = Y & X
        self.optimal = list(Y)
        if len(self.optimal) != 0:
          print ("\nThe pure strategies Nash equilibria are:")
          for k in range(0,len(self.optimal)):
              print (self.optimal[k])
        else:
            print ("\nThis game has no pure strategies Nash equilibria!")
            



GameA = ['A', # Player A
         ('C','C',3), # When B cooperates: if A cooperates, A receives 3 units of political gains
         ('C','D',5), # When B cooperates: if A defects, A receives 5 units of political gains
         ('D','C',0), # When B defects: if A cooperates, A receives 0 units of political gains
         ('D','D',1)] # When B defects: if A defects, A receives 1 unit of political gains
GameB = ['B', # Player B
         ('C','C',3), # When A cooperates: if B cooperates, B receives 3 units of political gains
         ('C','D',5), # When A cooperates: if B defects, B receives 5 units of political gains
         ('D','C',0), # When A defects: if B cooperates, B receives 0 units of political gains
         ('D','D',1)] # When A defects: if B defects, B receives 1 unit of political gains

#game(players,structure,plays,optimal)
Game = game(('A','B'),[GameA,GameB],None)


#player(self,name,order,strategySpace,payoffs,choice,suboptimal,strategies,state,gameplay):
PlayerA = player('A',(1,0), [], [], None, [], None, [0,0], [])
PlayerB = player('B',(0,1), [], [], None, [], None, [0,0], [])


Players = [PlayerA, PlayerB]

for i in range(0,len(Players)):
    Players[i].processGame(Game.structure)
    Players[i].evaluate()

GP = []

for i in range(0,len(Players)):
    X = Players[i].gameplay
    GP = GP + [X]

Game.Nash(GP)
