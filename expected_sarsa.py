import gym
import numpy as np
#loading evnironment
#this is off-policy TD method, behavior policy and target policy are diff
#epsilon softmax
class expected_sarsa:
    def __init__(self,epsilon,learning_rate,num_episodes,discount_factor):
         #loading environment
         self.env = gym.make('FrozenLake-v0')
         #inital the table of every state-action value,each state has number of actions value
         self.q_table = np.zeros([self.env.observation_space.n,self.env.action_space.n])
         self.lr = learning_rate
         self.df = discount_factor
         self.num_episodes = num_episodes
         self.epsilon = epsilon
    
    #generate the policy of a state
    #softmax
    def generate_target_policy(self,state):
        base_rate = (1-self.epsilon)/len(state)
        max_value = np.amax(state)
        func = lambda x:1 if x == max_value else 0
        policy_list = list(map(func,state))
        #get max arg length
        argmax_length = policy_list.count(1)
        max_rate = self.epsilon/argmax_length
        #get epsilon softmax policy, max_action = epsilon/num of max action + (1 - epsilon)/num of action, normal_action = (1 - epsilon)/num of action
        func = lambda x:max_rate + base_rate if x == 1 else base_rate
        #print(list(map(func,policy_list)),' ',state)
        return list(map(func,policy_list))
    
    def update_table(self,state,target_staete,action,reward,terminate):
      if not terminate:
        #the action distribution of target state
        policy_distribution = self.generate_target_policy(self.q_table[target_staete,:])
        expected_target_state_value = np.dot(list(self.q_table[target_staete,:]),policy_distribution)
        #update value, Q(St,A) = Q(S,A) + lr*(reward+ df * Q(expexted_St+1) - Q(S,A))
        self.q_table[state,action] += self.lr *(reward + self.df*expected_target_state_value - self.q_table[state,action])
      else:
        #if target state is terminate, Q(S target) = 0
        self.q_table[state,action] += self.lr *(reward - self.q_table[state,action])
    
    
    def rl(self):
        rList = []
        for i in range(self.num_episodes):
           #initial the enviornment
           state = self.env.reset()
           terminate = False
           episode = 0
           #maximun 100 steps
           steps = 0
           self.env.step(self.env.action_space.sample())
           while steps < 100:
               #refresh evn
               #self.env.render()
               #epsilon greedy
               if self.epsilon > np.random.uniform(0, 1):
                  action = np.argmax(self.q_table[state,:])
               else:
                  action = self.env.action_space.sample()
               #interact with environment, excute action
               target_state, reward, terminate, info = self.env.step(action)
               if target_state == 15:
                   reward = 1
               #update q table by using expected sarsa
               self.update_table(state,target_state,action,reward,terminate)
               #change current state
               state = target_state
               steps += 1
               if terminate:
                   #print(reward)
                   steps += 1
                   #self.env.render()
                   break
           if state == 15:
               print('total step of episode %dth is: %d'%(i,steps))
               

def main():
    s = expected_sarsa(0.90,0.9,10000,0.9)
    s.rl()
main()