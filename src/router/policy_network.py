import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import List
class RouterPolicy(nn.Module):
    def __init__(self,input_dim:int,hidden_dim:int,output_dim:int):
        super(RouterPolicy,self).__init__()
        self.fc1 = nn.Linear(input_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = F.dropout(x, p=0.1, training=self.training)
        
        x = torch.relu(self.fc2(x))
        x = F.dropout(x, p=0.1, training=self.training)
        
        logits = self.fc3(x)
        probabilities = torch.softmax(logits, dim= -1)
        return probabilities


class PolicyAgent:
    
    def __init__(self, input_dim:int, hidden_dim :int, output_dim:int, learningrate:float = 0.001):
        self.policynetwork = RouterPolicy(input_dim, hidden_dim, output_dim)
        self.optimizer = torch.optim.Adam(self.policynetwork.parameters(), lr=learningrate)
        
    def select_action(self, feature_vector:List[float], evaluate:bool = False) -> int:
        """
        1. select_action(feature_vector) - Uses the policy network to select an action based on the input feature vector.
        
        Args:
            feature_vector: A 1D list of numerical features representing the current state.
            evaluate: If True, selects the action with the highest probability (greedy). If False, samples from the distribution.
            """
        if evaluate:
            self.policynetwork.eval()
            
        else:
            self.policynetwork.train()
            
        state_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)
        
        with torch.set_grad_enabled(not evaluate):
            action_probabilities = self.policynetwork(state_tensor)
            
        if evaluate:
            action = torch.argmax(action_probabilities, dim=-1).item()
            return action, torch.tensor(0.0)
        
        dist = torch.distributions.Categorical(action_probabilities)
        action_tensor = dist.sample()
        action = int(action_tensor.item())
        log_prob = dist.log_prob(action_tensor)
        
        return action, log_prob
    
    
    def update_policy(self, log_prob: torch.Tensor, reward: float):
        """Minimal single-sample REINFORCE update, for demonstration only (see the
        __main__ block below).

        This is NOT what any real experiment in this repo uses. The actual training
        loop (src/router/train_reinforce.py::ReinforceTrainer._train_epoch) batches
        log_probs across a full epoch, subtracts an EMA baseline from the raw reward,
        adds an entropy bonus, and clips gradients before stepping the optimizer — all
        of which are missing here. Do not use this method to understand "the" REINFORCE
        update rule; read train_reinforce.py instead.
        """
        self.optimizer.zero_grad()

        loss = - log_prob * reward
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
            MOCK_FEATURE_DIM = 10 
            
            print("Testing Policy Network Initialization...")
       
            agent = PolicyAgent(input_dim=MOCK_FEATURE_DIM, hidden_dim=32, output_dim=3)    
            
            simulated_query_feature_vector = [0.5] * MOCK_FEATURE_DIM
            
            action, log_prob = agent.select_action(simulated_query_feature_vector, evaluate=False)
            
            simulated_reward = 1.0
            
            agent.update_policy(log_prob, reward=simulated_reward)
            
            action_eval, _ = agent.select_action(simulated_query_feature_vector, evaluate=True) 
            
            retriever_mapping = {0: "BM25", 1: "Dense", 2: "Hybrid"}
            print(f"-> Optimal Selector Choice: {action_eval} ({retriever_mapping[action_eval]})")
            
            
    
    

