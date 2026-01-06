import numpy as np

class ActorCriticAgent:
    """
    One-Step Actor-Critic (Episodic) implementation.
    Uses linear function approximation with Tile Coding.
    """
    def __init__(self, num_features, num_actions, alpha_theta=1e-3, alpha_w=1e-2, gamma=0.99):
        self.num_features = num_features
        self.num_actions = num_actions
        self.gamma = gamma
        
        # ACTOR: Policy parameters (theta)
        self.theta = np.zeros((num_features, num_actions))
        self.alpha_theta = alpha_theta # Learning rate for actor
        
        # CRITIC: Value function weights (w)
        self.w = np.zeros(num_features)
        self.alpha_w = alpha_w # Learning rate for critic

        self.I = 1.0 # Discount factor tracker

    def softmax(self, x):
        """Numerical stable softmax"""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def select_action(self, features):
        """Sample action from policy π(a|s)"""
        # Sum weights for active features
        logits = np.sum(self.theta[features], axis=0)
        probs = self.softmax(logits)
        return np.random.choice(self.num_actions, p=probs)

    def learn(self, state_features, action, reward, next_state_features, done):
        """
        Update Actor and Critic weights based on TD-Error.
        """
        # 1. Critic Estimate V(s)
        current_val = np.sum(self.w[state_features])
        
        # 2. Critic Estimate V(s')
        if done:
            next_val = 0
        else:
            next_val = np.sum(self.w[next_state_features])
            
        # 3. TD Error (delta) = r + gamma * V(s') - V(s)
        td_target = reward + self.gamma * next_val
        delta = td_target - current_val
        
        # 4. Update Critic (w)
        # w <- w + alpha_w * delta * gradient(V)
        # Gradient of linear V is just the features (1s at active indices)
        self.w[state_features] += self.alpha_w * delta
        
        # 5. Update Actor (theta)
        # theta <- theta + alpha_theta * I * delta * gradient(ln pi)
        logits = np.sum(self.theta[state_features], axis=0)
        probs = self.softmax(logits)
        
        # Gradient of ln_pi is (1 - pi) for taken action, -pi for others
        grad_ln_pi = -probs
        grad_ln_pi[action] += 1
        
        # Update only active features for the actor
        for feat_idx in state_features:
            self.theta[feat_idx] += self.alpha_theta * self.I * grad_ln_pi
            
        # Update discount tracker
        if done:
            self.I = 1.0
        else:
            self.I *= self.gamma
