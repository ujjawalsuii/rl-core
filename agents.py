import numpy as np

class ActorCriticLambdaAgent:
    """
    Actor-Critic with Eligibility Traces (AC-λ).
    Implements backward-view TD(λ) for efficient credit assignment.
    Reference: Sutton & Barto, Sec 13.5.
    """
    def __init__(self, num_features, num_actions, alpha_theta=1e-3, alpha_w=1e-2, gamma=0.99, lambd=0.8):
        self.num_features = num_features
        self.num_actions = num_actions
        self.gamma = gamma
        self.lambd = lambd  # Trace decay rate (λ)
        
        # Policy (Actor)
        self.theta = np.zeros((num_features, num_actions))
        self.alpha_theta = alpha_theta
        self.z_theta = np.zeros((num_features, num_actions)) # Eligibility Trace for Actor

        # Value Function (Critic)
        self.w = np.zeros(num_features)
        self.alpha_w = alpha_w
        self.z_w = np.zeros(num_features) # Eligibility Trace for Critic

        self.I = 1.0 # Discount factor tracker

    def softmax(self, x):
        """Numerically stable softmax"""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def select_action(self, features):
        """Sample action from policy π(a|s)"""
        logits = np.sum(self.theta[features], axis=0)
        probs = self.softmax(logits)
        return np.random.choice(self.num_actions, p=probs)

    def learn(self, state_features, action, reward, next_state_features, done):
        """
        Update with Eligibility Traces (Backward View).
        """
        # 1. Critic Evaluation
        current_val = np.sum(self.w[state_features])
        if done:
            next_val = 0
            td_error = reward - current_val
        else:
            next_val = np.sum(self.w[next_state_features])
            td_error = reward + self.gamma * next_val - current_val

        # 2. Update Critic Trace (Accumulating Trace)
        # Decay existing traces: z <- γλz
        self.z_w *= (self.gamma * self.lambd)
        # Add current features to trace: z <- z + 1 (for active features)
        self.z_w[state_features] += 1.0

        # 3. Update Critic Weights (w)
        # w <- w + α * δ * z
        self.w += self.alpha_w * td_error * self.z_w

        # 4. Update Actor Trace
        logits = np.sum(self.theta[state_features], axis=0)
        probs = self.softmax(logits)
        
        # Gradient of ln_pi
        grad_ln_pi = -probs
        grad_ln_pi[action] += 1.0
        
        # Decay Actor Trace
        self.z_theta *= (self.gamma * self.lambd)
        
        # Add current gradient to trace (only for active features)
        for feat_idx in state_features:
            self.z_theta[feat_idx] += self.I * grad_ln_pi

        # 5. Update Actor Weights (theta)
        # θ <- θ + α * δ * z_theta
        self.theta += self.alpha_theta * td_error * self.z_theta

        # 6. Update Discount Factor
        if done:
            self.I = 1.0
            self.reset_traces()
        else:
            self.I *= self.gamma

    def reset_traces(self):
        """Reset traces at start of episode"""
        self.z_theta.fill(0)
        self.z_w.fill(0)
