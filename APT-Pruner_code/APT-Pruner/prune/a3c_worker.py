import torch
import torch.multiprocessing as mp
import torch.optim as optim
from torch.distributions import Bernoulli
from reward_enhanced import HybridRewardCalculator


class A3CPruningWorker(mp.Process):
    def __init__(self, worker_id, args, global_net, optimizer, static_prior, prior_stats, dataset):
        super(A3CPruningWorker, self).__init__()
        self.worker_id = worker_id
        self.args = args
        self.global_net = global_net
        self.optimizer = optimizer  # Shared optimizer

        # Stage 1 Data
        self.static_prior = static_prior  # Matrix I
        self.prior_stats = prior_stats  # mu, sigma

        self.dataset = dataset
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    def construct_state(self, features, origin_index_map, layer_idx):
        """
        对应 Sec 3.3.1: State Construction
        s_{t,i} = Concat(Normalized_Prior, Global, Local, Pos)
        """
        B, N_t, D = features.shape

        # 1. Normalized Static Prior (Use Origin Map to look up)
        # I_p where p = P_{t-1}[i]
        # static_prior: [Total_N]
        batch_indices = origin_index_map  # [B, N_t]
        raw_priors = self.static_prior[batch_indices]  # [B, N_t]
        norm_priors = (raw_priors - self.prior_stats['mu']) / (self.prior_stats['sigma'] + 1e-6)
        norm_priors = norm_priors.unsqueeze(-1)  # [B, N_t, 1]

        # 2. Global Context (Mean pooling)
        global_ctx = features.mean(dim=1, keepdim=True).expand(-1, N_t, -1)  # [B, N_t, D]
        # (Assuming projection happens inside PolicyNet for simplicity, or add explicit projection here)

        # 3. Local Features
        local_ctx = features  # [B, N_t, D]

        # 4. Positional Encoding
        # In a real implementation, you would look up the sinusoidal PE based on origin_index_map
        # Here we simulate/placeholder it using the indices normalized
        pos_enc = origin_index_map.unsqueeze(-1).float() / 197.0  # [B, N_t, 1] (Simplified)

        # Concatenate: [B, N_t, 1 + D + D + 1] -> adjusted to match PolicyNet input
        # Note: Ideally, dimension reduction MLPs (phi_glob, phi_loc) should be used here
        # to match d_glob, d_loc args. We assume pre-processing or net handles it.
        state = torch.cat([norm_priors, global_ctx, local_ctx, pos_enc], dim=-1)
        return state

    def run(self):
        # Local Network
        local_net = self.args.policy_class(
            self.args.d_glob, self.args.d_loc, self.args.d_pos
        ).to(self.device)

        reward_calc = HybridRewardCalculator(budget_ratio=self.args.budget_ratio)
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.args.batch_size, shuffle=True)

        step = 0
        while step < self.args.max_steps:
            try:
                inputs, targets = next(iter(dataloader))
            except StopIteration:
                dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.args.batch_size, shuffle=True)
                inputs, targets = next(iter(dataloader))

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Sync with global
            local_net.load_state_dict(self.global_net.state_dict())

            # --- Episode Start ---
            # Simulate generic ViT forward pass with Pruning
            # Initial State
            # N_0 = 197, P_0 = [0, ..., 196]
            curr_features = self.args.get_initial_embeddings(inputs)  # Placeholder
            B, N, D = curr_features.shape
            origin_map = torch.arange(N).unsqueeze(0).expand(B, -1).to(self.device)

            log_probs = []
            values = []
            rewards = []
            entropies = []

            # Layer-wise Loop (MDP)
            for t in range(self.args.num_layers):
                # 1. Construct State
                state = self.construct_state(curr_features, origin_map, t)

                # 2. Action (Asymmetric Training: Bernoulli Sampling) [Sec 3.3.2]
                logits, value = local_net(state)
                probs = torch.sigmoid(logits)
                dist = Bernoulli(probs)

                action_mask = dist.sample()  # [B, N_t, 1]
                # Force keeping at least 1 token (or CLS)
                if action_mask.sum() == 0: action_mask[:, 0] = 1.0

                # 3. Sequence Compaction [Sec 3.3.3]
                # Using Gather to physically remove tokens
                # For simplicity in this snippet, we assume batch_size=1 or handle variable lengths via masking
                # Here we strictly follow "Gather" logic for B=1 or padded B>1

                # Compute Next Features (Simulated Block)
                # In real code: next_features = TransformerBlock(curr_features masked)
                # Here we calculate reward based on hypothetical reconstruction
                base_features = self.args.get_base_features(inputs, t + 1)  # Oracle

                # 4. Compute Reward [Sec 3.3.4]
                # We need to perform compaction to get N_{t+1}
                kept_indices = action_mask.squeeze(-1).bool()

                # Simplified for Batch processing (Masking instead of physical gather for parallel training efficiency)
                # But logical N_t changes
                current_N = kept_indices.sum(dim=1)  # [B]

                r_immediate = reward_calc.compute_reward(
                    curr_features, base_features, origin_map, current_N.mean(), N
                )

                rewards.append(r_immediate)
                log_probs.append(dist.log_prob(action_mask))
                values.append(value)
                entropies.append(dist.entropy())

                # Update State for next layer (Simulated)
                # Update Origin Map P_{t} = Gather(P_{t-1})
                # curr_features = next_features

            # --- A3C Update (Accumulate Gradients) ---
            R = 0  # Terminal Reward (Task Loss difference)
            loss = 0

            # GAE / N-step return calculation
            for i in reversed(range(len(rewards))):
                R = rewards[i] + self.args.gamma * R
                advantage = R - values[i].mean()  # Using mean value for simplicity

                # Actor Loss
                loss = loss - (log_probs[i] * advantage.detach()).mean()
                # Critic Loss
                loss = loss + 0.5 * (R - values[i].mean()).pow(2)
                # Entropy
                loss = loss - self.args.entropy_coef * entropies[i].mean()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(local_net.parameters(), 40)

            # Push grads to global
            for lp, gp in zip(local_net.parameters(), self.global_net.parameters()):
                if gp.grad is None:
                    gp.grad = lp.grad
                else:
                    gp.grad += lp.grad

            self.optimizer.step()
            step += 1
            if step % 10 == 0:
                print(f"Worker {self.worker_id} Step {step} Loss {loss.item()}")