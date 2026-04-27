# Step 1: Generate outputs (no gradients needed here)
with torch.no_grad():
    outputs = model.generate(prompts, temp=0.7, n=8)
    log_probs_old = model.forward(prompts, outputs).logprobs  # π_θ_old

# Step 2: Compute rewards and advantages
rewards = reward_function(outputs, utility_estimates)
advantages = (rewards - rewards.mean()) / rewards.std()

# Step 3: Forward pass WITH gradients (this builds the computation graph)
log_probs = model.forward(prompts, outputs).logprobs  # π_θ, gradient-connected

# Step 4: Compute loss
ratio = torch.exp(log_probs - log_probs_old)
clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
loss = -torch.mean(torch.min(ratio * advantages, clipped_ratio * advantages))
loss += beta * kl_divergence(model, ref_model, prompts, outputs)

# Step 5: Backprop — PyTorch computes ∂loss/∂W for every LoRA weight
loss.backward()

# Step 6: Update weights
optimizer.step()   # W_new = W - lr × gradient (via AdamW)
optimizer.zero_grad()