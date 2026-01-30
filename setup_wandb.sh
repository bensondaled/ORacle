#!/bin/bash
# Setup script for Weights & Biases (wandb)
# Run this script once to authenticate with your wandb account

echo "Setting up Weights & Biases for ORacle..."
echo ""
echo "You will be prompted to log in to your wandb account."
echo "Use your email: marcgh@stanford.edu"
echo ""

# Run wandb login
wandb login

echo ""
echo "Wandb setup complete!"
echo "Your runs will appear at: https://wandb.ai/your-username/flexible_bp_autoreg"
