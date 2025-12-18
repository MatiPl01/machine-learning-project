# Quick Fix for Benchmark Notebook

## The Issue
The model expects 36 features (28 + 8 PE) but receives only 9 features.

## Solution
Add this diagnostic cell right after loading the dataset (before precomputing PE):

```python
# DIAGNOSTIC: Check actual feature dimensions
sample = dataset[0]
print(f"Sample graph:")
print(f"  Nodes: {sample.num_nodes}")
print(f"  Node features shape: {sample.x.shape}")
print(f"  Feature dimension: {sample.x.shape[1]}")
print(f"  Has PE: {hasattr(sample, 'pe')}")
```

Then update the model configuration based on the output:

```python
# If ZINC has 9 features (common), update this:
ACTUAL_IN_CHANNELS = sample.x.shape[1]  # Use actual dimension

# Then when creating models, use:
goat_model = GOAT(
    in_channels=ACTUAL_IN_CHANNELS,  # <-- Use detected dimension
    hidden_channels=64,
    out_channels=1,
    num_layers=3,
    num_heads=4,
    num_virtual_nodes=1,
    pe_dim=PE_DIM,
    dropout=0.1,
    task_type='graph_classification',
).to(device)
```

## Quick Fix for Your Current Notebook

Replace the GOAT model creation cell with:

```python
# Create GOAT model
print("=" * 80)
print("TRAINING GOAT MODEL")
print("=" * 80)

# Auto-detect feature dimension
sample_data = train_loader.dataset[0] if hasattr(train_loader.dataset, '__getitem__') else dataset[0]
ACTUAL_IN_CHANNELS = sample_data.x.shape[1]
print(f"\nDetected node feature dimension: {ACTUAL_IN_CHANNELS}")
print(f"Positional encoding dimension: {PE_DIM}")
print(f"Total input dimension: {ACTUAL_IN_CHANNELS + PE_DIM}")

goat_model = GOAT(
    in_channels=ACTUAL_IN_CHANNELS,  # Auto-detected
    hidden_channels=64,
    out_channels=1,
    num_layers=3,
    num_heads=4,
    num_virtual_nodes=1,
    pe_dim=PE_DIM,
    dropout=0.1,
    task_type='graph_classification',
).to(device)

# Count parameters
goat_params = count_parameters(goat_model)
print(f"\nGOAT Model:")
print(f"  Total parameters: {goat_params['total']:,}")
print(f"  Trainable parameters: {goat_params['trainable']:,}")
print(f"  Model size: {goat_params['total_mb']:.2f} MB")

# Setup training
goat_optimizer = torch.optim.Adam(goat_model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

# Complexity tracker
goat_tracker = ComplexityTracker(goat_model, device)

print(f"\nStarting training for {NUM_EPOCHS} epochs...")
```

Do the same for Exphormer!


