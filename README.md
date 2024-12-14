# Steps to Run Locally
1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run tests:
   ```bash
   pytest tests/
   ```

4. Train model:
   ```bash
   python src/train.py
   ```

# To deploy to GitHub
1. Create a new GitHub repository
2. Initialize git in your local project:
   ```bash
   git init
   ```
3. Push your code to the new repository:
   ```bash
   git remote add origin https://github.com/your-username/your-repo.git
   git branch -M main
   git add .
   git commit -m "Initial commit"
   git push -u origin main
   ```

4. The GitHub Actions workflow will automatically trigger when you push to the repository. It will:
   - Set up the Python environment
   - Install dependencies
   - Run all tests to verify:
     - Model architecture
     - Parameter count
     - Input/output dimensions
     - Presence of batch normalization and dropout  

# Traning Logs
```
(venv) gitesh.grover@Giteshs-MacBook-Pro ai-era-assignment6 % python src/train.py                   

[INFO] Using device: mps
[STEP 1/5] Preparing datasets...
[INFO] Total training batches: 782
[INFO] Batch size: 64
[INFO] Training samples: 50000
[INFO] Test samples: 10000

[INFO] Validation samples: 10000

[STEP 2/5] Initializing model...
[INFO] Total parameters: 19340
[STEP 3-4/5] Starting training and Evaluation...

[INFO] Training of Epoch 1 started...
Epoch 1: 100%|████████████████████████████████████████████████████████████████████| 782/782 [00:11<00:00, 70.84it/s, loss=0.310, accuracy=89.86%]
[INFO] Training of Epoch 1 completed in 11.05 seconds
[INFO] Evaluating model...
Current learning rate: 0.1
Test Accuracy: 96.89%

[INFO] Training of Epoch 2 started...
Epoch 2: 100%|████████████████████████████████████████████████████████████████████| 782/782 [00:10<00:00, 72.97it/s, loss=0.092, accuracy=97.15%]
[INFO] Training of Epoch 2 completed in 22.41 seconds
[INFO] Evaluating model...
Current learning rate: 0.1
Test Accuracy: 98.26%

[INFO] Training of Epoch 3 started...
Epoch 3: 100%|████████████████████████████████████████████████████████████████████| 782/782 [00:10<00:00, 72.05it/s, loss=0.068, accuracy=97.86%]
[INFO] Training of Epoch 3 completed in 33.90 seconds
[INFO] Evaluating model...
Current learning rate: 0.1
Test Accuracy: 98.73%

[INFO] Training of Epoch 4 started...
Epoch 4: 100%|████████████████████████████████████████████████████████████████████| 782/782 [00:10<00:00, 73.50it/s, loss=0.062, accuracy=98.05%]
[INFO] Training of Epoch 4 completed in 45.18 seconds
[INFO] Evaluating model...
Current learning rate: 0.1
Test Accuracy: 98.73%

[INFO] Training of Epoch 5 started...
Epoch 5: 100%|████████████████████████████████████████████████████████████████████| 782/782 [00:10<00:00, 74.53it/s, loss=0.056, accuracy=98.30%]
[INFO] Training of Epoch 5 completed in 56.31 seconds
[INFO] Evaluating model...
Current learning rate: 0.1
Test Accuracy: 98.63%

[INFO] Training of Epoch 6 started...
Epoch 6: 100%|████████████████████████████████████████████████████████████████████| 782/782 [00:10<00:00, 73.66it/s, loss=0.052, accuracy=98.38%]
[INFO] Training of Epoch 6 completed in 67.57 seconds
[INFO] Evaluating model...
Current learning rate: 0.010000000000000002
Test Accuracy: 98.95%

[INFO] Training of Epoch 7 started...
Epoch 7: 100%|████████████████████████████████████████████████████████████████████| 782/782 [00:10<00:00, 73.53it/s, loss=0.032, accuracy=99.02%]
[INFO] Training of Epoch 7 completed in 78.84 seconds
[INFO] Evaluating model...
Current learning rate: 0.010000000000000002
Test Accuracy: 99.38%

[INFO] Training of Epoch 8 started...
Epoch 8: 100%|████████████████████████████████████████████████████████████████████| 782/782 [00:10<00:00, 73.58it/s, loss=0.030, accuracy=99.04%]
[INFO] Training of Epoch 8 completed in 90.10 seconds
[INFO] Evaluating model...
Current learning rate: 0.010000000000000002
Test Accuracy: 99.39%

[INFO] Training of Epoch 9 started...
Epoch 9: 100%|████████████████████████████████████████████████████████████████████| 782/782 [00:10<00:00, 76.55it/s, loss=0.028, accuracy=99.10%]
[INFO] Training of Epoch 9 completed in 100.95 seconds
[INFO] Evaluating model...
Current learning rate: 0.010000000000000002
Test Accuracy: 99.29%

[INFO] Training of Epoch 10 started...
Epoch 10: 100%|███████████████████████████████████████████████████████████████████| 782/782 [00:10<00:00, 73.55it/s, loss=0.028, accuracy=99.08%]
[INFO] Training of Epoch 10 completed in 112.21 seconds
[INFO] Evaluating model...
Current learning rate: 0.010000000000000002
Test Accuracy: 99.41%

[INFO] Training of Epoch 11 started...
Epoch 11: 100%|███████████████████████████████████████████████████████████████████| 782/782 [00:10<00:00, 74.15it/s, loss=0.026, accuracy=99.17%]
[INFO] Training of Epoch 11 completed in 123.40 seconds
[INFO] Evaluating model...
Current learning rate: 0.010000000000000002
Test Accuracy: 99.38%

[INFO] Training of Epoch 12 started...
Epoch 12: 100%|███████████████████████████████████████████████████████████████████| 782/782 [00:10<00:00, 74.33it/s, loss=0.026, accuracy=99.19%]
[INFO] Training of Epoch 12 completed in 134.56 seconds
[INFO] Evaluating model...
Current learning rate: 0.0010000000000000002
Test Accuracy: 99.38%

[INFO] Training of Epoch 13 started...
Epoch 13: 100%|███████████████████████████████████████████████████████████████████| 782/782 [00:10<00:00, 73.31it/s, loss=0.024, accuracy=99.21%]
[INFO] Training of Epoch 13 completed in 145.87 seconds
[INFO] Evaluating model...
Current learning rate: 0.0010000000000000002
Test Accuracy: 99.41%

[INFO] Training of Epoch 14 started...
Epoch 14: 100%|███████████████████████████████████████████████████████████████████| 782/782 [00:10<00:00, 73.55it/s, loss=0.026, accuracy=99.17%]
[INFO] Training of Epoch 14 completed in 157.14 seconds
[INFO] Evaluating model...
Current learning rate: 0.0010000000000000002
Test Accuracy: 99.40%

[INFO] Training of Epoch 15 started...
Epoch 15: 100%|███████████████████████████████████████████████████████████████████| 782/782 [00:10<00:00, 74.41it/s, loss=0.024, accuracy=99.22%]
[INFO] Training of Epoch 15 completed in 168.29 seconds
[INFO] Evaluating model...
Current learning rate: 0.0010000000000000002
Test Accuracy: 99.43%

[INFO] Training of Epoch 16 started...
Epoch 16: 100%|███████████████████████████████████████████████████████████████████| 782/782 [00:10<00:00, 73.46it/s, loss=0.025, accuracy=99.21%]
[INFO] Training of Epoch 16 completed in 179.57 seconds
[INFO] Evaluating model...
Current learning rate: 0.0010000000000000002
Test Accuracy: 99.44%

[INFO] Training of Epoch 17 started...
Epoch 17: 100%|███████████████████████████████████████████████████████████████████| 782/782 [00:10<00:00, 74.00it/s, loss=0.025, accuracy=99.19%]
[INFO] Training of Epoch 17 completed in 190.78 seconds
[INFO] Evaluating model...
Current learning rate: 0.0010000000000000002
Test Accuracy: 99.36%

[INFO] Training of Epoch 18 started...
Epoch 18: 100%|███████████████████████████████████████████████████████████████████| 782/782 [00:10<00:00, 73.94it/s, loss=0.025, accuracy=99.21%]
[INFO] Training of Epoch 18 completed in 201.99 seconds
[INFO] Evaluating model...
Current learning rate: 0.00010000000000000003
Test Accuracy: 99.43%

[INFO] Training of Epoch 19 started...
Epoch 19: 100%|███████████████████████████████████████████████████████████████████| 782/782 [00:10<00:00, 73.51it/s, loss=0.025, accuracy=99.19%]
[INFO] Training of Epoch 19 completed in 213.26 seconds
[INFO] Evaluating model...
Current learning rate: 0.00010000000000000003
Test Accuracy: 99.41%

[INFO] Training of Epoch 20 started...
Epoch 20: 100%|███████████████████████████████████████████████████████████████████| 782/782 [00:10<00:00, 74.30it/s, loss=0.025, accuracy=99.21%]
[INFO] Training of Epoch 20 completed in 224.43 seconds
[INFO] Evaluating model...
Current learning rate: 0.00010000000000000003
Test Accuracy: 99.43%

[STEP 5/5] Evaluating model against validation...
Test Accuracy: 99.51%
```

