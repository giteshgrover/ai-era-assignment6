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
