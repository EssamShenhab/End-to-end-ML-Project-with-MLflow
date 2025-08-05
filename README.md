# End-to-end-Machine-Learning-Project-with-MLflow


## Workflows

1. Update config.yaml
2. Update schema.yaml
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the app.py



# How to run?
### STEPS:

### Installation

```bash
git clone https://github.com/EssamShenhab/End-to-end-ML-Project-with-MLflow.git
cd End-to-end-ML-Project-with-MLflow
pip install -r requirements.txt
```

---

## Run Locally

```bash
python app.py
```

You can also run the main pipeline:

```bash
python main.py
```

---

## MLflow Tracking (Local)

##### Start the MLflow UI locally:

```bash
mlflow ui
```

Then go to:

```
http://127.0.0.1:5000
```

---

## MLflow with DAGsHub

### Step 1: Initialize DAGsHub

Inside your Python script (e.g., `main.py`), add:

```python
import dagshub
dagshub.init(repo_owner='EssamShenhab', repo_name='End-to-end-ML-Project-with-MLflow', mlflow=True)
```

Then:

```python
import mlflow

with mlflow.start_run():
    mlflow.log_param('parameter name', 'value')
    mlflow.log_metric('metric name', 1)
```

---

## Deploy to AWS EC2

1. Create an EC2 instance.
2. SSH into the instance:

   ```bash
   ssh -i your-key.pem ec2-user@your-ec2-public-ip
   ```
3. Clone the repo and install requirements.
4. Run your script or app:

   ```bash
   python main.py
   ```

---
