# MLOps-Model-Development-to-Production-deployment

**🧱Model Development**

- Processed NYC taxi trip data with engineered features like pickup/dropoff zones and trip distance.
- Tuned XGBoost and Random Forest models using Hyperopt, improving RMSE by ~30%.
- Tracked all experiments, metrics, and artifacts using MLflow.

**🧱 Pipeline Orchestration**

- Designed a modular ML pipeline using Apache Airflow to automate preprocessing and training stages.
- Configured Docker + CeleryExecutor for reproducible, distributed task execution.
- Used XCom for smooth data passing and monitoring through the DAG.
  
**🧱 Experiment Management**

- Logged models to MLflow Model Registry, versioned them, and promoted top models to "Production" stage.
- Enabled consistent experiment comparison and artifact retrieval.
  
**🧱 Deployment**

- Deployed the trained model as a REST API using Flask, allowing real-time predictions via user input.
- Containerized the app with Docker to simulate a real-world, always-on service.
