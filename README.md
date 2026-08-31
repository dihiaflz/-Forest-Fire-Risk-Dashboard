# Forest Fire Risk Dashboard

An interactive Streamlit dashboard designed to monitor and predict wildfire risks in Kendira Forest using IoT sensor data and AI.

The system integrates 8 strategically placed sensors that capture key environmental metrics (temperature, humidity, CO levels, etc.) in real time. The data is:

- Displayed on a live map where each sensor is visualized by a geolocated marker.

- Summarized in a dynamic table that shows sensor readings, coordinates, and model outputs.

- Analyzed by AI models:

  - A classification model to detect fire risk (yes/no).

  - A regression model to estimate the probability of a fire starting.

Historical data is automatically logged every 5 minutes to Google Drive, enabling time-series analysis and making sure the history survives app restarts and redeployments. The dashboard also provides trend charts for temperature and humidity evolution across all sensors, offering deeper insight into environmental dynamics.

This project combines IoT, Machine Learning, and Data Visualization into a single application, aiming to support smarter and faster decision-making for wildfire prevention.

# Notebook & Synthetic Data

The repository also includes the training notebook used to build the AI models.

- The notebook contains the synthetic data generation code used to simulate realistic sensor readings.

- These generated datasets were then exported and used for training the classification and regression models.

- A link to the full generated dataset is provided in the notebook (hosted on Google Drive).

# Setup: Google Drive Persistence

The dashboard logs its history to a `database.csv` file stored on Google Drive instead of locally, so the data isn't lost when the app restarts or gets redeployed. Before running the app for the first time, you need to:

1. Create a Google Cloud project and enable the **Google Drive API**.
2. Create a **Service Account** under APIs & Services > Credentials, and download its JSON key.
3. Upload a `database.csv` file to your Google Drive, then share it with the service account's email (found in the JSON key, field `client_email`) with **Editor** access.
4. Copy the file's ID from its Drive link (the part between `/d/` and `/view`).
5. Create a `.streamlit/secrets.toml` file inside the `app` folder with the following structure:

```toml
drive_file_id = "your_file_id_here"

[gcp_service_account]
type = "service_account"
project_id = "..."
private_key_id = "..."
private_key = """-----BEGIN PRIVATE KEY-----
...
-----END PRIVATE KEY-----
"""
client_email = "..."
client_id = "..."
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "..."
```

(All these values come from the JSON key downloaded in step 2, except `drive_file_id` which comes from step 4.)

**Note:** `secrets.toml` contains real credentials and is excluded via `.gitignore` — never commit it. If you're deploying to Streamlit Community Cloud, paste this same content into the app's Secrets settings instead of relying on the local file.

# HOW TO USE :
1. Clone the repository to your local machine.
2. Create a virtual environment using the command **python -m venv venv** and then activate it using **venv\Scripts\activate**
3. Move to the app folder using **cd app** and Install dependencies using **pip install -r requirements.txt**
4. Complete the **Setup: Google Drive Persistence** section above before running the app.
5. Once everything is installed and configured, you can run the app with **python -m streamlit run main.py** (considering you are in the app folder)

# Second Option : DOCKER
You can also run the app inside Docker:
1. Open your Docker Desktop App and ensure that it is activated
2. Make sure `.streamlit/secrets.toml` exists inside the app folder (see Setup section above) before building, since it needs to be available inside the container.
3. Build the Docker image using **docker build -t prevent-forest-fire .** after moving to the app folder using **cd app** (where is the dockerfile)
4. Run a container from the image using **docker run -p 8501:80 prevent-forest-fire:v1.0** or **docker run -p 8501:80 prevent-forest-fire:latest** according to the version displayed in your docker desktop -> Images section

You can now run the app through the Local URL **localhost:8501**

# Live Demo
Deployed on Streamlit Community Cloud: *https://fire-risk-dashboard.streamlit.app/*
