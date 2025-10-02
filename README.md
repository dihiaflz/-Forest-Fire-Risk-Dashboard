# Forest Fire Risk Dashboard

An interactive Streamlit dashboard designed to monitor and predict wildfire risks in Kendera Forest using IoT sensor data and AI.

The system integrates 8 strategically placed sensors that capture key environmental metrics (temperature, humidity, CO levels, etc.) in real time. The data is:

- Displayed on a live map where each sensor is visualized by a geolocated marker.

- Summarized in a dynamic table that shows sensor readings, coordinates, and model outputs.

- Analyzed by AI models:

  - A classification model to detect fire risk (yes/no).

  - A regression model to estimate the probability of a fire starting.

Historical data is automatically logged every 5 minutes, enabling time-series analysis. The dashboard also provides trend charts for temperature and humidity evolution across all sensors, offering deeper insight into environmental dynamics.

This project combines IoT, Machine Learning, and Data Visualization into a single application, aiming to support smarter and faster decision-making for wildfire prevention

# HOW TO USE :
1. Clone the repository to your local machine.
2. Create a virtual environment using the command **python -m venv venv** and then activate it using **venv\Scripts\activate**
3. Move to the app folder using **cd app** and Install dependencies using **pip install -r requirements.txt**
4. Once everything is installed, you can run the app with **python -m streamlit run main.py** (considering you are in the app folder)

# Second Option : DOCKER
You can also run the app inside Docker:
1. Open your Docker Desktop App and ensure that it is activated
2. Build the Docker image using **docker build -t prevent-forest-fire .** after moving to the app folder using **cd app** (where is the dockerfile)
3. Run a container from the image using **docker run -p 8501:80 prevent-forest-fire:v1.0** or **docker run -p 8501:80 prevent-forest-fire:latest** according to the version displayed in your docker desktop -> Images section

You can now run the app through the Local URL **localhost:8501**
