# SMART HOME DEVICE EFFICIENCY

## Overview
->This appliction aims at predicting the efficiency(whether efficient or not) of some smart home devices using various entities associated with them

### Prerequisites
- Python 3.x
- Pip (Python package installer)
- AWS CLI
- AWS Elastic Beanstalk CLI

### Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/steve601/smarthome.git
    cd smarthome
    ```

2. **Create a virtual environment and activate it**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Run the application locally**:
    ```sh
    python smarthome.py
    ```
Access the app at `http://127.0.0.1:5000`.

## Deployment to AWS Elastic Beanstalk

### Step-by-Step Deployment

1. **Initialize Elastic Beanstalk**:
    ```sh
    eb init
    ```

    Follow the prompts to configure your application. Choose the appropriate region and platform (Python).

2. **Create an Elastic Beanstalk environment**:
    ```sh
    eb create flask-env
    ```

3. **Deploy the application**:
    ```sh
    eb deploy
    ```

4. **Access the application**:
    After deployment, access your application using the URL provided by Elastic Beanstalk.

### Configuration File: `.ebextensions/python.config`

Ensure your configuration file is set up correctly:

```yaml
option_settings:
  aws:elasticbeanstalk:container:python:
    WSGIPath: smarthome:app
