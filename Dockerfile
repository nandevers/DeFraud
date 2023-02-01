# Use a base image with Python 3.11
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt /app/requirements.txt

# Install the required packages
RUN pip install -r requirements.txt

# Copy the rest of the app code to the container
COPY /app /app
COPY /data /data


# Expose port 8501 to run the Streamlit app
EXPOSE 8501

# Set the entrypoint to start Streamlit
ENTRYPOINT ["streamlit", "run", "app.py","--server.port=8501", "--server.address=0.0.0.0"]