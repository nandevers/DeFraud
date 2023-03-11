FROM python:3.11

RUN mkdir DeFraud
ADD data /DeFraud/data

# Create a directory called DeFraud and set it as the working directory
WORKDIR /DeFraud

# Copy all files in the current directory into the container's working directory
COPY preprocessing /DeFraud/preprocessing
COPY feature_engineering /DeFraud/feature_engineering
COPY requirements.txt /DeFraud/

VOLUME [ "/data" ]

# Install the required packages
RUN pip install -r requirements.txt

ENTRYPOINT [ "/bin/bash" ]