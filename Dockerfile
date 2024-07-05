# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /inference

# Copy all contents into the container
COPY . .

# Install necessary packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the Flask port
EXPOSE 5000

# Run the application
CMD ["python", "inference/app.py"]
