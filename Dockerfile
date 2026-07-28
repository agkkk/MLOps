# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Upgrade pip to the latest version to prevent hash errors
RUN pip install --upgrade pip

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project code into the container
COPY . .

# Expose the ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Make sure the run script is executable
RUN chmod +x run.sh

# Run the shell script when the container launches
CMD ["./run.sh"]