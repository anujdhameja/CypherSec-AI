# Start with a full-featured Ubuntu OS
FROM ubuntu:latest

# Set the environment to non-interactive to avoid install prompts
ENV DEBIAN_FRONTEND=noninteractive

# Update and install all compilers, runtimes, and our monitor
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    build-essential \
    gcc \
    g++ \
    python3 \
    python3-pip \
    default-jdk \
    php-cli \
    strace

# --- Install Trivy for CVE scanning ---
RUN wget https://github.com/aquasecurity/trivy/releases/download/v0.51.4/trivy_0.51.4_Linux-64bit.deb
RUN dpkg -i trivy_0.51.4_Linux-64bit.deb
RUN rm trivy_0.51.4_Linux-64bit.deb

# --- Install .NET SDK for C# ---
RUN wget https://packages.microsoft.com/config/ubuntu/$(lsb_release -rs)/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
RUN dpkg -i packages-microsoft-prod.deb
RUN rm packages-microsoft-prod.deb
RUN apt-get update && apt-get install -y dotnet-sdk-8.0

# --- NEW: Download the Trivy DB *during* the build ---
# This caches the database in the image so we can run offline
RUN trivy fs . --download-db-only

# --- Setup our working directory ---
WORKDIR /app

# Copy our detector script (which we'll create next) into the image
COPY detector.py .

# The container will start by running our detector script
CMD ["python3", "./detector.py"]