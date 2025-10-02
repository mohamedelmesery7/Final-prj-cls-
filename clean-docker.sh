#!/bin/bash

echo "Stopping all running containers..."
docker stop $(docker ps -a -q) 2>/dev/null || true

echo "Removing all containers..."
docker rm $(docker ps -a -q) 2>/dev/null || true

echo "Removing all unused networks..."
docker network prune -f

echo "Removing unused Docker images..."
docker image prune -a -f

echo "You can now run './run.sh' to start the application fresh."
