#!/bin/bash

docker image build -t iris-app . 
echo "✅ Docker iamge successfully created!"
docker run -p 8000:8000 -d iris-app # create random name for container
echo "✅ Docker Container successfully created!"
