# docker commands

1- to test id docker is installed correctly or not

$ docker run hello-world

2- to pull any image from docker hub (docker registery)

$ docker pull image_name , for ex : docker pull ubuntu:12.04

3- to know all images in your docker hub

$ docker images

4- to know all contaiers running

$ docker ps

5-  to know all containers you have

$ docker ps -a

6- to remove specific image

$ docker rmi image_id

7- to remove specific container

$ docker rm container_id

8- to remove all stopped container 

$ docker rm $(docker ps -a -q)

9- to build image from docker file

$ docker build -t image_name .

10- to run container

$ docker run -p host_port:container_port image_name

11- to run multi app container 

$ docker compose up

12- to shutdown multi app container 

$ docker compose down