# follow then slides to understand about the docker

- to build and run the docker with the ARG for the AWS credential use:
-  ` docker build . --rm --build-arg AWS_ACCESS_KEY_ID="your_credential" --build-arg AWS_SECRET_ACCESS_KEY="your credential" -t cat-dog:latest`

- running the docker file with the port
    - docker run -d -p 8000:8000 docker_image_name

- push to the docker hub
- docker tag project_name your_username/project
- docker push your_username/project