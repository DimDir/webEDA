# webEDA toolkit
This project is supposed to simplify data analysis process for data scientists or managers

## Installation using Docker
1) clone this repository to your git or just download
2) build an image (you've got to be inside directory webEda):
```
docker build -t webeda .
```
3) create and start container:
```
docker run -d -p 8501:8501 --name eda webeda:latest
```
4) Enjoy the app in your browswer: http://localhost:8501/

soon on dockerhub...
