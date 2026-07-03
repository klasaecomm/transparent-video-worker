pipeline {
    agent any
    options { disableConcurrentBuilds(); buildDiscarder(logRotator(numToKeepStr: '20')) }
    parameters {
        booleanParam(name: 'BUILD_IMAGE', defaultValue: true,  description: 'Build image from Dockerfile.')
        booleanParam(name: 'PUSH_IMAGE',  defaultValue: true,  description: 'Push image to REGISTRY.')
        string(name: 'REGISTRY',        defaultValue: 'localhost:32000',            description: 'Docker registry host.')
        string(name: 'IMAGE_NAME',      defaultValue: 'transparent-video-worker',   description: 'Image name.')
        string(name: 'IMAGE_TAG',       defaultValue: 'latest',                     description: 'Image tag.')
        string(name: 'DOCKER_PLATFORM', defaultValue: 'linux/amd64',                description: 'Build platform.')
    }
    stages {
        stage('Checkout') { steps { checkout scm } }
        stage('Docker Image') {
            when { expression { params.BUILD_IMAGE } }
            steps {
                script {
                    String image = "${params.REGISTRY}/${params.IMAGE_NAME}:${params.IMAGE_TAG}"
                    env.BUILT_IMAGE = image
                    if (params.PUSH_IMAGE) {
                        sh "docker buildx build --platform '${params.DOCKER_PLATFORM}' -t '${image}' --push ."
                    } else {
                        sh "docker buildx build --platform '${params.DOCKER_PLATFORM}' -t '${image}' --load ."
                    }
                }
            }
        }
    }
    post { success { echo "Built ${env.BUILT_IMAGE}" } }
}
