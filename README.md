# Flask_PyTorch_DL
This repo contains a minimal example and demo to deploy any PyTorch model into WebApps

## Requirements
- Flask
- Docker
- Python3.x

## 

## Systems Architecture

```mermaid
flowchart TB
    Build <--> Deploy
    subgraph PyTorch
    Build <--> Test
    Build --> Train
    Train <--> Train
    Test <--> Test
    Train --> Test
    end

    subgraph Onnx
    Deploy --> Model.pth
    Model --> Converted
    Converted <--> Model.pth    
    end

    subgraph Flask
    API <--> Production
    Model.pth --> API
    end
    subgraph Container
    Containerized --> Build
    Containerized --> Model.pth
    Containerized --> Production
    end    
```

# Flow

```mermaid
flowchart LR
   Build <--train--> Model
   Model <--test--> SaveModel
   SaveModel <--flask-->app
   app <--containerized-->docker
   webserver <--deploy-->app


```

## TODO



## References