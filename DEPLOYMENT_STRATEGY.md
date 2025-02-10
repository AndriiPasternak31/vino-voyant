# VinoVoyant Deployment Strategy
## End-to-End Deployment Plan

## 1. Infrastructure Setup

### AWS Resources Required
1. **EC2 Instance**
   - t2.medium or larger (for ML model handling)
   - Ubuntu Server 20.04 LTS
   - 16GB+ RAM recommended
   - Security group with ports 80/443 (HTTP/HTTPS)

2. **S3 Bucket Configuration**
   - Bucket: `vino-voyant-wine-origin-predictor`
   - Region: `eu-north-1`
   - Public access for dataset
   - CORS configuration for API access

3. **CloudWatch Setup**
   - Custom metrics for model performance
   - Application logs monitoring
   - Resource utilization alerts
   - Error rate tracking

4. **Load Balancer**
   - Application Load Balancer (ALB)
   - Health checks configuration
   - SSL/TLS certificate integration
   - Auto-scaling group integration

## 2. Application Containerization

### Docker Configuration
```dockerfile
# Base image with Python 3.8
FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Expose port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "streamlit_app.py"]
```

## 3. CI/CD Pipeline (GitHub Actions)

```yaml
name: VinoVoyant CI/CD

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v1
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: eu-north-1
    
    - name: Build and push Docker image
      run: |
        docker build -t vino-voyant .
        docker tag vino-voyant:latest ${{ secrets.ECR_REGISTRY }}/vino-voyant:latest
        docker push ${{ secrets.ECR_REGISTRY }}/vino-voyant:latest
    
    - name: Deploy to EC2
      run: |
        aws ecs update-service --cluster vino-voyant --service web --force-new-deployment
```

## 4. Deployment Process

### Initial Deployment
1. **Infrastructure Provisioning**
   ```bash
   # Initialize Terraform
   terraform init
   
   # Apply infrastructure
   terraform apply
   ```

2. **Application Deployment**
   ```bash
   # Build Docker image
   docker build -t vino-voyant .
   
   # Push to ECR
   docker push $ECR_REGISTRY/vino-voyant:latest
   
   # Deploy to ECS
   aws ecs update-service --cluster vino-voyant --service web --force-new-deployment
   ```

### Blue-Green Deployment Strategy
1. **Preparation**
   - Create new "green" environment
   - Deploy new version to green environment
   - Run health checks and tests

2. **Switching Traffic**
   - Gradually route traffic to green environment
   - Monitor for errors
   - If successful, decommission blue environment
   - If issues, quick rollback to blue environment

## 5. Monitoring and Maintenance

### CloudWatch Alarms
```json
{
  "Alarms": [
    {
      "AlarmName": "HighCPUUtilization",
      "MetricName": "CPUUtilization",
      "Threshold": 80,
      "Period": 300,
      "EvaluationPeriods": 2
    },
    {
      "AlarmName": "ModelLatency",
      "MetricName": "PredictionLatency",
      "Threshold": 2000,
      "Period": 60,
      "EvaluationPeriods": 3
    }
  ]
}
```

### Health Checks
1. **Application Health**
   - Endpoint: `/health`
   - Check frequency: 30 seconds
   - Timeout: 5 seconds
   - Healthy threshold: 2
   - Unhealthy threshold: 3

2. **Model Health**
   - Periodic test predictions
   - Model performance metrics
   - Resource utilization
   - Error rate monitoring

## 6. Scaling Strategy

### Auto Scaling Configuration
```json
{
  "AutoScalingGroup": {
    "MinSize": 2,
    "MaxSize": 10,
    "DesiredCapacity": 2,
    "HealthCheckGracePeriod": 300,
    "HealthCheckType": "ELB"
  },
  "ScalingPolicies": [
    {
      "PolicyName": "CPUScaling",
      "TargetValue": 70,
      "PredefinedMetricType": "ASGAverageCPUUtilization"
    }
  ]
}
```

## 7. Backup and Recovery

### Backup Strategy
1. **Model Artifacts**
   - Daily snapshots of model files
   - Version control in S3
   - Retention period: 30 days

2. **Application Data**
   - Continuous replication of S3 data
   - Cross-region backup
   - Point-in-time recovery

### Disaster Recovery
1. **Recovery Time Objective (RTO)**: 1 hour
2. **Recovery Point Objective (RPO)**: 5 minutes
3. **Recovery Steps**:
   ```bash
   # Restore from backup
   aws s3 sync s3://backup-bucket/models s3://production-bucket/models
   
   # Deploy application
   aws ecs update-service --cluster vino-voyant --service web --force-new-deployment
   ```

## 8. Security Measures

### Security Configuration
```json
{
  "SecurityGroup": {
    "Ingress": [
      {
        "FromPort": 80,
        "ToPort": 80,
        "Protocol": "tcp",
        "CidrIp": "0.0.0.0/0"
      },
      {
        "FromPort": 443,
        "ToPort": 443,
        "Protocol": "tcp",
        "CidrIp": "0.0.0.0/0"
      }
    ]
  },
  "S3Bucket": {
    "PublicAccessBlock": {
      "BlockPublicAcls": true,
      "BlockPublicPolicy": true,
      "IgnorePublicAcls": true,
      "RestrictPublicBuckets": true
    }
  }
}
```

## 9. Cost Optimization

### Resource Optimization
1. **EC2 Instances**
   - Use Reserved Instances for base load
   - Spot Instances for variable load
   - Auto-scaling based on demand

2. **S3 Storage**
   - Lifecycle policies for old data
   - Intelligent-Tiering for infrequent access
   - Compression for model artifacts

## 10. Maintenance Schedule

### Regular Maintenance
1. **Daily**
   - Log analysis
   - Performance monitoring
   - Backup verification

2. **Weekly**
   - Security patches
   - Model performance review
   - Resource optimization

3. **Monthly**
   - Full system backup
   - Infrastructure review
   - Cost analysis

---

This deployment strategy ensures a robust, scalable, and maintainable production environment for the VinoVoyant service. The strategy focuses on automation, monitoring, and quick recovery capabilities while maintaining security and cost-effectiveness. 