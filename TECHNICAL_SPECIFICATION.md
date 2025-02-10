# VinoVoyant - Wine Origin Prediction Service
## Technical Specification and Architecture Document

## Table of Contents
1. [System Overview](#system-overview)
2. [Data Analysis and Model Development](#data-analysis-and-model-development)
3. [Architecture Design](#architecture-design)
4. [Implementation Details](#implementation-details)
5. [Deployment Strategy](#deployment-strategy)
6. [Performance and Scalability](#performance-and-scalability)

## System Overview

VinoVoyant is an end-to-end wine origin prediction service that uses machine learning and AI to predict a wine's country of origin based on its description and characteristics. The system combines traditional ML approaches with modern LLM techniques to provide accurate predictions and insights.

### Key Features
- Multi-model prediction system (Traditional ML, Transformer, and LLM)
- Interactive web interface for predictions
- Advanced analytics dashboard
- Real-time insights generation
- AWS S3 integration for data storage
- Streamlit-based user interface

## Data Analysis and Model Development

### Dataset Characteristics
- Source: Wine Quality Review Dataset
- Features:
  - Country (Target): US, Spain, France, or Italy
  - Description: Free text wine description
  - Points: Review score (1-100)
  - Price: Bottle cost
  - Variety: Grape type

### Model Architecture

The system implements three distinct prediction approaches:

1. **Traditional ML Pipeline**
   - TF-IDF Vectorization for text processing
   - Logistic Regression classifier
   - Feature engineering including:
     - Text length analysis
     - Price categorization
     - Quality categorization
     - Variety encoding

2. **Transformer Model**
   - DistilBERT base (uncased)
   - Fine-tuned for wine description analysis
   - Embedding-based classification
   - Local CPU inference

3. **LLM-based Analysis**
   - GPT-4o-mini model
   - Prompt engineering for wine expertise
   - Detailed reasoning capabilities
   - API-based inference

### Model Performance Metrics
- Cross-validation accuracy
- Confidence scoring
- Real-time performance monitoring
- Error analysis and logging

## Architecture Design

### System Components Diagram
```
[Client Layer]
    │
    ▼
[Streamlit Web Application]
    │
    ├──► [Traditional ML Pipeline]
    │     └──► [TF-IDF + Logistic Regression]
    │
    ├──► [Transformer Pipeline]
    │     └──► [DistilBERT + Linear Classifier]
    │
    ├──► [LLM Pipeline]
    │     └──► [GPT-4o-mini API]
    │
    ├──► [Analytics Engine]
    │     └──► [Plotly Visualizations]
    │
    └──► [Data Layer]
          └──► [AWS S3 Storage]
```

### Component Interactions

1. **Data Flow**
   ```
   [AWS S3] ◄──► [Data Preprocessor] ◄──► [Model Pipeline] ◄──► [Web Interface]
   ```

2. **Prediction Flow**
   ```
   [User Input] ──► [Model Selection] ──► [Prediction Pipeline] ──► [Results Display]
   ```

3. **Analytics Flow**
   ```
   [S3 Data] ──► [Analytics Engine] ──► [Visualization] ──► [LLM Insights]
   ```

## Implementation Details

### Key Technologies
- **Frontend**: Streamlit
- **Backend**: Python 3.8+
- **ML Framework**: PyTorch, Transformers, Scikit-learn
- **Data Storage**: AWS S3
- **LLM Integration**: REST API
- **Visualization**: Plotly

### Code Structure
```
project/
├── src/
│   ├── models/
│   │   ├── advanced_predictors.py
│   │   └── country_predictor.py
│   ├── preprocessing.py
│   └── analytics.py
├── streamlit_app.py
└── requirements.txt
```

### Key Classes and Components

1. **WineDataPreprocessor**
   - Data cleaning and preparation
   - Feature engineering
   - Text preprocessing
   - S3 data integration

2. **TransformerPredictor**
   - DistilBERT model management
   - Embedding generation
   - Classification logic
   - Model persistence

3. **DetailedPromptPredictor**
   - LLM integration
   - Prompt engineering
   - Response parsing
   - Error handling

## Deployment Strategy

### Production Deployment Plan

1. **Infrastructure Setup**
   - AWS EC2 for application hosting
   - S3 for data storage
   - CloudWatch for monitoring
   - Load balancer for traffic management

2. **Deployment Process**
   - Docker containerization
   - CI/CD pipeline integration
   - Blue-green deployment strategy
   - Automated testing

3. **Monitoring and Maintenance**
   - Performance metrics tracking
   - Error logging and alerting
   - Model performance monitoring
   - Regular updates and maintenance

### Scaling Considerations
- Horizontal scaling for web interface
- Caching for frequent predictions
- Model optimization for performance
- Load balancing for API requests

## Performance and Scalability

### Performance Optimizations
1. **Model Caching**
   - Pre-trained model storage
   - Inference optimization
   - Batch prediction support

2. **Data Management**
   - Efficient S3 data access
   - Data preprocessing optimization
   - Caching frequently used data

3. **Application Performance**
   - Streamlit session state management
   - Efficient data loading
   - Optimized visualization rendering

### Monitoring and Metrics
- Response time tracking
- Model accuracy monitoring
- Resource utilization metrics
- Error rate tracking

## Security Considerations

1. **Data Security**
   - S3 bucket policies
   - API key management
   - Secure data transmission

2. **Application Security**
   - Input validation
   - Error handling
   - Rate limiting
   - Access control

## Future Enhancements

1. **Model Improvements**
   - Additional model architectures
   - Enhanced feature engineering
   - Model ensemble approaches

2. **Platform Features**
   - User authentication
   - Prediction history
   - Advanced analytics
   - Custom model training

3. **Infrastructure**
   - Multi-region deployment
   - Advanced caching
   - Auto-scaling
   - Disaster recovery

---

*Generated for Artificial Atlanta Technical Assessment*
*Author: Andrii Pasternak* 