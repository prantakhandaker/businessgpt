# 🚀 Ultimate RAG Pipeline

An advanced Retrieval-Augmented Generation (RAG) system specifically designed for business analytics and data querying. This pipeline extends your existing RAG system with cutting-edge features for enhanced query understanding, multi-stage retrieval, and intelligent analytics.

## ✨ Key Features

### 🔍 **Multi-Stage Retrieval System**
- **Semantic Search**: Vector-based similarity search using embeddings
- **Keyword Search**: BM25-like scoring for exact matches
- **Temporal Search**: Context-aware time-based retrieval
- **Hybrid Fusion**: Intelligent combination of all retrieval methods

### 🧠 **Advanced Query Understanding**
- **Intent Classification**: Automatically categorizes queries (analytical, operational, exploratory, reporting)
- **Entity Extraction**: Identifies business entities (depot, client, product, sales, profit)
- **Temporal Context Detection**: Recognizes time-based queries
- **Query Decomposition**: Breaks down complex questions into simpler sub-queries

### 🕸️ **Knowledge Graph Integration**
- **Dynamic Graph Building**: Automatically constructs relationships from your data
- **Entity Relationship Mapping**: Discovers connections between business entities
- **Context Path Finding**: Uses graph traversal for enhanced context understanding

### 🛠️ **Enhanced SQL Generation**
- **Execution Planning**: Creates step-by-step SQL execution plans
- **Validation & Safety**: Ensures SQL correctness and security
- **Performance Optimization**: Optimizes queries for BigQuery best practices
- **Error Handling**: Graceful fallback mechanisms

### 💬 **Conversation Memory**
- **Session Management**: Maintains conversation context across interactions
- **User Preferences**: Learns and adapts to user patterns
- **History Tracking**: Stores and retrieves previous interactions

### 📊 **Advanced Analytics**
- **Automated Insights**: Generates business insights from data
- **Statistical Analysis**: Performs comprehensive statistical analysis
- **Trend Detection**: Identifies patterns and trends in data
- **Anomaly Detection**: Spots unusual patterns and outliers

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Query           │───▶│  Multi-Stage    │
│                 │    │  Understanding   │    │  Retrieval      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Response       │◀───│  SQL Generation  │◀───│  Knowledge      │
│  Generation     │    │  & Validation    │    │  Graph          │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  Advanced        │
                       │  Analytics       │
                       └──────────────────┘
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements_ultimate.txt

# Ensure Qdrant is running
docker run -p 6333:6333 qdrant/qdrant
```

### 2. Configuration

Update the configuration in `ultimate_rag_pipeline.py`:

```python
BQ_PROJECT = "your-bigquery-project"
BQ_DATASET = "your-dataset"
VERTEX_PROJECT = "your-vertex-project"
# ... other configurations
```

### 3. Run the Pipeline

```bash
# Direct usage
python ultimate_rag_pipeline.py

# Or run as API server
python ultimate_rag_api.py
```

### 4. API Usage

```bash
# Start the API server
python ultimate_rag_api.py

# Test the API
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the top 10 clients by sales?"}'
```

## 📡 API Endpoints

### Main Endpoints

- **POST /ask** - Main question processing endpoint
- **GET /health** - Health check and system status
- **GET /conversation/<user_id>/<session_id>** - Get conversation history

### Advanced Endpoints

- **POST /insights** - Generate insights from data
- **POST /decompose** - Decompose complex queries
- **POST /intent** - Classify query intent

## 🔧 Configuration Options

### Retrieval Weights
```python
retrieval_weights = {
    'semantic': 0.6,    # Vector similarity weight
    'keyword': 0.3,     # Keyword matching weight
    'temporal': 0.1     # Temporal context weight
}
```

### Intent Patterns
```python
intent_patterns = {
    "analytical": ["trend", "analysis", "compare", "forecast"],
    "operational": ["current", "today", "status", "alert"],
    "exploratory": ["explore", "discover", "what if", "drill down"],
    "reporting": ["report", "summary", "dashboard", "export"]
}
```

## 📊 Example Queries

### Simple Queries
```python
# Basic sales query
"What are the top 10 clients by sales?"

# Time-based query
"Show me sales for this month"

# Comparison query
"Compare sales between depots"
```

### Complex Queries
```python
# Multi-dimensional analysis
"What are the trends in product sales by depot over the last quarter?"

# Exploratory query
"Help me understand the relationship between client size and sales volume"

# Operational query
"Alert me if any depot has sales below 80% of target this week"
```

## 🎯 Performance Features

### Caching Strategy
- **Query Result Caching**: Caches frequent query results
- **Embedding Caching**: Stores computed embeddings
- **SQL Plan Caching**: Caches execution plans
- **Conversation Caching**: Persists conversation history

### Optimization
- **Parallel Processing**: Concurrent execution of sub-queries
- **Smart Retrieval**: Adaptive retrieval based on query complexity
- **Resource Management**: Efficient memory and compute usage

## 🔒 Security Features

### Data Protection
- **SQL Validation**: Prevents malicious SQL injection
- **Access Control**: Role-based data access
- **Query Auditing**: Logs all queries for compliance
- **Rate Limiting**: Prevents system abuse

### Safety Measures
- **Forbidden Patterns**: Blocks dangerous SQL operations
- **Timeout Protection**: Prevents long-running queries
- **Error Handling**: Graceful error recovery

## 📈 Monitoring & Analytics

### System Metrics
- **Query Processing Time**: Track response times
- **Retrieval Accuracy**: Monitor retrieval quality
- **User Engagement**: Analyze usage patterns
- **Error Rates**: Track system reliability

### Business Insights
- **Query Patterns**: Understand user needs
- **Data Usage**: Track data access patterns
- **Performance Trends**: Monitor system performance
- **User Satisfaction**: Measure user experience

## 🛠️ Customization

### Adding New Retrieval Methods
```python
class CustomRetriever:
    def retrieve(self, query: str, top_k: int = 5):
        # Implement your custom retrieval logic
        pass

# Add to MultiStageRetriever
retriever.custom_retriever = CustomRetriever()
```

### Extending Analytics
```python
class CustomAnalytics:
    def generate_insights(self, results, query):
        # Add your custom insight generation
        pass

# Add to AdvancedAnalytics
analytics.custom_analytics = CustomAnalytics()
```

## 🐛 Troubleshooting

### Common Issues

1. **Qdrant Connection Error**
   ```bash
   # Ensure Qdrant is running
   docker ps | grep qdrant
   ```

2. **BigQuery Authentication**
   ```bash
   # Check credentials
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"
   ```

3. **Memory Issues**
   ```python
   # Reduce batch sizes in configuration
   BATCH_SIZE = 100  # Reduce from default
   ```

### Debug Mode
```python
# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your enhancements
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built on top of your existing RAG pipeline
- Uses Google Cloud BigQuery and Vertex AI
- Powered by Qdrant vector database
- Enhanced with scikit-learn and NetworkX

---

**Ready to revolutionize your business analytics? Start with the Ultimate RAG Pipeline! 🚀**
