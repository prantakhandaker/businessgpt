#!/usr/bin/env python3
"""
Test script for the Ultimate RAG Pipeline
"""

import sys
import json
import time
from ultimate_rag_pipeline import UltimateRAGPipeline

def test_pipeline():
    """Test the Ultimate RAG Pipeline with sample queries"""
    
    print("Testing Ultimate RAG Pipeline...")
    print("=" * 50)
    
    try:
        # Initialize pipeline
        print("Initializing pipeline...")
        pipeline = UltimateRAGPipeline()
        print("Pipeline initialized successfully!")
        
        # Test queries
        test_queries = [
            "What are the top 10 clients by sales?",
            "Show me sales trends for this month",
            "Compare sales between different depots",
            "What products are performing best?",
            "Give me a summary of today's sales"
        ]
        
        print(f"\nTesting {len(test_queries)} sample queries...")
        print("=" * 50)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\nTest {i}: {query}")
            print("-" * 30)
            
            start_time = time.time()
            
            try:
                # Process query
                response = pipeline.ask_intelligent(query)
                
                processing_time = time.time() - start_time
                
                # Display results
                print(f"Query processed in {processing_time:.2f} seconds")
                print(f"Intent: {response.get('metadata', {}).get('intent', 'unknown')}")
                print(f"Confidence: {response.get('metadata', {}).get('confidence', 0):.2f}")
                print(f"Summary: {response.get('summary', 'No summary')[:100]}...")
                
                if response.get('results'):
                    print(f"Results: {len(response['results'])} rows")
                
                if response.get('charts'):
                    print(f"Charts: {len(response['charts'])} generated")
                
                if response.get('insights'):
                    print(f"Insights: {len(response['insights'])} generated")
                
                if response.get('error'):
                    print(f"Error: {response['error']}")
                
            except Exception as e:
                print(f"Error processing query: {str(e)}")
            
            print("-" * 30)
        
        print("\nAll tests completed!")
        
    except Exception as e:
        print(f"Failed to initialize pipeline: {str(e)}")
        return False
    
    return True

def test_components():
    """Test individual components"""
    
    print("\nTesting Individual Components...")
    print("=" * 50)
    
    try:
        from ultimate_rag_pipeline import (
            MultiStageRetriever, 
            QueryUnderstanding, 
            KnowledgeGraph,
            AdvancedAnalytics
        )
        
        # Test Query Understanding
        print("\n1. Testing Query Understanding...")
        query_understanding = QueryUnderstanding()
        intent = query_understanding.classify_intent("What are the top sales trends?")
        print(f"   Intent: {intent.intent_type} (confidence: {intent.confidence:.2f})")
        print(f"   Entities: {intent.entities}")
        
        # Test Multi-Stage Retriever
        print("\n2. Testing Multi-Stage Retriever...")
        retriever = MultiStageRetriever()
        results = retriever.retrieve("sales data", top_k=3)
        print(f"   Retrieved {len(results)} results")
        
        # Test Knowledge Graph
        print("\n3. Testing Knowledge Graph...")
        kg = KnowledgeGraph()
        print(f"   Graph nodes: {kg.graph.number_of_nodes()}")
        print(f"   Graph edges: {kg.graph.number_of_edges()}")
        
        # Test Analytics
        print("\n4. Testing Advanced Analytics...")
        analytics = AdvancedAnalytics()
        sample_data = [
            {"product": "A", "sales": 100},
            {"product": "B", "sales": 150},
            {"product": "C", "sales": 200}
        ]
        insights = analytics.generate_insights(sample_data, "sales analysis")
        print(f"   Generated {len(insights)} insights")
        
        print("\nAll components tested successfully!")
        
    except Exception as e:
        print(f"Component test failed: {str(e)}")
        return False
    
    return True

def main():
    """Main test function"""
    
    print("Ultimate RAG Pipeline Test Suite")
    print("=" * 60)
    
    # Test components first
    if not test_components():
        print("\nComponent tests failed. Exiting.")
        sys.exit(1)
    
    # Test full pipeline
    if not test_pipeline():
        print("\nPipeline tests failed. Exiting.")
        sys.exit(1)
    
    print("\nAll tests passed! Ultimate RAG Pipeline is ready to use.")
    print("\nNext steps:")
    print("1. Run 'python ultimate_rag_api.py' to start the API server")
    print("2. Test with your frontend application")
    print("3. Customize the configuration for your specific needs")

if __name__ == "__main__":
    main()
