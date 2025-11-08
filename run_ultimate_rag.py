#!/usr/bin/env python3
"""
Simple runner script for the Ultimate RAG Pipeline
"""

import os
import sys
import subprocess
import time

def check_dependencies():
    """Check if required dependencies are installed"""
    import importlib

    # map pip package name -> python import path
    package_to_module = {
        'flask': 'flask',
        'flask-cors': 'flask_cors',
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',
        'networkx': 'networkx',
        'google-cloud-bigquery': 'google.cloud.bigquery',
        'qdrant-client': 'qdrant_client',
        'vertexai': 'vertexai',
    }

    missing_packages = []

    for pkg, module in package_to_module.items():
        try:
            importlib.import_module(module)
        except Exception:
            missing_packages.append(pkg)

    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall them with:")
        print("pip install -r requirements_ultimate.txt")
        return False

    print("✅ All required packages are installed!")
    return True

def check_services():
    """Check if required services are running"""
    print("\n🔍 Checking required services...")
    
    # Check Qdrant
    try:
        import requests
        response = requests.get("http://localhost:6333/collections", timeout=5)
        if response.status_code == 200:
            print("✅ Qdrant is running on localhost:6333")
        else:
            print("❌ Qdrant is not responding properly")
            return False
    except:
        print("❌ Qdrant is not running. Start it with:")
        print("docker run -p 6333:6333 qdrant/qdrant")
        return False
    
    # Check credentials
    if not os.path.exists("credentials"):
        print("❌ Credentials directory not found")
        print("Please ensure your Google Cloud credentials are in the 'credentials' directory")
        return False
    
    print("✅ Credentials directory found")
    return True

def run_tests():
    """Run the test suite"""
    print("\n🧪 Running tests...")
    try:
        result = subprocess.run([sys.executable, "test_ultimate_rag.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ All tests passed!")
            return True
        else:
            print("❌ Tests failed:")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return False

def start_api_server():
    """Start the API server"""
    print("\n🚀 Starting Ultimate RAG API Server...")
    print("Server will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        subprocess.run([sys.executable, "ultimate_rag_api.py"])
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")

def main():
    """Main function"""
    print("🚀 Ultimate RAG Pipeline Launcher")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check services
    if not check_services():
        sys.exit(1)
    
    # Ask user what to do
    print("\nWhat would you like to do?")
    print("1. Run tests")
    print("2. Start API server")
    print("3. Run tests and start server")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        if run_tests():
            print("\n✅ Tests completed successfully!")
        else:
            print("\n❌ Tests failed!")
            sys.exit(1)
    
    elif choice == "2":
        start_api_server()
    
    elif choice == "3":
        if run_tests():
            print("\n✅ Tests passed! Starting server...")
            time.sleep(2)
            start_api_server()
        else:
            print("\n❌ Tests failed! Not starting server.")
            sys.exit(1)
    
    elif choice == "4":
        print("👋 Goodbye!")
        sys.exit(0)
    
    else:
        print("❌ Invalid choice. Please run the script again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
