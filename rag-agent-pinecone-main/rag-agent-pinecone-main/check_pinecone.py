#!/usr/bin/env python3
"""
Simple script to check Pinecone account and show available indexes.
"""

import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()

def check_pinecone_account():
    """Check Pinecone account and show available indexes."""
    print("🔍 Checking Pinecone Account...")
    
    # Get API key
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("❌ PINECONE_API_KEY not found in .env file")
        return
    
    print(f"✅ PINECONE_API_KEY: {api_key[:20]}...")
    
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=api_key)
        
        # List all indexes
        print("\n📋 Available Indexes:")
        indexes = pc.list_indexes()
        
        if not indexes:
            print("   No indexes found")
        else:
            for index in indexes:
                print(f"   📁 {index.name}")
                print(f"      Environment: {index.host}")
                print(f"      Dimension: {index.dimension}")
                print(f"      Metric: {index.metric}")
                print(f"      Status: {index.status.state}")
                print()
        
        # Show account info
        print("🏢 Account Information:")
        print(f"   Project ID: {pc.whoami()}")
        
    except Exception as e:
        print(f"❌ Error connecting to Pinecone: {e}")
        print("\n💡 Make sure your PINECONE_API_KEY is correct")

if __name__ == "__main__":
    check_pinecone_account()
