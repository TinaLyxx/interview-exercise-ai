#!/usr/bin/env python3
"""
Example usage script for the Knowledge Assistant.
This script demonstrates how to interact with the API.
"""
import requests
import json
import time


def test_knowledge_assistant(base_url="http://localhost:8000"):
    """Test the Knowledge Assistant API with sample queries."""
    
    print("🧠 Knowledge Assistant API Test")
    print("=" * 50)
    
    # Test health endpoint
    print("\n1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✓ System is healthy")
        else:
            print("✗ System health check failed")
            return
    except Exception as e:
        print(f"✗ Cannot connect to API: {e}")
        print("Make sure the Knowledge Assistant is running on localhost:8000")
        return
    
    # Test sample tickets
    sample_tickets = [
        {
            "description": "Domain Suspension Issue",
            "ticket_text": "My domain was suspended and I didn't get any notice. How can I reactivate it?"
        },
        {
            "description": "WHOIS Information Update",
            "ticket_text": "I need to update my WHOIS information but can't find where to do it in my account."
        },
        {
            "description": "Domain Transfer Question",
            "ticket_text": "How do I transfer my domain to another registrar? What documents do I need?"
        },
        {
            "description": "DNS Configuration Issue",
            "ticket_text": "My website is not loading after I changed the DNS settings. Can you help me troubleshoot?"
        },
        {
            "description": "Billing Payment Problem",
            "ticket_text": "My payment was declined and now my domain renewal failed. What should I do?"
        }
    ]
    
    print(f"\n2. Testing resolve-ticket endpoint with {len(sample_tickets)} sample tickets...")
    
    for i, ticket in enumerate(sample_tickets, 1):
        print(f"\n--- Test {i}: {ticket['description']} ---")
        print(f"Query: {ticket['ticket_text']}")
        
        try:
            response = requests.post(
                f"{base_url}/resolve-ticket",
                json={"ticket_text": ticket['ticket_text']},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✓ Response generated successfully")
                print(f"Answer: {result['answer'][:100]}...")
                print(f"References: {result['references']}")
                print(f"Action Required: {result['action_required']}")
                
                # Debug: Check if references are empty
                if not result['references']:
                    print("⚠️  Warning: No references found - this might indicate:")
                    print("   - Similarity threshold too high (current: 0.7)")
                    print("   - Query doesn't match any documents")
                    print("   - Documents not properly indexed")
            else:
                print(f"✗ Error {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"✗ Request failed: {e}")
        
        # Small delay between requests
        time.sleep(1)
    
    # Test stats endpoint
    print(f"\n3. Testing system statistics...")
    try:
        response = requests.get(f"{base_url}/stats")
        if response.status_code == 200:
            stats = response.json()
            print("✓ Statistics retrieved")
            print(f"System Status: {stats.get('status', 'unknown')}")
            if 'components' in stats:
                retriever = stats['components'].get('retriever', {})
                print(f"Documents Indexed: {retriever.get('total_documents', 'unknown')}")
                print(f"LLM Model: {stats['components'].get('llm_model', 'unknown')}")
        else:
            print(f"✗ Stats request failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Stats request failed: {e}")
    
    # Test rebuild knowledge base endpoint
    print(f"\n4. Testing rebuild knowledge base...")
    try:
        print("   Rebuilding knowledge base...")
        response = requests.post(f"{base_url}/rebuild-knowledge-base")
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Knowledge base rebuilt successfully")
            print(f"Status: {result.get('status', 'unknown')}")
            print(f"Message: {result.get('message', 'No message')}")
            
            # Display rebuild statistics if available
            if 'stats' in result:
                stats = result['stats']
                print(f"Rebuild Statistics:")
                print(f"  - Total documents: {stats.get('total_documents', 'unknown')}")
                print(f"  - Embedding dimension: {stats.get('embedding_dimension', 'unknown')}")
                print(f"  - Index size: {stats.get('index_size', 'unknown')}")
        else:
            print(f"✗ Rebuild request failed: {response.status_code}")
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"✗ Rebuild request failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Knowledge Assistant test completed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the Knowledge Assistant API")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="Base URL of the Knowledge Assistant API")
    
    args = parser.parse_args()
    test_knowledge_assistant(args.url)
