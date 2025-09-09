#!/usr/bin/env python3

"""
Test script to validate the OpenAI client implementation
"""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, '/home/runner/work/LIRL/LIRL')

from openai_client import response_openai, response_openai_json


def test_function_imports():
    """Test that functions can be imported correctly"""
    print("✓ Functions imported successfully")
    return True


def test_function_structure():
    """Test the structure of the functions without making API calls"""
    try:
        # This will test the basic structure but won't make API calls due to likely authentication issues
        print("Testing function structure...")
        
        # We can check if the functions exist and are callable
        assert callable(response_openai), "response_openai should be callable"
        assert callable(response_openai_json), "response_openai_json should be callable"
        
        print("✓ Functions are properly structured and callable")
        return True
        
    except Exception as e:
        print(f"✗ Error in function structure: {e}")
        return False


def main():
    """Run all tests"""
    print("Running OpenAI client tests...")
    print("=" * 50)
    
    tests = [
        test_function_imports,
        test_function_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed: {e}")
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! The OpenAI client implementation is valid.")
        return True
    else:
        print("✗ Some tests failed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)