#!/usr/bin/env python3
"""Command line interface for LIRL models."""

import argparse
import json
import sys
from typing import Optional

from lirl.models import list_available_models, get_model_info, get_available_model_types, _registry


def print_models_table(models: list, show_details: bool = False):
    """Print models in a formatted table."""
    if not models:
        print("No models found matching the criteria.")
        return
    
    if show_details:
        print(f"{'Name':<20} {'Type':<20} {'Version':<10} {'Status':<15} {'Description'}")
        print("-" * 85)
        for model_name in models:
            model_info = get_model_info(model_name)
            if model_info:
                print(f"{model_info['name']:<20} {model_info['model_type']:<20} {model_info['version']:<10} {model_info['status']:<15} {model_info['description']}")
    else:
        print("Available Models:")
        for i, model_name in enumerate(models, 1):
            print(f"{i:2d}. {model_name}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="LIRL Models CLI - List and inspect available models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # List all available models
  %(prog)s --type reinforcement_learning  # List RL models only
  %(prog)s --status available              # List only available models
  %(prog)s --details                       # Show detailed information
  %(prog)s --info DQN                      # Show detailed info for DQN model
  %(prog)s --types                         # List available model types
  %(prog)s --json                          # Output in JSON format
        """
    )
    
    parser.add_argument(
        "--type", "-t",
        help="Filter by model type"
    )
    
    parser.add_argument(
        "--status", "-s",
        choices=["available", "experimental", "deprecated"],
        help="Filter by model status"
    )
    
    parser.add_argument(
        "--details", "-d",
        action="store_true",
        help="Show detailed information in table format"
    )
    
    parser.add_argument(
        "--info", "-i",
        metavar="MODEL_NAME",
        help="Show detailed information for a specific model"
    )
    
    parser.add_argument(
        "--types",
        action="store_true",
        help="List available model types"
    )
    
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output in JSON format"
    )
    
    args = parser.parse_args()
    
    try:
        # Show model types
        if args.types:
            types = get_available_model_types()
            if args.json:
                print(json.dumps(types, indent=2))
            else:
                print("Available Model Types:")
                for model_type in types:
                    print(f"  - {model_type}")
            return
        
        # Show specific model info
        if args.info:
            model_info = get_model_info(args.info)
            if model_info:
                if args.json:
                    print(json.dumps(model_info, indent=2))
                else:
                    print(f"Model: {model_info['name']}")
                    print(f"Description: {model_info['description']}")
                    print(f"Type: {model_info['model_type']}")
                    print(f"Version: {model_info['version']}")
                    print(f"Status: {model_info['status']}")
                    if model_info['parameters']:
                        print("Parameters:")
                        for key, value in model_info['parameters'].items():
                            print(f"  {key}: {value}")
                    if model_info['requirements']:
                        print("Requirements:")
                        for req in model_info['requirements']:
                            print(f"  - {req}")
            else:
                print(f"Model '{args.info}' not found.")
                sys.exit(1)
            return
        
        # List models
        models = list_available_models(model_type=args.type, status=args.status)
        
        if args.json:
            # Get full model info for JSON output
            models_info = []
            for model_name in models:
                model_info = get_model_info(model_name)
                if model_info:
                    models_info.append(model_info)
            print(json.dumps(models_info, indent=2))
        else:
            print_models_table(models, show_details=args.details)
            
            # Show summary
            total = len(models)
            if args.type or args.status:
                filters = []
                if args.type:
                    filters.append(f"type='{args.type}'")
                if args.status:
                    filters.append(f"status='{args.status}'")
                filter_str = f" (filtered by {', '.join(filters)})"
            else:
                filter_str = ""
            
            print(f"\nTotal: {total} models{filter_str}")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()