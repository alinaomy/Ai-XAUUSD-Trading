#!/usr/bin/env python3
"""
arXiv Paper Search Helper

This script helps you search for papers on arXiv, including your own submission.
"""

import requests
import json
import sys
from datetime import datetime

def search_arxiv(query, max_results=10, sort_by="submittedDate", sort_order="descending"):
    """
    Search arXiv for papers matching the query.

    Args:
        query: Search query (title, author, keywords)
        max_results: Maximum number of results to return
        sort_by: Sort field (submittedDate, relevance, etc.)
        sort_order: Sort order (ascending, descending)
    """
    base_url = "http://export.arxiv.org/api/query"

    params = {
        "search_query": query,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": sort_order
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()

        # Parse the XML response (simplified)
        print(f"🔍 Searching arXiv for: '{query}'")
        print(f"📊 Found results (API response received)")
        print(f"🌐 Full search URL: {response.url}")
        print()

        # For now, just show the URL for manual checking
        print("📋 To view results manually:")
        print(f"   Visit: https://arxiv.org/search/?query={query.replace(' ', '+')}")
        print()

        return True

    except Exception as e:
        print(f"❌ Search failed: {e}")
        return False

def search_your_paper():
    """Search for your specific paper."""
    print("🔍 Searching for your AI-XAUUSD Trading paper...")
    print()

    # Search by title
    title_query = 'ti:"AI-Driven XAUUSD Trading System"'
    print("1. Searching by title:")
    search_arxiv(title_query, max_results=5)

    # Search by author
    author_query = 'au:JonusNattapong'
    print("2. Searching by author (JonusNattapong):")
    search_arxiv(author_query, max_results=5)

    # Search by keywords
    keyword_query = 'ti:XAUUSD AND ti:trading AND ti:reinforcement'
    print("3. Searching by keywords:")
    search_arxiv(keyword_query, max_results=5)

def search_related_papers():
    """Search for related research papers."""
    print("🔍 Searching for related research papers...")
    print()

    queries = [
        'ti:reinforcement AND ti:learning AND ti:trading',
        'ti:ensemble AND ti:methods AND ti:finance',
        'ti:market AND ti:regime AND ti:detection',
        'ti:XAUUSD OR ti:gold AND ti:algorithmic'
    ]

    for i, query in enumerate(queries, 1):
        print(f"{i}. Related work:")
        search_arxiv(query, max_results=3)
        print()

def main():
    """Main function."""
    print("🚀 arXiv Paper Search Helper")
    print("=" * 40)

    if len(sys.argv) > 1:
        # Custom search query
        query = " ".join(sys.argv[1:])
        search_arxiv(query)
    else:
        # Default searches
        search_your_paper()
        print("\n" + "=" * 40)
        search_related_papers()

    print("📖 Additional search tips:")
    print("• Use quotes for exact phrases: 'reinforcement learning'")
    print("• Combine terms: ti:title AND au:author")
    print("• Search prefixes: ti:, au:, abs:, cat:")
    print("• Visit: https://arxiv.org/help/api/user-manual#query_details")

if __name__ == "__main__":
    main()