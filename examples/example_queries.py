"""
Example Queries for Different Document Types
==========================================

This module contains sample queries categorized by document type
to help users test the RAG system effectively.

Usage:
    from examples.example_queries import get_queries_by_type
    
    queries = get_queries_by_type("technical")
    for query in queries:
        answer = rag.query(query)
        print(f"Q: {query}")
        print(f"A: {answer}\n")
"""

from typing import Dict, List

# Technical Documentation Queries
TECHNICAL_QUERIES = [
    "What are the system requirements mentioned in this document?",
    "What are the technical specifications?",
    "How should the system architecture be designed?",
    "What are the performance benchmarks or criteria?",
    "What are the hardware requirements?",
    "What software dependencies are mentioned?",
    "How should the system be configured?",
    "What are the integration guidelines?",
    "What are the API specifications?",
    "What testing procedures are recommended?",
    "What are the deployment requirements?",
    "How should monitoring be implemented?",
    "What are the scalability considerations?",
    "What troubleshooting steps are provided?",
    "What are the maintenance procedures?"
]

# Security and Compliance Queries
SECURITY_QUERIES = [
    "What are the security requirements mentioned?",
    "How should access control be implemented?",
    "What authentication methods are required?",
    "What are the data protection measures?",
    "How should sensitive data be handled?",
    "What encryption standards are mentioned?",
    "What are the compliance requirements?",
    "How should security audits be conducted?",
    "What are the incident response procedures?",
    "What backup and recovery measures are required?",
    "How should user permissions be managed?",
    "What are the network security requirements?",
    "What logging and monitoring is required for security?",
    "How should security updates be handled?",
    "What are the data retention policies?"
]

# Policy and Legal Queries
POLICY_QUERIES = [
    "What are the main policies outlined in this document?",
    "What are the compliance requirements?",
    "What are the legal obligations mentioned?",
    "How should disputes be resolved?",
    "What are the terms and conditions?",
    "What are the liability limitations?",
    "How should data privacy be protected?",
    "What are the intellectual property rights?",
    "What are the termination clauses?",
    "How should amendments be handled?",
    "What are the governing laws?",
    "What notification requirements exist?",
    "How should confidential information be handled?",
    "What are the penalty or sanction provisions?",
    "What reporting obligations are specified?"
]

# Business and Operational Queries
BUSINESS_QUERIES = [
    "What are the business objectives outlined?",
    "What are the key performance indicators (KPIs)?",
    "How should processes be implemented?",
    "What are the operational procedures?",
    "What are the resource requirements?",
    "How should quality control be managed?",
    "What are the cost considerations?",
    "How should project management be approached?",
    "What are the timeline and milestones?",
    "How should communication be handled?",
    "What are the training requirements?",
    "How should documentation be maintained?",
    "What are the risk management strategies?",
    "How should vendor relationships be managed?",
    "What are the success criteria?"
]

# Medical and Healthcare Queries
MEDICAL_QUERIES = [
    "What are the medical procedures described?",
    "What are the diagnostic criteria mentioned?",
    "What treatment options are available?",
    "What are the contraindications or warnings?",
    "How should medications be administered?",
    "What are the side effects or adverse reactions?",
    "What monitoring is required during treatment?",
    "What are the patient care guidelines?",
    "How should emergencies be handled?",
    "What are the infection control measures?",
    "What documentation is required?",
    "How should equipment be maintained?",
    "What are the quality assurance requirements?",
    "What training is required for staff?",
    "What are the ethical considerations?"
]

# Research and Academic Queries
RESEARCH_QUERIES = [
    "What is the main research question or hypothesis?",
    "What methodology was used in this study?",
    "What are the key findings or results?",
    "What are the limitations of this research?",
    "How was data collected and analyzed?",
    "What are the statistical methods used?",
    "What are the implications of the findings?",
    "How does this research compare to previous studies?",
    "What are the recommendations for future research?",
    "What are the practical applications?",
    "What ethical considerations were addressed?",
    "How was validity and reliability ensured?",
    "What are the theoretical contributions?",
    "What are the sample characteristics?",
    "What are the conclusions drawn?"
]

# General Analysis Queries
GENERAL_QUERIES = [
    "What are the main topics covered in this document?",
    "Can you provide a summary of the key points?",
    "What are the most important sections?",
    "How is this document structured?",
    "What are the primary recommendations?",
    "What problems or challenges are identified?",
    "What solutions are proposed?",
    "Who is the target audience for this document?",
    "What actions are required from readers?",
    "What are the next steps mentioned?",
    "What references or sources are cited?",
    "What appendices or supplementary materials exist?",
    "What definitions or terminology are provided?",
    "What examples or case studies are included?",
    "What contact information is provided?"
]

# Query categories mapping
QUERY_CATEGORIES = {
    "technical": TECHNICAL_QUERIES,
    "security": SECURITY_QUERIES,
    "policy": POLICY_QUERIES,
    "business": BUSINESS_QUERIES,
    "medical": MEDICAL_QUERIES,
    "research": RESEARCH_QUERIES,
    "general": GENERAL_QUERIES
}

def get_queries_by_type(document_type: str) -> List[str]:
    """
    Get sample queries for a specific document type.
    
    Args:
        document_type: Type of document ("technical", "security", "policy", 
                      "business", "medical", "research", "general")
    
    Returns:
        List of sample queries for the specified document type
    
    Raises:
        ValueError: If document_type is not recognized
    """
    if document_type.lower() not in QUERY_CATEGORIES:
        available_types = ", ".join(QUERY_CATEGORIES.keys())
        raise ValueError(f"Unknown document type: {document_type}. Available types: {available_types}")
    
    return QUERY_CATEGORIES[document_type.lower()]

def get_all_categories() -> List[str]:
    """Get list of all available query categories."""
    return list(QUERY_CATEGORIES.keys())

def get_random_queries(document_type: str, count: int = 5) -> List[str]:
    """
    Get a random sample of queries for a document type.
    
    Args:
        document_type: Type of document
        count: Number of queries to return
        
    Returns:
        Random sample of queries
    """
    import random
    
    queries = get_queries_by_type(document_type)
    return random.sample(queries, min(count, len(queries)))

def demo_queries_by_type():
    """Demonstrate queries for each document type."""
    print("📋 Sample Queries by Document Type")
    print("=" * 50)
    
    for doc_type in get_all_categories():
        print(f"\n🔍 {doc_type.upper()} DOCUMENT QUERIES:")
        print("-" * 30)
        
        queries = get_queries_by_type(doc_type)
        for i, query in enumerate(queries[:3], 1):  # Show first 3 queries
            print(f"{i}. {query}")
        
        if len(queries) > 3:
            print(f"   ... and {len(queries) - 3} more queries")

if __name__ == "__main__":
    # Run demonstration
    demo_queries_by_type()
    
    # Example usage
    print("\n" + "=" * 50)
    print("📝 Example Usage:")
    print("=" * 50)
    
    print(f"\n# Get technical queries")
    print(f"technical_queries = get_queries_by_type('technical')")
    print(f"# Returns {len(get_queries_by_type('technical'))} queries")
    
    print(f"\n# Get random sample")
    print(f"random_queries = get_random_queries('business', 3)")
    sample_queries = get_random_queries('business', 3)
    for i, query in enumerate(sample_queries, 1):
        print(f"# {i}. {query}")
