import os
from graphdatascience import GraphDataScience
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

# host = st.secrets["NEO4J_URI"]
# user = st.secrets["NEO4J_USER"]
# password = st.secrets["NEO4J_PASSWORD"]
# db = st.secrets["NEO4J_DB"]

host = os.getenv('NEO4J_URI')
user = os.getenv('NEO4J_USER')
password = os.getenv('NEO4J_PASSWORD')
db = os.getenv('NEO4J_DB')

gds = GraphDataScience(
    host,
    auth=(user, password),
    aura_ds=False)

gds.set_database("neo4j")

def run_query(query, params=None):
    return gds.run_cypher(query, params=params)

if __name__ == "__main__":
    query = """
    MATCH (p:Person)-[:HAS_SYMPTOM]->(s:Symptom)
    RETURN p.name AS person, COUNT(s) AS symptom_count
    ORDER BY symptom_count DESC
    LIMIT 1
    """
    results = run_query(query)
    print(results)

