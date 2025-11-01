# from langchain.chains import GraphCypherQAChain
# from langchain_community.chains import GraphCypherQAChain
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

from langchain_community.graphs import Neo4jGraph
# from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from retry import retry
from timeit import default_timer as timer
from neo4j_driver import run_query 
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import os

host = os.getenv('NEO4J_URI')
user = os.getenv('NEO4J_USER')
password = os.getenv('NEO4J_PASSWORD')
db = os.getenv('NEO4J_DB')

azure_deployment = os.getenv("azure_deployment")
azure_endpoint = os.getenv("azure_endpoint")
openai_api_version = os.getenv("openai_api_version")
openai_api_key = os.getenv("openai_api_key")
openai_api_type = os.getenv("openai_api_type")
    

CYPHER_GENERATION_TEMPLATE = """You are an expert Neo4j Cypher translator who understands the question in english and convert to Cypher strictly based on the Neo4j Schema provided and following the instructions below:
1. Generate Cypher query compatible ONLY for Neo4j Version 5
2. Do not use EXISTS, SIZE keywords in the cypher. Use alias when using the WITH keyword
3. Use only Nodes and relationships mentioned in the schema
4. Always enclose the Cypher output inside 3 backticks
5. Always do a case-insensitive and fuzzy search for any properties related search. Eg: to search for a Team name use `toLower(t.name) contains 'neo4j'`
6. Always use aliases to refer the node in the query
7. Cypher is NOT SQL. So, do not mix and match the syntaxes

Schema:
Nodes:
(:Case)
(:Person)
(:Symptom)
(:Disease)
(:BodySystem)
(:Diagnosis)
(:Biological)

Relationships:
(:Case)-[:FOR]->(:Person)
(:Person)-[:HAS_SYMPTOM]->(:Symptom)
(:Person)-[:HAS_DISEASE]->(:Disease)
(:Case)-[:SEEN_ON]->(:BodySystem)
(:Disease)-[:AFFECTS]->(:BodySystem)
(:Person)-[:HAS_DIAGNOSIS]->(:Diagnosis)
(:Person)-[:SHOWED]->(:Symptom)

Samples:
Question: Which patient has the most number of symptoms?
Answer: ```MATCH (n:Person)-[:HAS_SYMPTOM]->(s:Symptom) return n.id,n.age, n.gender,count(s) as symptoms 
order by symptoms desc```
Question: Which disease affect most of my patients?
Answer: ```MATCH (d:Disease) RETURN d.name as disease, SIZE([(d)-[]-(p:Person) | p]) AS affected_patients ORDER BY affected_patients DESC LIMIT 1```
Question: Which of patients have cough?
Answer: ```MATCH (p:Person)-[:HAS_SYMPTOM]->(s:Symptom) WHERE toLower(s.description) CONTAINS 'cough' RETURN p.id, p.age, p.location, p.gender```

Question: {question}"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["question"], template=CYPHER_GENERATION_TEMPLATE
)

@retry(tries=5, delay=5)
def get_results(messages):
    start = timer()
    try:
        graph = Neo4jGraph(
            url=host, 
            username=user, 
            password=password
        )
        chain = GraphCypherQAChain.from_llm(
            AzureChatOpenAI(azure_deployment=azure_deployment,
                              azure_endpoint=azure_endpoint,
                              openai_api_version=openai_api_version,
                              openai_api_key=openai_api_key,
                              openai_api_type=openai_api_type), 
            graph=graph, verbose=True,
            return_intermediate_steps=True,
            cypher_prompt=CYPHER_GENERATION_PROMPT,
            allow_dangerous_requests=True 
        )

        if messages:
            question = messages.pop()
        else: 
            question = 'How many cases are there?'

        result = chain.invoke({"query": question})
        if result:
            generated_query = result['intermediate_steps'][0]['query']
            print("🔎 Generated Cypher:", generated_query)

            # Execute the query using your neo4j_driver
            query_results = run_query(generated_query)
            print("✅ Query Results:", query_results)
        else:
            print("❌ Chain returned None — check your schema or query.")

        return result
    except Exception as ex:
        print(ex)
    #     return "LLM Quota Exceeded. Please try again"
    finally:
        print('Cypher Generation Time : {}'.format(timer() - start))

if __name__ == "__main__":
    # question = "Which disease affect most of my patients?"
    question = "Which patient has the most number of symptoms"
    response = get_results([question])
    print("🔎 Intermediate Steps:", response.get("intermediate_steps"))
    print("✅ Final Answer:", response.get("result"))
