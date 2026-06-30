##for testing purpose

from app import ask_npa_question, ask_with_sql

print("*"*20)
print("\nResults from rag(Semantic Search) pipeline")
print(ask_npa_question("Which bank had highest NPA in 2024?"))
print("\n--------------------------------------------------------\n")
print("Exact results from Sql generation pipeline")
print(ask_with_sql("Which bank had highest NPA in 2024?"))
