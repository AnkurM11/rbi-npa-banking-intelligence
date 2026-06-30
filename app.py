import streamlit as st
import pandas as pd
import re
from urllib.parse import quote_plus
import os
from sqlalchemy import create_engine
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


load_dotenv(override=True)


##supabase Db connection system
# Fetch variables
USER = os.getenv("user")
PASSWORD = quote_plus(os.getenv("password"))
HOST = os.getenv("host")
PORT = os.getenv("port")
DBNAME = os.getenv("dbname")

print(HOST)
# Construct the SQLAlchemy connection string
DATABASE_URL = f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}?sslmode=require"

# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URL)
# If using Transaction Pooler or Session Pooler, we want to ensure we disable SQLAlchemy client side pooling -
# https://docs.sqlalchemy.org/en/20/core/pooling.html#switching-pool-implementations
# engine = create_engine(DATABASE_URL, poolclass=NullPool)

# Test the connection
try:
    with engine.connect() as connection:
        print("Connection successful!")
except Exception as e:
    print(f"Failed to connect: {e}")


##groq client connection setup
##groq api key load from .env file
client= Groq(api_key=os.getenv("GROQ_API_KEY"))

##chromadb setup with catching
@st.cache_resource
def setup_chromadb():
    ##chromadb client creation
    client = chromadb.Client()

    ##collection creation means schema for storing data in chromadb 
    collection = client.create_collection("npa_data")

    ##model load for sentence trasfromer 
    model = SentenceTransformer('all-MiniLM-L6-v2')

    ##RAG pipeline sentence creation from the structured data story creation
    query = '''
                select year, bank_name, gross_npa, gross_advances, npa_ratio ,bank_type, data_quality_flag, risk_category, rnk
                from gold_bank_ranking order by year, rnk
            '''
    
    df = pd.read_sql(query,engine)

    ##now we will create a story by adding one column as text into the dataframe with apply function row by row using axis=1 
    df['text'] =  df.apply(lambda row :f"{row['bank_name']} had gross npa of {row['gross_npa']} crores"
                            f"with npa ratio of {row['npa_ratio']}% in {row['year']}. "
                            f"it is a {row['bank_type']} ranked {row['rnk']} with risk category of {row['risk_category']}.",
                            axis=1)
    ##now this column 1st convert to list then provide them a unique ids to identify each and every sentence and embedding witht his ids
    texts = df['text'].tolist()
    ##now add ids to each list text
    ids = [f"npa_{x}"  for x in range(len(df))]

    ##embeddings creation ##texts list ko encode karke wapis se list me convert karo 
    embedding = model.encode(texts).tolist()

    ##now to store this in chromadb we need three things documents(orginal text), embeddings, ids for each embedding row wise
    collection.add(
        documents=texts,
        embeddings=embedding,
        ids=ids
    )

    return collection, model


##function ke bahar dono variables use ke liye ready
collection,model = setup_chromadb()


##1st rag function for user query and semantic search results ko consolidate karke llm ko denge to get appropriate response
##ist lets create a function in which user will ask question and semantic search will happen from chromadb and will get resutls and no llm involved 
def ask_npa_question(question):
    try:
        results = collection.query(
            query_texts= [question],
            n_results=5
        )
        ##isse jo resutls aayenge vo dictionary me aayegenge key value like documents, distances and ids
        ##so unn results ko combine karna hai line by line and groq(llm) ko share karna hai 

        context=""
        for x in results['documents'][0]:
            context += x + "\n"
        
        ##now iss context variable me 1st query ke resutls as string stored hai 
        ##this we will share with groq to get consolidated responses
        ##niche messages variable jo hai volist of dictionary hai 

        response = client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[{"role": "system", "content": "You are a bank analyst. Answer in natural language based on data shared in String Format."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ]
        )
    except Exception as e:
        print(f"Sorry , Couldn't generate a valid response.")
    return response.choices[0].message.content


##above rag was having issues as it works with cosine similarity 
##but it gives expected resutls not the exact resutls it is good when we are having unstructured data and for structered data we have created one sql query generator which will give us better results on structured data
##sql generation function##
def sql_query_generator(question):
    try:
        schema = """ Table name : gold_bank_ranking,
                    Schema : [{'year' : 'int8'}, {'bank_name' : 'text'}, {'gross_npa' : 'float8'}, {'gross_advances' : 'float8'},{'npa_ratio' : 'float8'}, {'bank_type' : 'text'}, {'data_quality_flag' : 'text'}, {'risk_category' : 'text'}, {'rnk'} :'int8']
                """
        
        response = client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[
                {"role": "system", "content": """You are a SQL expert. Generate direct PostgreSQL query, no explanation. 
                    Always SELECT the relevant metric/value column being asked about along with bank_name when comparing banks - never return just the name without the value being compared.
                    IMPORTANT: When using aggregate functions like COUNT, SUM, AVG, MAX, MIN - only include other columns in SELECT if they are also in GROUP BY clause. If the question doesn't need a specific column breakdown, keep the query simple with just the aggregate.
                    IMPORTANT: Always add NULLS LAST when using ORDER BY DESC, and NULLS FIRST when using ORDER BY ASC."""},
                {"role": "user", "content": f"Schema: {schema}\nQuestion: {question}"}
            ],
            temperature=0
        )
        ##aab yaha se jo response aayega vo bhut hi bekar aayega usko parse karke sql query me karna padega using re
        raw_sql = response.choices[0].message.content

        ##ye jo cleaned sql hai isse query karke supabase database se results fetch karke hum log output nikal ke phir usko de denge llm ko to get exact response
        cleaned_sql = re.sub(r'```sql\n?|```', '', raw_sql)
        cleaned_sql = cleaned_sql.strip().rstrip(';')
    except Exception as e:
        print(f"Sorry, Couldn't generate a valid response.")

    return cleaned_sql

def ask_with_sql(question):
    clean_sql = sql_query_generator(question)
    result_df = pd.read_sql(clean_sql,engine)
    result_text = result_df.to_string(index=False)

    response = client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[
            {"role": "system", "content": "You are a bank analyst. Answer in natural language based on data shared in String Format."},
            {"role": "user", "content": f"Question: {question}\nData: {result_text}"}
        ],
        temperature=0
    )

    return response.choices[0].message.content


##now streamlit ui finally

# ===== STREAMLIT UI =====
st.set_page_config(
    page_title="RBI NPA Intelligence Chatbot", 
    page_icon="🏦",
    layout="wide"  # ← Full width use karega
)

# Center alignment ke liye
st.markdown("""
    <div style='text-align: center;'>
        <h1>🏦 RBI NPA Banking Intelligence Chatbot</h1>
        <p style='font-size: 18px;'>Ask questions about Indian Bank NPA data (2004-2025)</p>
        <p style='color: gray;'>Powered by dbt + Supabase + ChromaDB + Groq LLM</p>
    </div>
""", unsafe_allow_html=True)

st.divider()

question = st.text_input("💬 Ask your question:", placeholder="e.g. Which bank had highest NPA in 2024?")

if question:
    st.divider()
    col1, gap, col2 = st.columns([1, 0.1, 1])  # ← Beech mein gap column
    
    with col1:
        st.subheader("🔍 RAG (Semantic Search)")
        with st.spinner("Searching..."):
            rag_answer = ask_npa_question(question)
        st.info(rag_answer)
    
    with col2:
        st.subheader("📊 Text-to-SQL (Exact Query)")
        with st.spinner("Generating SQL..."):
            sql_answer = ask_with_sql(question)
        st.success(sql_answer)

st.divider()
st.caption("Built by Ankur Madhukar | Data Engineering Portfolio Project")

