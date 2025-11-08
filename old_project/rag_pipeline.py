# import os
# import json
# import time
# import hashlib
# import logging
# import numpy as np
# from difflib import get_close_matches
# from google.cloud import bigquery, aiplatform
# from google.oauth2 import service_account
# from qdrant_client import QdrantClient
# from qdrant_client.models import VectorParams, Distance, PointStruct
# from vertexai.language_models import TextEmbeddingModel
# from vertexai.generative_models import GenerativeModel

# # -----------------------------
# # CONFIGURATION
# # -----------------------------
# BQ_PROJECT = "skf-bq-analytics-hub"
# BQ_DATASET = "mrep_skf"
# VERTEX_PROJECT = "tcl-vertex-ai"
# BQ_CREDENTIALS = "credentials/bm-409@skf-bq-analytics-hub.iam.gserviceaccount.com.json"
# VERTEX_CREDENTIALS = "credentials/tcl-vertex-ai-58aa168440df.json"
# LOCATION = "asia-southeast1"

# EMBEDDING_MODEL = "gemini-embedding-001"
# GENERATION_MODEL = "gemini-2.5-flash"
# QDRANT_COLLECTION = "pharma_ultimate_embeddings"
# TABLES = ["sales_details"]
# KEYWORD_CACHE_FILE = "keyword_cache.json"

# logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# # -----------------------------
# # 1️⃣ Initialize BigQuery & Vertex AI
# # -----------------------------
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = BQ_CREDENTIALS
# bq_client = bigquery.Client(project=BQ_PROJECT)

# vertex_creds = service_account.Credentials.from_service_account_file(VERTEX_CREDENTIALS)
# aiplatform.init(project=VERTEX_PROJECT, location=LOCATION, credentials=vertex_creds)

# embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
# llm = GenerativeModel(GENERATION_MODEL)

# # -----------------------------
# # 2️⃣ Initialize Qdrant
# # -----------------------------
# qdrant_client = QdrantClient("localhost", port=6333)
# existing_collections = [c.name for c in qdrant_client.get_collections().collections]
# if QDRANT_COLLECTION not in existing_collections:
#     logging.info(f"Creating Qdrant collection '{QDRANT_COLLECTION}'")
#     qdrant_client.create_collection(
#         collection_name=QDRANT_COLLECTION,
#         vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
#     )

# # -----------------------------
# # 3️⃣ Helpers
# # -----------------------------
# def clean_sql(sql: str) -> str:
#     return sql.replace("```sql", "").replace("```", "").strip()

# def get_partition_columns(tables):
#     partition_cols = {}
#     for t in tables:
#         table_ref = f"{BQ_DATASET}.{t}"
#         table = bq_client.get_table(table_ref)
#         if table.time_partitioning and table.time_partitioning.field:
#             partition_cols[t] = table.time_partitioning.field
#     return partition_cols

# PARTITION_COLUMNS = get_partition_columns(TABLES)

# # -----------------------------
# # 4️⃣ Embed tables + schema
# # -----------------------------
# def embed_schema(tables):
#     points = []
#     for idx, table_name in enumerate(tables):
#         table_ref = f"{BQ_DATASET}.{table_name}"
#         table = bq_client.get_table(table_ref)
#         schema_text = ", ".join([f"{f.name} ({f.field_type})" for f in table.schema])
#         embedding = embedding_model.get_embeddings([schema_text])[0].values

#         points.append(PointStruct(
#             id=idx,
#             vector=embedding,
#             payload={"table": table_name, "schema": schema_text}
#         ))

#     if points:
#         qdrant_client.recreate_collection(
#             collection_name=QDRANT_COLLECTION,
#             vectors_config=VectorParams(size=len(points[0].vector), distance=Distance.COSINE)
#         )
#         qdrant_client.upsert(collection_name=QDRANT_COLLECTION, points=points)
#     return points

# # -----------------------------
# # 5️⃣ Dynamic keyword embeddings
# # -----------------------------
# def get_dynamic_keywords():
#     keywords = set()
#     for table_name in TABLES:
#         sql = f"""
#         SELECT DISTINCT depot_name AS keyword FROM `{BQ_PROJECT}.{BQ_DATASET}.{table_name}`
#         UNION DISTINCT
#         SELECT DISTINCT client_name AS keyword FROM `{BQ_PROJECT}.{BQ_DATASET}.{table_name}`
#         UNION DISTINCT
#         SELECT DISTINCT item_name AS keyword FROM `{BQ_PROJECT}.{BQ_DATASET}.{table_name}`
#         """
#         results = bq_client.query(sql).result()
#         for row in results:
#             if row.keyword:
#                 keywords.add(row.keyword.strip())
#     return sorted(list(keywords))

# def compute_schema_hash(tables):
#     m = hashlib.md5()
#     for t in tables:
#         table = bq_client.get_table(f"{BQ_DATASET}.{t}")
#         m.update(json.dumps([f.to_api_repr() for f in table.schema]).encode())
#     return m.hexdigest()

# def embed_keywords_with_cache(keywords):
#     schema_hash = compute_schema_hash(TABLES)
#     if os.path.exists(KEYWORD_CACHE_FILE):
#         with open(KEYWORD_CACHE_FILE, "r") as f:
#             cache_data = json.load(f)
#         if cache_data.get("hash") == schema_hash:
#             logging.info("✅ Loaded keyword embeddings from cache.")
#             return {k: np.array(v) for k, v in cache_data["embeddings"].items()}

#     logging.info("⏳ Generating keyword embeddings...")
#     embeddings = embedding_model.get_embeddings(keywords)
#     cache = {kw: emb.values for kw, emb in zip(keywords, embeddings)}
#     with open(KEYWORD_CACHE_FILE, "w") as f:
#         json.dump({"hash": schema_hash, "embeddings": cache}, f)
#     logging.info(f"✅ Cached {len(keywords)} keyword embeddings.")
#     return {k: np.array(v) for k, v in cache.items()}

# # -----------------------------
# # 6️⃣ Semantic correction
# # -----------------------------
# def semantic_correct(word, keyword_embeddings):
#     # normalize input
#     word_norm = word.strip().lower()
    
#     # fuzzy match first
#     keyword_norm_map = {k.lower(): k for k in keyword_embeddings.keys()}
#     fuzzy_matches = get_close_matches(word_norm, keyword_norm_map.keys(), n=1, cutoff=0.7)
#     if fuzzy_matches:
#         return keyword_norm_map[fuzzy_matches[0]]

#     # Fall back to embedding similarity
#     word_emb = np.array(embedding_model.get_embeddings([word])[0].values)
#     word_emb /= np.linalg.norm(word_emb)
#     best_match, best_score = word, -1
#     for kw, emb in keyword_embeddings.items():
#         sim = np.dot(word_emb, emb/np.linalg.norm(emb))
#         if sim > best_score:
#             best_score, best_match = sim, kw
#     return best_match


# def preprocess_query_semantic(query, keyword_embeddings):
#     words = query.split()
#     corrected_words = []
#     i = 0
#     while i < len(words):
#         for n in [3,2,1]:
#             if i+n <= len(words):
#                 phrase = " ".join(words[i:i+n])
#                 corrected_phrase = semantic_correct(phrase, keyword_embeddings)
#                 if corrected_phrase != phrase:
#                     corrected_words.append(corrected_phrase)
#                     i += n
#                     break
#         else:
#             corrected_words.append(words[i])
#             i += 1
#     return " ".join(corrected_words)

# # -----------------------------
# # 7️⃣ RAG retrieval
# # -----------------------------
# def retrieve_context(query, top_k=5):
#     query_emb = embedding_model.get_embeddings([query])[0].values
#     hits = qdrant_client.query_points(
#         collection_name=QDRANT_COLLECTION, 
#         query=query_emb, 
#         limit=top_k
#     ).points
#     return [{"table": h.payload["table"], "schema": h.payload["schema"]} for h in hits]

# # -----------------------------
# # 8️⃣ SQL generation
# # -----------------------------
# def generate_sql(query, context, max_retries=2):
#     for c in context:
#         c["table"] = f"{BQ_DATASET}.{c['table']}"
#     context_text = json.dumps(context, indent=2)
#     partition_text = json.dumps(PARTITION_COLUMNS, indent=2)

#     prompt = f"""
# You are a BigQuery SQL expert. Your name is Sabbir.
# User question: "{query}"

# Relevant tables and schema:
# {context_text}

# Partitioned tables info (table: partition_column):
# {partition_text}

# Instructions:
# - Write a valid BigQuery SQL using fully qualified table names.
# - Include joins, aggregations, and partition filters if needed.
# - If table is partitioned, always filter using BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD'.
# - Do not explain, do not add markdown fences.
# - Also give a perfect name of every column in the SQL query.
# - CRITICAL: If using UNION ALL, ensure all columns have compatible data types by using CAST() functions.
# - For UNION ALL queries, cast all columns to consistent types (e.g., CAST(column AS STRING) or CAST(column AS FLOAT64)).
# - Always use proper data type casting to prevent "incompatible types" errors.
# """
#     for attempt in range(max_retries):
#         response = llm.generate_content(prompt)
#         if response.candidates:
#             sql = response.candidates[0].content.parts[0].text.strip()
#             if sql:
#                 return clean_sql(sql)
#         logging.warning(f"SQL generation attempt {attempt+1} failed. Retrying...")
#     raise ValueError("Failed to generate SQL after retries.")

# # -----------------------------
# # 9️⃣ SQL execution & summary
# # -----------------------------
# # def execute_and_summarize(sql):
# #     sql = clean_sql(sql)
# #     try:
# #         job = bq_client.query(sql, job_config=bigquery.QueryJobConfig(dry_run=False))
# #         results = [dict(r) for r in job.result()]
# #     except Exception as e:
# #         return {"error": str(e), "sql": sql, "charts": []}

# #     if not results:
# #         return {"results": [], "summary": "No results found.", "sql": sql, "charts": []}

# #     summary_prompt = f"Summarize the following results in plain business terms:\n{json.dumps(results, indent=2)}"
# #     summary = llm.generate_content(summary_prompt).candidates[0].content.parts[0].text.strip()

# #     charts = []
# #     numeric_keys = [k for k in results[0] if isinstance(results[0][k], (int, float))]
# #     cat_keys = [k for k in results[0] if not isinstance(results[0][k], (int, float))]
# #     x_axis = cat_keys[0] if cat_keys else list(results[0].keys())[0]

# #     for key in numeric_keys:
# #         if len(results) <= 5:
# #             chart_type = "pie"
# #         elif "date" in x_axis.lower() or "time" in x_axis.lower():
# #             chart_type = "line"
# #         else:
# #             chart_type = "bar"

# #         charts.append({
# #             "type": chart_type,
# #             "title": key,
# #             "labels": [str(r[x_axis]) for r in results],
# #             "values": [r[key] for r in results]
# #         })

# #     return {"results": results, "summary": summary, "sql": sql, "charts": charts}


# def execute_and_summarize(sql, user_chart_type=None):
#     sql = clean_sql(sql)
#     try:
#         job = bq_client.query(sql, job_config=bigquery.QueryJobConfig(dry_run=False))
#         results = [dict(r) for r in job.result()]
#     except Exception as e:
#         return {"error": str(e), "sql": sql, "charts": []}

#     if not results:
#         return {"results": [], "summary": "No results found.", "sql": sql, "charts": []}

#     # Summary
#     summary_prompt = f"Summarize the following results in plain business terms:\n{json.dumps(results, indent=2)}"
#     summary = llm.generate_content(summary_prompt).candidates[0].content.parts[0].text.strip()

#     # Default chart heuristic
#     charts = []
#     numeric_keys = [k for k in results[0] if isinstance(results[0][k], (int, float))]
#     cat_keys = [k for k in results[0] if not isinstance(results[0][k], (int, float))]
#     x_axis = cat_keys[0] if cat_keys else list(results[0].keys())[0]

#     for key in numeric_keys:
#         # Use user choice if provided
#         chart_type = user_chart_type if user_chart_type else (
#             "pie" if len(results) <= 5 else
#             "line" if "date" in x_axis.lower() or "time" in x_axis.lower() else
#             "bar"
#         )

#         charts.append({
#             "type": chart_type,
#             "title": key,
#             "labels": [str(r[x_axis]) for r in results],
#             "values": [r[key] for r in results]
#         })

#     return {"results": results, "summary": summary, "sql": sql, "charts": charts}




# # -----------------------------
# # 10️⃣ Question classification
# # -----------------------------
# def classify_question(question: str) -> str:
#     prompt = f"""
# Classify the following question into one of these:
# 1. business → sales, revenue, profit, etc.
# 2. smalltalk → greetings, thanks, chatting.
# 3. non_business → unrelated (weather, politics, math)
# Question: "{question}"
# Answer with one word: business, smalltalk, non_business
# """
#     response = llm.generate_content(prompt)
#     text = response.text.strip().lower().split()[0]
#     return text

# # -----------------------------
# # 11️⃣ Main RAG controller
# # -----------------------------


# # def ask_intelligent(question: str):
# #     category = classify_question(question)
# #     logging.info(f"[Classifier] → {category}")

# #     if category == "smalltalk":
# #         return {"summary": "Hello! Ask a business-related question.", "results": None, "sql": None, "charts": []}
# #     if category == "non_business":
# #         return {"summary": "Please ask business-related questions only.", "results": None, "sql": None, "charts": []}

# #     context = retrieve_context(question)
# #     sql = generate_sql(question, context)
# #     logging.info(f"Generated SQL:\n{sql}")
# #     return execute_and_summarize(sql)



# def ask_intelligent(question: str):
#     category = classify_question(question)
#     print(f"[Classifier] → {category}")

#     if category == "smalltalk":
#         if "name" in question.lower():
#             return {"summary": "My name is sabbir — your Business Insights Assistant.", "results": None, "sql": None, "charts": []}
#         return {"summary": "Hello! Please ask a business-related question.", "results": None, "sql": None, "charts": []}

#     if category == "non_business":
#         return {"summary": "Please ask business-related questions only.", "results": None, "sql": None, "charts": []}

#     context = retrieve_context(question)
#     sql = generate_sql(question, context)
#     print(f"Generated SQL:\n{sql}")
#     return execute_and_summarize(sql)

# # -----------------------------
# # 12️⃣ Initialization & loop
# # -----------------------------
# if __name__ == "__main__":
#     logging.info("Embedding tables & KPIs...")
#     embed_schema(TABLES)

#     logging.info("Preparing dynamic keyword embeddings...")
#     ALL_KEYWORDS = get_dynamic_keywords()
#     KEYWORD_EMBEDDINGS = embed_keywords_with_cache(ALL_KEYWORDS)

#     logging.info("✅ System ready!")
#     while True:
#         user_question = input("Ask me anything (type 'exit' to quit): ")
#         if user_question.lower() == "exit":
#             break

#         corrected_query = preprocess_query_semantic(user_question, KEYWORD_EMBEDDINGS)
#         if corrected_query != user_question:
#             logging.info(f"[Auto-Corrected] {user_question} → {corrected_query}")

#         # Ask user for chart type
#         print("Select chart type (press Enter for auto): bar, line, pie")
#         user_chart_type = input("Chart type: ").strip().lower()
#         if user_chart_type not in ["bar", "line", "pie"]:
#             user_chart_type = None

#         response = ask_intelligent(corrected_query)
        
#         # Override chart type if user selected
#         for chart in response["charts"]:
#             if user_chart_type:
#                 chart["type"] = user_chart_type

#         print(json.dumps(response, indent=2))

