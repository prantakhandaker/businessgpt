# # from flask import Flask, request, jsonify, send_from_directory
# # from rag_pipeline import ask_business_question, embed_schema, TABLES

# # app = Flask(__name__, static_folder='frontend')

# # # Embed tables once at startup
# # print("Embedding tables & KPIs...")
# # embed_schema(TABLES)
# # print("✅ Embeddings ready!")

# # @app.route('/')
# # def index():
# #     return send_from_directory(app.static_folder, 'index.html')

# # @app.route('/ask', methods=['POST'])
# # def ask():
# #     data = request.json
# #     question = data.get('question', '')
# #     if not question:
# #         return jsonify({"error": "No question provided."})
# #     response = ask_business_question(question)
# #     return jsonify(response)

# # if __name__ == "__main__":
# #     app.run(debug=True)







# # from flask import Flask, request, jsonify, render_template
# # import rag_pipeline as rag
# # import os

# # app = Flask(__name__, template_folder="frontend")  # Ensure your index.html is in a folder named 'templates'

# # # -----------------------------
# # # 🏠 Serve Frontend
# # # -----------------------------
# # @app.route("/")
# # def home():
# #     return render_template("index.html")  # Render the HTML page

# # # -----------------------------
# # # 💬 Main Ask Endpoint
# # # -----------------------------
# # @app.route("/ask", methods=["POST"])
# # def ask():
# #     try:
# #         data = request.get_json()
# #         if not data or "question" not in data:
# #             return jsonify({"error": "Missing 'question' in request body."}), 400

# #         question = data["question"]
# #         print(f"[User] → {question}")

# #         # Use the intelligent handler from rag_pipeline.py
# #         response = rag.ask_intelligent(question)

# #         return jsonify(response)

# #     except Exception as e:
# #         print(f"[Error] {e}")
# #         return jsonify({"error": str(e)}), 500

# # # -----------------------------
# # # ⚙️ Server Runner
# # # -----------------------------
# # if __name__ == "__main__":
# #     # Initialize embeddings when server starts
# #     print("🚀 Initializing Pharma Insights GPT backend...")
# #     rag.embed_schema(rag.TABLES)
# #     print("✅ Embeddings are ready. Server is live on http://127.0.0.1:5000")
# #     app.run(host="0.0.0.0", port=5000, debug=True)






# from flask import Flask, request, jsonify, render_template
# import rag_pipeline as rag
# import os

# # -----------------------------
# # Flask App
# # -----------------------------
# app = Flask(__name__, template_folder="frontend")  # <-- Put index.html here

# # -----------------------------
# # 🏠 Serve Frontend
# # -----------------------------
# @app.route("/")
# def home():
#     return render_template("index.html")

# # -----------------------------
# # 💬 Main Ask Endpoint
# # -----------------------------
# @app.route("/ask", methods=["POST"])
# def ask():
#     try:
#         data = request.get_json()
#         if not data or "question" not in data:
#             return jsonify({"error": "Missing 'question' in request body."}), 400

#         question = data["question"]
#         print(f"[User] → {question}")

#         # Use the intelligent RAG + AI chart pipeline
#         response = rag.ask_intelligent(question)

#         # Ensure charts key exists
#         if "charts" not in response:
#             response["charts"] = []

#         return jsonify(response)

#     except Exception as e:
#         print(f"[Error] {e}")
#         return jsonify({"error": str(e), "charts": []}), 500

# # -----------------------------
# # ⚙️ Server Runner
# # -----------------------------
# if __name__ == "__main__":
#     print("🚀 Initializing Pharma Insights GPT backend...")
#     rag.embed_schema(rag.TABLES)
#     print("✅ Embeddings are ready. Server is live on http://127.0.0.1:5000")
#     app.run(host="0.0.0.0", port=5000, debug=True)
