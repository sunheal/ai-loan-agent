# ===============================
# Query Classification Prompt
# ===============================
CLASSIFY_QUERY_PROMT = """
Classify the following loan-related user query into one category.

Categories:
- informational
- eligibility
- rate
- document
- unsupported

Query:
"{query}"

Return ONLY the category name.
""".strip()


# ===============================
# Retrieval Validation Prompt
# ===============================
VALIDATE_RETRIEVAL_SYSTEM_RPOMT = (
  "You are a lending policy assistant."
  "You must answer strictly using the provided context."
  "Do not guess, infer, or add information that is not explicitly stated."
)

VALIDATE_RETRIEVAL_USER_RPOMT = """
Context:
{context}

Question:
{query}

If the answer cannot be fully derived from the context, response with EXACTLY this phrase:

INSUFFICIENT_CONTEXT
""".strip()


# ===============================
# Answer Formatting Prompt
# ===============================
FORMAT_ANSWER_PROMT = """
Rewrite the following answer in a clear, professional, customer-friendly tone. Do NOT add new information or modify the meaning.

Answer:
{answer}
""".strip()