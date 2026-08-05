from google import genai

from app.config import (
    GEMINI_API_KEY,
    MODEL_NAME
)

client = genai.Client(api_key=GEMINI_API_KEY)


class LLM:

    def answer_question(self, question, context):

        prompt = f"""
You are an expert AI assistant.

Use ONLY the provided document context to answer.

Instructions:
1. Answer clearly and accurately.
2. If the answer exists in the context, explain it in your own words.
3. Do not invent information.
4. If the answer is not available in the context, reply:
   "I couldn't find the answer in the uploaded document."

======================
DOCUMENT CONTEXT
======================

{context}

======================
USER QUESTION
======================

{question}

======================
ANSWER
======================
"""

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        return response.text