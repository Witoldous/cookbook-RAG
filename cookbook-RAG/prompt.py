import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from config import CHROMA_PATH
from embedding import get_embedding_function

PROMPT_TEMPLATE = """
You are professional cooker.

{context}

---

Select the best matching recipe based on the following available ingredients: {ingredients}
- You MUST use recipes from the provided context only.
- Do NOT modify the recipes.
- FORMAT the response with clear sections: Name, Ingredients, Instructions.
- If you can't find a recipe that meets this criteria, you should respond with "I can't find a recipe matching the ingredients".
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ingredients", type=str, help="Comma-separated list of ingredients.")
    args = parser.parse_args()
    query_rag(args.ingredients)

def query_rag(ingredients: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    results = db.similarity_search_with_score(ingredients, k=5)
    
    if not results:
        print("No matching recipes found.")
        return
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, ingredients=ingredients)
    
    model = OllamaLLM(model="mistral")
    response_text = model.invoke(prompt)
    
    sources = [doc.metadata.get("id", None) for doc, _ in results]
    formatted_response = f"Response:\n{response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text

if __name__ == "__main__":
    main()
