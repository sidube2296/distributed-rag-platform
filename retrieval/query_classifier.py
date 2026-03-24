def classify_query(query):
    query = query.lower()

    if "nlp" in query or "language" in query:
        return "nlp"
    elif "physics" in query or "light" in query:
        return "physics"
    else:
        return "general"
