import rag_pipeline

question = "What is scaled dot-product attention?"
result = rag_pipeline.answer_question(question)

print("=" * 60)
print("FULL ANSWER:")
print(result["answer"])
print()
print("STATUS:", result["status"])
print()
print("=" * 60)
print("RETRIEVED SOURCES:")
for i, s in enumerate(result["sources"]):
    print(f"\n--- Source {i+1} (page {s['metadata'].get('page') + 1}) ---")
    print(s["text"])
