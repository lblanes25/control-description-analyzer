import spacy

def test_finance_manager_detection():
    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_md")
        print("✅ Using en_core_web_md")
    except OSError:
        nlp = spacy.load("en_core_web_sm")
        print("✅ Using en_core_web_sm")
    
    text = "The Finance Manager reviews and reconciles monthly bank statements within 5 business days of month-end to ensure accuracy and identify any unauthorized transactions. Discrepancies are investigated and resolved within 2 business days, with findings reported to the CFO."
    
    doc = nlp(text)
    
    print("\n🔍 FULL TOKEN ANALYSIS:")
    print(f"{'Token':<20} {'POS':<10} {'DEP':<15} {'HEAD':<20}")
    print("-" * 65)
    for token in doc:
        print(f"{token.text:<20} {token.pos_:<10} {token.dep_:<15} {token.head.text:<20}")
    
    print("\n📦 NOUN CHUNKS:")
    for chunk in doc.noun_chunks:
        print(f"  '{chunk.text}' (root: {chunk.root.text}, start: {chunk.start}, end: {chunk.end})")
    
    print("\n🔍 LOOKING FOR FINANCE MANAGER:")
    # Check tokens 0-3 specifically
    for i in range(min(4, len(doc))):
        token = doc[i]
        print(f"  Token {i}: '{token.text}' - POS: {token.pos_}, DEP: {token.dep_}")

if __name__ == "__main__":
    test_finance_manager_detection()