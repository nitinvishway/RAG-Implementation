import urllib.request
import json
import time

time.sleep(2)

def test_query(question):
    req = urllib.request.Request(
        'http://127.0.0.1:5000/api/chat',
        data=json.dumps({'question': question}).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    res = json.loads(urllib.request.urlopen(req).read().decode('utf-8'))
    print(f"Q: {question}")
    print(f"A: {res.get('answer', 'NO ANSWER')}")
    print(f"Provider: {res.get('provider', 'UNKNOWN')}")
    # Show retrieved chunks
    chunks = res.get('retrieved_chunks', [])
    for i, c in enumerate(chunks[:2]):
        print(f"  Chunk {i}: {c['content'][:100]}...")
    print()

test_query("What is the admission fee?")
test_query("When does admission open?")
test_query("What is the hostel fee?")
test_query("What is the library timing?")
test_query("What is RAG?")
test_query("Do students need ID card?")
