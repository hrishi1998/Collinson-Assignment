import requests

# Base URL of the API
base_url = "http://localhost:8000"

# Text to be stored
text_to_store = """
There are broadly two types of extractive summarization tasks depending on what the summarization program focuses on. 
The first is generic summarization, which focuses on obtaining a generic summary or abstract of the collection 
(whether documents, or sets of images, or videos, news stories etc.). The second is query relevant summarization, 
sometimes called query-based summarization, which summarizes objects specific to a query. Summarization systems are 
able to create both query relevant text summaries and generic machine-generated summaries depending on what the user needs.
An example of a summarization problem is document summarization, which attempts to automatically produce an abstract 
from a given document. Sometimes one might be interested in generating a summary from a single source document, while 
others can use multiple source documents (for example, a cluster of articles on the same topic). This problem is called 
multi-document summarization. A related application is summarizing news articles. Imagine a system, which automatically 
pulls together news articles on a given topic (from the web), and concisely represents the latest news as a summary.
Image collection summarization is another application example of automatic summarization. It consists in selecting a 
representative set of images from a larger set of images.[3] A summary in this context is useful to show the most representative
 images of results in an image collection exploration system. Video summarization is a related domain, where the system 
 automatically creates a trailer of a long video. This also has applications in consumer or personal videos, where one 
 might want to skip the boring or repetitive actions. Similarly, in surveillance videos, one would want to extract important 
 and suspicious activity, while ignoring all the boring and redundant frames captured.
"""

# Endpoint for storing text
store_text_endpoint = f"{base_url}/store_text/"

# Endpoint for retrieving summary
retrieve_summary_endpoint = f"{base_url}/retrieve_summary/"

# Example of storing text
response = requests.post(store_text_endpoint, json={"text": text_to_store})
print("Store Text Response:", response.json())  # Print the JSON response for debugging
document_id = response.json().get("document_id")
if document_id is None:
    print("Document ID not found in response.")
else:
    print(f"Document ID: {document_id}")

# Example of retrieving summary
response = requests.get(f"{retrieve_summary_endpoint}{document_id}")
print("Retrieve Summary Response:", response.json())  # Print the JSON response for debugging
summary = response.json().get("summary")
if summary is None:
    print("Summary not found in response.")
else:
    print(f"Summary: {summary}")
