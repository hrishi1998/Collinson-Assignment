from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

app = FastAPI()

nlp = spacy.load("en_core_web_sm")

class TextRequest(BaseModel):
    text: str

class SummaryResponse(BaseModel):
    summary: str

stored_text = ""

@app.post("/store_text/")
async def store_text(req: TextRequest):
    global stored_text
    stored_text = req.text
    return {"message": "Text stored successfully"}

@app.get("/retrieve_text/")
async def retrieve_text():
    return {"text": stored_text}

@app.get("/retrieve_summary/")
async def retrieve_summary():
    global stored_text
    if not stored_text:
        raise HTTPException(status_code=404, detail="Text not found")

    doc = nlp(stored_text)

    stopwords = list(STOP_WORDS)
    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1

    max_frequency = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / max_frequency

    sentence_tokens = [sent for sent in doc.sents]

    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]

    select_length = int(len(sentence_tokens) * 0.3)

    summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)

    final_summary = [word.text for word in summary]
    summary_text = " ".join(final_summary)

    return {"summary": summary_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
