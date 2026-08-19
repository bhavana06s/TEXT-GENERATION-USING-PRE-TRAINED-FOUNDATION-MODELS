from transformers import pipeline
#---------- Sentiment Analysis---------
sentiment_analyzer = pipeline("sentiment-analysis")
reviews = [ "The new smartphone has an amazing camera and battery life!", "The delivery was late and the packaging was damaged."]
forreviewinreviews: result=sentiment_analyzer(review)[0]
print(f"Review:{review}\n->{result['label']}({round(result['score'],3)})\n")
#----------DocumentClassification(Zero-Shot)---------
classifier=pipeline("zero-shot-classification",model="facebook/bart-large-mnli")
document="Thecentralbankraisedinterestratestocontrolrisinginflation."
candidate_labels=["Politics","Economy","Sports","Technology"]
classification=classifier(document,candidate_labels)
print("Document:",document)
forlabel,scoreinzip(classification["labels"],classification["scores"]):
print(f"{label}:{round(score,3)}")
