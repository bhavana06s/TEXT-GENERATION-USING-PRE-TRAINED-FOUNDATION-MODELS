import gradio as gr
from transformers import pipeline
import evaluate
#----------1.BuildandDeploytheApp---------
summarizer=pipeline("summarization",model="facebook/bart-large-cnn")
defsummarize_text(input_text):
    result=summarizer(input_text,max_length=45,min_length=15,do_sample=False)
    returnresult[0]["summary_text"]
demo=gr.Interface( fn=summarize_text, inputs=gr.Textbox(lines=8,label="Entertexttosummarize"), outputs=gr.Textbox(label="GeneratedSummary"), title="GenAITextSummarizer", description="Acloud-deployableGenerativeAIsummarizationappbuiltwithGradio." )
demo.launch(share=True)
#share=TruegeneratesapubliccloudURL
#----------2.EvaluateGeneratedOutput---------
rouge=evaluate.load("rouge")
generated_summaries=[ "AImodelsgeneratenewcontentsuchastextandimages.", ]
reference_summaries=[ "GenerativeAImodelsarecapableofproducingnewcontentincludingtextandimages.", ]
scores=rouge.compute(predictions=generated_summaries,references=reference_summaries)
print("ROUGEEvaluationScores:",scores)
