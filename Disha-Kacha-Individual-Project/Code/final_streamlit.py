import streamlit as st
from groq import Groq
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import torch
import re
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Model Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

filepath1 = "speeches.xlsx"
filepath2 = "speeches_russian_PM.xlsx"

data1 = pd.read_excel(filepath1)
data2 = pd.read_excel(filepath2)
data2 = data2.rename(columns={"transcript_filtered": "transcript"})

data = pd.concat([data1, data2], ignore_index=True)
data['transcript'] = data['transcript'].fillna("").astype(str)


def presidential_speech_chat_completion(client, model, user_question, relevant_excerpts):
    system_prompt = '''
    Given the user's question and relevant excerpts from presidential speeches, answer the question thinking like a president. 
    React as if you are the presindent yourself.
    '''

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"User Question: {user_question}\n\nRelevant Speech Excerpt(s):\n\n{relevant_excerpts}",
            },
        ],
        model=model,
    )

    return chat_completion.choices[0].message.content


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


class SentenceTransformerWrapper:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_tensor=True).tolist()

    def embed_query(self, text):
        return self.model.encode(text, convert_to_tensor=True).tolist()


# Using the chroma database which was created earlier. Currently uses recursive character splitter
embedding_function = SentenceTransformerWrapper(embedding_model)
vectorstore_dir = "./chromadb_combined_data"

chroma_db = Chroma(
    persist_directory=vectorstore_dir,
    embedding_function=embedding_function
)


def get_relevant_context(question, top_n=5):
    docs = chroma_db.similarity_search(question, k=top_n)
    combined_context = " ".join([doc.page_content for doc in docs])
    return combined_context

##############################
# Summarization
##############################

summarizer = pipeline("summarization", model="facebook/bart-large-cnn",
                      device=0 if torch.cuda.is_available() else -1)

def summarize_response(response):
    if len(response.split()) <= 20:
        return response

    summarized = summarizer(response, max_length=75, min_length=10, do_sample=False)
    return summarized[0]['summary_text']

###############################
# Sentiment Analysis
###############################

sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english",
                              device=0 if torch.cuda.is_available() else -1)


def display_sentiment_meter(response):
    """
    Display a sentiment emoji (happy for positive, sad for negative) with size based on the sentiment score.
    """
    # Perform sentiment analysis
    sentiment_result = sentiment_analyzer(response)

    # Get the sentiment score and label
    score = sentiment_result[0]['score']
    label = sentiment_result[0]['label']

    # Scale emoji size based on the sentiment score (larger for stronger sentiment)
    emoji_size = int(50 + 150 * score)  # Scale emoji size from 50 to 200 based on score

    # Choose emoji based on sentiment
    emoji = "😊" if label == 'POSITIVE' else "😞"

    # Display the emoji with the appropriate size
    st.markdown(f"<h1 style='text-align: center; font-size: {emoji_size}px;'>{emoji}</h1>", unsafe_allow_html=True)

    # Display the sentiment label and score
    st.write(f"Sentiment: {label}")
    st.write(f"Score: {score:.2f}")

###############################
# Named Entity Recognition (NER)
###############################

ner_pipeline = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    device=0 if torch.cuda.is_available() else -1,
    aggregation_strategy="simple"
)

def extract_named_entities(text):
    """
    Extract Named Entities (NER) from the text, deduplicate them, and generate a word cloud
    with colors based on the entity types, including a legend for the entity types.
    """
    ner_results = ner_pipeline(text)

    # Map entity types to colors
    entity_color_map = {
        "PER": "#3498db",  # Person
        "LOC": "#2ecc71",  # Location
        "ORG": "#e74c3c",  # Organization
        "MISC": "#9b59b6",  # Miscellaneous
        "DEFAULT": "#95a5a6"  # Default color
    }

    # Prepare word frequencies and colors
    word_frequencies = {}
    word_colors = {}
    seen = set()

    for entity in ner_results:
        word = entity['word'].strip()
        if word.startswith("##") or "#" in word or word in seen:
            continue
        seen.add(word)

        # Get the entity type and score
        entity_type = entity['entity_group'] if 'entity_group' in entity else entity['entity']
        score = entity['score']

        # Add word frequency and color
        word_frequencies[word] = score  # Use score as frequency
        word_colors[word] = entity_color_map.get(entity_type, entity_color_map["DEFAULT"])

    # Custom color function for word cloud
    def color_func(word, **kwargs):
        return word_colors.get(word, "#95a5a6")  # Default gray if not found

    # Generate word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="black"
    ).generate_from_frequencies(word_frequencies)

    # Create legend
    fig, ax = plt.subplots(figsize=(12, 6))

    # Display the word cloud
    ax.imshow(wordcloud.recolor(color_func=color_func), interpolation="bilinear")
    ax.axis("off")

    # Add legend
    handles = []
    for entity_type, color in entity_color_map.items():
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=entity_type))

    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1, 0.5), title="Entity Types")
    ax.set_title("Named Entity Word Cloud", fontsize=16, color="white")

    # Display the plot in Streamlit
    st.pyplot(fig)

###############################
# Chat History Functionality
###############################

chat_history = {}

def clean_chat_history():
    """
    Remove entries in chat history that are older than 15 minutes.
    """
    current_time = datetime.now()
    fifteen_minutes_ago = current_time - timedelta(minutes=15)
    keys_to_remove = [key for key, (timestamp, _) in chat_history.items() if
                      timestamp < fifteen_minutes_ago]
    for key in keys_to_remove:
        del chat_history[key]


# Streamlit UI
def main():
    groq_api_key = "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"

    client = Groq(api_key=groq_api_key)
    st.set_page_config(
        page_title="President Q&A 🤖",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    # Header image
    st.image('url.jpeg', use_container_width=True, caption="The wisdom of Presidents at your fingertips!")

    # Main title and description
    # -webkit - text - stroke: 1px white;
    st.markdown(
        """
        <div style="text-align: center; font-family: 'Arial', sans-serif; color: #B2BEB5;">
            <h1 style="font-size: 3em; font-weight: bold;">Talk to a President!!!</h1>
            <p style="font-size: 1.2em;">
                Welcome! Dive into the wisdom of Presidents. Ask questions like:
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Sidebar for model selection
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align: center; font-family: 'Arial', sans-serif;">
                <h3 style="color: #95A5A6;">⚙️ Choose Your Model</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
        model = st.selectbox(
            'Select a Model',
            ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it', 'fine_tuned_president_model']
        )
        st.markdown(
            """
            <p style="text-align: center; color: #95A5A6; font-size: 0.9em;">
                Tip: Choose the <strong>"fine_tuned_president_model"</strong> for tuned answers on presidential topics!
            </p>
            """,
            unsafe_allow_html=True
        )


    user_question = st.text_input(
        "Your question:",
        placeholder="e.g., What did John F. Kennedy emphasize in his inaugural speech?",
        help="Ask about any U.S. President's views, quotes, or policies!"
    )

    # Placeholder for the AI response
    if user_question:
        st.markdown(
            """
            <div style="margin-top: 30px; padding: 20px; background: #D5DBDB; border-radius: 10px;">
                <h3 style="color: #16A085; font-family: 'Arial', sans-serif; text-align: center;">🤔 AI Response</h3>
                <p style="text-align: justify; color: #2C3E50;">
                    Here is your AI-generated answer to the question!
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="margin-top: 30px; padding: 20px; background: #F2F3F4; border-radius: 10px;">
                <h3 style="color: #E74C3C; font-family: 'Arial', sans-serif; text-align: center;">⏳ Waiting for Your Question</h3>
                <p style="text-align: center; color: #7F8C8D;">
                    Type a question above to get started!
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    if user_question:
        if model == 'fine_tuned_president_model':
            output_dir = "./fine_tuned_president_5_epochs"
            fine_tuned_model = GPT2LMHeadModel.from_pretrained(output_dir).to(device)
            fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained(output_dir)

            fine_tuned_generator = pipeline(
                "text-generation",
                model=fine_tuned_model,
                tokenizer=fine_tuned_tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )

            if fine_tuned_tokenizer.pad_token is None:
                fine_tuned_tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
                fine_tuned_model.resize_token_embeddings(len(fine_tuned_tokenizer))


            def clean_text(text):
                text = re.sub(r'\s+', ' ', text.strip())
                text = re.sub(r'[^\x00-\x7F]+', '', text)
                text = re.sub(r'(?:(\b\w+\b)\s+)+\1\b', r'\1', text)
                return text

            def remove_repeated_phrases(response):
                """
                Removes overly repeated phrases or sentences in the generated response.
                """
                sentences = response.split('. ')
                seen = set()
                filtered_sentences = []
                for sentence in sentences:
                    cleaned_sentence = clean_text(sentence)
                    if cleaned_sentence not in seen:
                        filtered_sentences.append(cleaned_sentence)
                        seen.add(cleaned_sentence)
                return '. '.join(filtered_sentences)


            def answer_question(question):
                relevant_context = get_relevant_context(question)
                # if not is_relevant:
                #     return "I'm sorry, I couldn't find a direct match in the speeches. Please try rephrasing your question or asking something more specific."

                truncated_context = relevant_context[:1000]
                prompt = (
                    f"The following context is strictly derived from the speeches dataset:\n\n"
                    f"Context: {truncated_context}\n\n"
                    f"Based on this context, answer the following question:\n"
                    f"Question: {question}\nAnswer:"
                )

                print(f"Prompt length: {len(prompt)}")

                response = fine_tuned_generator(
                    prompt,
                    max_new_tokens=300,
                    num_return_sequences=1,
                    pad_token_id=fine_tuned_tokenizer.eos_token_id,
                    truncation=True
                )
                response = response[0]['generated_text'].split('Answer:')[-1].strip()
                return response

            ###############################
            # Chat Response Handling
            ###############################

            def get_response(question):
                """
                Handle chat history, summarization, and sentiment analysis for a question.
                """
                clean_chat_history()

                if question in chat_history:
                    timestamp, response = chat_history[question]
                    print("\nResponse from history!")
                    return response

                # relevant_context, is_relevant = get_relevant_context(question)
                # if not is_relevant:
                #     return "Please ask something directly related to the speeches."

                answer = answer_question(question)

                summarized_answer = summarize_response(answer)

                # sentiment = display_sentiment_meter(answer)

                # ner_entities = extract_named_entities(answer)

                chat_history[question] = (datetime.now(), (answer, summarized_answer))

                return answer, summarized_answer

            response = get_response(user_question)  # Use custom model
            if isinstance(response, tuple):
                answer = response[0]
                st.subheader("President's Answer")
                st.write(answer)
                st.subheader("Summarized Response")
                summarized_answer = response[1]
                st.write(summarized_answer)
                st.subheader("Sentiment (Top Emotions)")
                display_sentiment_meter(answer)
                st.subheader("Named Entities")
                extract_named_entities(answer)

            else:
                # If result is a string (no relevant context)
                st.write(response)

        else:
            # relevant_excerpts = get_relevant_excerpts(user_question, docsearch)
            relevant_excerpts = get_relevant_context(user_question, 5)

            response = presidential_speech_chat_completion(client, model, user_question, relevant_excerpts)
            st.subheader("President's Answer")
            st.write(response)
            st.subheader("Summarized Response")
            summarized_answer = summarize_response(response)
            st.write(summarized_answer)
            st.subheader("Sentiment (Top Emotions)")
            display_sentiment_meter(response)
            st.subheader("Named Entities")
            extract_named_entities(response)


if __name__ == "__main__":
    main()
