import gradio as gr
from transformers import pipeline
from collections import defaultdict, Counter
import emoji
import re

def preprocess_for_sarcasm(text):
    # Convert emojis to their corresponding names and make the text lowercase
    return emoji.demojize(text, delimiters=(" "," ")).lower()

# Emotion models
emotion_roberta = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

emotion_bert = pipeline(
    "text-classification",
    model="bhadresh-savani/bert-base-uncased-emotion",
    return_all_scores=True
)

# Custom sarcasm classifier from saved model
sarcasm_classifier = pipeline(
    "text-classification",
    model="/content/drive/MyDrive/Sentimentana",      
    tokenizer="/content/drive/MyDrive/Sentimentana"
)

# Whisper ASR for speech-to-text
asr = pipeline("automatic-speech-recognition", model="openai/whisper-base")

# Emoji to emotion mapping
emoji_to_emotion = {
    "😂": "joy", "😄": "joy", "😊": "joy", "😁": "joy", "😃": "joy", "😆": "joy",
    "😢": "sadness", "😭": "sadness", "😞": "sadness", "😔": "sadness", "😟": "sadness", "😓": "sadness",
    "😡": "anger", "😠": "anger", "🤬": "anger",
    "😱": "fear", "😨": "fear", "😰": "fear", "😖": "fear",
    "😲": "surprise", "😮": "surprise", "😳": "surprise",
    "🤢": "disgust", "🤮": "disgust", "😷": "disgust",
    "❤️": "love", "😍": "love", "😘": "love", "💖": "love", "💕": "love", "💘": "love",
    "😎": "confidence", "🕶️": "confidence",
    "🤔": "confusion", "😕": "confusion",
    "😤": "frustration", "😒": "frustration", "🙄": "frustration",
    "🤩": "excitement", "😜": "excitement", "😝": "excitement",
    "☺️": "shyness", "😳": "shyness",
    "👍": "approval", "👎": "disapproval", "🙏": "gratitude", "👏": "appreciation",
    "🤝": "agreement", "👌": "agreement", "✌️": "peace", "🤘": "confidence", "👊": "support",
    "✋": "stop", "🖐️": "hello", "👋": "hello", "🤚": "hello",
    "😴": "tired", "🥱": "tired", "😶": "speechless", "😯": "speechless"
}

slang_to_emotion = {
    "lol": "joy", "lmao": "joy", "rofl": "joy", "omg": "surprise", "btw": "neutral",
    "brb": "neutral", "ttyl": "neutral", "idk": "confusion", "smh": "disgust",
    "omfg": "surprise", "wtf": "anger", "yolo": "confidence", "fomo": "fear",
    "bff": "love", "lmfao": "joy", "haha": "joy", "heh": "joy", "oops": "embarrassment"
}

def extract_emojis(text):
    return [char for char in text if char in emoji.EMOJI_DATA]

def boost_emotions_from_punctuation(text, scores):
    if "!" in text:
        for key in scores:
            scores[key] += 0.01
    return scores

def detect_and_boost_from_slang(text, emotion_scores):
    slang_found = re.findall(r'\b(?:' + '|'.join(slang_to_emotion.keys()) + r')\b', text.lower())
    for slang in slang_found:
        emotion = slang_to_emotion.get(slang)
        if emotion:
            emotion_scores[emotion] = emotion_scores.get(emotion, 0) + 0.05
    return emotion_scores

def combine_emotion_scores(roberta_scores, bert_scores, emoji_list, text, emoji_boost=True):
    combined = defaultdict(list)
    for e in roberta_scores:
        combined[e['label'].lower()].append(e['score'])
    for e in bert_scores:
        combined[e['label'].lower()].append(e['score'])

    averaged = {label: sum(scores) / len(scores) for label, scores in combined.items()}

    if emoji_boost and emoji_list:
        emoji_counts = Counter(emoji_list)
        for emo in emoji_counts:
            emotion = emoji_to_emotion.get(emo)
            if emotion:
                averaged[emotion] = averaged.get(emotion, 0) + 0.20 * emoji_counts[emo]

    averaged = boost_emotions_from_punctuation(text, averaged)
    averaged = detect_and_boost_from_slang(text, averaged)

    sorted_emotions = sorted(averaged.items(), key=lambda x: x[1], reverse=True)
    return "\n".join([f"- {label.capitalize()}: {score:.2f}" for label, score in sorted_emotions]) or "No emotions detected."

def emoji_only_scores(emoji_list):
    emotion_scores = defaultdict(float)
    emoji_counts = Counter(emoji_list)
    for emo in emoji_counts:
        emotion = emoji_to_emotion.get(emo)
        if emotion:
            emotion_scores[emotion] += 0.1 * emoji_counts[emo]
    sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
    return "\n".join([f"- {label.capitalize()}" for label, score in sorted_emotions]) or "Could not determine emotion from emojis."

def summarize_emotion(emotions_dict):
    if not emotions_dict:
        return "The emotion expressed is unclear or neutral.\n"
    sorted_emotions = sorted(emotions_dict.items(), key=lambda x: x[1], reverse=True)
    top_emotions = [label for label, score in sorted_emotions if score > 0.1][:2]
    if not top_emotions:
        return "The emotion expressed is unclear or neutral.\n"
    return f"The emotion expressed is mainly {' and '.join(top_emotions)}.\n"

def interpret_sarcasm(sarcastic_score, nonsarcastic_score):
    diff = abs(sarcastic_score - nonsarcastic_score)

    if diff < 0.15:
        return "It's uncertain whether this is sarcastic."

    if sarcastic_score > nonsarcastic_score:
        if sarcastic_score > 0.85:
            return "This sentence is mostly sarcastic."
        elif sarcastic_score > 0.70:
            return "This sentence is likely sarcastic."
        else:
            return "Possibly sarcastic, but not clear."
    else:
        if nonsarcastic_score > 0.85:
            return "This sentence is mostly not sarcastic."
        elif nonsarcastic_score > 0.70:
            return "This sentence is likely not sarcastic."
        else:
            return "Possibly not sarcastic, but uncertain."

import nltk
from nltk.corpus import words
nltk.download('words')

english_vocab = set(words.words())

def is_meaningful(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    meaningful_words = [word for word in tokens if word in english_vocab]
    return len(meaningful_words) >= max(1, len(tokens) // 2)

def analyze_text(text):
    if not text.strip():
        return "Please enter some text."

    emojis = extract_emojis(text)
    is_emoji_only = all(char in emojis for char in text.strip()) and len(emojis) > 0

    slang_found = re.findall(r'\b(?:' + '|'.join(slang_to_emotion.keys()) + r')\b', text.lower())
    is_slang_only = slang_found and all(word.lower() in slang_to_emotion for word in re.findall(r'\w+', text))
    
    if not is_emoji_only and not is_slang_only:
        if not is_meaningful(text):
            return "Please enter a meaningful sentence."

    # Emoji-only input handling
    if is_emoji_only:
        emotion_scores = defaultdict(float)
        emoji_counts = Counter(emojis)
        for emo in emoji_counts:
            emotion = emoji_to_emotion.get(emo)
            if emotion:
                emotion_scores[emotion] += 0.1 * emoji_counts[emo]
        sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
        emotion_summary = "\n".join([f"- {label.capitalize()}" for label, _ in sorted_emotions]) or "Could not determine emotion from emojis."
        summary = summarize_emotion(emotion_scores)
        return f"""**Sentence Analysis**:
        {emotion_summary}

        {summary.strip()}
        (Sarcasm detection skipped for emoji-only input.)"""

    # Slang-only input handling
    elif is_slang_only:
        slang_emotions = defaultdict(float)
        for slang in slang_found:
            emotion = slang_to_emotion.get(slang)
            if emotion:
                slang_emotions[emotion] += 0.1
        sorted_emotions = sorted(slang_emotions.items(), key=lambda x: x[1], reverse=True)
        emotion_summary = "\n".join([f"- {label.capitalize()}" for label, _ in sorted_emotions]) or "Could not determine emotion from slang."
        summary = summarize_emotion(slang_emotions)
        return f"""**Sentence Analysis**:
        {emotion_summary}

        {summary.strip()}
        (Sarcasm detection skipped for slang-only input.)"""

    else:
        # Emotion processing
        roberta_scores = emotion_roberta(text)[0]
        bert_scores = emotion_bert(text)[0]
        final_scores = defaultdict(list)

        for e in roberta_scores:
            final_scores[e['label'].lower()].append(e['score'])
        for e in bert_scores:
            final_scores[e['label'].lower()].append(e['score'])

        averaged = {label: sum(scores) / len(scores) for label, scores in final_scores.items()}

        if emojis:
            emoji_counts = Counter(emojis)
            for emo in emoji_counts:
                emotion = emoji_to_emotion.get(emo)
                if emotion:
                    averaged[emotion] = averaged.get(emotion, 0) + 0.05 * emoji_counts[emo]

        averaged = boost_emotions_from_punctuation(text, averaged)
        averaged = detect_and_boost_from_slang(text, averaged)
        emotion_summary = summarize_emotion(averaged)

        # Sarcasm prediction using both label scores
        processed_text = preprocess_for_sarcasm(text)
        sarcasm_result = sarcasm_classifier(processed_text)

        scores = {res["label"]: res["score"] for res in sarcasm_result}

        # Extract individual scores
        sarcastic_score = scores.get("LABEL_1", 0)
        nonsarcastic_score = scores.get("LABEL_0", 0)

        sarcasm_summary = interpret_sarcasm(sarcastic_score, nonsarcastic_score)

        # TEMPORARY DEBUG DISPLAY
        debug_scores = f"(Sarcastic: {sarcastic_score:.2f}, Not Sarcastic: {nonsarcastic_score:.2f})"
        # Adjust emotion based on sarcasm
        adjusted_emotion = emotion_summary

        if sarcastic_score > nonsarcastic_score and sarcastic_score > 0.95:
            if "joy" in emotion_summary.lower():
                adjusted_emotion = "The emotion expressed is likely anger or frustration"
            elif "surprise" in emotion_summary.lower():
                adjusted_emotion = "The emotion expressed is likely disappointment"
            elif "love" in emotion_summary.lower():
                adjusted_emotion = "The emotion expressed is likely mockery or insincerity"
            elif "neutral" in emotion_summary.lower():
                adjusted_emotion = "The emotion expressed is likely underlying annoyance"

        return f"""**Sentence Analysis**:
         {adjusted_emotion.strip()}

        **Sarcastic.....?**: {sarcasm_summary} {debug_scores}"""


text_input = gr.Textbox(lines=4, placeholder="Type or speak something...")
audio_input = gr.Audio(type="filepath")

def process_input(text=None, audio=None):
    if audio and not text:
        transcription = asr(audio)["text"]
        analysis = analyze_text(transcription)
        return f"**Transcribed Text:**\n{transcription}\n\n{analysis}"
    return analyze_text(text or "")

iface = gr.Interface(
    fn=process_input,
    inputs=[text_input, audio_input],
    outputs="markdown",
    title="🎤 Multi-Modal Emotion, Emoji & Sarcasm Analyzer",
    description="Type or speak a sentence. Detects emotions (with emoji support) and classifies sarcasm using a custom-trained model."
)

iface.launch()
