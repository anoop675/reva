import re
import torch
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .config import DEFAULT_TEST_IMAGES_ROOT


def _truncate_short_answer(text: str) -> str:
    """Keep the first line / first sentence for short-phrase region probes."""
    text = text.strip().split("\n")[0].strip()
    for sep in (". ", "; "):
        head = text.split(sep)[0].strip()
        if sep in text and len(head) <= 80:
            text = head
            break
    return text.rstrip(".")


def _generate_from_region_tokens( region_tokens, frozen_qwen, qwen_tokenizer, config, text, max_new_tokens=16, repetition_penalty=1.2, truncate=True, ):
    """[K region tokens] + text -> decoded answer (no chat template)."""
    device = config.device
    text_tokens = qwen_tokenizer(text, return_tensors="pt").to(device)
    text_embeds = frozen_qwen.get_input_embeddings()(text_tokens["input_ids"])
    input_embeds = torch.cat([region_tokens, text_embeds], dim=1)
    attn_mask = torch.ones(1, input_embeds.shape[1], dtype=torch.long, device=device)
    generated_ids = frozen_qwen.generate(
        inputs_embeds=input_embeds,
        attention_mask=attn_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=qwen_tokenizer.pad_token_id,
        eos_token_id=qwen_tokenizer.eos_token_id,
        repetition_penalty=repetition_penalty,
    )
    raw = qwen_tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    return _truncate_short_answer(raw) if truncate else raw


def generate_caption_for_image(pil_image, frozen_vit, projection_head_b, frozen_qwen, clip_image_processor, qwen_tokenizer, config, max_new_tokens: int = 64) -> str:
    
    """Inference-time caption generation. Builds the projected image token sequence, then lets Qwen generate autoregressively."""
    frozen_vit.eval()
    projection_head_b.eval()
    frozen_qwen.eval()

    with torch.no_grad():

        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
            
        processed_image = clip_image_processor(images=pil_image, return_tensors='pt')['pixel_values'].to(config.device, dtype=config.compute_dtype)

        vit_output = frozen_vit(pixel_values=processed_image)
        image_patch_features = vit_output.last_hidden_state[:, 1:, :]  # (1, 256, 1024)
        projected_image_tokens = projection_head_b(image_patch_features)  # (1, 256, 3584)

        bos_token_id = qwen_tokenizer.bos_token_id or qwen_tokenizer.eos_token_id
        bos_embedding = frozen_qwen.get_input_embeddings()(torch.tensor([[bos_token_id]], device=config.device))  # (1, 1, 3584)

        initial_input_embeds = torch.cat([projected_image_tokens, bos_embedding], dim=1)

        generated_token_ids = frozen_qwen.generate(
            inputs_embeds=initial_input_embeds,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=qwen_tokenizer.pad_token_id,
            eos_token_id=qwen_tokenizer.eos_token_id
        )

        generated_caption = qwen_tokenizer.decode(generated_token_ids[0], skip_special_tokens=True)

    return generated_caption
    

def run_vqa_inference(pil_image, question_text: str, format_prompt: str, frozen_vit, projection_head_b, frozen_qwen, clip_image_processor, qwen_tokenizer, eval_config) -> str:
    """
    Single-sample VQA inference.
    Constructs the prompt as: [Global Image Tokens] + [Question] + [Format prompt]
    then generates autoregressively with greedy decoding.
    """
    frozen_vit.eval()
    projection_head_b.eval()
    frozen_qwen.eval()

    with torch.no_grad():
        processed_image = clip_image_processor(images=pil_image, return_tensors="pt")['pixel_values']
        processed_image = processed_image.to(eval_config.device, dtype=eval_config.compute_dtype)
        vit_output = frozen_vit(pixel_values=processed_image)
        image_patch_features = vit_output.last_hidden_state[:, 1:, :]  # (1, 256, 1024)
        projected_image_tokens = projection_head_b(image_patch_features)  # (1, 256, 3584)

        messages = [{"role": "user", "content": f"{question_text} {format_prompt}"}]
        formatted_prompt = qwen_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokenised = qwen_tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)
        question_token_ids = tokenised["input_ids"].to(eval_config.device)

        question_embeddings = frozen_qwen.get_input_embeddings()(question_token_ids)

        input_embeds = torch.cat([projected_image_tokens, question_embeddings], dim=1)

        generated_ids = frozen_qwen.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=eval_config.max_new_tokens,
            do_sample=eval_config.do_sample,
            pad_token_id=qwen_tokenizer.pad_token_id,
            eos_token_id=qwen_tokenizer.eos_token_id,
        )

        generated_answer = qwen_tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

    return generated_answer


# Mirrors logic from the official GT-Vision-Lab VQAEval (vqaEval.py)

_MANUAL_MAP = {
    'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3',
    'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8',
    'nine': '9', 'ten': '10'
}

_ARTICLES = {'a', 'an', 'the'}

_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
    "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't",
    "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
    "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm", "Ive": "I've",
    "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've",
    "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've",
    "oclock": "o'clock", "oughtnt": "oughtn't", "shant": "shan't",
    "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's",
    "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",           
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
    "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've",
    "someone'dve": "someone'd've", "someonell": "someone'll", "someones": "someone's",
    "somethingd": "something'd", "somethingd've": "something'd've",
    "something'dve": "something'd've", "somethingll": "something'll",
    "thats": "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres": "there's",
    "theyd": "they'd", "theyd've": "they'd've", "they'dve": "they'd've",
    "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas",
    "wasnt": "wasn't", "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've",
    "werent": "weren't", "whatll": "what'll", "whatre": "what're", "whats": "what's",
    "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's",
    "whereve": "where've", "whod": "who'd", "whod've": "who'd've",
    "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've",
    "whyll": "why'll", "whyre": "why're", "whys": "why's", "wont": "won't",
    "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll",
    "y'allll": "y'all'll", "yall'd've": "y'all'd've", "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've",
    "you'dve": "you'd've", "youll": "you'll", "youre": "you're", "youve": "you've",
}

_PUNCT = [';', '/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\',
          '_', '-', '>', '<', '@', '`', ',', '?', '!']

_PERIOD_STRIP = re.compile(r'(?!<=\d)(\.)(?!\d)')
_COMMA_STRIP  = re.compile(r'(\d)(\,)(\d)')


def _process_punctuation(text: str) -> str:
    out = text
    for p in _PUNCT:
        if (p + ' ' in text) or (' ' + p in text) or (_COMMA_STRIP.search(text) is not None):
            out = out.replace(p, '')
        else:
            out = out.replace(p, ' ')
    out = _PERIOD_STRIP.sub('', out, re.UNICODE)
    return out


def _process_digit_article(text: str) -> str:
    words = []
    for word in text.lower().split():
        word = _MANUAL_MAP.get(word, word)
        if word not in _ARTICLES:
            words.append(word)
    words = [_CONTRACTIONS.get(w, w) for w in words]
    return ' '.join(words)


def vqa_soft_score(predicted: str, ground_truth_answers: list) -> float:

    # Strip whitespace on both sides
    predicted = predicted.replace('\n', ' ').replace('\t', ' ').strip()
    gt_answers = [gt.replace('\n', ' ').replace('\t', ' ').strip() for gt in ground_truth_answers]

    # The official conditional in vqaEval.py line 67 states that only normalise if ground truths are not all identical
    if len(set(gt_answers)) > 1:
        predicted = _process_punctuation(predicted)
        predicted = _process_digit_article(predicted)
        gt_answers = [_process_digit_article(_process_punctuation(gt)) for gt in gt_answers]

    # Leave-one-out soft scoring
    per_annotator_scores = []
    for i in range(len(gt_answers)):
        others = [gt_answers[j] for j in range(len(gt_answers)) if j != i]
        match_count = sum(1 for gt in others if gt == predicted)
        per_annotator_scores.append(min(match_count / 3.0, 1.0))

    return sum(per_annotator_scores) / len(per_annotator_scores)

def normalise_answer(answer: str) -> str:
    """
    Normalise a GQA answer string for exact match evaluation
    Applies unconditional punctuation and digit/article processing
    Used for GQA evaluation only for applying symmetrically to both predicted and ground truth answers before comparing
    Not to be used for VQAv2 scoring, use vqa_soft_score() which applies conditional normalisation following vqaEval.py """
    answer = answer.replace('\n', ' ').replace('\t', ' ').strip()
    answer = _process_punctuation(answer)
    answer = _process_digit_article(answer)
    return answer


def interactive_inference_loop(frozen_vit, projection_head_b, frozen_qwen, clip_image_processor, qwen_tokenizer, config):

    while True:
        image_source = input("Image path or URL (or 'quit'): ").strip()
        if image_source.lower() == 'quit':
            break
        try:
            if image_source.startswith("http"):
                response = requests.get(image_source, timeout=10)
                pil_image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                pil_image = Image.open(image_source).convert('RGB')
            print(f"Image loaded of size: {pil_image.size[0]} x {pil_image.size[1]}px\n")
        except Exception as e:
            print(f"Could not load image: {e}\n")
            continue

        while True:
            question = input("Question (or 'image' to change image, 'quit' to exit): ").strip()
            if question.lower() == 'quit':
                break
            if question.lower() == 'image':
                break
            if not question:
                continue

            frozen_vit.eval()
            projection_head_b.eval()
            frozen_qwen.eval()

            with torch.no_grad():
                processed_image = clip_image_processor(
                    images=pil_image, return_tensors="pt"
                )['pixel_values'].to(config.device, dtype=config.compute_dtype)

                image_patch_features = frozen_vit(pixel_values=processed_image).last_hidden_state[:, 1:, :]
                projected_image_tokens = projection_head_b(image_patch_features)

                messages = [{"role": "user", "content": question}]
                formatted_prompt = qwen_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                tokenised = qwen_tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)
                question_embeds = frozen_qwen.get_input_embeddings()(tokenised["input_ids"].to(config.device))
                bos_token_id = qwen_tokenizer.bos_token_id or qwen_tokenizer.eos_token_id
                bos_embedding = frozen_qwen.get_input_embeddings()(
                    torch.tensor([[bos_token_id]], device=config.device)
                )
                input_embeds = torch.cat([projected_image_tokens, bos_embedding, question_embeds], dim=1)

                answer = qwen_tokenizer.decode(
                    frozen_qwen.generate(
                        inputs_embeds=input_embeds,
                        max_new_tokens=200,
                        do_sample=False,
                        pad_token_id=qwen_tokenizer.pad_token_id,
                        eos_token_id=qwen_tokenizer.eos_token_id,
                        repetition_penalty=1.2,
                    )[0],
                    skip_special_tokens=True,
                ).strip()

            print(f"Answer: {answer}\n")


def _load_pil_image(image_path_or_url: str) -> Image.Image:
    if image_path_or_url.startswith('http'):
        response = requests.get(image_path_or_url, timeout=10)
        return Image.open(BytesIO(response.content)).convert('RGB')
    return Image.open(image_path_or_url).convert('RGB')


def _embed_text(text: str, qwen_tokenizer, frozen_qwen, device):
    tokens = qwen_tokenizer(text, return_tensors='pt').to(device)
    embeds = frozen_qwen.get_input_embeddings()(tokens['input_ids'])
    return embeds, tokens['attention_mask']


def _slugify_region_label(label: str) -> str:
    slug = re.sub(r'[^a-z0-9]+', '-', label.lower().strip())
    return slug.strip('-') or 'object'


def _box336_to_normalized(box_336) -> list:
    x1, y1, x2, y2 = [float(v) for v in box_336]
    return [x1 / 336.0, y1 / 336.0, x2 / 336.0, y2 / 336.0]


def parse_manual_boxes(text: str) -> list:
    """Parse manual boxes from 'x1,y1,x2,y2' or semicolon-separated lists (original pixels)."""
    text = text.strip()
    if not text:
        return []
    boxes = []
    for part in text.split(';'):
        part = part.strip()
        if not part:
            continue
        coords = [float(v.strip()) for v in part.split(',')]
        if len(coords) != 4:
            raise ValueError(f"Each box needs 4 values (x1,y1,x2,y2), got: {part!r}")
        boxes.append(coords)
    return boxes


def manual_boxes_to_336(manual_boxes, img_w, img_h, device):
    """Convert raw-pixel [x1,y1,x2,y2] boxes to 336px CLIP space."""
    scale_x, scale_y = 336.0 / img_w, 336.0 / img_h
    scaled = []
    for x1, y1, x2, y2 in manual_boxes:
        x1 = max(0.0, min(float(x1), img_w - 1))
        y1 = max(0.0, min(float(y1), img_h - 1))
        x2 = max(x1 + 1, min(float(x2), img_w))
        y2 = max(y1 + 1, min(float(y2), img_h))
        scaled.append([x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y])
    return torch.tensor(scaled, dtype=torch.float32, device=device)


def _draw_inference_boxes(pil_image, display_boxes, scores, labels):
    img_w, img_h = pil_image.size
    fig, ax = plt.subplots(1, figsize=(8, 8 * img_h / img_w))
    ax.imshow(pil_image)
    scale_back_x = img_w / 336.0
    scale_back_y = img_h / 336.0
    for i in range(display_boxes.shape[0]):
        bx1, by1, bx2, by2 = display_boxes[i].tolist()
        bx1, bx2 = bx1 * scale_back_x, bx2 * scale_back_x
        by1, by2 = by1 * scale_back_y, by2 * scale_back_y
        color = 'red' if i == 0 else 'cyan'
        lw = 2.5 if i == 0 else 1.0
        alpha = 1.0 if i == 0 else 0.4
        rect = patches.Rectangle(
            (bx1, by1), bx2 - bx1, by2 - by1,
            linewidth=lw, edgecolor=color, facecolor='none', alpha=alpha,
        )
        ax.add_patch(rect)
        label = labels[i] if i < len(labels) else f"region{i}"
        if scores is not None:
            ax.text(bx1, by1 - 5, f"{label} ({scores[i]:.2f})", color=color, fontsize=8)
        else:
            ax.text(bx1, by1 - 5, label, color=color, fontsize=8)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def _format_tags_for_dino(tags: list) -> str:
    """Join tags into a Grounding DINO caption ('tag1 . tag2 . tag3')."""
    return " . ".join(t.strip() for t in tags if t and t.strip())


def _normalize_selected_tags(raw: str, ram_tags: list) -> list:
    """Map Qwen output back to tags from the RAM list (exact or partial match)."""
    ram_lower = {t.lower(): t for t in ram_tags}
    parts = [p.strip() for p in re.split(r'\s*\.\s*|\|', raw) if p.strip()]
    selected = []
    for phrase in parts:
        pl = phrase.lower()
        if pl in ram_lower:
            selected.append(ram_lower[pl])
            continue
        for tag_lower, original in ram_lower.items():
            if tag_lower in pl or pl in tag_lower:
                selected.append(original)
                break
    seen = set()
    out = []
    for tag in selected:
        key = tag.lower()
        if key not in seen:
            seen.add(key)
            out.append(tag)
    return out


def select_ram_tags_for_question_llm( question: str, ram_tags: list, frozen_qwen, qwen_tokenizer, config, max_select: int = None, ) -> list:
    """Pick the RAM tags most relevant to answering the question."""
    if not ram_tags:
        return []
    max_select = max_select or getattr(config, 'max_qwen_selected_tags', 3)
    tag_list = ", ".join(ram_tags)
    messages = [
        {
            "role": "system",
            "content": (
                "You are given a question about an image and a list of object tags "
                "detected in that image. The tags are question-agnostic. Select ONLY "
                "tags from the provided list that are needed to answer the question. "
                f"Pick at most {max_select} tags. "
                "Respond with ONLY the selected tags from the list, separated by ' . '. "
                "Do not invent tags that are not in the list."
            ),
        },
        {
            "role": "user",
            "content": "Detected tags: dog, grass, tree, sky, fence\nQuestion: What color is the dog?",
        },
        {"role": "assistant", "content": "dog"},
        {
            "role": "user",
            "content": "Detected tags: car, street, building, person, traffic light\nQuestion: How many red cars are there?",
        },
        {"role": "assistant", "content": "car"},
        {
            "role": "user",
            "content": "Detected tags: ball, chair, table, lamp, floor\nQuestion: What color is the ball next to the chair?",
        },
        {"role": "assistant", "content": "ball . chair"},
        {
            "role": "user",
            "content": f"Detected tags: {tag_list}\nQuestion: {question}",
        },
    ]
    text = qwen_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = qwen_tokenizer(text, return_tensors='pt').to(config.device)
    output_ids = frozen_qwen.generate(
        **inputs,
        max_new_tokens=24,
        do_sample=False,
        repetition_penalty=1.1,
    )
    generated = qwen_tokenizer.decode(
        output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
    )
    raw = generated.strip().split("\n")[0].strip()
    selected = _normalize_selected_tags(raw, ram_tags)
    return selected[:max_select]


def _tags_from_grounding_prompt(text: str) -> list:
    if not text or not str(text).strip():
        return []
    return [t.strip() for t in str(text).split(" . ") if t.strip()]


def _merge_grounding_tags(*tag_lists) -> list:
    """Case-insensitive union preserving first-seen order."""
    seen = set()
    merged = []
    for tags in tag_lists:
        for tag in tags:
            cleaned = str(tag).strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(cleaned)
    return merged


_SPACY_NLP = False  # False = not tried; None = unavailable; otherwise loaded nlp
_SPACY_UNAVAILABLE_REASON = None

_GROUNDING_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "am", "be", "been", "being",
    "do", "does", "did", "you", "your", "see", "there", "any", "many", "much",
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "this", "that", "these", "those", "in", "on", "at", "of", "to", "for",
    "with", "from", "by", "as", "or", "and", "not", "no", "yes", "if", "than",
    "have", "has", "had", "can", "could", "would", "should", "will", "shall",
    "may", "might", "must", "it", "its", "he", "she", "they", "we", "i", "me",
    "my", "his", "her", "their", "our", "them", "him", "us",
    "image", "picture", "photo", "pic", "show", "shown", "look", "like",
    "color", "type", "kind", "number", "count", "total", "left", "right",
}


def _get_spacy_nlp():
    """Return a loaded spaCy pipeline, or None if unavailable in this environment."""
    global _SPACY_NLP, _SPACY_UNAVAILABLE_REASON
    if _SPACY_NLP is not False:
        return _SPACY_NLP

    try:
        import spacy

        _SPACY_NLP = spacy.load("en_core_web_sm")
    except OSError:
        _SPACY_NLP = None
        _SPACY_UNAVAILABLE_REASON = (
            "spaCy model 'en_core_web_sm' is missing. "
            "Install with: python -m spacy download en_core_web_sm"
        )
    except (ImportError, ValueError, AttributeError):
        _SPACY_NLP = None
        _SPACY_UNAVAILABLE_REASON = (
            "spaCy/thinc failed to import (often a numpy binary mismatch). "
            "Try: pip install --upgrade --force-reinstall 'numpy<2' spacy thinc && "
            "python -m spacy download en_core_web_sm"
        )
    return _SPACY_NLP


def _extract_grounding_nouns_heuristic(question: str) -> list:
    """Lightweight noun-ish fallback when spaCy is unavailable."""
    words = re.findall(r"[A-Za-z]+", str(question).lower())
    tags = []
    for word in words:
        if word in _GROUNDING_STOPWORDS or len(word) <= 1:
            continue
        if word.endswith("ies") and len(word) > 4:
            lemma = word[:-3] + "y"
        elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
            lemma = word[:-1]
        else:
            lemma = word
        tags.append(lemma)
    return _merge_grounding_tags(tags)


def extract_grounding_nouns_spacy(question: str, *, verbose: bool = False) -> tuple[list, str]:
    """
    Extract grounding tags from a question.

    Returns (tags, extractor_name) where extractor_name is 'spaCy' or 'heuristic'.
    """
    if not question or not str(question).strip():
        return [], "spaCy"

    nlp = _get_spacy_nlp()
    if nlp is not None:
        doc = nlp(str(question))
        tags = []
        for chunk in doc.noun_chunks:
            lemma = chunk.root.lemma_.lower().strip()
            if lemma and len(lemma) > 1:
                tags.append(lemma)
        for token in doc:
            if token.pos_ in ("NOUN", "PROPN") and not token.is_stop:
                lemma = token.lemma_.lower().strip()
                if lemma and len(lemma) > 1:
                    tags.append(lemma)
        return _merge_grounding_tags(tags), "spaCy"

    if verbose and _SPACY_UNAVAILABLE_REASON:
        print(f"spaCy unavailable — using heuristic nouns. {_SPACY_UNAVAILABLE_REASON}")
    return _extract_grounding_nouns_heuristic(question), "heuristic"


def _build_automatic_grounding_prompt( pil_image, *, question=None, ram_proposer=None, config=None, box_source=None, verbose=True, ) -> str:
    """Build a Grounding DINO caption from spaCy nouns and/or RAM++ tags."""
    if box_source is None:
        box_source = getattr(config, "box_source", "hybrid")
    box_source = str(box_source).lower().strip()
    if box_source not in {"question", "ram", "hybrid"}:
        raise ValueError(
            f"box_source must be 'question', 'ram', or 'hybrid', got {box_source!r}"
        )

    question_tags = []
    if question:
        question_tags, noun_extractor = extract_grounding_nouns_spacy(
            question, verbose=verbose
        )
        if verbose:
            print(
                f"Question nouns ({noun_extractor}): "
                f"{_format_tags_for_dino(question_tags) or '(none)'}"
            )

    ram_tags = []
    if box_source in {"ram", "hybrid"} and ram_proposer is not None:
        max_ram = getattr(config, "max_ram_tags", 20)
        ram_tags = ram_proposer.get_tags(pil_image, max_tags=max_ram)
        if verbose:
            preview = ", ".join(ram_tags[:12])
            suffix = " ..." if len(ram_tags) > 12 else ""
            print(f"RAM tags ({len(ram_tags)}): {preview}{suffix}")

    if box_source == "question":
        merged = question_tags
    else:
        merged = _merge_grounding_tags(question_tags, ram_tags)

    if not merged:
        if verbose:
            print("No grounding tags found — fallback to 'object'")
        merged = ["object"]

    grounding_prompt = _format_tags_for_dino(merged)
    if verbose:
        mode_label = {
            "question": "spaCy question nouns",
            "ram": "spaCy ∪ RAM",
            "hybrid": "spaCy ∪ RAM",
        }[box_source]
        print(f"Grounding prompt ({mode_label}): {grounding_prompt}")
    return grounding_prompt


def resolve_region_boxes( pil_image, *, manual_boxes=None, grounding_dino=None, ram_proposer=None, text_prompt=None, question=None, frozen_qwen=None, qwen_tokenizer=None, config=None, max_dino_boxes=None, box_source=None, verbose=True, ):
    """
    Resolve region boxes for inference.

    Priority: manual_boxes (raw pixel coords) > automatic proposals.

    box_source (defaults to config.box_source or 'hybrid'):
      - 'hybrid' / 'ram': spaCy question nouns ∪ RAM++ tags → DINO
      - 'question': spaCy question nouns only → DINO

    text_prompt, when set, bypasses automatic tag building.

    Returns (boxes_336, display_boxes_cpu, scores_or_none, labels, grounding_prompt).
    """
    device = config.device
    grounding_prompt = text_prompt
    if box_source is None:
        box_source = getattr(config, "box_source", "hybrid")

    if manual_boxes:
        boxes_336 = manual_boxes_to_336(manual_boxes, pil_image.size[0], pil_image.size[1], device)
        labels = [f"manual{i}" for i in range(len(manual_boxes))]
        if verbose:
            print(f"Using {len(manual_boxes)} manual box(es) in original pixel space")
        return boxes_336, boxes_336.cpu(), None, labels, None

    if grounding_dino is None:
        raise ValueError(
            "Provide manual_boxes (list of [x1,y1,x2,y2] in original pixels) "
            "or pass grounding_dino for automatic proposals"
        )

    if grounding_prompt is None:
        box_source = str(box_source).lower().strip()
        if box_source in {"ram", "hybrid"} and ram_proposer is None:
            raise ValueError(f"box_source={box_source!r} requires RAM++")
        if box_source == "question" and not question:
            raise ValueError("box_source='question' requires a question")
        grounding_prompt = _build_automatic_grounding_prompt(
            pil_image,
            question=question,
            ram_proposer=ram_proposer,
            config=config,
            box_source=box_source,
            verbose=verbose,
        )

    max_boxes = config.dino_max_boxes if max_dino_boxes is None else max_dino_boxes
    per_tag = getattr(config, 'dino_per_tag', True)
    tags_for_dino = (
        [t.strip() for t in grounding_prompt.split(' . ') if t.strip()]
        if grounding_prompt else []
    )
    if per_tag and len(tags_for_dino) > 1:
        max_per_tag = getattr(config, 'dino_max_boxes_per_tag', 1)
        if verbose:
            print(f"Per-tag DINO mode: {len(tags_for_dino)} tags")
        boxes, scores, phrases = grounding_dino.propose_per_tags(
            pil_image,
            tags_for_dino,
            max_boxes_per_tag=max_per_tag,
            max_total_boxes=max_boxes,
            verbose=verbose,
        )
    else:
        boxes, scores, phrases = grounding_dino.propose(
            pil_image, grounding_prompt, max_boxes=max_boxes
        )
    if verbose:
        print(f"Grounding DINO proposed {boxes.shape[0]} boxes for '{grounding_prompt}'")
        for i, (score, phrase) in enumerate(zip(scores.tolist(), phrases)):
            coords = [round(v, 1) for v in boxes[i].tolist()]
            print(f"  [{i}] '{phrase}' score={score:.3f} box={coords}")
    return boxes.to(device), boxes, scores, list(phrases), grounding_prompt

DEFAULT_REGION_PROMPT = (
    "Answer in one word or a short phrase.\n"
    "What is in this region?"
)

@torch.no_grad()
def sanity_check_region_alignment( region_extractor, frozen_vit, frozen_qwen, clip_image_processor, qwen_tokenizer, test_image_path, manual_boxes=None, config=None, region_names=None, prompt=DEFAULT_REGION_PROMPT, max_new_tokens=16, repetition_penalty=1.2, show_boxes=False, *, question=None, grounding_dino=None, ram_proposer=None, text_prompt=None, box_source=None, ):
    """
    Sanity check: one short phrase per region box (describe what each region is).

    **Manual boxes:**
        Pass `manual_boxes` as list of [x1, y1, x2, y2] in original image pixels.

    **Automatic boxes:**
        Omit `manual_boxes` and pass `grounding_dino` (and `ram_proposer` for ram/hybrid).
        Tag building follows ``box_source`` (spaCy nouns ± RAM++).

    Returns list of dicts: name, box_px, box_336, response, (score if from DINO).
    """
    region_extractor.eval()
    frozen_vit.eval()
    frozen_qwen.eval()

    pil_image = _load_pil_image(test_image_path)
    img_w, img_h = pil_image.size
    print(f"Image loaded: {img_w}x{img_h}")

    pixel_values = clip_image_processor(images=pil_image, return_tensors="pt")["pixel_values"]
    pixel_values = pixel_values.to(config.device, dtype=config.compute_dtype)
    vit_out = frozen_vit(pixel_values=pixel_values, output_hidden_states=True)

    scores = None
    if manual_boxes:
        boxes_336 = manual_boxes_to_336(manual_boxes, img_w, img_h, config.device)
        labels = region_names or [f"region{i}" for i in range(len(manual_boxes))]
        box_px_list = [list(b) for b in manual_boxes]
    else:
        if grounding_dino is None:
            raise ValueError(
                "Provide manual_boxes, or pass grounding_dino + ram_proposer "
                "for automatic RAM->DINO box proposals"
            )
        if ram_proposer is None and text_prompt is None:
            raise ValueError(
                "Automatic mode needs ram_proposer (RAM->DINO) or an explicit text_prompt"
            )
        # Default: RAM all tags -> DINO (no question). Hybrid only if question given.
        if box_source is None:
            box_source = getattr(config, "box_source", "hybrid")
        boxes_336, display_boxes, scores, default_labels, grounding_prompt = resolve_region_boxes(
            pil_image,
            manual_boxes=None,
            grounding_dino=grounding_dino,
            ram_proposer=ram_proposer,
            text_prompt=text_prompt,
            question=question,
            frozen_qwen=frozen_qwen,
            qwen_tokenizer=qwen_tokenizer,
            config=config,
            box_source=box_source,
        )
        if boxes_336.shape[0] == 0:
            raise ValueError("No region boxes resolved")
        labels = region_names or default_labels
        scale_x, scale_y = img_w / 336.0, img_h / 336.0
        box_px_list = [
            [b[0] * scale_x, b[1] * scale_y, b[2] * scale_x, b[3] * scale_y]
            for b in boxes_336.cpu().tolist()
        ]
        if show_boxes:
            _draw_inference_boxes(pil_image, display_boxes, scores, labels)

    if manual_boxes and show_boxes:
        _draw_inference_boxes(pil_image, boxes_336.cpu(), None, labels)

    results = []
    n_boxes = boxes_336.shape[0]
    for i in range(n_boxes):
        name = labels[i] if i < len(labels) else (region_names[i] if region_names and i < len(region_names) else f"region{i}")
        region_tokens = region_extractor(vit_out.hidden_states, boxes_336[i:i + 1])
        response = _generate_from_region_tokens(
            region_tokens,
            frozen_qwen,
            qwen_tokenizer,
            config,
            text=prompt,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            truncate=True,
        )
        box_336 = boxes_336[i].tolist()
        box_px = box_px_list[i]
        score_str = f"  score={scores[i]:.3f}" if scores is not None and i < len(scores) else ""
        print(f"\n=== {name}{score_str} ===")
        print(f"  pixels: {[round(v, 1) for v in box_px]}  ->  336-space: {[round(v, 1) for v in box_336]}")
        print(f"Response : {response}")
        entry = {
            "name": name,
            "box_px": box_px,
            "box_336": box_336,
            "response": response,
        }
        if scores is not None and i < len(scores):
            entry["score"] = float(scores[i])
        results.append(entry)
    return results


@torch.no_grad()
def region_projection_a_inference( image_path_or_url, question, frozen_vit, region_extractor, frozen_qwen, qwen_tokenizer, clip_image_processor, config, manual_boxes=None, grounding_dino=None, ram_proposer=None, text_prompt=None, region_labels=None, format_prompt="Answer in one word or a short phrase.", show_boxes=True, max_new_tokens=32, repetition_penalty=1.2, max_dino_boxes=None, box_source=None, ):
    """
    Region-only (Projection-A) inference: many boxes + one question -> one answer.

    Automatic boxes follow ``box_source`` (spaCy nouns ± RAM++ tags).
    """
    region_extractor.eval()
    frozen_vit.eval()
    frozen_qwen.eval()

    if manual_boxes is not None and len(manual_boxes) == 0:
        manual_boxes = None

    pil_image = _load_pil_image(image_path_or_url)
    img_w, img_h = pil_image.size
    print(f"Image loaded: {img_w}x{img_h}")

    pixel_values = clip_image_processor(images=pil_image, return_tensors="pt")["pixel_values"]
    pixel_values = pixel_values.to(config.device, dtype=config.compute_dtype)
    vit_out = frozen_vit(pixel_values=pixel_values, output_hidden_states=True)

    boxes_336, display_boxes, scores, default_labels, grounding_prompt = resolve_region_boxes(
        pil_image,
        manual_boxes=manual_boxes,
        grounding_dino=grounding_dino,
        ram_proposer=ram_proposer,
        text_prompt=text_prompt,
        question=question,
        frozen_qwen=frozen_qwen,
        qwen_tokenizer=qwen_tokenizer,
        config=config,
        max_dino_boxes=max_dino_boxes,
        box_source=box_source,
    )
    labels = region_labels or default_labels

    if boxes_336.shape[0] == 0:
        raise ValueError("No region boxes resolved — pass manual_boxes or grounding_dino")

    if show_boxes:
        _draw_inference_boxes(pil_image, display_boxes, scores, labels)

    assembled_embeds = []
    assembled_attn = []
    header_tokens = qwen_tokenizer("Detected visual regions:\n", return_tensors="pt").to(config.device)
    assembled_embeds.append(frozen_qwen.get_input_embeddings()(header_tokens["input_ids"]))
    assembled_attn.append(header_tokens["attention_mask"])

    for i in range(boxes_336.shape[0]):
        lbl_tokens = qwen_tokenizer(f"Region {i}: ", return_tensors="pt").to(config.device)
        assembled_embeds.append(frozen_qwen.get_input_embeddings()(lbl_tokens["input_ids"]))
        assembled_attn.append(lbl_tokens["attention_mask"])
        rt = region_extractor(vit_out.hidden_states, boxes_336[i:i + 1])
        assembled_embeds.append(rt)
        assembled_attn.append(torch.ones(1, rt.shape[1], dtype=torch.long, device=config.device))
        nl_tokens = qwen_tokenizer("\n", return_tensors="pt").to(config.device)
        assembled_embeds.append(frozen_qwen.get_input_embeddings()(nl_tokens["input_ids"]))
        assembled_attn.append(nl_tokens["attention_mask"])

    footer = (
        f"\n{format_prompt}\n\nQuestion: {question}\nAnswer:\n"
        if format_prompt else f"\nQuestion: {question}\nAnswer:\n"
    )
    q_tokens = qwen_tokenizer(footer, return_tensors="pt").to(config.device)
    assembled_embeds.append(frozen_qwen.get_input_embeddings()(q_tokens["input_ids"]))
    assembled_attn.append(q_tokens["attention_mask"])

    input_embeds = torch.cat(assembled_embeds, dim=1)
    full_attn = torch.cat(assembled_attn, dim=1)
    output_ids = frozen_qwen.generate(
        inputs_embeds=input_embeds,
        attention_mask=full_attn,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        repetition_penalty=repetition_penalty,
        pad_token_id=qwen_tokenizer.pad_token_id,
        eos_token_id=qwen_tokenizer.eos_token_id,
    )
    raw = qwen_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return _truncate_short_answer(raw) if format_prompt else raw


def extract_grounding_prompt_llm(question, frozen_qwen, qwen_tokenizer, config) -> str:
    """Extract Grounding DINO noun phrases from a question using Qwen."""
    messages = [
        {
            "role": "system",
            "content": (
                "You extract the key object noun phrase(s) from a question for object detection. "
                "Respond with ONLY the noun phrase(s), nothing else. "
                "If multiple objects, separate with ' . '."
            ),
        },
        {"role": "user", "content": "How many red cars are there?"},
        {"role": "assistant", "content": "red car"},
        {"role": "user", "content": "Is there a cop in the image?"},
        {"role": "assistant", "content": "cop"},
        {"role": "user", "content": "What color is the ball next to the chair?"},
        {"role": "assistant", "content": "ball . chair"},
        {"role": "user", "content": question},
    ]
    text = qwen_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = qwen_tokenizer(text, return_tensors='pt').to(config.device)
    output_ids = frozen_qwen.generate(
        **inputs,
        max_new_tokens=15,
        do_sample=False,
        repetition_penalty=1.1,
    )
    generated = qwen_tokenizer.decode(
        output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
    )
    extracted = generated.strip().split("\n")[0].strip()
    return extracted if extracted else "object"


@torch.no_grad()
def boxes_336_to_px(boxes_336, img_w, img_h):
    sx, sy = img_w / 336.0, img_h / 336.0
    return [[x1 * sx, y1 * sy, x2 * sx, y2 * sy] for x1, y1, x2, y2 in boxes_336.cpu().tolist()]


@torch.no_grad()
def run_concat576_inference( image_path_or_url, question, *, frozen_vit, projection_head_b, region_extractor, qwen, qwen_tokenizer, clip_image_processor, config, boxes_px=None, format_prompt=None, max_new_tokens=32, include_regions=True, ):
    """
    concat576 inference.

    Prefix: [576 globals] + optional [K×16 regions] + format/Q.
    include_regions=False => global576-only ablation.
    """
    from .stage3_dataset import DEFAULT_FORMAT_PROMPT
    from .stage3_train import build_stage3_training_inputs, scale_boxes_to_336

    format_prompt = format_prompt or DEFAULT_FORMAT_PROMPT
    device = config.device
    pil_image = _load_pil_image(image_path_or_url)
    img_w, img_h = pil_image.size
    pixel_values = clip_image_processor(images=pil_image, return_tensors="pt")["pixel_values"]
    pixel_values = pixel_values.to(device, dtype=config.compute_dtype)
    vit_out = frozen_vit(pixel_values=pixel_values, output_hidden_states=True)
    max_boxes = getattr(config, "max_boxes_per_image", 20)
    boxes_336 = scale_boxes_to_336((boxes_px or [])[:max_boxes], img_w, img_h)

    input_embeds, attention_mask, _, aux = build_stage3_training_inputs(
        vit_out=vit_out,
        boxes_336=boxes_336,
        question=question,
        answer="",
        format_prompt=format_prompt,
        projection_head_b=projection_head_b,
        region_extractor=region_extractor,
        qwen=qwen,
        qwen_tokenizer=qwen_tokenizer,
        config=config,
        max_answer_tokens=1,
        include_regions=include_regions,
    )
    input_embeds = input_embeds[:, :-1, :]
    attention_mask = attention_mask[:, :-1]

    output_ids = qwen.generate(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=qwen_tokenizer.pad_token_id,
        eos_token_id=[qwen_tokenizer.eos_token_id, qwen_tokenizer.encode(".")[0]],
    )
    pred = qwen_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return pred, aux


@torch.no_grad()
def run_combined_inference( image_path_or_url, question, frozen_vit, projection_head_b, region_extractor, frozen_qwen, qwen_tokenizer, clip_image_processor, config, manual_boxes=None, grounding_dino=None, ram_proposer=None, text_prompt=None, region_labels=None, format_prompt=None, show_boxes=False, max_new_tokens=64, verbose=True, box_source=None, include_regions=True, ):
    """
    concat576 inference after resolving region boxes.

    Automatic proposals follow ``box_source``:
      hybrid/ram → spaCy question nouns ∪ RAM++ tags; question → spaCy nouns only.
    manual_boxes: optional list of [x1, y1, x2, y2] in original image pixels.
    """
    from .stage3_dataset import DEFAULT_FORMAT_PROMPT

    format_prompt = format_prompt or DEFAULT_FORMAT_PROMPT
    frozen_vit.eval()
    projection_head_b.eval()
    region_extractor.eval()
    frozen_qwen.eval()

    pil_image = _load_pil_image(image_path_or_url)
    img_w, img_h = pil_image.size
    if verbose:
        print(f"Image loaded: {img_w}x{img_h}")

    if box_source is None:
        box_source = getattr(config, "box_source", "hybrid")

    max_boxes = getattr(config, "max_boxes_per_image", 20)
    boxes_336, display_boxes, scores, default_labels, grounding_prompt = resolve_region_boxes(
        pil_image,
        manual_boxes=manual_boxes,
        grounding_dino=grounding_dino,
        ram_proposer=ram_proposer,
        text_prompt=text_prompt,
        question=question,
        frozen_qwen=frozen_qwen,
        qwen_tokenizer=qwen_tokenizer,
        config=config,
        max_dino_boxes=max_boxes,
        box_source=box_source,
        verbose=verbose,
    )
    labels = region_labels or default_labels

    if show_boxes:
        _draw_inference_boxes(pil_image, display_boxes, scores, labels)

    boxes_px = boxes_336_to_px(boxes_336, img_w, img_h) if boxes_336.numel() else []

    answer, aux = run_concat576_inference(
        image_path_or_url,
        question,
        frozen_vit=frozen_vit,
        projection_head_b=projection_head_b,
        region_extractor=region_extractor,
        qwen=frozen_qwen,
        qwen_tokenizer=qwen_tokenizer,
        clip_image_processor=clip_image_processor,
        config=config,
        boxes_px=boxes_px,
        format_prompt=format_prompt,
        max_new_tokens=max_new_tokens,
        include_regions=include_regions,
    )

    return {
        "answer": answer,
        "grounding_prompt": grounding_prompt,
        "boxes_336": boxes_336.cpu(),
        "boxes_px": boxes_px,
        "labels": labels,
        "global_token_count": aux.get("num_global_tokens", 576),
        "num_region_tokens": aux.get("num_region_tokens", 0),
        "num_regions": len(boxes_px),
    }


@torch.no_grad()
def run_concat576_ram_dino_inference( image_path_or_url, question, *, frozen_vit, projection_head_b, region_extractor, qwen, qwen_tokenizer, clip_image_processor, config, grounding_dino, ram_proposer, format_prompt=None, max_new_tokens=64, manual_boxes=None, text_prompt=None, show_boxes=False, verbose=True, box_source=None, ):
    """End-to-end spaCy/RAM tag building → Grounding DINO → concat576 inference."""
    from .stage3_dataset import DEFAULT_FORMAT_PROMPT

    format_prompt = format_prompt or DEFAULT_FORMAT_PROMPT
    pil_image = _load_pil_image(image_path_or_url)
    img_w, img_h = pil_image.size
    max_boxes = getattr(config, "max_boxes_per_image", 20)

    boxes_336, display_boxes, scores, labels, grounding_prompt = resolve_region_boxes(
        pil_image,
        manual_boxes=manual_boxes,
        grounding_dino=grounding_dino,
        ram_proposer=ram_proposer,
        text_prompt=text_prompt,
        question=question,
        frozen_qwen=qwen,
        qwen_tokenizer=qwen_tokenizer,
        config=config,
        max_dino_boxes=max_boxes,
        box_source=box_source,
        verbose=verbose,
    )

    if show_boxes and boxes_336.shape[0] > 0:
        _draw_inference_boxes(pil_image, display_boxes, scores, labels)

    boxes_px = boxes_336_to_px(boxes_336, img_w, img_h) if boxes_336.numel() else []
    answer, aux = run_concat576_inference(
        image_path_or_url,
        question,
        frozen_vit=frozen_vit,
        projection_head_b=projection_head_b,
        region_extractor=region_extractor,
        qwen=qwen,
        qwen_tokenizer=qwen_tokenizer,
        clip_image_processor=clip_image_processor,
        config=config,
        boxes_px=boxes_px,
        format_prompt=format_prompt,
        max_new_tokens=max_new_tokens,
        include_regions=True,
    )

    return {
        "answer": answer,
        "grounding_prompt": grounding_prompt,
        "boxes_px": boxes_px,
        "labels": labels,
        "scores": scores.cpu().tolist() if scores is not None else None,
        "num_global_tokens": aux.get("num_global_tokens", 576),
        "num_region_tokens": aux.get("num_region_tokens", 0),
    }


def interactive_combined_inference( frozen_vit, projection_head_b, region_extractor, frozen_qwen, qwen_tokenizer, clip_image_processor, config, grounding_dino=None, ram_proposer=None, images_root=None, box_source=None, ):
    """Interactive loop for concat576 global + region inference."""
    if images_root is None:
        images_root = DEFAULT_TEST_IMAGES_ROOT

    while True:
        image_input = input("Image path or URL (or 'quit'): ").strip()
        if image_input.lower() == 'quit':
            break

        if image_input.startswith('http'):
            image_path = image_input
        elif Path(image_input).is_file():
            image_path = image_input
        else:
            image_path = str(images_root / image_input)

        box_str = input(
            "Manual boxes as x1,y1,x2,y2 (semicolon-separated for multiple), "
            "or press Enter for Grounding DINO: "
        ).strip()
        manual_boxes = parse_manual_boxes(box_str) if box_str else None

        text_prompt = None
        if not manual_boxes:
            custom_prompt = input(
                "Grounding prompt (Enter to auto-extract from question): "
            ).strip()
            text_prompt = custom_prompt or None

        question = input("Question (or 'quit'): ").strip()
        if question.lower() == 'quit':
            break
        if not question:
            continue

        result = run_combined_inference(
            image_path, question,
            frozen_vit, projection_head_b, region_extractor, frozen_qwen,
            qwen_tokenizer, clip_image_processor, config,
            manual_boxes=manual_boxes,
            grounding_dino=grounding_dino,
            ram_proposer=ram_proposer,
            text_prompt=text_prompt,
            show_boxes=True,
            box_source=box_source,
        )
        print(f"Answer: {result['answer']}\n")


def interactive_projection_a_test(frozen_vit, region_extractor, frozen_qwen, qwen_tokenizer, clip_image_processor, config, grounding_dino=None, ram_proposer=None, box_source=None):
    """Interactive Projection-A test with manual boxes or Grounding DINO."""
    while True:
        image_path = input("Image path or URL (or 'quit'): ").strip()
        if image_path.lower() == 'quit':
            break

        box_str = input(
            "Manual boxes as x1,y1,x2,y2 (semicolon-separated), "
            "or press Enter for Grounding DINO: "
        ).strip()
        manual_boxes = parse_manual_boxes(box_str) if box_str else None

        text_prompt = None
        if not manual_boxes:
            custom_prompt = input(
                "Grounding prompt (Enter to auto-extract from question): "
            ).strip()
            text_prompt = custom_prompt or None

        while True:
            question = input(
                "Description/question for this region "
                "(or 'box' to change boxes, 'quit' to exit): "
            ).strip()
            if question.lower() == 'quit':
                return
            if question.lower() == 'box':
                break
            if not question:
                continue

            answer = region_projection_a_inference(
                image_path, question,
                frozen_vit, region_extractor, frozen_qwen, qwen_tokenizer,
                clip_image_processor, config,
                manual_boxes=manual_boxes,
                grounding_dino=grounding_dino,
                ram_proposer=ram_proposer,
                text_prompt=text_prompt,
                show_boxes=False,
                box_source=box_source,
            )
            print(f"Answer: {answer}\n")